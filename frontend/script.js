const { createApp } = Vue;

createApp({
    data() {
        const defaultApiBase = window.location.protocol === 'file:'
            ? 'http://localhost:8000'
            : window.location.origin;
        return {
            messages: [],
            sessions: [],
            currentSessionId: 'default_session',
            userInput: '',
            isLoading: false,
            token: localStorage.getItem('accessToken') || '',
            currentUser: null,
            authMode: 'login',
            authForm: {
                username: '',
                password: '',
                role: 'user',
                admin_code: ''
            },
            authLoading: false,
            showUserMenu: false,
            recentQueries: [],
            apiBase: window.__API_BASE__ || defaultApiBase,
            quickSymptoms: [
                '喉咙痛',
                '头痛',
                '发烧',
                '咳嗽',
                '腹泻',
                '疲劳',
                '肌肉酸痛',
                '失眠'
            ]
        };
    },
    computed: {
        isAuthenticated() {
            return !!this.token && !!this.currentUser;
        }
    },
    async mounted() {
        this.configureMarked();
        if (this.token) {
            try {
                await this.fetchMe();
                await this.loadSessions();
            } catch (_) {
                this.handleLogout();
            }
        }
        this.loadRecentQueries();
    },
    methods: {
        configureMarked() {
            marked.setOptions({
                highlight: function(code, lang) {
                    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                    return hljs.highlight(code, { language }).value;
                },
                langPrefix: 'hljs language-',
                breaks: true,
                gfm: true
            });
        },

        parseMarkdown(text) {
            return marked.parse(text);
        },

        apiUrl(path) {
            return new URL(path, this.apiBase).toString();
        },

        authHeaders(extra = {}) {
            const headers = { ...extra };
            if (this.token) {
                headers.Authorization = `Bearer ${this.token}`;
            }
            return headers;
        },

        async authFetch(url, options = {}) {
            const opts = { ...options };
            opts.headers = this.authHeaders(opts.headers || {});
            const response = await fetch(this.apiUrl(url), opts);
            if (response.status === 401) {
                this.handleLogout();
                throw new Error('登录已过期，请重新登录');
            }
            return response;
        },

        async fetchMe() {
            const response = await this.authFetch('/auth/me');
            if (!response.ok) throw new Error('认证失败');
            this.currentUser = await response.json();
        },

        async handleAuthSubmit() {
            if (this.authLoading) return;
            const username = this.authForm.username.trim();
            const password = this.authForm.password.trim();
            if (!username || !password) {
                alert('用户名和密码不能为空');
                return;
            }

            this.authLoading = true;
            try {
                const endpoint = this.authMode === 'login' ? '/auth/login' : '/auth/register';
                const payload = {
                    username,
                    password
                };
                if (this.authMode === 'register') {
                    payload.role = this.authForm.role;
                    payload.admin_code = this.authForm.admin_code || null;
                }

                const response = await fetch(this.apiUrl(endpoint), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(data.detail || '认证失败');
                }

                this.token = data.access_token;
                this.currentUser = { username: data.username, role: data.role };
                localStorage.setItem('accessToken', this.token);
                this.authForm.password = '';
                this.authForm.admin_code = '';
                this.messages = [];
                this.currentSessionId = 'default_session';
                await this.loadSessions();
            } catch (error) {
                alert(error.message);
            } finally {
                this.authLoading = false;
            }
        },

        handleLogout() {
            this.token = '';
            this.currentUser = null;
            this.messages = [];
            this.sessions = [];
            this.currentSessionId = 'default_session';
            localStorage.removeItem('accessToken');
            this.showUserMenu = false;
        },

        makeSessionId() {
            return `session-${Date.now()}`;
        },

        async loadSessions() {
            if (!this.isAuthenticated) return;
            try {
                const response = await this.authFetch('/sessions');
                if (!response.ok) throw new Error('加载会话失败');
                const data = await response.json();
                this.sessions = data.sessions || [];
            } catch (error) {
                console.error('loadSessions error:', error);
            }
        },

        async openSession(sessionId) {
            if (!sessionId || sessionId === this.currentSessionId) return;
            try {
                const response = await this.authFetch(`/sessions/${encodeURIComponent(sessionId)}`);
                if (!response.ok) throw new Error('加载会话失败');
                const data = await response.json();
                this.currentSessionId = sessionId;
                this.messages = (data.messages || []).map((msg) => {
                    const ragTrace = msg.rag_trace || null;
                    return {
                        role: msg.type === 'human' ? 'user' : 'assistant',
                        content: msg.content || '',
                        ragSteps: [],
                        ragTrace,
                        retrievedDocs: this.buildRetrievedDocsFromTrace(ragTrace),
                        searchResults: [],
                        references: [],
                        relationships: [],
                        streaming: false,
                        processExpanded: false,
                        resultsExpanded: false,
                        referencesExpanded: false,
                        relationshipsExpanded: false
                    };
                });
                this.showUserMenu = false;
                await this.loadSessions();
                this.scrollToBottom();
            } catch (error) {
                alert(error.message);
            }
        },

        async createNewSession() {
            this.currentSessionId = this.makeSessionId();
            this.messages = [];
            this.showUserMenu = false;
            await this.loadSessions();
        },

        async deleteSession(sessionId) {
            if (!sessionId) return;
            if (!confirm(`确定删除会话 ${sessionId} 吗？此操作不可恢复。`)) return;
            try {
                const response = await this.authFetch(`/sessions/${encodeURIComponent(sessionId)}`, {
                    method: 'DELETE'
                });
                if (!response.ok) {
                    const data = await response.json().catch(() => ({}));
                    throw new Error(data.detail || '删除会话失败');
                }
                if (sessionId === this.currentSessionId) {
                    this.currentSessionId = this.makeSessionId();
                    this.messages = [];
                }
                await this.loadSessions();
            } catch (error) {
                alert(error.message);
            }
        },

        addSymptom(symptom) {
            if (this.userInput) {
                this.userInput += '、' + symptom;
            } else {
                this.userInput = symptom;
            }
        },

        async sendMessage() {
            if (!this.isAuthenticated) {
                alert('请先登录');
                return;
            }

            const text = this.userInput.trim();
            if (!text || this.isLoading) return;

            // 保存查询历史
            if (!this.recentQueries.includes(text)) {
                this.recentQueries.unshift(text);
                if (this.recentQueries.length > 20) {
                    this.recentQueries.pop();
                }
                this.saveRecentQueries();
            }

            // 添加用户消息
            this.messages.push({
                role: 'user',
                content: text
            });

            this.userInput = '';
            this.isLoading = true;

            try {
                const response = await this.authFetch('/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text, session_id: this.currentSessionId })
                });

                if (!response.ok) {
                    throw new Error('请求失败');
                }

                // 读取流式响应
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let assistantMessage = {
                    role: 'assistant',
                    content: '',
                    ragSteps: [],
                    ragTrace: null,
                    retrievedDocs: [],
                    searchResults: [],
                    references: [],
                    relationships: [],
                    streaming: true,
                    processExpanded: false,
                    resultsExpanded: false,
                    referencesExpanded: false,
                    relationshipsExpanded: false
                };

                this.messages.push(assistantMessage);

                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const payload = line.slice(6).trim();
                                if (payload === '[DONE]') {
                                    assistantMessage.streaming = false;
                                    continue;
                                }
                                const data = JSON.parse(payload);
                                
                                if (data.type === 'content') {
                                    const chunk = data.data ?? data.content ?? '';
                                    assistantMessage.content += chunk;
                                    await this.yieldToBrowser();
                                } else if (data.type === 'rag_step') {
                                    const step = data.step || {};
                                    assistantMessage.ragSteps.push({
                                        icon: step.icon || '•',
                                        label: step.label || '处理中',
                                        detail: step.detail || '',
                                        step_index: step.step_index ?? assistantMessage.ragSteps.length + 1,
                                        timestamp: step.timestamp || '',
                                        elapsed_ms: step.elapsed_ms ?? null,
                                        delta_ms: step.delta_ms ?? null,
                                        layer: step.layer || '',
                                        items: Array.isArray(step.items) ? step.items : [],
                                        relation: step.relation || '',
                                        hits: Array.isArray(step.hits) ? step.hits : [],
                                        source_entities: Array.isArray(step.source_entities) ? step.source_entities : [],
                                        evidence_sources: Array.isArray(step.evidence_sources) ? step.evidence_sources : []
                                    });
                                } else if (data.type === 'trace') {
                                    assistantMessage.ragTrace = data.rag_trace ?? null;
                                    this.mergeRetrievedDocsFromTrace(assistantMessage);
                                } else if (data.type === 'retrieved_docs') {
                                    assistantMessage.retrievedDocs.push({
                                        stage: data.stage || 'initial',
                                        query: data.query || '',
                                        docs: Array.isArray(data.docs) ? data.docs : [],
                                        meta: data.meta || {}
                                    });
                                } else if (data.type === 'search_results') {
                                    assistantMessage.searchResults = data.data ?? [];
                                } else if (data.type === 'references') {
                                    assistantMessage.references = data.data ?? [];
                                } else if (data.type === 'relationships') {
                                    assistantMessage.relationships = data.data ?? [];
                                } else if (data.type === 'error') {
                                    const err = data.data ?? data.content ?? '未知错误';
                                    assistantMessage.content += `\n\n⚠️ ${err}`;
                                }
                                
                                // 触发 Vue 更新
                                await this.$nextTick();
                                this.scrollToBottom();
                            } catch (e) {
                                console.error('Parse error:', e);
                            }
                        }
                    }
                }
                assistantMessage.streaming = false;
            } catch (error) {
                console.error('Error:', error);
                this.messages.push({
                    role: 'assistant',
                    content: '抱歉，出现错误：' + error.message
                });
            } finally {
                this.isLoading = false;
                if (this.isAuthenticated) {
                    await this.loadSessions();
                }
                this.scrollToBottom();
            }
        },

        yieldToBrowser() {
            return new Promise((resolve) => {
                if (typeof requestAnimationFrame === 'function') {
                    requestAnimationFrame(() => resolve());
                    return;
                }
                setTimeout(resolve, 0);
            });
        },

        buildRetrievedDocsFromTrace(trace) {
            if (!trace || typeof trace !== 'object') return [];
            const blocks = [];
            const initialDocs = Array.isArray(trace.initial_retrieved_chunks)
                ? trace.initial_retrieved_chunks
                : [];
            const expandedDocs = Array.isArray(trace.expanded_retrieved_chunks)
                ? trace.expanded_retrieved_chunks
                : [];

            if (initialDocs.length > 0) {
                blocks.push({
                    stage: 'initial',
                    query: trace.query || '',
                    docs: initialDocs,
                    meta: this.pickTraceRetrievalMeta(trace)
                });
            }
            if (expandedDocs.length > 0) {
                blocks.push({
                    stage: 'expanded',
                    query: trace.expanded_query || trace.rewrite_query || trace.query || '',
                    docs: expandedDocs,
                    meta: this.pickTraceRetrievalMeta(trace)
                });
            }
            if (blocks.length === 0 && Array.isArray(trace.retrieved_chunks) && trace.retrieved_chunks.length > 0) {
                blocks.push({
                    stage: trace.retrieval_stage || 'retrieved',
                    query: trace.expanded_query || trace.query || '',
                    docs: trace.retrieved_chunks,
                    meta: this.pickTraceRetrievalMeta(trace)
                });
            }
            return blocks;
        },

        mergeRetrievedDocsFromTrace(message) {
            const blocks = this.buildRetrievedDocsFromTrace(message.ragTrace);
            if (!blocks.length) return;
            const existingStages = new Set((message.retrievedDocs || []).map((block) => block.stage));
            const additions = blocks.filter((block) => !existingStages.has(block.stage));
            if (additions.length > 0) {
                message.retrievedDocs = [...(message.retrievedDocs || []), ...additions];
            }
        },

        pickTraceRetrievalMeta(trace) {
            if (!trace || typeof trace !== 'object') return {};
            const keys = [
                'retrieval_mode',
                'candidate_k',
                'leaf_retrieve_level',
                'rerank_enabled',
                'rerank_attempted',
                'rerank_applied',
                'rerank_model',
                'rerank_endpoint',
                'rerank_error',
                'auto_merge_enabled',
                'auto_merge_applied',
                'auto_merge_threshold',
                'auto_merge_replaced_chunks',
                'auto_merge_steps'
            ];
            return keys.reduce((meta, key) => {
                if (trace[key] !== undefined && trace[key] !== null && trace[key] !== '') {
                    meta[key] = trace[key];
                }
                return meta;
            }, {});
        },

        handleClearChat() {
            if (confirm('确定要清空当前对话吗？')) {
                this.messages = [];
                this.userInput = '';
            }
        },

        scrollToBottom() {
            this.$nextTick(() => {
                const container = this.$refs.messagesContainer;
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            });
        },

        getTraceEntries(trace) {
            if (!trace || typeof trace !== 'object') return [];
            const entries = [];
            const order = [
                'tool_used',
                'tool_name',
                'query',
                'expanded_query',
                'rewrite_strategy',
                'rewrite_query',
                'grade_score',
                'grade_route',
                'retrieval_mode',
                'candidate_k',
                'leaf_retrieve_level',
                'rerank_enabled',
                'rerank_attempted',
                'rerank_applied',
                'rerank_model',
                'rerank_endpoint',
                'rerank_error',
                'auto_merge_enabled',
                'auto_merge_applied',
                'auto_merge_threshold',
                'auto_merge_replaced_chunks',
                'auto_merge_steps'
            ];
            for (const key of order) {
                if (trace[key] !== undefined && trace[key] !== null && trace[key] !== '') {
                    entries.push({ key, value: trace[key] });
                }
            }
            const hiddenKeys = new Set([
                'retrieved_chunks',
                'initial_retrieved_chunks',
                'expanded_retrieved_chunks'
            ]);
            const consumed = new Set([...order, ...hiddenKeys]);
            for (const [key, value] of Object.entries(trace)) {
                if (consumed.has(key)) continue;
                if (value !== undefined && value !== null && value !== '') {
                    entries.push({ key, value });
                }
            }
            return entries;
        },

        formatTimestamp(timestamp) {
            if (!timestamp) return '';
            const date = new Date(timestamp);
            if (Number.isNaN(date.getTime())) return String(timestamp);
            return date.toLocaleTimeString('zh-CN', { hour12: false });
        },

        formatMs(value) {
            if (value === null || value === undefined || value === '') return '';
            return `${value} ms`;
        },

        formatJson(value) {
            try {
                return JSON.stringify(value, null, 2);
            } catch (_) {
                return String(value);
            }
        },

        formatStageLabel(stage) {
            if (stage === 'initial') return '初始检索';
            if (stage === 'expanded') return '扩展检索';
            return stage || '检索';
        },

        formatTraceValue(value) {
            if (Array.isArray(value)) {
                return value.length ? `${value.length} 项` : '0 项';
            }
            if (value && typeof value === 'object') {
                return JSON.stringify(value, null, 2);
            }
            return String(value);
        },

        saveRecentQueries() {
            localStorage.setItem('recentQueries', JSON.stringify(this.recentQueries));
        },

        loadRecentQueries() {
            const saved = localStorage.getItem('recentQueries');
            if (saved) {
                this.recentQueries = JSON.parse(saved);
            }
        }
    }
}).mount('#app');
