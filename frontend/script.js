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
            episodicUploadStatus: null,
            semanticUploadStatus: null,
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
                // 使用 Vue 响应式 API 创建消息对象，确保实时更新
                let assistantMessage = Vue.reactive({
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
                    relationshipsExpanded: false,
                    retrievalBubble: {
                        steps: [],
                        summary: '',
                        isComplete: false
                    },
                    retrievalBubbleExpanded: true,
                    refsExpanded: false
                });

                this.messages.push(assistantMessage);

                let buffer = '';
                console.log('[RAG DEBUG] Starting SSE read loop');
                while (true) {
                    const { done, value } = await reader.read();
                    console.log('[RAG DEBUG] Reader read:', done ? 'done' : `bytes: ${value?.length || 0}`);
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    console.log('[RAG DEBUG] Buffer after decode:', buffer.length, 'chars');
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        console.log('[RAG DEBUG] Processing line:', line.substring(0, 50), '...');
                        if (line.startsWith('data: ')) {
                            try {
                                const payload = line.slice(6).trim();
                                console.log('[RAG DEBUG] Payload:', payload.substring(0, 100), '...');
                                if (payload === '[DONE]') {
                                    assistantMessage.streaming = false;
                                    if (assistantMessage.retrievalBubble) {
                                        assistantMessage.retrievalBubble.isComplete = true;
                                        assistantMessage.retrievalBubble.steps.forEach(s => s.isComplete = true);
                                    }
                                    continue;
                                }
                                const data = JSON.parse(payload);
                                console.log('[RAG DEBUG] Parsed data type:', data.type);
                                
                                if (data.type === 'content') {
                                    const chunk = data.data ?? data.content ?? '';
                                    assistantMessage.content += chunk;
                                    await this.yieldToBrowser();
                                } else if (data.type === 'rag_step') {
                                    console.log('[RAG DEBUG] Received rag_step event:', data);
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
                                        evidence_sources: Array.isArray(step.evidence_sources) ? step.evidence_sources : [],
                                        docs: step.docs || [],
                                        extra: step.extra || {}
                                    });

                                    this.buildRetrievalBubble(assistantMessage, step);
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

        buildRetrievalBubble(message, step) {
            if (!message.retrievalBubble) {
                message.retrievalBubble = { steps: [], summary: '', isComplete: false };
            }
            const bubble = message.retrievalBubble;
            const stepIndex = step.step_index ?? bubble.steps.length + 1;

            const layer = step.layer || '';
            const icon = step.icon || '•';
            let label = step.label || '处理中';
            let description = step.detail || '';
            const details = [];

            if (layer === 'entity') {
                const items = Array.isArray(step.items) ? step.items : [];
                if (items.length > 0) {
                    label = `🔍 提取实体 (${items.length} 个)`;
                    description = '从用户输入中识别医疗实体';
                    details.push({ label: '实体类型', value: items.slice(0, 5).join('、') + (items.length > 5 ? '...' : '') });
                }
            } else if (layer === 'relation') {
                const relation = step.relation || '';
                label = `🔗 关系匹配`;
                description = '查找实体之间的关系';
                if (relation) {
                    details.push({ label: '关系模板', value: relation });
                }
            } else if (layer === 'evidence') {
                const hits = Array.isArray(step.hits) ? step.hits : [];
                label = `📄 证据检索`;
                description = '获取支撑答案的证据片段';
                if (hits.length > 0) {
                    details.push({ label: '检索结果', value: `${hits.length} 个相关片段` });
                }
            } else if (stepIndex === 1 || label.includes('query') || label.includes('Query')) {
                label = `✨ 分析问题`;
                description = step.detail || '正在理解您的问题';
            } else if (label.includes('rewrite') || label.includes('Rewrite')) {
                label = `🔄 优化查询`;
                description = step.detail || '改进搜索策略';
            } else if (label.includes('search') || label.includes('Search') || label.includes('retriev')) {
                label = `🔍 语义检索`;
                description = step.detail || '在知识库中搜索相关内容';
            } else if (label.includes('rerank') || label.includes('Rerank') || label.includes('rank')) {
                label = `📊 重排序`;
                description = step.detail || '对检索结果进行相关性排序';
            } else if (label.includes('grade') || label.includes('Grade')) {
                label = `⭐ 质量评估`;
                description = step.detail || '评估文档相关性';
            } else if (label.includes('merge') || label.includes('Merge')) {
                label = `🔗 文档合并`;
                description = step.detail || '合并相关文档片段';
            }

            const stepDocs = step.docs || (step.extra && step.extra.docs) || [];
            console.log('[RAG DEBUG] stepDocs length:', stepDocs.length, 'step.docs:', step.docs, 'step.extra:', step.extra);
            if (stepDocs.length > 0) {
                stepDocs.forEach((doc, idx) => {
                    const filename = doc.filename || '未知来源';
                    const page = doc.page_number || '?';
                    const score = doc.score ? doc.score.toFixed(2) : '-';
                    const text = (doc.text || '').substring(0, 100);
                    details.push({
                        label: `文档 ${idx + 1}`,
                        value: `${filename} (P${page}) | 相关度: ${score}`,
                        preview: text + (text.length >= 100 ? '...' : '')
                    });
                });
                console.log('[RAG DEBUG] Added details:', details);
            }

            const existingIndex = bubble.steps.findIndex(s => s.step_index === stepIndex);
            const newStep = {
                step_index: stepIndex,
                label,
                description,
                details,
                isActive: !bubble.isComplete,
                isComplete: bubble.isComplete
            };

            if (existingIndex >= 0) {
                bubble.steps.splice(existingIndex, 1, newStep);
            } else {
                bubble.steps.push(newStep);
            }

            bubble.steps.sort((a, b) => a.step_index - b.step_index);
            for (let i = 0; i < bubble.steps.length; i++) {
                if (i < bubble.steps.length - 1) {
                    bubble.steps[i].isComplete = true;
                    bubble.steps[i].isActive = false;
                } else {
                    bubble.steps[i].isActive = !bubble.isComplete;
                }
            }

            if (bubble.steps.length > 0 && !bubble.isComplete) {
                const completedCount = bubble.steps.filter(s => s.isComplete).length;
                const totalCount = bubble.steps.length;
                if (completedCount > 0) {
                    bubble.summary = `已完成 ${completedCount}/${totalCount} 个检索步骤，正在获取答案...`;
                }
            }
        },

        saveRecentQueries() {
            localStorage.setItem('recentQueries', JSON.stringify(this.recentQueries));
        },

        loadRecentQueries() {
            const saved = localStorage.getItem('recentQueries');
            if (saved) {
                this.recentQueries = JSON.parse(saved);
            }
        },

        async handleEpisodicUpload(event) {
            const file = event.target.files[0];
            if (!file) return;

            this.episodicUploadStatus = {
                isComplete: false,
                message: '正在上传文件...',
                jobId: null
            };

            try {
                const formData = new FormData();
                formData.append('file', file);

                const response = await this.authFetch('/documents/episodic/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const data = await response.json().catch(() => ({}));
                    throw new Error(data.detail || '上传失败');
                }

                const data = await response.json();
                this.episodicUploadStatus = {
                    isComplete: false,
                    message: data.message || '文件已上传，正在处理...',
                    jobId: data.job_id
                };

                // 轮询任务状态
                await this.pollJobStatus(data.job_id, 'episodic');
            } catch (error) {
                this.episodicUploadStatus = {
                    isComplete: false,
                    message: '上传失败：' + error.message,
                    jobId: null
                };
            }

            // 清空 input
            event.target.value = '';
        },

        async handleSemanticUpload(event) {
            const file = event.target.files[0];
            if (!file) return;

            this.semanticUploadStatus = {
                isComplete: false,
                message: '正在上传文件...',
                jobId: null
            };

            try {
                const formData = new FormData();
                formData.append('file', file);

                const response = await this.authFetch('/documents/semantic/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const data = await response.json().catch(() => ({}));
                    throw new Error(data.detail || '上传失败');
                }

                const data = await response.json();
                this.semanticUploadStatus = {
                    isComplete: false,
                    message: data.message || '文件已上传，正在处理...',
                    jobId: data.job_id
                };

                // 轮询任务状态
                await this.pollJobStatus(data.job_id, 'semantic');
            } catch (error) {
                this.semanticUploadStatus = {
                    isComplete: false,
                    message: '上传失败：' + error.message,
                    jobId: null
                };
            }

            // 清空 input
            event.target.value = '';
        },

        async pollJobStatus(jobId, type) {
            const maxAttempts = 60;
            const interval = 2000;

            for (let i = 0; i < maxAttempts; i++) {
                try {
                    const response = await this.authFetch(`/documents/upload/jobs/${encodeURIComponent(jobId)}`);
                    if (!response.ok) {
                        await new Promise(r => setTimeout(r, interval));
                        continue;
                    }

                    const job = await response.json();
                    
                    const status = type === 'episodic' ? this.episodicUploadStatus : this.semanticUploadStatus;
                    
                    if (job) {
                        const steps = job.steps || {};
                        const currentStep = Object.keys(steps).find(s => steps[s] && steps[s].status === 'running');
                        
                        if (currentStep) {
                            status.message = steps[currentStep].message || '正在处理...';
                        }

                        if (job.is_complete) {
                            status.isComplete = true;
                            status.message = job.final_message || '处理完成！';
                            return;
                        }

                        if (job.failed) {
                            status.isComplete = false;
                            status.message = '处理失败：' + (job.error_message || '未知错误');
                            return;
                        }
                    }
                } catch (error) {
                    console.error('Poll error:', error);
                }

                await new Promise(r => setTimeout(r, interval));
            }

            const status = type === 'episodic' ? this.episodicUploadStatus : this.semanticUploadStatus;
            status.message = '处理超时，请稍后检查';
        }
    }
}).mount('#app');
