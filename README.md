# MedChat - 医疗智能问答系统

> 基于大语言模型的医疗领域智能问答系统，融合知识图谱、向量检索与智能 Agent 技术，为用户提供专业、准确的医疗健康咨询服务。

[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/icey1287/SuperMew)

## 🌟 项目简介

MedChat 是一款面向医疗健康领域的智能问答系统，旨在为用户提供专业、可靠的医疗咨询服务。系统融合了**知识图谱检索**、**向量数据库检索**和**智能决策 Agent** 三大核心技术，能够根据用户问题的类型和意图，智能选择最优的检索策略，提供精准、全面的回答。

### 🎯 核心能力

| 能力 | 描述 |
|------|------|
| **智能意图识别** | 自动识别用户问题类型（事实性、解释性、建议性、比较性） |
| **多源检索** | 知识图谱 + 双向量数据库（情景记忆 + 语义记忆） |
| **智能决策** | Agent 根据问题类型自动选择检索策略 |
| **实时检索可视化** | 检索过程实时展示，让用户了解回答来源 |
| **迭代优化** | 支持答案质量评估与自动重新检索 |

### 📊 应用场景

- **患者健康咨询**：用户可以询问疾病症状、治疗方法、用药建议等
- **病例辅助分析**：结合用户上传的病例记录，提供个性化医疗建议
- **药品信息查询**：查询药品说明书、用法用量、注意事项等
- **健康知识科普**：提供疾病预防、健康保养等方面的知识

---

## 🏗️ 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端界面 (Vue 3)                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI 后端服务                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   API 层     │  │  智能 Agent  │  │    检索引擎          │  │
│  │  (认证/会话) │  │(意图识别/    │  │(知识图谱/向量检索)   │  │
│  │              │  │ 策略决策)    │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   PostgreSQL     │ │    Milvus        │ │     Neo4j        │
│  (用户/会话/消息) │ │ (向量数据库)     │ │  (知识图谱)      │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

### 检索架构

系统采用**双向量数据库 + 知识图谱**的混合检索架构：

| 数据源 | 类型 | 存储内容 | 用途 |
|--------|------|----------|------|
| **情景记忆** | 向量库 | 用户上传的病例、就诊记录 | 个性化医疗咨询 |
| **语义记忆** | 向量库 | 药品说明书、医学文献 | 专业知识查询 |
| **知识图谱** | 图数据库 | 疾病、症状、药品关系 | 结构化事实查询 |

### 智能检索流程

```
用户提问 → 意图分类 → 实体识别 → 策略决策 → 多源检索 → 结果融合 → 答案生成
              ↓              ↓
           [事实性问题]    [疾病/症状/药品]
              ↓              ↓
        知识图谱优先      混合检索策略
```

---

## 🚀 快速开始

### 环境要求

- Python `3.12+`
- Docker / Docker Compose（用于启动依赖服务）
- 包管理：推荐使用 `uv`

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/medchat.git
cd medchat

# 2. 安装依赖
uv sync

# 3. 启动依赖服务（PostgreSQL、Redis、Milvus、Neo4j）
docker compose up -d

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，配置您的 API Key 和模型参数

# 5. 启动服务
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### 访问地址

- **前端页面**：http://localhost:8000
- **API 文档**：http://localhost:8000/docs
- **Milvus 管理界面**：http://localhost:8080

---

## 📁 项目结构

```
medchat/
├── backend/                    # 后端代码
│   ├── app.py                  # FastAPI 入口
│   ├── api.py                  # API 接口定义
│   ├── agent.py                # 智能 Agent 核心逻辑
│   ├── intelligent_retrieval.py # 智能检索器
│   ├── smart_retrieval_agent.py # 检索策略决策器
│   ├── medical_ner.py          # 医疗实体识别
│   ├── dual_vector_store.py    # 双向量数据库管理
│   ├── tools.py                # 工具函数（天气、检索等）
│   ├── rag_pipeline.py         # RAG 检索管道
│   ├── database.py             # 数据库配置
│   ├── models.py               # ORM 模型
│   ├── auth.py                 # 认证模块
│   └── cache.py                # Redis 缓存
├── frontend/                   # 前端代码
│   ├── index.html              # 主页面
│   ├── script.js               # Vue 逻辑
│   └── style.css               # 样式
├── data/                       # 数据目录
│   ├── bm25_state.json         # BM25 稀疏向量统计
│   └── documents/              # 上传文档存储
├── docker-compose.yml          # Docker 配置
├── pyproject.toml              # 项目依赖配置
└── .env                        # 环境变量
```

---

## 🔧 核心模块

### 1. 智能检索 Agent

`backend/intelligent_retrieval.py` - 智能检索器，负责：
- 问题类型分类（事实性/解释性/建议性/比较性）
- 医疗实体识别（疾病、症状、药品、检查项目）
- 检索策略决策（图谱优先/向量优先/混合检索）
- 多源检索结果融合

### 2. 医疗实体识别

`backend/medical_ner.py` - 医疗领域命名实体识别：
- 支持疾病、症状、药品、治疗方法、检查项目等实体类型
- 规则匹配 + 预训练模型双模式
- 高准确率的实体抽取

### 3. 双向量数据库

`backend/dual_vector_store.py` - 情景记忆与语义记忆分离：
- **情景记忆**：存储用户个人病例数据
- **语义记忆**：存储通用医疗知识库
- 支持独立检索与混合检索

### 4. 知识图谱检索

`backend/tools.py` - Neo4j 医疗知识图谱查询：
- 支持多种查询意图（症状、病因、治疗、用药等）
- 实体模糊匹配与关系推理
- 结构化知识返回

---

## 📡 API 接口

### 认证接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/auth/register` | POST | 用户注册 |
| `/auth/login` | POST | 用户登录 |
| `/auth/me` | GET | 获取当前用户信息 |

### 聊天接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/chat` | POST | 非流式聊天 |
| `/chat/stream` | POST | 流式聊天（推荐） |

### 会话接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/sessions` | GET | 获取会话列表 |
| `/sessions/{id}` | GET | 获取会话详情 |
| `/sessions/{id}` | DELETE | 删除会话 |

### 文档接口（管理员）

| 接口 | 方法 | 说明 |
|------|------|------|
| `/documents` | GET | 获取文档列表 |
| `/documents/upload` | POST | 上传文档 |
| `/documents/{name}` | DELETE | 删除文档 |

---

## ⚙️ 环境变量配置

```env
# ===== 模型配置 =====
ARK_API_KEY=your-api-key
MODEL=your-model-name
BASE_URL=https://api.example.com/v1

# ===== 向量配置 =====
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu
DENSE_EMBEDDING_DIM=1024

# ===== Milvus 配置 =====
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530

# ===== 情景记忆数据库 =====
MILVUS_HOST_EPISODIC=127.0.0.1
MILVUS_PORT_EPISODIC=19530
MILVUS_COLLECTION_EPISODIC=user_cases

# ===== 语义记忆数据库 =====
MILVUS_HOST_SEMANTIC=127.0.0.1
MILVUS_PORT_SEMANTIC=19531
MILVUS_COLLECTION_SEMANTIC=medical_documents

# ===== Neo4j 配置 =====
NEO4J_URL=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# ===== 数据库配置 =====
DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379/0

# ===== 认证配置 =====
JWT_SECRET_KEY=your-secret-key
ADMIN_INVITE_CODE=admin-code
```

---

## 🎨 前端功能

- **聊天界面**：支持流式打字机效果，实时显示检索过程
- **检索可视化**：展示检索步骤、来源、相关性评分
- **会话管理**：查看历史会话、切换会话
- **文档上传**：支持 PDF、Word 等格式文档上传
- **响应式设计**：适配桌面端和移动端

---

## 🔬 技术特点

1. **智能检索策略**：根据问题类型自动选择最优检索路径
2. **双向量架构**：情景记忆与语义记忆分离，支持个性化服务
3. **实时可观测**：检索过程实时展示，增强用户信任度
4. **知识图谱集成**：结构化医疗知识，提供精准事实回答
5. **向量检索优化**：支持三级分块、Auto-merging、RRF 排序

---

## 📝 更新日志

### 2026-05-21 智能检索 Agent 升级
- 新增智能检索决策器，支持自动判断检索策略
- 实现双向量数据库架构（情景记忆 + 语义记忆）
- 集成医疗实体识别工具（NER）
- 优化检索流程，支持迭代反思

### 2026-04-08 本地嵌入与 BM25 持久化
- 切换为本地 HuggingFace 嵌入模型
- BM25 稀疏向量统计持久化
- 支持增量更新与删除同步

### 2026-03-21 后端服务建设
- 新增用户认证与权限系统
- 聊天历史迁移到 PostgreSQL
- 引入 Redis 缓存优化性能

---

## 📄 许可证

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

*MedChat - 让医疗咨询更智能* 💊🤖