<div align="center">

<img src="logo.png" alt="Fin Customer Agent" width="720" />

# 💼 Fin Customer Agent

### 面向金融场景的智能客服多 Agent 系统

**Supervisor 编排 · RAG 知识检索 · 合规双层审查 · 全链路可观测**

<br/>

![Version](https://img.shields.io/badge/version-1.0.0-5b6cff?style=for-the-badge&logo=git&logoColor=white)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3%2B-1A1A2E?style=for-the-badge&logo=langchain&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-00C853?style=for-the-badge&logo=open-source-initiative&logoColor=white)

<br/>

[**快速启动**](#-快速启动) · [**系统架构**](#-系统架构) · [**API 接口**](#-api-接口) · [**前端预览**](#-前端预览) · [**路线图**](#-路线图) · [**贡献**](#-贡献)

<br/>

> ✨ **本项目为学习与演示用途**。真实金融业务需额外具备身份认证、权限控制、日志审计、真实风控、人工复核及完整监管合规流程。

</div>

---

## 📑 目录

- [项目简介](#-项目简介)
- [核心亮点](#-核心亮点)
- [前端预览](#-前端预览)
- [系统架构](#-系统架构)
- [快速启动](#-快速启动)
- [环境变量](#-环境变量)
- [API 接口](#-api-接口)
- [核心模块](#-核心模块说明)
- [项目结构](#-项目结构)
- [测试与质量保障](#-测试与质量保障)
- [部署](#-部署)
- [路线图](#-路线图)
- [致谢与参考](#-致谢与参考)
- [许可证](#-许可证)

---

## 🎯 项目简介

Fin Customer Agent 是一个面向**金融场景**的智能客服系统，采用 **Supervisor 多 Agent 编排架构**。系统接收用户自然语言请求后，由 Supervisor Agent 自动识别意图并路由至专业子 Agent，所有回复均经过金融合规审查后返回。

适合作为以下场景的参考实现：

- 多 Agent 协作系统的设计与开发参考
- LLM 在强合规领域的工程化落地
- 智能客服类应用的端到端模板

## ✨ 核心亮点

| 模块 | 能力 | 工程要点 |
|---|---|---|
| 🤖 **多 Agent 编排** | Supervisor + 3 专业 Agent | LangGraph `StateGraph` 有向图调度，每节点独立重试与降级 |
| 📚 **RAG 知识问答** | Query 改写 → 向量检索 → LLM 重排序 → 生成 | 混合评分（向量 35% + 词法 65%），FAISS 索引持久化与启动复用 |
| 🛡️ **金融合规审查** | 规则引擎 + LLM 深度审查 + PII 脱敏 | 敏感词检测、手机号/身份证/银行卡识别、两段式审计 |
| 🧠 **三级记忆体系** | 工作记忆 / 短期记忆（Redis）/ 长期记忆（FAISS） | TTL 自动过期、Redis 故障自动回退内存、并发安全 |
| 🌊 **SSE 流式响应** | 逐节点实时推送处理进度 | 节点进度条可视化、延迟统计、合规元信息 |
| 🔌 **MCP 工具协议** | 工具注册 / 发现 / 调用 + 沙箱 | 输入校验、权限检查、超时、审计日志 |
| 📊 **可观测性** | OpenTelemetry 全链路追踪 + 自研指标 | Jaeger 集成、Agent 调用统计、压缩比例指标 |
| 🔐 **可选鉴权** | `API_KEYS` 环境变量启用 | 未配置时开发模式，配置后所有写接口强制 `X-API-Key` |
| 🧪 **质量保障** | 单元测试 + 离线业务评测 | pytest 覆盖核心模块、合成 benchmark 持续验证 |

## 🎨 前端预览

内置单页应用 (`/static/index.html`)，零额外依赖即可在浏览器体验完整功能：

- 🌓 **明暗双主题**：自动跟随系统，支持一键切换并持久化
- 💬 **会话管理**：历史会话列表、新建/切换/删除
- 📝 **Markdown 渲染**：粗体、列表、引用、链接、代码块
- 🎯 **意图可视化**：每条 AI 回复附带路由标签（知识库/工单/合规）
- ⚡ **实时进度**：节点进度条展示 Agent 调用顺序
- 🎨 **响应式设计**：玻璃拟态 + 渐变光斑背景 + 移动端适配

启动服务后访问 `http://localhost:8000` 即可。

## 🏗️ 系统架构

### 1. Agent 协作流程

```
                          ┌──────────────────────────┐
                          │   用户消息 (Web / API)    │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │      Supervisor Node      │
                          │  (LLM 路由 · 意图识别)    │
                          └─────────────┬────────────┘
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            ▼                           ▼                           ▼
    ┌───────────────┐          ┌─────────────────┐          ┌──────────────────┐
    │  knowledge_rag│          │  ticket_handler │          │ compliance_check │
    │  知识库问答    │          │  工单处理        │          │  合规问询回复     │
    └───────┬───────┘          └────────┬────────┘          └────────┬─────────┘
            │                           │                           │
            │   ┌───────────────────────┴────────┐                  │
            └─▶ │        compliance_check        │ ◀────────────────┘
                │   规则引擎 + LLM 双层审查       │
                │   PII 自动脱敏 · 风险定级       │
                └───────────────┬────────────────┘
                                ▼
                      ┌──────────────────┐
                      │    synthesize    │
                      │   结果汇总生成    │
                      └────────┬─────────┘
                               ▼
                         最终回复（SSE）
```

### 2. 技术栈

| 层级 | 选型 | 说明 |
|---|---|---|
| **Web 框架** | FastAPI + Uvicorn | REST API + SSE 流式 + 静态资源 |
| **Agent 编排** | LangGraph `StateGraph` | 多 Agent 有向图调度 |
| **LLM 接入** | LangChain + OpenAI 兼容 API | 兼容 OpenAI / DeepSeek / 通义千问 等 |
| **向量检索** | FAISS (`IndexFlatIP`) | 长期记忆语义检索 |
| **文本嵌入** | Qwen3-Embedding-0.6B | OpenAI 兼容 API，支持批量 |
| **会话存储** | Redis (async) | 短期对话历史，30 分钟滑动 TTL |
| **工具协议** | MCP (JSON-RPC) | 工具注册、发现与调用 |
| **可观测性** | OpenTelemetry + Jaeger | 全链路追踪 + 自研指标 |
| **容器编排** | Docker Compose | Redis + 后端 + Jaeger 一键启动 |
| **测试** | pytest + pytest-asyncio | 单元测试覆盖核心模块 |

### 3. 目录结构

```
FinCustomerAgent/
├── app/
│   ├── main.py                       # FastAPI 入口，路由与生命周期
│   ├── config.py                     # Pydantic Settings 配置中心
│   ├── agents/
│   │   ├── supervisor.py             # Supervisor 编排 (LangGraph)
│   │   ├── knowledge_rag.py          # RAG 知识检索 Agent
│   │   ├── ticket_handler.py         # 工单 CRUD Agent
│   │   └── compliance_checker.py     # 合规审查 Agent
│   ├── memory/
│   │   ├── working_memory.py         # 工作记忆 (asyncio.Lock)
│   │   ├── short_term.py             # 短期记忆 (Redis + 内存回退)
│   │   └── long_term.py              # 长期记忆 (FAISS + 词法回退)
│   ├── mcp/
│   │   └── mcp_server.py             # MCP 工具服务器
│   ├── evaluation/
│   │   ├── rag_metrics.py            # RAG 离线评测 (Recall/MRR/nDCG)
│   │   └── business_metrics.py       # 业务指标评测 + Markdown 报告
│   ├── skills/
│   │   └── registry.py               # Skill 能力注册表
│   ├── tracing/
│   │   └── otel_config.py            # OpenTelemetry 配置
│   ├── harness/
│   │   ├── auth.py                   # API Key 鉴权
│   │   ├── health.py                 # 综合健康检查
│   │   ├── middleware.py             # 请求ID + 全局异常处理
│   │   └── sandbox.py                # 工具调用沙箱
│   └── static/                       # 前端单页应用
│       ├── index.html
│       ├── app.js
│       └── styles.css
├── tests/                            # 单元测试 (pytest)
├── knowledge_base/                   # 金融知识库 (8 个领域)
├── scripts/
│   └── run_business_evaluation.py    # 业务评测 CLI
├── embedding_server.py               # 本地 Embedding 服务 (Qwen3-Embedding-0.6B)
├── business_evaluation_report.md     # 业务评测报告（自动生成）
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── pytest.ini
├── requirements.txt
├── requirements_embedding.txt
├── .env.example
└── README.md
```

## 🚀 快速启动

### 前置条件

- **Python 3.11+**
- 一个 OpenAI 兼容的 LLM API（OpenAI / DeepSeek / 通义千问 等）
- （可选）Docker Desktop
- （可选）Redis 7.x — 未安装时自动回退到内存存储

### 方式一：本地运行（推荐）

```bash
# 1. 克隆项目
git clone <repository-url>
cd fin-customer-agent

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，至少设置 OPENAI_API_KEY 和 OPENAI_BASE_URL

# 5. （可选）启动本地 Embedding 服务
python embedding_server.py    # 默认监听 http://localhost:6008

# 6. 启动主服务
python -m app.main
```

打开浏览器访问 `http://localhost:8000`。

> 💡 本地开发热重载：`APP_RELOAD=true python -m app.main`
> 🔐 生产部署鉴权：设置 `API_KEYS=key1,key2`，写接口将强制要求 `X-API-Key` 请求头。

### 方式二：Docker Compose

```bash
cp .env.example .env
# 编辑 .env，填写 API Key

docker-compose up -d
```

启动后访问：

| 服务 | 地址 |
|---|---|
| 🖥️ 聊天界面 | http://localhost:8000 |
| 📚 API 文档 | http://localhost:8000/docs |
| 🔍 健康检查 | http://localhost:8000/health |
| 📈 Jaeger 追踪 | http://localhost:16686 |

## ⚙️ 环境变量

所有配置统一通过 [`app/config.py`](app/config.py) 管理，按职责拆分为 6 个分组：

| 分组 | 关键变量 | 说明 | 默认值 |
|---|---|---|---|
| **LLM** | `OPENAI_API_KEY`<br>`OPENAI_BASE_URL`<br>`MODEL_NAME`<br>`MODEL_TEMPERATURE` | API 密钥<br>API 地址<br>模型名<br>采样温度 | —<br>`https://api.openai.com/v1`<br>`gpt-4o-mini`<br>`0` |
| **Embedding** | `EMBEDDING_API_BASE`<br>`EMBEDDING_API_KEY`<br>`EMBEDDING_MODEL_NAME`<br>`EMBEDDING_DIM`<br>`EMBEDDING_COOLDOWN_SECONDS` | API 地址<br>API 密钥<br>模型名<br>向量维度<br>失败冷却秒数 | —<br>—<br>`Qwen/Qwen3-Embedding-0.6B`<br>`1024`<br>`60` |
| **Memory** | `REDIS_URL`<br>`FAISS_INDEX_PATH`<br>`SHORT_TERM_MAX_TURNS`<br>`SHORT_TERM_TTL_SECONDS`<br>`WORKING_MEMORY_MAX_ENTRIES`<br>`CLEANUP_INTERVAL_SECONDS`<br>`CLEANUP_MAX_AGE_SECONDS` | Redis 地址<br>FAISS 路径<br>短期记忆轮数<br>短期记忆 TTL<br>工作记忆容量<br>清理间隔<br>会话过期时间 | `redis://localhost:6379/0`<br>`./vector_store/faiss_index`<br>`20`<br>`1800`<br>`50`<br>`300`<br>`1800` |
| **Tracing** | `OTEL_SERVICE_NAME`<br>`OTEL_EXPORTER_OTLP_ENDPOINT` | 服务名<br>OTLP 上报地址 | `fin-customer-agent`<br>空（不上报） |
| **Server** | `HOST`<br>`PORT`<br>`CORS_ORIGINS`<br>`APP_RELOAD` | 监听地址<br>监听端口<br>跨域来源<br>是否热重载 | `0.0.0.0`<br>`8000`<br>`http://localhost:8000`<br>`false` |
| **Harness** | `API_KEYS`<br>`SANDBOX_TIMEOUT_SECONDS` | API Key 列表（逗号分隔）<br>工具沙箱超时 | 空（开发模式）<br>`30` |

## 🔌 API 接口

> 🔐 设置 `API_KEYS` 后，标注 `[鉴权]` 的接口要求请求头 `X-API-Key: <key>`；只读元数据接口保持公开。

### 聊天

**同步接口** `[鉴权]`：

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"user_id": "user_001", "message": "理财产品A的收益率是多少？"}'
```

**SSE 流式接口** `[鉴权]`：

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"user_id": "user_001", "message": "理财产品A的收益率是多少？"}'
```

SSE 事件类型：

| 事件 | 说明 | 关键字段 |
|---|---|---|
| `node_update` | 节点开始处理 | `node`, `display`, `intent` |
| `final` | 处理完成 | `response`, `session_id`, `intent`, `compliance_passed` |
| `error` | 处理异常 | `error` |
| `done` | 流结束 | — |

### 工具与技能

```bash
# MCP 工具列表
curl http://localhost:8000/api/tools

# 调用工具 `[鉴权]`
curl -X POST http://localhost:8000/api/tools/call \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"name": "risk_check", "arguments": {"user_id": "u1", "action": "purchase", "amount": 60000}}'

# Skill 能力列表
curl http://localhost:8000/api/skills
```

### 监控与评测

```bash
# 系统指标
curl http://localhost:8000/api/metrics

# 会话统计
curl http://localhost:8000/api/sessions/stats

# 手动清理过期会话 `[鉴权]`
curl -X POST http://localhost:8000/api/sessions/cleanup

# RAG 离线评测 `[鉴权]`
curl -X POST http://localhost:8000/api/evaluate/rag \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"question": "理财产品A的投资期限是多少？", "gold_sources": ["product_faq.md"], "gold_answer_points": ["6个月至3年"]}'

# 业务指标离线评测 `[鉴权]`
curl -X POST http://localhost:8000/api/evaluate/business

# 生成 Markdown 评测报告
python scripts/run_business_evaluation.py

# 跳过 Supervisor 实时路由评测
python scripts/run_business_evaluation.py --routing-mode skip

# 控制路由评测并发与超时
ROUTING_EVAL_TIMEOUT_SECONDS=10 ROUTING_EVAL_CONCURRENCY=5 \
  python scripts/run_business_evaluation.py --routing-mode live
```

完整接口契约参见 `http://localhost:8000/docs`。

## 🧩 核心模块说明

### 多 Agent 编排

系统使用 LangGraph `StateGraph` 构建有向图，由 Supervisor Agent 根据用户意图路由至对应子 Agent：

- **`knowledge_rag`** — 知识库问答（产品查询、政策咨询、流程指引）
- **`ticket_handler`** — 工单业务（退款申请、理赔报案、开户预约）
- **`compliance_checker`** — 合规审查（敏感词检测、PII 保护、风险提示）

所有回复均经过合规审查后才返回用户。

### 三级记忆体系

| 层级 | 存储方式 | 生命周期 | 用途 |
|---|---|---|---|
| 工作记忆 | 进程内存 (`asyncio.Lock`) | 请求级 | Agent 路由决策上下文、中间状态 |
| 短期记忆 | Redis（内存回退） | 30 分钟滑动 TTL | 多轮对话历史，最多 20 轮 |
| 长期记忆 | FAISS + 持久化 | 磁盘 | 知识库文档语义检索 + 启动复用 |

### RAG 知识检索

完整流程：**Query 改写 → 向量检索（top-5） → LLM 重排序（top-3） → 上下文注入 → 生成回答**

- 嵌入模型：`Qwen/Qwen3-Embedding-0.6B`（OpenAI 兼容 API），支持批量向量化
- 检索采用混合评分策略：**向量相似度（35%）+ 词法匹配（65%）**
- 索引持久化：`ntotal == len(chunks)` 时跳过重新嵌入，避免重复消耗 Embedding API
- 降级策略：Embedding 不可用时自动回退纯词法检索

### 金融合规审查

两阶段审查机制：

1. **规则引擎**（毫秒级）— 正则匹配 PII（手机号、身份证、银行卡、邮箱）+ 关键词检测违规金融用语 + 语义风险模式
2. **LLM 深度审查** — 规则引擎未拦截时，由大模型进行语义级别的合规判断

PII 信息自动脱敏处理，违规内容标记风险等级（`low` / `medium` / `high` / `critical`）。

### SSE 流式响应

流式接口会按节点推送事件，前端据此实时显示：

1. **意图识别** (`supervisor_route`) — 决定路由到哪个 Agent
2. **专业处理** (`knowledge_rag` / `ticket_handler`) — 实际处理
3. **合规审查** (`compliance_check`) — 必经节点
4. **汇总生成** (`synthesize`) — 拼接最终回复

## 🧪 测试与质量保障

```bash
# 安装测试依赖（已包含在 requirements.txt）
pip install -r requirements.txt

# 运行所有单元测试
pytest -v

# 生成 HTML 报告
pytest --html=report.html --self-contained-html
```

当前覆盖：

- ✅ `tests/test_auth.py` — API Key 鉴权 + Settings CSV 解析
- ✅ `tests/test_compliance_checker.py` — 规则引擎 + PII 检测与脱敏
- ✅ `tests/test_long_term_memory.py` — 词法评分、查询词条提取、文本分块
- ✅ `tests/test_working_memory.py` — 异步锁、并发更新、过期清理

离线业务评测：

```bash
python scripts/run_business_evaluation.py
# 生成 business_evaluation_report.md
```

## 🐳 部署

### Docker

镜像特点：

- 🛡️ **非 root 用户** — 创建专用 `app:app` 用户运行容器
- 🏥 **HEALTHCHECK** — 内置 `/health` 端点健康检查
- 📦 **瘦镜像** — `.dockerignore` 排除 `.git`、报告、评测脚本等

```bash
# 构建镜像
docker build -t fin-customer-agent:latest .

# 直接运行
docker run -p 8000:8000 --env-file .env fin-customer-agent:latest
```

### Docker Compose

`docker-compose.yml` 默认启动：

- `redis` — 短期记忆后端
- `python-agent` — 主应用（自动构建当前目录）
- `jaeger` — 链路追踪 UI（http://localhost:16686）

```bash
docker-compose up -d
```

### 生产建议

- ✅ 设置 `API_KEYS` 启用鉴权
- ✅ 反向代理层（nginx / Caddy）启用 HTTPS
- ✅ 为 Embedding / LLM 调用配置独立的速率限制
- ✅ 使用 PostgreSQL + pgvector 替换内存工单与 FAISS

## 🗺️ 路线图

| 状态 | 计划 | 说明 |
|---|---|---|
| ✅ | Supervisor 多 Agent 编排 | 已完成 |
| ✅ | RAG + 双层合规审查 | 已完成 |
| ✅ | SSE 流式响应 + 前端可视化 | 已完成 |
| ✅ | API Key 鉴权 | 已完成 |
| ✅ | Docker 非 root 部署 | 已完成 |
| ✅ | 单元测试基础覆盖 | 已完成 |
| 🚧 | Milvus / pgvector 切换 | 抽象接口已预留 |
| 🚧 | 多用户隔离 + RBAC | 框架已具备，权限矩阵待补全 |
| 📋 | 工单持久化（Postgres） | 当前为内存存储 |
| 📋 | LLM-as-Judge 评测 | 引入自动化的答案质量评估 |
| 📋 | 多语言支持（English / 粤语 / 繁体） | 知识库需要扩充 |
| 📋 | 真实金融场景压力测试 | 与合作伙伴联合验证 |

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

开发约定：

- 提交前运行 `pytest tests/` 确保全部通过
- 保持代码风格一致（参考 `app/` 现有模块）
- 新增 Agent / Memory 后端请在 `app/agents/__init__.py` 与 `app/memory/__init__.py` 中暴露
- 环境变量请同步更新 `.env.example` 与本 README

## 🙏 致谢与参考

- [LangGraph](https://langchain-ai.github.io/langgraph/) — 多 Agent 编排框架
- [LangChain](https://python.langchain.com/) — LLM 应用开发工具链
- [FastAPI](https://fastapi.tiangolo.com/) — 高性能 Python Web 框架
- [FAISS](https://github.com/facebookresearch/faiss) — 向量相似度检索
- [OpenTelemetry](https://opentelemetry.io/) — 可观测性标准
- [Qwen3-Embedding](https://huggingface.co/Qwen) — 阿里达摩院开源嵌入模型

灵感与最佳实践参考：

- Anthropic 的 [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- LangChain 的 [Multi-Agent Supervisor Tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)

## 📄 许可证

[MIT License](LICENSE) © 2026 Fin Customer Agent Contributors

---

<div align="center">

如果这个项目对你有帮助，欢迎点 ⭐ 支持！

<sub>Built with ❤️ using LangGraph · FastAPI · FAISS</sub>

</div>