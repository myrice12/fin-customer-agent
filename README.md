<div align="center">

![]( logo.png)

# Fin Customer Agent

<font size=5>基于 LangGraph 的金融智能客服多 Agent 系统</font>

**v1.0** · 2026-05-05

<br>

![Version](https://img.shields.io/badge/Version-v1.0-2196F3?style=flat)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3+-1A1A2E?logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-CPU-FF6F00?logo=meta&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7.x-DC382D?logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00C853?logo=open-source-initiative&logoColor=white)

<br>

[快速启动](#快速启动) · [系统架构](#系统架构) · [API 文档](#api-接口) · [核心模块](#核心模块说明) · [版本记录](#版本记录)

</div>

---

## 项目简介

本项目是一个面向金融场景的智能客服系统，采用 Supervisor 多 Agent 编排架构。系统接收用户自然语言请求后，由 Supervisor Agent 自动识别意图并路由至专业子 Agent 处理，所有回复均经过金融合规审查后返回。

**核心能力：**

- 多 Agent 协同编排（Supervisor + 3 专业 Agent）
- RAG 知识库问答（Query 改写 → 向量检索 → 重排序 → 生成）
- 金融合规双层审查（规则引擎 + LLM 深度审查）
- 三级记忆体系（工作记忆 / 短期记忆 / 长期记忆）
- SSE 实时流式响应（逐节点推送处理进度）
- MCP 标准工具协议（统一工具注册、发现与调用）
- OpenTelemetry 全链路追踪

> **声明：** 本项目为学习与演示用途，不构成可直接上线的生产系统。真实金融业务需额外具备身份认证、权限控制、日志审计、真实风控、人工复核及完整监管合规流程。

---

## 系统架构

### Agent 协作流程

```
用户消息
  │
  ▼
┌─────────────────┐
│  Supervisor Agent │  ← 意图识别 + 任务路由
│  (supervisor.py)  │
└────────┬────────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
┌──────┐┌──────┐┌──────────┐
│知识检索││工单处理││合规审查  │
│ RAG  ││Ticket ││Compliance│
└──┬───┘└──┬───┘└────┬─────┘
   │       │         │
   └───────┴────┬────┘
                ▼
         ┌────────────┐
         │  合规终审    │
         └──────┬─────┘
                ▼
         ┌────────────┐
         │  结果汇总    │
         └──────┬─────┘
                ▼
           最终回复
```

### 目录结构

```
app/
├── main.py                        # FastAPI 入口，路由与生命周期管理
├── agents/
│   ├── supervisor.py              # Supervisor 编排 Agent（LangGraph StateGraph）
│   ├── intent_router.py           # 意图识别 Agent
│   ├── knowledge_rag.py           # 知识库 RAG 问答 Agent
│   ├── ticket_handler.py          # 工单创建/查询 Agent
│   └── compliance_checker.py      # 金融合规审查 Agent
├── memory/
│   ├── working_memory.py          # 工作记忆（进程内，请求级）
│   ├── short_term.py              # 短期记忆（Redis，30 分钟 TTL）
│   └── long_term.py               # 长期记忆（FAISS 向量检索 + API Embedding）
├── mcp/
│   └── mcp_server.py              # MCP 工具服务器（4 个内置工具）
├── evaluation/
│   └── rag_metrics.py             # RAG 离线评测指标（Recall@K, MRR, nDCG）
├── skills/
│   └── registry.py                # Skill 能力注册表
├── tracing/
│   └── otel_config.py             # OpenTelemetry 链路追踪配置
└── static/                        # 前端单页应用
    ├── index.html
    ├── app.js
    └── styles.css
```

### 技术栈

| 组件 | 技术选型 | 职责 |
|------|----------|------|
| Web 框架 | FastAPI + Uvicorn | REST API 与静态文件服务 |
| Agent 编排 | LangGraph (StateGraph) | 多 Agent 有向图调度 |
| LLM 调用 | LangChain + OpenAI 兼容 API | 意图识别、RAG 生成、合规审查 |
| 向量检索 | FAISS (IndexFlatIP) | 长期记忆语义相似度检索 |
| 文本嵌入 | Qwen3-Embedding-4B (API) | 文档与查询的向量化 |
| 会话存储 | Redis (async) | 短期对话历史，30 分钟滑动 TTL |
| 工具协议 | MCP (Model Context Protocol) | 标准化工具注册与调用 |
| 可观测性 | OpenTelemetry + Jaeger | 分布式链路追踪与指标采集 |
| 容器编排 | Docker Compose | Redis + 后端 + Jaeger 一键部署 |

---

## 快速启动

### 前置条件

- Python 3.11+
- OpenAI 兼容的 LLM API（如 DeepSeek、OpenAI 等）

### 本地运行（推荐）

```bash
# 克隆项目
git clone <repository-url>
cd Financial-customer-sgent

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填写 LLM API Key 和 Embedding API 地址

# 启动服务
python -m app.main
```

> 未安装 Redis 时系统自动回退到内存存储，功能不受影响，适合本地开发与学习。

### Docker Compose（可选）

```bash
cp .env.example .env
# 编辑 .env，填写 API Key

docker-compose up -d
```

> Docker 方式会同时启动 Redis、后端服务和 Jaeger 链路追踪。需确保 Docker Desktop 已启动且网络可访问 Docker Hub。

启动成功后访问：

| 服务 | 地址 |
|------|------|
| 聊天界面 | http://localhost:8000 |
| API 文档 | http://localhost:8000/docs |
| Jaeger 追踪 | http://localhost:16686（仅 Docker 方式） |
| 健康检查 | http://localhost:8000/health |

---

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | LLM API 密钥 | — |
| `OPENAI_BASE_URL` | LLM API 地址 | `https://api.deepseek.com` |
| `MODEL_NAME` | LLM 模型名称 | `deepseek-v4-flash` |
| `EMBEDDING_API_BASE` | Embedding API 地址 | — |
| `EMBEDDING_API_KEY` | Embedding API 密钥 | — |
| `EMBEDDING_MODEL_NAME` | Embedding 模型名称 | `Qwen3-Embedding-4B` |
| `EMBEDDING_DIM` | 嵌入向量维度 | `1024` |
| `REDIS_URL` | Redis 连接地址 | `redis://localhost:6379/0` |
| `FAISS_INDEX_PATH` | FAISS 索引存储路径 | `./vector_store/faiss_index` |
| `OTEL_SERVICE_NAME` | 链路追踪服务名 | `fin-customer-agent` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP 上报地址 | 空（不上报） |
| `HOST` | 服务监听地址 | `0.0.0.0` |
| `PORT` | 服务端口 | `8000` |

---

## API 接口

### 聊天

**同步接口：**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_001", "message": "理财产品A的收益率是多少？"}'
```

**SSE 流式接口：**

```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_001", "message": "理财产品A的收益率是多少？"}'
```

流式事件类型：

| 事件 | 说明 |
|------|------|
| `node_update` | Agent 节点开始处理，携带节点名称与意图 |
| `final` | 处理完成，携带最终回复与元数据 |
| `error` | 处理异常 |
| `done` | 流结束 |

### 工具与技能

```bash
# MCP 工具列表
curl http://localhost:8000/api/tools

# 调用工具
curl -X POST http://localhost:8000/api/tools/call \
  -H "Content-Type: application/json" \
  -d '{"name": "risk_check", "arguments": {"user_id": "u1", "action": "purchase", "amount": 60000}}'

# Skill 能力列表
curl http://localhost:8000/api/skills
```

### 监控与评测

```bash
# 系统指标（Agent 调用统计、工具日志、Skill 列表）
curl http://localhost:8000/api/metrics

# 会话统计
curl http://localhost:8000/api/sessions/stats

# 手动清理过期会话
curl -X POST http://localhost:8000/api/sessions/cleanup

# RAG 离线评测
curl -X POST http://localhost:8000/api/evaluate/rag \
  -H "Content-Type: application/json" \
  -d '{
    "question": "理财产品A的投资期限是多少？",
    "gold_sources": ["product_faq.md"],
    "gold_answer_points": ["6个月至3年"]
  }'

# 业务指标离线评测
curl -X POST http://localhost:8000/api/evaluate/business

# 跳过 Supervisor 实时路由评测
curl -X POST 'http://localhost:8000/api/evaluate/business?routing_mode=skip'

# 生成 Markdown 评测报告
python scripts/run_business_evaluation.py

# 未配置可用 LLM API 时，可跳过 Supervisor 实时路由评测
python scripts/run_business_evaluation.py --routing-mode skip

# live 路由评测可通过环境变量控制超时与并发
ROUTING_EVAL_TIMEOUT_SECONDS=10 ROUTING_EVAL_CONCURRENCY=5 \
  python scripts/run_business_evaluation.py --routing-mode live
```

---

## 核心模块说明

### 多 Agent 编排

系统使用 LangGraph `StateGraph` 构建有向图，由 Supervisor Agent 根据用户意图路由至对应子 Agent：

- **knowledge_rag** — 知识库问答（产品查询、政策咨询、流程指引）
- **ticket_handler** — 工单业务（退款申请、理赔报案、开户预约）
- **compliance_checker** — 合规审查（敏感词检测、PII 保护、风险提示）

所有回复均经过合规审查后才返回用户。

### 三级记忆体系

| 层级 | 存储方式 | 生命周期 | 用途 |
|------|----------|----------|------|
| 工作记忆 | 进程内存 | 请求级 | Agent 路由决策上下文、中间状态 |
| 短期记忆 | Redis（内存回退） | 30 分钟滑动 TTL | 多轮对话历史，最多 20 轮 |
| 长期记忆 | FAISS + API Embedding | 持久化 | 知识库文档语义检索 |

### RAG 知识检索

完整流程：Query 改写 → 向量检索（top-5） → LLM 重排序（top-3） → 上下文注入 → 生成回答

嵌入模型采用 Qwen3-Embedding-4B，通过 OpenAI 兼容 API 调用，支持批量向量化。检索采用混合评分策略：向量相似度（35%）+ 词法匹配（65%）。

### 金融合规审查

两阶段审查机制：

1. **规则引擎**（毫秒级）— 正则匹配 PII（手机号、身份证、银行卡、邮箱）+ 关键词检测违规金融用语
2. **LLM 深度审查** — 规则引擎未拦截时，由大模型进行语义级别的合规判断

PII 信息自动脱敏处理，违规内容标记风险等级（low / medium / high / critical）。

---

## 快速验证

启动服务后，在聊天界面输入以下问题：

| 输入 | 预期路由 | 预期行为 |
|------|----------|----------|
| 理财产品A的收益率是多少？ | Knowledge RAG | 返回产品收益率、期限、风险提示 |
| 退款政策是什么？ | Knowledge RAG | 返回退款政策详情 |
| 我想创建一个退款工单 | Ticket Handler | 创建模拟工单，返回工单号 |
| 保证收益吗？ | Compliance Checker | 触发合规说明，提示不能承诺保本保收益 |

---

## 版本记录

### v1.0（2026-05-05）

首个正式版本，包含完整的多 Agent 编排、RAG 知识检索、金融合规审查和可观测性能力。

**核心功能：**

- Supervisor 多 Agent 编排（LangGraph StateGraph）
- Knowledge RAG Agent（Query 改写 → 向量检索 → 重排序 → 生成）
- Ticket Handler Agent（工单创建与查询）
- Compliance Checker Agent（规则引擎 + LLM 双层合规审查）
- 三级记忆体系（工作记忆 / 短期记忆 / 长期记忆）
- SSE 实时流式响应（逐节点推送处理进度）
- MCP 工具服务器（4 个内置模拟工具）
- RAG 离线评测接口（Recall@K、MRR、nDCG）
- OpenTelemetry 全链路追踪
- 会话过期自动清理机制

**嵌入模型：**

- 接入 Qwen3-Embedding-4B，通过 OpenAI 兼容 API 调用
- 支持批量向量化，自动检测维度变化并重建索引

**前端：**

- 单页聊天界面，支持 SSE 流式消息展示
- 实时显示 Agent 处理进度与意图分类

---

## 许可证

[MIT License](LICENSE)

---
