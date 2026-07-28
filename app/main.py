"""
FastAPI入口 — 提供REST API + SSE流式响应
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.agents.knowledge_rag import KnowledgeRAGAgent
from app.agents.supervisor import create_supervisor_graph
from app.config import get_settings
from app.evaluation.business_metrics import run_business_evaluation
from app.evaluation.rag_metrics import RAGEvalCase, evaluate_rag_case
from app.harness.auth import auth_enabled, verify_api_key
from app.harness.health import comprehensive_health_check
from app.harness.middleware import setup_middleware
from app.mcp.mcp_server import MCPToolServer, create_default_tools
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.memory.working_memory import WorkingMemory
from app.skills.registry import create_default_skill_registry
from app.tracing.otel_config import get_agent_metrics, init_tracer, shutdown_tracer

logger = logging.getLogger(__name__)


def _build_components():
    """根据配置中心构建全局组件实例。"""
    settings = get_settings()

    working_memory = WorkingMemory(
        max_entries_per_session=settings.memory.working_memory_max_entries,
    )
    short_term_memory = ShortTermMemory(
        redis_url=settings.memory.redis_url,
        max_turns=settings.memory.short_term_max_turns,
        ttl_seconds=settings.memory.short_term_ttl_seconds,
    )
    long_term_memory = LongTermMemory(
        index_path=settings.memory.faiss_index_path,
        embedding_dim=settings.embedding.embedding_dim,
        embedding_api_base=settings.embedding.embedding_api_base,
        embedding_api_key=settings.embedding.embedding_api_key,
        embedding_model=settings.embedding.embedding_model_name,
    )
    mcp_server = create_default_tools(MCPToolServer())
    skill_registry = create_default_skill_registry()
    metrics = get_agent_metrics()
    return working_memory, short_term_memory, long_term_memory, mcp_server, skill_registry, metrics


components = _build_components()
working_memory, short_term_memory, long_term_memory, mcp_server, skill_registry, metrics = components
graph = None
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global graph
    settings = get_settings()

    init_tracer(
        service_name=settings.tracing.otel_service_name,
        otlp_endpoint=settings.tracing.otel_exporter_otlp_endpoint or None,
    )

    llm = ChatOpenAI(
        model=settings.llm.model_name,
        temperature=settings.llm.model_temperature,
        api_key=settings.llm.openai_api_key or None,
        base_url=settings.llm.openai_base_url,
    )
    graph = create_supervisor_graph(
        llm=llm,
        working_memory=working_memory,
        short_term_memory=short_term_memory,
        long_term_memory=long_term_memory,
        enable_checkpointing=False,
    )

    kb_count = await long_term_memory.load_knowledge_base("knowledge_base")
    logger.info("Knowledge base loaded: %d new chunks", kb_count)

    async def _cleanup_loop():
        while True:
            await asyncio.sleep(settings.memory.cleanup_interval_seconds)
            wm_cleaned = await working_memory.cleanup_expired(
                max_age_seconds=settings.memory.cleanup_max_age_seconds,
            )
            stm_cleaned = await short_term_memory.cleanup_expired()
            if wm_cleaned or stm_cleaned:
                logger.info(
                    "Session cleanup: working=%d, short_term=%d",
                    wm_cleaned, stm_cleaned,
                )

    cleanup_task = asyncio.create_task(_cleanup_loop())

    if auth_enabled():
        logger.info("API key authentication is ENABLED")
    else:
        logger.warning(
            "API key authentication is DISABLED (API_KEYS not set) "
            "— running in dev mode. DO NOT expose this instance publicly."
        )

    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    await long_term_memory.close()
    shutdown_tracer()


app = FastAPI(
    title="智能客服多Agent系统",
    description="基于LangGraph的Supervisor编排多Agent智能客服系统",
    version="1.0.0",
    lifespan=lifespan,
)

_origins = [o.strip() for o in get_settings().server.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_middleware(app)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    compliance_passed: bool


class RAGEvalRequest(BaseModel):
    question: str
    gold_sources: list[str]
    gold_answer_points: list[str] = []


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict = {}


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    """主聊天接口"""
    if graph is None:
        raise HTTPException(status_code=503, detail="系统初始化中")

    session_id = request.session_id or str(uuid.uuid4())
    await short_term_memory.add_message(session_id, "user", request.message)

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "session_id": session_id,
        "intent": "",
        "sub_results": {},
        "compliance_passed": True,
        "final_response": "",
        "current_agent": "",
        "retry_count": 0,
    }
    config = {"configurable": {"thread_id": session_id}}

    try:
        result = await asyncio.wait_for(
            graph.ainvoke(initial_state, config=config),
            timeout=120.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="请求处理超时，请稍后重试")

    final_response = result.get("final_response", "系统处理异常，请稍后重试")
    await short_term_memory.add_message(session_id, "assistant", final_response)

    return ChatResponse(
        response=final_response,
        session_id=session_id,
        intent=result.get("intent", "unknown"),
        compliance_passed=result.get("compliance_passed", True),
    )


NODE_DISPLAY_NAMES = {
    "supervisor_route": "正在识别意图...",
    "knowledge_rag": "正在检索知识库...",
    "ticket_handler": "正在处理工单...",
    "compliance_check": "正在合规审查...",
    "synthesize": "正在生成回复...",
}


@app.post("/api/chat/stream", dependencies=[Depends(verify_api_key)])
async def chat_stream(request: ChatRequest):
    """SSE流式聊天接口 — 逐节点推送处理进度"""
    if graph is None:
        raise HTTPException(status_code=503, detail="系统初始化中")

    session_id = request.session_id or str(uuid.uuid4())
    await short_term_memory.add_message(session_id, "user", request.message)

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "session_id": session_id,
        "intent": "",
        "sub_results": {},
        "compliance_passed": True,
        "final_response": "",
        "current_agent": "",
        "retry_count": 0,
    }
    config = {"configurable": {"thread_id": session_id}}

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk in graph.astream(initial_state, config=config, stream_mode="updates"):
                for node_name, state_update in chunk.items():
                    display = NODE_DISPLAY_NAMES.get(node_name, f"正在处理: {node_name}")
                    intent = state_update.get("intent", "")
                    payload = {
                        "node": node_name,
                        "display": display,
                        "intent": intent,
                    }
                    yield f"event: node_update\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

            snapshot = await graph.aget_state(config)
            final_state = snapshot.values if snapshot else {}
            final_response = final_state.get("final_response", "系统处理异常，请稍后重试")
            await short_term_memory.add_message(session_id, "assistant", final_response)

            final_payload = {
                "response": final_response,
                "session_id": session_id,
                "intent": final_state.get("intent", "unknown"),
                "compliance_passed": final_state.get("compliance_passed", True),
            }
            yield f"event: final\ndata: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("SSE chat stream failed: %s", e)
            error_payload = {"error": "stream failed"}
            yield f"event: error\ndata: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
        finally:
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/history/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_history(session_id: str):
    """获取对话历史"""
    history = await short_term_memory.get_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.get("/api/tools")
async def list_tools():
    """MCP工具发现接口（只读元数据，保持公开）"""
    return {"tools": mcp_server.list_tools()}


@app.get("/api/skills")
async def list_skills(tag: str | None = None):
    """Skill能力发现接口（只读元数据，保持公开）"""
    return {"skills": skill_registry.list_skills(tag=tag)}


@app.post("/api/tools/call", dependencies=[Depends(verify_api_key)])
async def call_tool(request: ToolCallRequest):
    """MCP工具调用接口"""
    result = await mcp_server.call_tool(
        name=request.name,
        arguments=request.arguments,
    )
    return {
        "success": result.success,
        "result": result.result,
        "error": result.error,
        "duration_ms": result.duration_ms,
    }


@app.get("/api/metrics")
async def get_metrics():
    """获取系统指标"""
    return {
        "agent_metrics": metrics.get_summary(),
        "tool_call_log": mcp_server.get_call_log(last_n=20),
        "skills": skill_registry.list_skills(),
    }


@app.post("/api/evaluate/rag", dependencies=[Depends(verify_api_key)])
async def evaluate_rag(request: RAGEvalRequest):
    """对单条RAG样本进行离线指标评测"""
    if graph is None:
        raise HTTPException(status_code=503, detail="系统初始化中")

    agent = KnowledgeRAGAgent(llm=None, long_term_memory=long_term_memory)  # type: ignore[arg-type]
    retrieved_docs = await agent.retrieve_documents(request.question, top_k=5)
    reranked_docs = retrieved_docs[:3]
    answer = "\n".join(
        f"{doc.get('content', '')}\n来源: {doc.get('source', '未知')}"
        for doc in reranked_docs
    )

    case = RAGEvalCase(
        question=request.question,
        gold_sources=request.gold_sources,
        gold_answer_points=request.gold_answer_points,
    )
    return {
        "question": request.question,
        "metrics": evaluate_rag_case(case, retrieved_docs, reranked_docs, answer),
        "retrieved_sources": [doc.get("source", "") for doc in retrieved_docs],
        "reranked_sources": [doc.get("source", "") for doc in reranked_docs],
    }


@app.post("/api/evaluate/business", dependencies=[Depends(verify_api_key)])
async def evaluate_business_metrics(routing_mode: Literal["live", "skip"] = "live"):
    """离线业务指标评测：问答命中、人工耗时估算、路由、合规和上下文压缩。"""
    return await run_business_evaluation(kb_dir="knowledge_base", routing_mode=routing_mode)


@app.get("/health")
async def health_check():
    health = await comprehensive_health_check(
        short_term_memory=short_term_memory,
        long_term_memory=long_term_memory,
    )
    return {**health, "version": "1.0.0"}


@app.get("/api/sessions/stats")
async def session_stats():
    """获取会话统计信息"""
    return {
        "working_memory_sessions": working_memory.session_count,
        "short_term_fallback_sessions": short_term_memory.fallback_session_count,
    }


@app.post("/api/sessions/cleanup", dependencies=[Depends(verify_api_key)])
async def session_cleanup():
    """手动触发过期会话清理"""
    settings = get_settings()
    wm_cleaned = await working_memory.cleanup_expired(
        max_age_seconds=settings.memory.cleanup_max_age_seconds,
    )
    stm_cleaned = await short_term_memory.cleanup_expired()
    return {
        "working_memory_cleaned": wm_cleaned,
        "short_term_cleaned": stm_cleaned,
    }


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=os.getenv("APP_RELOAD", "false").lower() in ("1", "true", "yes"),
    )