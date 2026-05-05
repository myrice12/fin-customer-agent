"""
FastAPI入口 — 提供REST API + SSE流式响应
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agents.supervisor import create_supervisor_graph
from app.memory.working_memory import WorkingMemory
from app.memory.short_term import ShortTermMemory
from app.memory.long_term import LongTermMemory
from app.mcp.mcp_server import MCPToolServer, create_default_tools
from app.tracing.otel_config import init_tracer, get_agent_metrics, shutdown_tracer
from app.evaluation.rag_metrics import RAGEvalCase, evaluate_rag_case
from app.evaluation.business_metrics import run_business_evaluation
from app.skills.registry import create_default_skill_registry
from app.harness.middleware import setup_middleware
from app.harness.auth import load_api_keys
from app.harness.health import comprehensive_health_check

load_dotenv()


working_memory = WorkingMemory()
short_term_memory = ShortTermMemory(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
long_term_memory = LongTermMemory(index_path=os.getenv("FAISS_INDEX_PATH", "./vector_store/faiss_index"))
mcp_server = create_default_tools(MCPToolServer())
skill_registry = create_default_skill_registry()
metrics = get_agent_metrics()
graph = None
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global graph

    init_tracer(
        service_name=os.getenv("OTEL_SERVICE_NAME", "fin-customer-agent"),
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
    )

    graph = create_supervisor_graph(
        working_memory=working_memory,
        short_term_memory=short_term_memory,
        long_term_memory=long_term_memory,
    )

    kb_count = await long_term_memory.load_knowledge_base("knowledge_base")
    import logging
    logging.getLogger(__name__).info("Knowledge base loaded: %d chunks", kb_count)

    async def _cleanup_loop():
        while True:
            await asyncio.sleep(300)
            wm_cleaned = working_memory.cleanup_expired(max_age_seconds=1800)
            stm_cleaned = await short_term_memory.cleanup_expired()
            if wm_cleaned or stm_cleaned:
                import logging
                logging.getLogger(__name__).info(
                    "Session cleanup: working=%d, short_term=%d",
                    wm_cleaned, stm_cleaned,
                )

    cleanup_task = asyncio.create_task(_cleanup_loop())

    load_api_keys()

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8000").split(","),
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


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """主聊天接口"""
    if graph is None:
        raise HTTPException(status_code=503, detail="系统初始化中")

    session_id = request.session_id or str(uuid.uuid4())

    await short_term_memory.add_message(session_id, "user", request.message)

    from langchain_core.messages import HumanMessage

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
        result = await asyncio.wait_for(graph.ainvoke(initial_state, config=config), timeout=120.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="请求处理超时，请稍后重试")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

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


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE流式聊天接口 — 逐节点推送处理进度"""
    if graph is None:
        raise HTTPException(status_code=503, detail="系统初始化中")

    session_id = request.session_id or str(uuid.uuid4())

    await short_term_memory.add_message(session_id, "user", request.message)

    from langchain_core.messages import HumanMessage

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
            error_payload = {"error": str(e)}
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


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """获取对话历史"""
    history = await short_term_memory.get_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.get("/api/tools")
async def list_tools():
    """MCP工具发现接口"""
    return {"tools": mcp_server.list_tools()}


@app.get("/api/skills")
async def list_skills(tag: str | None = None):
    """Skill能力发现接口"""
    return {"skills": skill_registry.list_skills(tag=tag)}


@app.post("/api/tools/call")
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


@app.post("/api/evaluate/rag")
async def evaluate_rag(request: RAGEvalRequest):
    """对单条RAG样本进行离线指标评测"""
    from app.agents.knowledge_rag import KnowledgeRAGAgent

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


@app.post("/api/evaluate/business")
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


@app.post("/api/sessions/cleanup")
async def session_cleanup():
    """手动触发过期会话清理"""
    wm_cleaned = working_memory.cleanup_expired(max_age_seconds=1800)
    stm_cleaned = await short_term_memory.cleanup_expired()
    return {
        "working_memory_cleaned": wm_cleaned,
        "short_term_cleaned": stm_cleaned,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
