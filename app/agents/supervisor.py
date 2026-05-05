"""
Supervisor编排Agent — 中央协调者
负责接收用户请求，根据意图路由到对应子Agent，汇总结果返回。
采用LangGraph StateGraph实现，支持并行调度和Human-in-the-Loop断点。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from tenacity import retry, stop_after_attempt, wait_exponential

from app.agents.intent_router import IntentRouterAgent
from app.agents.knowledge_rag import KnowledgeRAGAgent
from app.agents.ticket_handler import TicketHandlerAgent
from app.agents.compliance_checker import ComplianceCheckerAgent
from app.memory.working_memory import WorkingMemory
from app.memory.short_term import ShortTermMemory
from app.memory.long_term import LongTermMemory
from app.tracing.otel_config import trace_agent_call

logger = logging.getLogger(__name__)


# ─── 状态定义 ───

class AgentState(TypedDict):
    """Supervisor编排的全局状态"""
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    intent: str
    sub_results: dict[str, Any]
    compliance_passed: bool
    final_response: str
    current_agent: str
    retry_count: int


# ─── Supervisor节点 ───

SUPERVISOR_SYSTEM_PROMPT = """你是一个智能客服系统的Supervisor（主管编排Agent）。
你的职责是：
1. 分析用户意图，决定分发给哪个子Agent处理
2. 汇总子Agent的处理结果，生成最终回复
3. 确保所有回复都经过合规审查

可用的子Agent：
- intent_router: 意图识别和分类
- knowledge_rag: 知识库检索和回答
- ticket_handler: 工单创建和查询
- compliance_checker: 合规审查和敏感词检测

根据用户消息，决定下一步路由到哪个Agent。
"""


class SupervisorNode:
    """Supervisor决策节点"""

    def __init__(
        self,
        llm: ChatOpenAI,
        working_memory: WorkingMemory,
        short_term_memory: ShortTermMemory | None = None,
    ):
        self.llm = llm
        self.working_memory = working_memory
        self.short_term_memory = short_term_memory

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _call_llm(self, messages):
        return await asyncio.wait_for(self.llm.ainvoke(messages), timeout=30.0)

    @trace_agent_call("supervisor")
    async def route_decision(self, state: AgentState) -> AgentState:
        """分析用户意图，决定路由"""
        messages = state["messages"]
        session_id = state.get("session_id", "default")

        context = self.working_memory.get_context(session_id)
        optimized_context = {}
        if self.short_term_memory is not None:
            optimized_context = await self.short_term_memory.get_optimized_context(
                session_id,
                max_tokens=2000,
            )

        routing_prompt = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            SystemMessage(content=f"当前工作记忆上下文: {context}"),
            SystemMessage(content=f"短期记忆优化上下文: {optimized_context.get('context', '')}"),
            *messages,
            HumanMessage(content=(
                "请分析用户的最新消息，返回应该路由到的Agent名称。"
                "只返回以下之一: knowledge_rag, ticket_handler, compliance_checker"
            )),
        ]

        try:
            response = await self._call_llm(routing_prompt)
            intent = response.content.strip().lower()
        except Exception as e:
            logger.warning("Supervisor routing LLM failed, defaulting to knowledge_rag: %s", e)
            intent = "knowledge_rag"

        valid_intents = {"knowledge_rag", "ticket_handler", "compliance_checker"}
        if intent not in valid_intents:
            intent = "knowledge_rag"

        self.working_memory.update(
            session_id,
            {
                "last_intent": intent,
                "context_metrics": {
                    "raw_tokens": optimized_context.get("raw_tokens", 0),
                    "injected_tokens": optimized_context.get("injected_tokens", 0),
                    "compression_ratio": optimized_context.get("compression_ratio", 0.0),
                },
            },
        )

        return {
            **state,
            "intent": intent,
            "current_agent": "supervisor",
        }

    @trace_agent_call("supervisor_synthesize")
    async def synthesize_response(self, state: AgentState) -> AgentState:
        """汇总子Agent结果，生成最终回复"""
        sub_results = state.get("sub_results", {})
        compliance_passed = state.get("compliance_passed", True)

        if not compliance_passed:
            final_response = (
                "抱歉，您的请求涉及敏感内容，已转交人工客服处理。"
                "工单编号已自动生成，请留意后续通知。"
            )
        else:
            result_parts = []
            for agent_name, result in sub_results.items():
                if isinstance(result, str) and result.strip():
                    result_parts.append(result)
            final_response = "\n\n".join(result_parts) if result_parts else "抱歉，暂时无法处理您的请求，请稍后重试。"

        return {
            **state,
            "final_response": final_response,
            "messages": [AIMessage(content=final_response)],
        }


# ─── 路由函数 ───

def route_to_agent(state: AgentState) -> str:
    """根据意图路由到对应Agent节点"""
    intent = state.get("intent", "knowledge_rag")
    route_map = {
        "knowledge_rag": "knowledge_rag",
        "ticket_handler": "ticket_handler",
        "compliance_checker": "compliance_check",
    }
    return route_map.get(intent, "knowledge_rag")


def should_check_compliance(state: AgentState) -> str:
    """所有回复都需经过合规审查"""
    return "compliance_check"


# ─── 构建Graph ───

def create_supervisor_graph(
    llm: ChatOpenAI | None = None,
    working_memory: WorkingMemory | None = None,
    short_term_memory: ShortTermMemory | None = None,
    long_term_memory: LongTermMemory | None = None,
    enable_checkpointing: bool = True,
) -> StateGraph:
    """
    构建Supervisor编排的多Agent StateGraph。

    这是整个系统的核心入口，将4个子Agent通过有向图连接起来，
    由Supervisor节点负责路由决策和结果汇总。

    Args:
        llm: 语言模型实例
        working_memory: 工作记忆
        short_term_memory: 短期记忆
        long_term_memory: 长期记忆
        enable_checkpointing: 是否启用检查点（支持断点恢复）
    """
    if llm is None:
        llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4o"),
            temperature=float(os.getenv("MODEL_TEMPERATURE", "0")),
        )
    if working_memory is None:
        working_memory = WorkingMemory()

    supervisor = SupervisorNode(llm, working_memory, short_term_memory)

    intent_router = IntentRouterAgent(llm)
    knowledge_agent = KnowledgeRAGAgent(llm, long_term_memory)
    ticket_agent = TicketHandlerAgent(llm)
    compliance_agent = ComplianceCheckerAgent(llm)

    graph = StateGraph(AgentState)

    graph.add_node("supervisor_route", supervisor.route_decision)
    graph.add_node("knowledge_rag", knowledge_agent.process)
    graph.add_node("ticket_handler", ticket_agent.process)
    graph.add_node("compliance_check", compliance_agent.process)
    graph.add_node("synthesize", supervisor.synthesize_response)

    graph.set_entry_point("supervisor_route")

    graph.add_conditional_edges(
        "supervisor_route",
        route_to_agent,
        {
            "knowledge_rag": "knowledge_rag",
            "ticket_handler": "ticket_handler",
            "compliance_check": "compliance_check",
        },
    )

    graph.add_edge("knowledge_rag", "compliance_check")
    graph.add_edge("ticket_handler", "compliance_check")
    graph.add_edge("compliance_check", "synthesize")
    graph.add_edge("synthesize", END)

    checkpointer = MemorySaver() if enable_checkpointing else None
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled
