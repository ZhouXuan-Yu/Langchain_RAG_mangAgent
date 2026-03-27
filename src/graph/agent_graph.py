"""Agent Graph — 06+08: LangGraph 组装与 ReAct Agent 构建."""

import logging
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.language_models import BaseChatModel

from src.graph.state import AgentState
from src.graph.nodes import (
    router_node,
    retrieve_memory,
    memory_update_node,
    retrieve_web,
    browse_page_node,
    retrieve_memory_async,
    retrieve_web_async,
    browse_page_node_async,
    image_process_node,
    reason_node,
    memory_reflect_node,
    should_continue,
    set_llm,
)
from src.graph.router import route_decision, after_retrieve_or_web
from src.memory.chroma_store import ChromaMemoryStore
from src.memory.sqlite_store import get_sqlite_checkpointer
from src.tools.memory_tools import memory_search, save_memory, knowledge_base_search
from src.tools.browser_tools import web_search, browse_page
from src.tools.calc_tools import calculator
from src.tools.multimodal_tools import process_image
from src.middleware.pii_redactor import pii_pre_model_hook

logger = logging.getLogger(__name__)


def build_agent_graph(
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    enable_middleware: bool = True,
) -> StateGraph:
    """
    构建完整的 LangGraph Agent — 包含高级编排和主动记忆.

    包含节点：
    - router_node: 动态路由决策
    - retrieve_memory: 记忆检索
    - memory_update_node: 记忆更新
    - retrieve_web: 网页搜索
    - browse_page_node: 深度页面抓取
    - image_process_node: 图片处理
    - reason_node: LLM 推理生成回复
    - memory_reflect_node: 主动记忆 Upsert

    Args:
        llm: LangChain LLM 实例（DeepSeek）
        checkpointer: 状态持久化器（SqliteSaver）
        enable_middleware: 是否启用 PII 脱敏和输入验证中间件

    Returns:
        编译后的 StateGraph
    """
    # 定义所有工具
    tools = [memory_search, save_memory, knowledge_base_search, web_search, browse_page, calculator, process_image]
    llm_with_tools = llm.bind_tools(tools)

    # 注入 LLM 到 nodes 模块（传两个版本：裸 LLM + 带工具的 LLM）
    set_llm(llm, llm_with_tools)

    # 构建图
    builder = StateGraph(AgentState)

    # ── 添加节点 ───────────────────────────────────────────────────────────
    # 全部使用异步版本，确保 astream_events 能正确捕获工具调用事件
    builder.add_node("router", router_node)
    builder.add_node("retrieve_memory", retrieve_memory_async)
    builder.add_node("memory_update", memory_update_node)
    builder.add_node("retrieve_web", retrieve_web_async)
    builder.add_node("browse_page", browse_page_node_async)
    builder.add_node("image_process", image_process_node)
    builder.add_node("reason_node", reason_node)
    builder.add_node("memory_reflect", memory_reflect_node)

    # ── 定义边 ─────────────────────────────────────────────────────────────
    # 启动 -> router
    builder.add_edge(START, "router")

    # router -> 条件分支
    builder.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve_memory": "retrieve_memory",
            "memory_update": "memory_update",
            "retrieve_web": "retrieve_web",
            "image_process": "image_process",
            "reason_node": "reason_node",
        },
    )

    # 记忆/网页检索 -> reason 或 browse_page
    builder.add_edge("retrieve_memory", "reason_node")
    builder.add_conditional_edges(
        "retrieve_web",
        after_retrieve_or_web,
        {"reason_node": "reason_node", "browse_page": "browse_page"},
    )
    builder.add_edge("browse_page", "reason_node")
    builder.add_edge("memory_update", "reason_node")
    builder.add_edge("image_process", "reason_node")

    # reason_node -> memory_reflect -> 循环控制
    builder.add_edge("reason_node", "memory_reflect")

    # memory_reflect -> 继续循环或结束
    builder.add_conditional_edges(
        "memory_reflect",
        should_continue,
        {"router": "router", "reason_node": "reason_node", "__end__": END},
    )

    # ── 编译 ──────────────────────────────────────────────────────────────
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("[agent_graph] compiled successfully")
    return graph


def build_react_agent(
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    system_prompt: str = "",
    enable_middleware: bool = True,
) -> Any:
    """
    构建轻量级 ReAct Agent — 基于 create_react_agent（用于快速验证 01-06 阶段）.

    对话策略：
    - 当 state 中最后一条为 HumanMessage 时，bind_tools(..., tool_choice=\"required\")，
      强制本轮至少调用一个工具（满足「每轮先检索/再回答」）。
    - 工具返回后最后一条为 ToolMessage，使用 tool_choice=\"auto\"，允许模型继续多轮工具或输出最终答复。

    Args:
        llm: LangChain LLM
        checkpointer: SqliteSaver
        system_prompt: System Prompt 字符串
        enable_middleware: 启用 PII 脱敏

    Returns:
        ReAct Agent Executor
    """
    from langchain_core.messages import HumanMessage
    from langgraph.prebuilt import create_react_agent

    tools = [
        web_search,
        browse_page,
        calculator,
        memory_search,
        knowledge_base_search,
        save_memory,
        process_image,
    ]

    pre_hook = pii_pre_model_hook if enable_middleware else None

    def _bound_model_for_turn(state: Any, runtime: Any) -> Any:
        msgs = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        last = msgs[-1] if msgs else None
        tool_choice = "required" if isinstance(last, HumanMessage) else "auto"
        try:
            return llm.bind_tools(tools, tool_choice=tool_choice, parallel_tool_calls=True)
        except Exception as e:
            logger.warning("[react_agent] bind_tools(tool_choice=%s) failed: %s; fallback auto", tool_choice, e)
            return llm.bind_tools(tools, tool_choice="auto")

    agent = create_react_agent(
        _bound_model_for_turn,
        tools=tools,
        checkpointer=checkpointer,
        prompt=system_prompt or None,
        pre_model_hook=pre_hook,
    )

    logger.info("[react_agent] compiled (dynamic tool_choice: HumanMessage→required, else→auto)")
    return agent
