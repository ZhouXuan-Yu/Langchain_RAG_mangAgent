"""Graph state definition — 05+16: LangGraph AgentState."""

from typing import TypedDict, Annotated, Any, Sequence
from operator import add as operator_add

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.memory.memory_schema import MemoryRecord


def add_messages(left: list[BaseMessage], right: list[BaseMessage]) -> list[BaseMessage]:
    """
    自定义消息合并函数，替代 langchain_core.messages.add_messages。
    将右侧消息追加到左侧列表。
    """
    return left + right


class AgentState(TypedDict):
    """
    LangGraph Agent 完整状态定义.

    包含：
    - messages: 对话历史（通过 add_messages 合并）
    - thread_id: 会话隔离 ID
    - memory_context: ChromaDB 检索到的记忆上下文
    - web_context: 网页搜索结果
    - pending_memory: 待评估存入的新事实列表
    - memory_updated: 本轮是否更新了记忆
    - last_tool_result: 上次工具调用结果
    - turn_count: 回合计数器（防止无限循环）
    - route_decision: 当前路由决策（debug 用）
    """

    # 对话消息历史，自动追加合并
    messages: Annotated[list[BaseMessage], add_messages]

    # 会话隔离
    thread_id: str

    # 外部知识上下文
    memory_context: Annotated[list[str], operator_add]
    web_context: Annotated[list[str], operator_add]

    # 主动记忆
    pending_memory: list[MemoryRecord]

    # 状态标志
    memory_updated: bool
    last_tool_result: str | None
    turn_count: int
    route_decision: str | None
