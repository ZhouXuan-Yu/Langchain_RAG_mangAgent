"""Input guard — 10-12: 工具输入验证中间件."""

import logging
from typing import Any, Optional

from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)


def validate_search_query(query: str) -> Optional[str]:
    """
    验证 web_search 工具的输入 — 防止 Agent 生成空或异常关键词.

    Returns:
        None 如果验证通过
        str 错误消息 如果验证失败（中间件会跳过工具调用）
    """
    if not query or not query.strip():
        return "[系统] 搜索关键词不能为空，已跳过搜索步骤。"
    if len(query) > 500:
        return f"[系统] 搜索关键词过长（{len(query)}字符 > 500），已跳过。"
    # 过滤明显的注入尝试
    if any(char in query for char in ["--", "UNION", "DROP ", "DELETE ", "<script"]):
        return f"[系统] 搜索关键词包含可疑字符，已跳过。"
    return None


def validate_memory_fact(fact: str) -> Optional[str]:
    """验证 save_memory 工具的输入."""
    if not fact or not fact.strip():
        return "[系统] 记忆内容不能为空。"
    if len(fact) > 5000:
        return f"[系统] 记忆内容过长（{len(fact)}字符），已截断。"
    return None


def before_tool(state: dict, config: dict | None = None) -> dict:
    """
    LangGraph before_tool 中间件 — 在工具调用前验证参数.

    当前验证规则：
    - web_search: query 不能为空、不能过长、不能含注入字符
    - save_memory: fact 不能为空、不能过长

    验证失败时返回 ToolMessage 错误响应，跳过实际工具执行。
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1]

    # 从 last_Additional_kwargs 获取 tool_calls
    tool_calls: list[dict] = []
    if hasattr(last_msg, "additional_kwargs") and last_msg.additional_kwargs:
        tool_calls = last_msg.additional_kwargs.get("tool_calls", [])
    elif hasattr(last_msg, "tool_calls"):
        tool_calls = getattr(last_msg, "tool_calls", [])

    for tc in tool_calls:
        tool_name = tc.get("name", "")
        args = tc.get("args", {})

        if tool_name == "web_search":
            err = validate_search_query(args.get("query", ""))
            if err:
                logger.warning(f"[input_guard] web_search blocked: {err}")
                return {
                    "messages": [
                        ToolMessage(
                            content=err,
                            tool_call_id=tc.get("id", ""),
                            name=tool_name,
                        )
                    ]
                }

        elif tool_name == "save_memory":
            err = validate_memory_fact(args.get("fact", ""))
            if err:
                logger.warning(f"[input_guard] save_memory blocked: {err}")
                return {
                    "messages": [
                        ToolMessage(
                            content=err,
                            tool_call_id=tc.get("id", ""),
                            name=tool_name,
                        )
                    ]
                }

    return {}
