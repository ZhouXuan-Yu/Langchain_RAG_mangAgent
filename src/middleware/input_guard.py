"""Input guard — 10-12: 工具输入验证中间件."""

import logging
from typing import Any, Optional, Callable

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

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


_TOOL_VALIDATORS: dict[str, Callable[[dict], Optional[str]]] = {
    "web_search": lambda args: validate_search_query(args.get("query", "")),
    "save_memory": lambda args: validate_memory_fact(args.get("fact", "")),
}


def wrap_tool_validation(tool: BaseTool) -> BaseTool:
    """
    包装工具，在调用前验证参数。

    验证失败时直接返回 ToolMessage 错误，不执行原工具。
    langgraph 1.0+ 的 create_react_agent 已移除 before_tool 参数，
    改用工具包装器实现输入验证。
    """
    tool_name = tool.name
    validator = _TOOL_VALIDATORS.get(tool_name)
    if not validator:
        return tool

    original_invoke = tool.invoke
    original_ainvoke = getattr(tool, "ainvoke", None)

    def safe_invoke(input_: Any, config: Any = None, **kwargs: Any) -> Any:
        args = input_ if isinstance(input_, dict) else {"input": input_}
        err = validator(args)
        if err:
            logger.warning("[input_guard] %s blocked: %s", tool_name, err)
            return ToolMessage(content=err, name=tool_name, tool_call_id="")
        return original_invoke(input_, config=config, **kwargs)

    tool.invoke = safe_invoke

    if original_ainvoke is not None:

        async def safe_ainvoke(input_: Any, config: Any = None, **kwargs: Any) -> Any:
            args = input_ if isinstance(input_, dict) else {"input": input_}
            err = validator(args)
            if err:
                logger.warning("[input_guard] %s blocked (async): %s", tool_name, err)
                return ToolMessage(content=err, name=tool_name, tool_call_id="")
            return await original_ainvoke(input_, config=config, **kwargs)

        tool.ainvoke = safe_ainvoke

    return tool
