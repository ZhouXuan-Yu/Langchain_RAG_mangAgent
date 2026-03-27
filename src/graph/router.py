"""Router — 16: LangGraph 条件路由逻辑."""

import logging
from typing import Literal

from src.graph.state import AgentState

logger = logging.getLogger(__name__)


def route_decision(state: AgentState) -> Literal["retrieve_memory", "memory_update", "retrieve_web", "reason_node", "image_process"]:
    """
    条件路由入口 — 根据 router_node 设置的 route_decision 决定下一跳.

    对应关系：
    - memory_retrieve → retrieve_memory
    - memory_update → memory_update
    - web_search → retrieve_web
    - image_process → image_process
    - direct_reason → reason_node
    """
    route = state.get("route_decision", "direct_reason")

    route_map = {
        "memory_retrieve": "retrieve_memory",
        "memory_update": "memory_update",
        "web_search": "retrieve_web",
        "image_process": "image_process",
        "direct_reason": "reason_node",
        "reason_node": "reason_node",
    }

    next_node = route_map.get(route, "reason_node")
    logger.info(f"[route_decision] {route} -> {next_node}")
    return next_node


def after_retrieve_or_web(state: AgentState) -> Literal["reason_node", "browse_page"]:
    """
    记忆/网页检索后的路由 — 判断是否需要深度抓取页面.

    若 web_context 包含"深度抓取"或"详细内容"请求，则跳转 browse_page。
    否则直接进入 reason_node。
    """
    last_msg = ""
    messages = state.get("messages", [])
    if messages:
        last_item = messages[-1]
        if hasattr(last_item, "content"):
            last_msg = last_item.content.lower() if isinstance(last_item.content, str) else ""
        elif isinstance(last_item, dict) and "content" in last_item:
            last_msg = last_item["content"].lower() if isinstance(last_item["content"], str) else ""

    if any(k in last_msg for k in ["详细", "深度", "完整", "具体代码", "怎么写", "详细说明"]):
        return "browse_page"

    return "reason_node"
