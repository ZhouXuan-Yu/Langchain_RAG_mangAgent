"""LangSmith client — 22-24: 全链路追踪集成."""

import logging
from typing import Any

from src.config import LANGSMITH_API_KEY, LANGSMITH_PROJECT, LANGSMITH_TRACING

logger = logging.getLogger(__name__)


def setup_langsmith() -> None:
    """
    配置 LangSmith 环境变量 — 启用全链路追踪.

    启用后，LangGraph 的所有节点输入输出都会自动上报到 LangSmith Dashboard，
    可用于调试 Agent 的"幻觉"问题和分析思考路径。

    使用方式：
        from src.supervision.langsmith_client import setup_langsmith
        setup_langsmith()
        # 然后正常使用 build_agent_graph()
    """
    if not LANGSMITH_API_KEY:
        logger.warning(
            "LANGSMITH_API_KEY not set — LangSmith tracing disabled. "
            "Set LANGSMITH_API_KEY in .env to enable tracing."
        )
        return

    import os
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    logger.info(f"LangSmith tracing enabled — project: {LANGSMITH_PROJECT}")


def get_langsmith_config() -> dict[str, Any]:
    """返回 LangSmith 追踪配置字典（用于 LangGraph callbacks）."""
    if not LANGSMITH_TRACING:
        return {}
    return {
        "configurable": {
            "callbacks": [],  # LangGraph auto-uses LANGCHAIN_TRACING_V2 env
        }
    }
