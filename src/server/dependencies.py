"""Dependencies — 全局 Agent 单例、模型切换、共享状态.

避免每次请求重建 LangGraph agent，使用单例缓存。
"""

import asyncio
import logging
from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver

logger = logging.getLogger(__name__)

from src.config import (
    DEFAULT_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    CHECKPOINT_PATH,
    CHROMA_PATH,
    USER_NAME,
    USER_TECH_STACK,
    USER_HARDWARE,
    USER_PROJECTS,
)
from src.llm import init_deepseek_llm
from src.memory.sqlite_store import get_sqlite_checkpointer, get_async_sqlite_checkpointer
from src.graph import build_react_agent

# ── 全局 Agent 注册表（按模型缓存）────────────────────────────────────────────
_agent_registry: dict[str, Any] = {}

# ── 当前活动模型 ──────────────────────────────────────────────────────────────
_current_model: str = DEFAULT_MODEL
_current_temperature: float = TEMPERATURE
_current_max_tokens: int = MAX_TOKENS

# ── 全局异步 Checkpointer（跨模型共享）────────────────────────────────────────
_async_checkpointer: Optional[BaseCheckpointSaver] = None
_async_checkpointer_lock = asyncio.Lock()


def set_temperature(temp: float) -> None:
    """更新 temperature 参数，下次请求时生效."""
    global _current_temperature
    _current_temperature = float(temp)
    # 清除 agent 缓存，使新 agent 使用新参数
    _agent_registry.clear()
    logger.info(f"[config] temperature set to {temp}")


def set_max_tokens(max_tokens: int) -> None:
    """更新 max_tokens 参数，下次请求时生效."""
    global _current_max_tokens
    _current_max_tokens = int(max_tokens)
    _agent_registry.clear()
    logger.info(f"[config] max_tokens set to {max_tokens}")


async def _get_async_checkpointer() -> BaseCheckpointSaver:
    """懒加载异步 SQLite checkpointer（延迟到首次请求，避免启动时磁盘错误）。"""
    global _async_checkpointer
    if _async_checkpointer is None:
        async with _async_checkpointer_lock:
            if _async_checkpointer is None:
                _async_checkpointer = await get_async_sqlite_checkpointer(str(CHECKPOINT_PATH))
    return _async_checkpointer


async def get_agent(model: Optional[str] = None) -> Any:
    """
    获取（或按需构建）指定模型的 ReAct Agent.

    使用 create_react_agent 构建，完整支持 LLM 工具调用和 astream_events。
    """
    key = model or _current_model
    if key not in _agent_registry:
        logger.info(f"[agent_registry] building new agent for model={key}")
        llm = init_deepseek_llm(
            model=key,
            temperature=_current_temperature,
            max_tokens=_current_max_tokens,
            streaming=True,
        )
        checkpointer = await _get_async_checkpointer()
        system_prompt = (
            f"你是一个专业的 AI 助手，用户名为「{USER_NAME}」。\n"
            f"用户的技术栈: {', '.join(USER_TECH_STACK)}。\n"
            f"用户的 GPU 硬件: {USER_HARDWARE}。\n"
            f"用户的项目: {', '.join(USER_PROJECTS)}。\n"
            "请结合用户的背景信息，提供精准、有帮助的回答。\n"
            "当需要搜索最新信息时，使用 web_search 工具。\n"
            "当需要检索用户个人长期记忆/偏好时，使用 memory_search 工具。\n"
            "当问题涉及用户在知识库中上传的文档（PDF/Word 等）时，使用 knowledge_base_search 工具。\n"
            "当需要保存重要信息到记忆时，使用 save_memory 工具。\n"
            "当需要计算时，使用 calculator 工具。"
        )
        _agent_registry[key] = build_react_agent(
            llm,
            checkpointer=checkpointer,
            system_prompt=system_prompt,
            enable_middleware=True,
        )
        logger.info(f"[agent_registry] agent cached: key={key}")
    return _agent_registry[key]


def switch_model(new_model: str) -> dict:
    """
    切换 LLM 模型。

    清除旧 agent 缓存，下次请求时按需重建。
    """
    global _current_model
    _current_model = new_model
    logger.info(f"[model_switch] switching to: {new_model}")
    _agent_registry.clear()
    return {"status": "ok", "current_model": _current_model}


def get_current_model() -> str:
    return _current_model


async def get_checkpointer() -> BaseCheckpointSaver:
    """暴露 checkpointer 供 API 路由使用."""
    return await _get_async_checkpointer()
