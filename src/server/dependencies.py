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
)
from src.llm import init_deepseek_llm
from src.memory.sqlite_store import get_sqlite_checkpointer, get_async_sqlite_checkpointer
from src.graph import build_agent_graph

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
    获取（或按需构建）指定模型的编译 Agent Graph.

    同一模型的 Agent 只构建一次，后续复用。
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
        _agent_registry[key] = build_agent_graph(
            llm,
            checkpointer=checkpointer,
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
