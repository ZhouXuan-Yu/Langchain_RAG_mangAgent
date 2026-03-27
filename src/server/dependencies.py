"""Dependencies — 全局 Agent 单例、模型切换、共享状态.

避免每次请求重建 LangGraph agent，使用单例缓存。
支持多 Provider 模型（DeepSeek / Claude / OpenAI / Gemini）。
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
    RUNTIME_API_KEYS,
)
from src.memory.sqlite_store import get_sqlite_checkpointer, get_async_sqlite_checkpointer
from src.graph import build_react_agent

# ── 全局 Agent 注册表（按 "provider/model" 缓存）──────────────────────────────
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


def _parse_model_key(model: str) -> tuple[str, str]:
    """
    解析模型字符串，返回 (provider, model_name)。

    支持两种格式：
      - "deepseek-chat"             → ("deepseek", "deepseek-chat")
      - "claude/claude-3-5-sonnet"  → ("claude", "claude-3-5-sonnet")
    """
    if "/" in model:
        parts = model.split("/", 1)
        return parts[0], parts[1]
    # Default: treat as deepseek if it's a known deepseek model, else "deepseek"
    return "deepseek", model


def _init_llm_for_model(
    provider: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    streaming: bool = True,
) -> Any:
    """根据 provider 初始化对应的 LLM client."""
    if provider == "claude":
        from src.llm.claude_client import init_claude_llm

        return init_claude_llm(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )
    elif provider == "openai":
        from src.llm.openai_client import init_openai_llm

        return init_openai_llm(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )
    elif provider == "gemini":
        from src.llm.google_client import init_gemini_llm

        return init_gemini_llm(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )
    else:
        # Default to DeepSeek
        from src.llm.deepseek_client import init_deepseek_llm

        return init_deepseek_llm(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )


async def get_agent(model: Optional[str] = None) -> Any:
    """
    获取（或按需构建）指定模型的 ReAct Agent.

    模型字符串格式：支持 "provider/model" 或纯模型名（默认 deepseek）。
    内部按 "provider/model" 缓存 agent 实例。
    """
    key = model or _current_model
    if key not in _agent_registry:
        logger.info(f"[agent_registry] building new agent for model={key}")
        provider, model_name = _parse_model_key(key)
        llm = _init_llm_for_model(
            provider=provider,
            model_name=model_name,
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
            "\n"
            "## 工具使用（强制）\n"
            "在收到用户每一轮新问题时，你必须先调用工具获取事实依据，禁止在未使用工具的情况下凭猜测作答。\n"
            "优先组合使用多种工具：例如同时 memory_search + knowledge_base_search；需要时效信息时再加 web_search。\n"
            "仅在工具结果返回后，再基于结果用中文给出简洁、结构化的最终回答。\n"
            "\n"
            "## 输出格式\n"
            "使用 Markdown，但列表请用「顶格」或「最多一级」缩进：用 `- ` 开头，子项也用 `- ` 并尽量少缩进空格，"
            "不要使用多层嵌套缩进列表，避免前端渲染成阶梯状。\n"
            "需要分点时用 `1.` `2.` 有序列表或短段落小标题（###）。\n"
            "\n"
            "工具说明：web_search 联网检索；browse_page 抓取网页正文；memory_search 检索用户长期记忆；"
            "knowledge_base_search 检索知识库文档；save_memory 保存重要事实；calculator 计算；process_image 图像相关。"
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
