"""Anthropic Claude LLM client — via langchain-anthropic."""

from typing import Any

from langchain_core.language_models import BaseChatModel

from src.config import (
    ANTHROPIC_API_KEY,
    RUNTIME_API_KEYS,
    DEFAULT_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
)


def init_claude_llm(
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    streaming: bool = True,
) -> BaseChatModel:
    """Initialize Anthropic Claude chat model via langchain-anthropic."""
    api_key = RUNTIME_API_KEYS.get("claude", ANTHROPIC_API_KEY)
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        api_key=api_key,
    )
