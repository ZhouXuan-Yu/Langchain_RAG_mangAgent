"""OpenAI-compatible LLM client — uses ChatOpenAI with configurable base_url.

Supports OpenAI itself as well as any OpenAI-compatible API (e.g. proxies,
local models, or any provider that implements the OpenAI chat completions API).
"""

from typing import Optional

from langchain_core.language_models import BaseChatModel

from src.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    RUNTIME_API_KEYS,
    DEFAULT_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
)


def init_openai_llm(
    model: str = "gpt-4o",
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    streaming: bool = True,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """Initialize an OpenAI-compatible chat model via ChatOpenAI.

    If base_url is None, falls back to the OPENAI_BASE_URL config.
    If api_key is None, uses the RUNTIME_API_KEYS["openai"] value.
    """
    from langchain_openai import ChatOpenAI

    effective_api_key = api_key if api_key else RUNTIME_API_KEYS.get("openai", OPENAI_API_KEY)
    effective_base_url = base_url if base_url else OPENAI_BASE_URL

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        api_key=effective_api_key,
        base_url=effective_base_url,
    )
