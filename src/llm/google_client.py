"""Google Gemini LLM client — via langchain-google-genai."""

from typing import Any

from langchain_core.language_models import BaseChatModel

from src.config import (
    GOOGLE_API_KEY,
    RUNTIME_API_KEYS,
    TEMPERATURE,
    MAX_TOKENS,
)


def init_gemini_llm(
    model: str = "gemini-2.0-flash",
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    streaming: bool = True,
) -> BaseChatModel:
    """Initialize Google Gemini model via langchain-google-genai."""
    api_key = RUNTIME_API_KEYS.get("gemini", GOOGLE_API_KEY)
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        streaming=streaming,
        google_api_key=api_key,
    )
