"""DeepSeek LLM client — 01-03: DeepSeek API 调用与消息结构."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.outputs import ChatResult

from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEFAULT_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
)


def init_deepseek_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    streaming: bool = True,
) -> BaseChatModel:
    """Initialize DeepSeek chat model via OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )


def build_messages(
    system_prompt: str,
    user_input: str,
    history: list[BaseMessage] | None = None,
) -> list[BaseMessage]:
    """Assemble System + History + Human message chain."""
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    if history:
        messages.extend(history)
    messages.append(HumanMessage(content=user_input))
    return messages


def chat(
    llm: BaseChatModel,
    system_prompt: str,
    user_input: str,
    history: list[BaseMessage] | None = None,
) -> BaseMessage:
    """Single-turn chat: build messages and invoke LLM."""
    messages = build_messages(system_prompt, user_input, history)
    return llm.invoke(messages)


def chat_streaming(
    llm: BaseChatModel,
    system_prompt: str,
    user_input: str,
    history: list[BaseMessage] | None = None,
) -> Any:
    """Streaming chat: return the stream generator."""
    messages = build_messages(system_prompt, user_input, history)
    return llm.stream(messages)


def trim_conversation(
    messages: list[BaseMessage],
    max_tokens: int = MAX_TOKENS,
    llm: BaseChatModel | None = None,
) -> list[BaseMessage]:
    """
    Trim conversation history to fit within context window.
    Uses the LLM as token counter if provided.
    """
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",
        token_counter=llm,
        include_system=True,
    )
