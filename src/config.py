"""Configuration management for the LangChain RAG Agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")

# ── HTTP Server ───────────────────────────────────────────────────────────────
APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

# ── Model Config ──────────────────────────────────────────────────────────────
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "deepseek-chat")
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "8000"))

# DeepSeek base URL (OpenAI-compatible endpoint)
DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"

# ── Storage Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
CHROMA_PATH: Path = PROJECT_ROOT / os.getenv("CHROMA_PATH", "data/chroma_db")
CHECKPOINT_PATH: Path = PROJECT_ROOT / os.getenv("CHECKPOINT_PATH", "data/checkpointer/checkpoints.db")

# ── ChromaDB Config ──────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME: str = "user_memory"
UPSERT_SIMILARITY_THRESHOLD: float = 0.85

# ── Memory Config ─────────────────────────────────────────────────────────────
MAX_CONTEXT_TOKENS: int = 8000   # trim_messages threshold
WEB_CONTENT_MAX_CHARS: int = 3000  # summarizer max chars

# ── PDF OCR ──────────────────────────────────────────────────────────────────
PDF_OCR_ENABLED: bool = os.getenv("PDF_OCR_ENABLED", "true").lower() in ("1", "true", "yes")
PDF_OCR_MAX_PAGES: int = int(os.getenv("PDF_OCR_MAX_PAGES", "40"))
PDF_OCR_DPI: float = float(os.getenv("PDF_OCR_DPI", "150"))
PDF_OCR_GPU: bool = os.getenv("PDF_OCR_GPU", "").lower() in ("1", "true", "yes")

# ── User Profile ──────────────────────────────────────────────────────────────
USER_NAME: str = "周轩"
USER_TECH_STACK: list[str] = ["Python", "Rust", "FastAPI", "LangChain", "LangGraph"]
USER_HARDWARE: str = "RTX 5060"
USER_PROJECTS: list[str] = ["人工智能", "深度学习", "计算机视觉", "自然语言处理"]

# ── LangSmith ────────────────────────────────────────────────────────────────
LANGSMITH_PROJECT: str = "deepseek-rag-agent"
LANGSMITH_TRACING: bool = bool(LANGSMITH_API_KEY)

# ── Model Providers ─────────────────────────────────────────────────────────
# Maps provider ID → config dict
# api_key is loaded at runtime (may be overridden via /api/config/update)
MODEL_PROVIDERS: dict[str, dict] = {
    "deepseek": {
        "name": "DeepSeek",
        "requires_api_key": True,
        "default_models": [
            {"id": "deepseek-chat",       "name": "DeepSeek V3",  "description": "最新 DeepSeek V3 模型，适合日常对话"},
            {"id": "deepseek-reasoner",   "name": "DeepSeek R1",  "description": "推理模型，适合复杂问题"},
        ],
        "base_url": "https://api.deepseek.com/v1",
        "client": "openai_compat",   # uses ChatOpenAI
    },
    "claude": {
        "name": "Claude (Anthropic)",
        "requires_api_key": True,
        "default_models": [
            {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "description": "能力与速度均衡的主力模型"},
            {"id": "claude-3-5-haiku-20241007", "name": "Claude 3.5 Haiku",  "description": "轻量快速，适合简单任务"},
            {"id": "claude-3-opus-20240229",    "name": "Claude 3 Opus",     "description": "最强能力，适合复杂推理"},
        ],
        "base_url": "https://api.anthropic.com/v1",
        "client": "anthropic",
    },
    "openai": {
        "name": "OpenAI",
        "requires_api_key": True,
        "default_models": [
            {"id": "gpt-4o",        "name": "GPT-4o",        "description": "最新多模态旗舰模型"},
            {"id": "gpt-4o-mini",   "name": "GPT-4o Mini",   "description": "轻量快速，成本低"},
            {"id": "gpt-4-turbo",   "name": "GPT-4 Turbo",   "description": "快速版 GPT-4"},
        ],
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "client": "openai_compat",
    },
    "gemini": {
        "name": "Gemini (Google)",
        "requires_api_key": True,
        "default_models": [
            {"id": "gemini-2.0-flash",  "name": "Gemini 2.0 Flash",  "description": "最新快速模型"},
            {"id": "gemini-1.5-flash",  "name": "Gemini 1.5 Flash",  "description": "高性价比快速模型"},
            {"id": "gemini-1.5-pro",   "name": "Gemini 1.5 Pro",   "description": "最强能力，适合复杂任务"},
        ],
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "client": "google_genai",
    },
}


# ── Runtime API Keys (can be updated via /api/config/update) ─────────────────
# These mirror the .env values but are mutable at runtime
RUNTIME_API_KEYS: dict[str, str] = {
    "deepseek": DEEPSEEK_API_KEY,
    "claude":   ANTHROPIC_API_KEY,
    "openai":   OPENAI_API_KEY,
    "gemini":   GOOGLE_API_KEY,
}


def get_runtime_api_key(provider: str) -> str:
    """Return the current runtime API key for a provider."""
    return RUNTIME_API_KEYS.get(provider, "")


def set_runtime_api_key(provider: str, key: str) -> None:
    """Update the runtime API key for a provider (in-memory)."""
    RUNTIME_API_KEYS[provider] = key


def is_provider_configured(provider: str) -> bool:
    """Return True if the provider has a non-empty API key."""
    return bool(get_runtime_api_key(provider))


# ── Validation ───────────────────────────────────────────────────────────────
if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. "
        "Copy .env.example to .env and fill in your API keys."
    )
