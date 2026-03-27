"""Configuration management for the LangChain RAG Agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ────────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")

# ── Model Config ──────────────────────────────────────────────────────────
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "deepseek-chat")
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "8000"))

# DeepSeek base URL (OpenAI-compatible endpoint)
DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"

# ── Storage Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
CHROMA_PATH: Path = PROJECT_ROOT / os.getenv("CHROMA_PATH", "data/chroma_db")
CHECKPOINT_PATH: Path = PROJECT_ROOT / os.getenv("CHECKPOINT_PATH", "data/checkpointer/checkpoints.db")

# ── ChromaDB Config ───────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME: str = "user_memory"
UPSERT_SIMILARITY_THRESHOLD: float = 0.85

# ── Memory Config ─────────────────────────────────────────────────────────
MAX_CONTEXT_TOKENS: int = 8000   # trim_messages threshold
WEB_CONTENT_MAX_CHARS: int = 3000  # summarizer max chars

# ── User Profile (customizable) ────────────────────────────────────────────
USER_NAME: str = "周暄"
USER_TECH_STACK: list[str] = ["Python", "Rust", "FastAPI", "LangChain", "LangGraph"]
USER_HARDWARE: str = "RTX 5060"
USER_PROJECTS: list[str] = ["智程导航", "智眸千析", "火灾检测", "手语识别"]

# ── LangSmith ─────────────────────────────────────────────────────────────
LANGSMITH_PROJECT: str = "deepseek-rag-agent"
LANGSMITH_TRACING: bool = bool(LANGSMITH_API_KEY)

# ── Validation ────────────────────────────────────────────────────────────
if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. "
        "Copy .env.example to .env and fill in your API keys."
    )
