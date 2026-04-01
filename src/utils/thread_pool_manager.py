"""Global thread pool lifecycle manager — decouples LLM/DB/vector I/O from FastAPI's async event loop."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any

from src.config import (
    THREAD_POOL_LLM_SIZE,
    THREAD_POOL_DB_SIZE,
    THREAD_POOL_VECTOR_SIZE,
)

logger = logging.getLogger(__name__)


class ThreadPoolManager:
    """
    Global thread pool lifecycle manager.

    Three dedicated pools:
      - llm_pool:    LLM inference calls (network I/O + JSON parsing)
      - db_pool:     SQLite / file I/O operations
      - vector_pool: ChromaDB / embedding operations
    """

    def __init__(self):
        self.llm_pool = ThreadPoolExecutor(
            max_workers=THREAD_POOL_LLM_SIZE,
            thread_name_prefix="llm_worker_",
        )
        self.db_pool = ThreadPoolExecutor(
            max_workers=THREAD_POOL_DB_SIZE,
            thread_name_prefix="db_worker_",
        )
        self.vector_pool = ThreadPoolExecutor(
            max_workers=THREAD_POOL_VECTOR_SIZE,
            thread_name_prefix="vector_worker_",
        )
        logger.info(
            "[pool_manager] initialized — llm=%d, db=%d, vector=%d",
            THREAD_POOL_LLM_SIZE,
            THREAD_POOL_DB_SIZE,
            THREAD_POOL_VECTOR_SIZE,
        )

    def submit_async(self, pool_name: str, fn: Callable, *args: Any, **kwargs: Any):
        """
        Submit a synchronous callable to the named pool, returning an asyncio.Future.

        pool_name: "llm" | "db" | "vector"
        """
        pool = {"llm": self.llm_pool, "db": self.db_pool, "vector": self.vector_pool}.get(pool_name)
        if pool is None:
            raise ValueError(f"Unknown pool: {pool_name}")

        async def _wrapper():
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(pool, lambda: fn(*args, **kwargs))

        return _wrapper()

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully shut down all thread pools."""
        logger.info("[pool_manager] shutting down thread pools (wait=%s)...", wait)
        self.llm_pool.shutdown(wait=wait)
        self.db_pool.shutdown(wait=wait)
        self.vector_pool.shutdown(wait=wait)
        logger.info("[pool_manager] shutdown complete")

    # ── Convenience shortcuts ────────────────────────────────────────────────────

    def run_in_llm_pool(self, fn: Callable, *args: Any, **kwargs: Any):
        """Submit sync fn to llm_pool, return asyncio.Future."""
        return self.submit_async("llm", fn, *args, **kwargs)

    def run_in_db_pool(self, fn: Callable, *args: Any, **kwargs: Any):
        """Submit sync fn to db_pool, return asyncio.Future."""
        return self.submit_async("db", fn, *args, **kwargs)

    def run_in_vector_pool(self, fn: Callable, *args: Any, **kwargs: Any):
        """Submit sync fn to vector_pool, return asyncio.Future."""
        return self.submit_async("vector", fn, *args, **kwargs)


# ── Global singleton ────────────────────────────────────────────────────────────
_pool_manager: ThreadPoolManager | None = None


def get_pool_manager() -> ThreadPoolManager:
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = ThreadPoolManager()
    return _pool_manager
