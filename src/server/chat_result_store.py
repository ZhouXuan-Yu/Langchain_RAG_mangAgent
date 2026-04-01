"""Thread-safe result store for chat tasks — lives in memory, cleaned up after TTL."""

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Results older than this many seconds are cleaned up
_RESULT_TTL_SECONDS = 3600  # 1 hour


class ChatResultStore:
    """
    In-memory, asyncio.Lock-protected result store for chat tasks.

    Each job_id maps to:
        {
            "status": "pending" | "running" | "done" | "failed",
            "chunks": [...],          # text chunks appended during streaming
            "error": None | str,
            "created_at": float,
            "updated_at": float,
        }
    """

    def __init__(self):
        self._store: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    # ── Mutation ────────────────────────────────────────────────────────────────

    async def create(self, job_id: str) -> None:
        """Initialize a new result entry for job_id."""
        now = time.time()
        async with self._lock:
            self._store[job_id] = {
                "status": "pending",
                "chunks": [],
                "error": None,
                "created_at": now,
                "updated_at": now,
            }

    async def set_status(self, job_id: str, status: str) -> None:
        """Update the status of a job."""
        async with self._lock:
            if job_id in self._store:
                self._store[job_id]["status"] = status
                self._store[job_id]["updated_at"] = time.time()

    async def append_chunk(self, job_id: str, chunk: str) -> None:
        """Append a text chunk to a running job."""
        async with self._lock:
            if job_id in self._store:
                self._store[job_id]["chunks"].append(chunk)
                self._store[job_id]["updated_at"] = time.time()

    async def set_done(self, job_id: str) -> None:
        async with self._lock:
            if job_id in self._store:
                self._store[job_id]["status"] = "done"
                self._store[job_id]["updated_at"] = time.time()

    async def set_failed(self, job_id: str, error: str) -> None:
        async with self._lock:
            if job_id in self._store:
                self._store[job_id]["status"] = "failed"
                self._store[job_id]["error"] = error
                self._store[job_id]["updated_at"] = time.time()

    # ── Query ──────────────────────────────────────────────────────────────────

    async def get(self, job_id: str) -> dict | None:
        """Return a snapshot of the job's result data (deep copy)."""
        async with self._lock:
            entry = self._store.get(job_id)
            if entry is None:
                return None
            # Return a copy so callers can't mutate the store
            return {
                "status": entry["status"],
                "chunks": list(entry["chunks"]),
                "error": entry["error"],
                "created_at": entry["created_at"],
                "updated_at": entry["updated_at"],
            }

    async def exists(self, job_id: str) -> bool:
        async with self._lock:
            return job_id in self._store

    # ── Cleanup ────────────────────────────────────────────────────────────────

    async def cleanup_expired(self) -> int:
        """Remove results older than _RESULT_TTL_SECONDS. Returns count of removed entries."""
        now = time.time()
        removed = 0
        async with self._lock:
            expired = [
                jid for jid, data in self._store.items()
                if now - data["updated_at"] > _RESULT_TTL_SECONDS
                and data["status"] in ("done", "failed")
            ]
            for jid in expired:
                del self._store[jid]
                removed += 1
        if removed:
            logger.debug("[result_store] cleaned up %d expired entries", removed)
        return removed

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def stats(self) -> dict[str, Any]:
        async with self._lock:
            total = len(self._store)
            by_status = {}
            for entry in self._store.values():
                by_status[entry["status"]] = by_status.get(entry["status"], 0) + 1
            return {"total": total, "by_status": by_status}
