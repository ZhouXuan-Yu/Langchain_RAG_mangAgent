"""Chat task executor — runs agent.astream_events() in the LLM thread pool so FastAPI's event loop stays free."""

import asyncio
import json
import logging
import threading
import uuid
from typing import Optional

from src.server.chat_result_store import ChatResultStore
from src.utils.thread_pool_manager import ThreadPoolManager, get_pool_manager

logger = logging.getLogger(__name__)

# Singleton instance
_executor: Optional["ChatTaskExecutor"] = None


def get_chat_executor() -> "ChatTaskExecutor":
    global _executor
    if _executor is None:
        _executor = ChatTaskExecutor(get_pool_manager())
    return _executor


class ChatTaskExecutor:
    """
    Executes chat tasks in the LLM thread pool.

    Usage:
        job_id = await executor.submit(message="...", thread_id="...", model="deepseek-chat")
        # ... later, in SSE endpoint:
        result = await executor.get_result(job_id)
    """

    def __init__(self, pool_manager: ThreadPoolManager):
        self._pool = pool_manager
        self._result_store = ChatResultStore()
        # Maps job_id -> threading.Event for cancellation
        self._cancel_tokens: dict[str, threading.Event] = {}
        self._token_lock = threading.Lock()

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    async def submit(
        self,
        message: str,
        thread_id: str,
        model: Optional[str] = None,
    ) -> str:
        """
        Submit a chat task to the LLM thread pool.

        Returns immediately with a job_id. Call get_result(job_id) to poll status.
        """
        job_id = f"chat_{uuid.uuid4().hex[:12]}"
        await self._result_store.create(job_id)

        # Register cancellation token
        cancel_event = threading.Event()
        with self._token_lock:
            self._cancel_tokens[job_id] = cancel_event

        # Kick off execution in the thread pool
        asyncio.create_task(self._run(job_id, message, thread_id, model))

        # Start cleanup loop lazily
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("[chat_executor] submitted job_id=%s thread_id=%s", job_id, thread_id)
        return job_id

    async def get_result(self, job_id: str) -> dict:
        """
        Return current snapshot of job result.

        Returns:
            {
                "status": "pending" | "running" | "done" | "failed" | "not_found",
                "chunks": [...],
                "error": None | str,
            }
        """
        result = await self._result_store.get(job_id)
        if result is None:
            return {"status": "not_found", "chunks": [], "error": None}
        return result

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job. Returns True if the job was found and cancelled."""
        with self._token_lock:
            token = self._cancel_tokens.pop(job_id, None)
        if token is None:
            return False
        token.set()
        await self._result_store.set_status(job_id, "failed")
        await self._result_store.set_failed(job_id, "Cancelled by user")
        logger.info("[chat_executor] cancelled job_id=%s", job_id)
        return True

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _run(
        self,
        job_id: str,
        message: str,
        thread_id: str,
        model: Optional[str],
    ) -> None:
        """
        The actual work runs synchronously in the LLM thread pool.
        We use run_in_executor so this async method doesn't block the event loop.
        """
        cancel_event: threading.Event
        with self._token_lock:
            cancel_event = self._cancel_tokens.get(job_id)

        def _sync_run():
            """Synchronous work — must run in thread pool."""
            from langchain_core.messages import HumanMessage
            from src.server.dependencies import get_agent

            # Get (or build) agent in thread — get_agent is thread-safe (cached)
            agent = get_agent(model)
            config = {"configurable": {"thread_id": thread_id}}

            seen_tool_starts: set[str] = set()
            seen_tool_ends: set[str] = set()

            def _event_run_id(ev: dict) -> str:
                return str(ev.get("run_id") or "")

            try:
                for event in agent.astream_events(
                    {"messages": [HumanMessage(content=message)], "thread_id": thread_id},
                    config=config,
                    version="v2",
                ):
                    # Poll cancellation
                    if cancel_event is not None and cancel_event.is_set():
                        logger.info("[chat_executor][_run] job_id=%s cancelled", job_id)
                        return

                    ev_type = event.get("event", "")

                    # on_chat_model_start — noop
                    if ev_type in ("on_chat_model_start", "chat_model_start"):
                        pass

                    # on_chat_model_stream — token chunk
                    elif ev_type in ("on_chat_model_stream", "chat_model_stream"):
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            content = chunk.content
                            if isinstance(content, list):
                                text_parts = []
                                for part in content:
                                    if isinstance(part, str):
                                        text_parts.append(part)
                                    elif isinstance(part, dict) and part.get("type") == "text":
                                        text_parts.append(part.get("text") or "")
                                content = "".join(text_parts)
                            skip = content.strip() in ("Tool", "Tool/use", "Invoking tool:", "=" * 20) or (
                                content.strip().startswith("=") and len(content.strip()) < 5
                            )
                            if not skip:
                                _append_chunk(job_id, content)

                    # on_tool_start
                    elif ev_type == "on_tool_start":
                        tool_name = event.get("name", "unknown")
                        tool_input = event.get("data", {}).get("input", {})
                        rid = _event_run_id(event)
                        dedup_key = rid or f"{tool_name}:{json.dumps(tool_input, ensure_ascii=False)[:120]}"
                        if dedup_key not in seen_tool_starts:
                            seen_tool_starts.add(dedup_key)
                            _append_chunk(job_id, f"[TOOL_START:{tool_name}]")

                    # on_tool_end
                    elif ev_type == "on_tool_end":
                        tool_name = event.get("name", "unknown")
                        tool_output = event.get("data", {}).get("output", "")
                        rid = _event_run_id(event)
                        dedup_key = rid or tool_name
                        if dedup_key not in seen_tool_ends:
                            seen_tool_ends.add(dedup_key)
                            output_str = str(tool_output)[:200]
                            _append_chunk(job_id, f"[TOOL_END:{tool_name}:{output_str}]")

                    # on纳德_end / on_text_end / finish — noop
                    elif ev_type in ("on纳德_end", "on_text_end", "finish"):
                        pass

            except Exception as e:
                logger.error("[chat_executor][_run] job_id=%s error: %s", job_id, e)
                _set_failed(job_id, str(e))

        def _append_chunk(jid: str, chunk: str):
            """Called from the worker thread — schedule async write via threadsafe."""
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self._result_store.append_chunk(jid, chunk))
                )
            except RuntimeError:
                pass

        def _set_failed(jid: str, error: str):
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self._result_store.set_failed(jid, error))
                )
            except RuntimeError:
                pass

        await self._result_store.set_status(job_id, "running")

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._pool.llm_pool, _sync_run)
        except Exception as e:
            logger.error("[chat_executor][_run] job_id=%s run_in_executor failed: %s", job_id, e)
            await self._result_store.set_failed(job_id, str(e))
        finally:
            await self._result_store.set_done(job_id)
            with self._token_lock:
                self._cancel_tokens.pop(job_id, None)

    # ── Cleanup ────────────────────────────────────────────────────────────────

    async def _cleanup_loop(self) -> None:
        """Periodically remove expired results."""
        while True:
            await asyncio.sleep(300)  # every 5 minutes
            removed = await self._result_store.cleanup_expired()
            if removed:
                logger.info("[chat_executor] cleanup removed %d expired entries", removed)
