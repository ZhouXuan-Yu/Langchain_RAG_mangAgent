# -*- coding: utf-8 -*-
"""FastAPI main server."""

import atexit
import logging
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import APP_PORT
from src.server.api import router


def _open_browser(url: str) -> None:
    try:
        if sys.platform == "win32":
            subprocess.Popen(["cmd", "/c", "start", "", url], shell=False, detached=True)
        else:
            import webbrowser
            webbrowser.open(url)
    except Exception:
        pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Resource cleanup ──────────────────────────────────────────────────────────
def _cleanup_resources():
    """Stop all background threads/processes so uvicorn can exit gracefully."""
    logger.info("[cleanup] stopping background resources...")

    # Log shutdown tracker
    try:
        from src.server.api import _SHUTDOWN_TRACKER
        logger.info(f"[cleanup] shutdown tracker: pending={_SHUTDOWN_TRACKER['pending_tasks']}, max={_SHUTDOWN_TRACKER['max_pending']}")
    except Exception:
        pass

    # Check for running orchestrator jobs
    try:
        from src.server.orch_jobs import get_job_manager, JobStatus
        jm = get_job_manager()
        running = [j.job_id for j in jm.list_jobs() if j.status == JobStatus.RUNNING]
        if running:
            logger.warning(f"[cleanup] {len(running)} running orchestrator jobs: {running}")
        else:
            logger.info("[cleanup] no running orchestrator jobs")
    except Exception as e:
        logger.warning(f"[cleanup] job manager check failed: {e}")

    # 1. TaskScheduler ThreadPoolExecutor
    try:
        from src.server.task_scheduler import get_scheduler
        sched = get_scheduler()
        sched._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("[cleanup] TaskScheduler executor stopped")
    except Exception as e:
        logger.warning(f"[cleanup] TaskScheduler: {e}")

    # 2. Playwright browser process
    try:
        from src.tools.browser_tools import _browser_pool, _pool_lock
        with _pool_lock:
            for key, pool in list(_browser_pool.items()):
                try:
                    pool["browser"].close()
                    pool["playwright"].stop()
                    logger.info(f"[cleanup] browser {key} closed")
                except Exception:
                    pass
            _browser_pool.clear()
    except Exception as e:
        logger.warning(f"[cleanup] browser pool: {e}")

    # 3. ChromaDB singleton reference
    try:
        from src.memory.chroma_store import ChromaMemoryStore
        ChromaMemoryStore._instance = None
        logger.info("[cleanup] ChromaMemoryStore cleared")
    except Exception as e:
        logger.warning(f"[cleanup] ChromaMemoryStore: {e}")

    logger.info("[cleanup] done")


async def _cleanup_resources_async():
    """
    Cancel all pending asyncio tasks during shutdown.
    Does NOT call loop.stop() or loop.close() — those are uvicorn's responsibility
    and calling them from here raises CancelledError that breaks the shutdown chain.
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    current_task = asyncio.current_task(loop)
    pending = [t for t in asyncio.all_tasks(loop) if t is not current_task]
    if pending:
        for t in pending:
            t.cancel()
        try:
            await asyncio.wait(pending, timeout=3.0)
        except asyncio.exceptions.CancelledError:
            # uvicorn may cancel the cleanup task itself — suppress so sync
            # _cleanup_resources() still runs and the shutdown chain completes.
            return


atexit.register(_cleanup_resources)


# ── FastAPI app ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[lifespan] server starting...")
    yield
    logger.info("[lifespan] server shutting down...")
    import asyncio as _asyncio
    try:
        await _cleanup_resources_async()
    except _asyncio.exceptions.CancelledError:
        pass  # fall through to sync cleanup
    _cleanup_resources()


app = FastAPI(
    title="LangGraph RAG Agent",
    description="DeepSeek + ChromaDB + LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    index_path = PROJECT_ROOT / "pages" / "index.html"
    if index_path.exists():
        return Response(index_path.read_text(encoding="utf-8"), media_type="text/html")
    return {"message": "pages/index.html not found"}


_PAGE_ROUTES = [
    "tasks", "kb", "agents", "costs", "sessions", "settings", "orchestrate",
]


def _make_page_handler(name: str):
    async def handler():
        page_path = PROJECT_ROOT / "pages" / f"{name}.html"
        if page_path.exists():
            return Response(page_path.read_text(encoding="utf-8"), media_type="text/html")
        return Response(f"pages/{name}.html not found", status_code=404)
    return handler


for _name in _PAGE_ROUTES:
    app.add_api_route(
        f"/{_name}",
        _make_page_handler(_name),
        methods=["GET"],
        include_in_schema=False,
    )


app.include_router(router)

static_root = PROJECT_ROOT / "pages"
if static_root.exists():
    app.mount("/", StaticFiles(directory=str(static_root), html=True), name="pages")


def _open_browser_server():
    _open_browser(f"http://localhost:{APP_PORT}")


if __name__ == "__main__":
    logger.info("Starting LangGraph RAG Agent (port %s)...", APP_PORT)
    threading.Timer(1.0, _open_browser_server).start()
    uvicorn.run(
        "src.server.main_server:app",
        host="localhost",
        port=APP_PORT,
    )
