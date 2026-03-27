"""FastAPI 主服务器 — 静态文件服务 + CORS + 自动打开浏览器."""

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.server.api import router


def _open_browser(url: str) -> None:
    """跨平台打开浏览器."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(["cmd", "/c", "start", "", url], shell=False, detached=True)
        else:
            import webbrowser
            webbrowser.open(url)
    except Exception:
        pass

# ── 日志配置 ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI 应用 ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="LangGraph RAG Agent",
    description="DeepSeek + ChromaDB + LangGraph — 带 Web UI 的 AI 智能体",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载 API 路由
app.include_router(router)

# 挂载静态文件（pages 目录，每个功能对应一个独立 HTML 页面）
static_root = PROJECT_ROOT / "pages"
if static_root.exists():
    app.mount("/", StaticFiles(directory=str(static_root), html=True), name="pages")

# ── 根路径：返回 index.html ───────────────────────────────────────────────────
@app.get("/")
async def root():
    """返回前端主页面（对话主页）."""
    index_path = PROJECT_ROOT / "pages" / "index.html"
    if index_path.exists():
        return Response(index_path.read_text(encoding="utf-8"), media_type="text/html")
    return {"message": "pages/index.html not found"}

# ── 健康检查 ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── 浏览器自动打开 ────────────────────────────────────────────────────────────
def _open_browser_server():
    _open_browser("http://localhost:8501")


if __name__ == "__main__":
    logger.info("启动 LangGraph RAG Agent Web UI...")
    threading.Timer(1.0, _open_browser_server).start()
    uvicorn.run(
        "src.server.main_server:app",
        host="localhost",
        port=8501,
        reload=True,
        reload_dirs=[str(PROJECT_ROOT)],
    )
