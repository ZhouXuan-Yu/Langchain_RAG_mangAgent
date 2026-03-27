"""一键启动 LangGraph RAG Agent + Web UI."""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.resolve()
load_dotenv(PROJECT_ROOT / ".env")


def _open_browser(url: str) -> None:
    """跨平台打开浏览器 — 解决 Windows PowerShell 环境下 webbrowser.open 失效的问题."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(["cmd", "/c", "start", "", url], shell=False, detached=True)
        else:
            import webbrowser
            webbrowser.open(url)
    except Exception:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass


def _is_port_open(host: str, port: int) -> bool:
    """检测端口是否开放."""
    sock = socket.socket()
    try:
        sock.settimeout(1)
        sock.connect((host, port))
        sock.close()
        return True
    except (OSError, ConnectionRefusedError):
        return False


def main():
    # 1. 检查环境
    if not (PROJECT_ROOT / ".env").exists():
        print("[ERROR] .env 文件不存在，请先 copy .env.example 为 .env 并填入 API Key")
        return

    # 2. 确保数据目录存在
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "chroma_db").mkdir(parents=True, exist_ok=True)
    (data_dir / "checkpointer").mkdir(parents=True, exist_ok=True)

    # 3. 固定端口（与 main_server / .env 中 APP_PORT 一致）
    # 注意：子进程若使用 stdout=PIPE 且主进程在等待循环中不读管道，uvicorn 日志会填满缓冲区，
    # 在 Windows 上会导致子进程阻塞、端口永不监听 —— 因此必须继承标准输出或使用 DEVNULL。
    port = int(os.getenv("APP_PORT", "8000"))
    if _is_port_open("localhost", port):
        print(
            f"[ERROR] 端口 {port} 已被占用（可能仍有旧 uvicorn/python 在运行）。"
            f"请结束占用进程后重试，或在 .env 中修改 APP_PORT。"
        )
        return

    # 4. 启动后端（子进程日志直接打印到当前终端，避免 PIPE 死锁）
    print(f"[INFO] 正在启动后端服务 (FastAPI + LangGraph) 端口 {port} ...")
    server = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "src.server.main_server:app",
            "--host", "localhost",
            "--port", str(port),
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    # 5. 等待服务就绪
    # 说明：uvicorn 主进程在 Windows 上启动子进程运行 HTTP 服务。
    # 当子进程崩溃（如 import 错误），主进程保持 alive 但端口无监听。
    # 因此用 /health HTTP 检查确认，而非只看端口是否 open。
    import urllib.request
    import urllib.error

    health_url = f"http://localhost:{port}/health"
    print(f"[INFO] 等待服务启动 (端口 {port})...")

    for i in range(60):
        time.sleep(0.5)
        if server.poll() is not None:
            print(f"[ERROR] 后端进程异常退出 (exit {server.returncode})，请查看上方 uvicorn 报错。")
            return
        try:
            # 用 /health 确认服务真正在响应，而非端口被旧进程占着
            req = urllib.request.Request(health_url, headers={"User-Agent": "python"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    break
        except (urllib.error.URLError, urllib.error.HTTPError, Exception):
            pass
        if i % 10 == 0:  # 每 5 秒才打印一次，避免刷屏
            print(f"[INFO]  等待中... ({i+1}/60)")
    else:
        print("[ERROR] 服务启动超时（约 30 秒），uvicorn 可能因 import 错误无法启动，请查看上方报错。")
        server.terminate()
        server.wait()
        return

    # 6. 打开浏览器
    url = f"http://localhost:{port}"
    print(f"[INFO] 服务已就绪，正在打开浏览器: {url}")
    _open_browser(url)

    # 7. 阻塞直到子进程结束（日志已由子进程继承的 stdout 输出）
    print("[INFO] 服务运行中，按 Ctrl+C 停止所有服务\n")
    try:
        server.wait()
    except KeyboardInterrupt:
        print("\n[INFO] 正在关闭服务...")
        server.terminate()
        server.wait()
        print("[INFO] 服务已关闭。")


if __name__ == "__main__":
    main()
