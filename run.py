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
    """跨平台打开浏览器."""
    try:
        if sys.platform == "win32":
            DETACHED_PROCESS = 0x00000008
            CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen(
                ["cmd", "/c", "start", "", url],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
            )
        else:
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
    port = int(os.getenv("APP_PORT", "8000"))
    if _is_port_open("localhost", port):
        print(
            f"[ERROR] 端口 {port} 已被占用（可能仍有旧 uvicorn/python 在运行）。"
            f"请结束占用进程后重试，或在 .env 中修改 APP_PORT。"
        )
        return

    # 4. 启动后端（stdout/stderr 直接继承父进程，避免 PIPE 死锁）
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
    import urllib.request
    import urllib.error
    health_url = f"http://localhost:{port}/health"
    print(f"[INFO] 等待服务启动 (端口 {port})...")

    for i in range(60):
        time.sleep(0.5)
        rc = server.poll()
        if rc is not None:
            print(f"\n[ERROR] 后端进程异常退出 (exit {rc})")
            return
        try:
            req = urllib.request.Request(health_url, headers={"User-Agent": "python"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    break
        except (urllib.error.URLError, urllib.error.HTTPError, Exception):
            pass
        if i % 10 == 0:
            print(f"[INFO]  等待中... ({i+1}/60)")
    else:
        print("[ERROR] 服务启动超时（约 30 秒）")
        server.terminate()
        server.wait()
        return

    # 6. 打开浏览器
    url = f"http://localhost:{port}"
    print(f"[INFO] 服务已就绪，正在打开浏览器: {url}")
    _open_browser(url)

    # 7. 等待子进程退出
    print("[INFO] 服务运行中，按 Ctrl+C 停止所有服务\n")
    try:
        server.wait()
    except KeyboardInterrupt:
        print("\n[INFO] 收到 Ctrl+C，正在关闭...")
        server.terminate()
        try:
            server.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

    rc = server.returncode
    if rc == 0:
        print("[INFO] 服务已关闭。")
    else:
        print(f"[INFO] 服务已关闭 (exit {rc})。")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
