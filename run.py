"""一键启动 LangGraph RAG Agent + Web UI."""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.resolve()


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

    # 3. 检查端口是否已被占用，尝试复用已有服务
    port = 8501
    if _is_port_open("localhost", port):
        print(f"[INFO] 端口 {port} 已有服务运行，尝试复用...")
        url = f"http://localhost:{port}"
        print(f"[INFO] 正在打开浏览器: {url}")
        _open_browser(url)
        print("[INFO] 服务运行中，按 Ctrl+C 停止\n")
        return

    # 端口未被占用，直接启动服务
    # 4. 启动后端（后台进程）
    print(f"[INFO] 正在启动后端服务 (FastAPI + LangGraph) 端口 {port} ...")
    server = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "src.server.main_server:app",
            "--host", "localhost",
            "--port", str(port),
            "--reload",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # 5. 等待服务就绪
    print(f"[INFO] 等待服务启动 (端口 {port})...")
    for i in range(20):
        time.sleep(1)
        if server.poll() is not None:
            output = server.stdout.read().decode(errors="replace")
            print(f"[ERROR] 后端进程异常退出 (exit {server.returncode})")
            print(output[-1000:])
            return
        if _is_port_open("localhost", port):
            break
        print(f"[INFO]  等待中... ({i+1}/20)")
    else:
        print("[ERROR] 服务启动超时（20秒）")
        server.terminate()
        server.wait()
        return

    # 6. 打开浏览器
    url = f"http://localhost:{port}"
    print(f"[INFO] 服务已就绪，正在打开浏览器: {url}")
    _open_browser(url)

    # 7. 监控进程
    print("[INFO] 服务运行中，按 Ctrl+C 停止所有服务\n")
    try:
        for line in server.stdout:
            text = line.decode(errors="replace")
            print(text, end="")
    except KeyboardInterrupt:
        print("\n[INFO] 正在关闭服务...")
        server.terminate()
        server.wait()
        print("[INFO] 服务已关闭。")


if __name__ == "__main__":
    main()
