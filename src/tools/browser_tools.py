"""Browser tools — 04-06: Tavily web_search + Playwright browse_page."""

import os
import logging
import threading
from typing import Any, Optional

from langchain_core.tools import tool

from src.config import TAVILY_API_KEY
from src.utils.markdown_cleaner import clean_markdown
from src.utils.summarizer import compress_web_content

logger = logging.getLogger(__name__)

# ── Browser Pool ───────────────────────────────────────────────────────────────
# 单例 Playwright + Chromium 浏览器，避免每次新建的高昂启动开销（~2-5s/次）
_browser_pool: dict[str, Any] = {}
_pool_lock = threading.Lock()


def _get_browser() -> Any:
    """
    获取复用浏览器实例（线程安全）。

    启动时创建一次，之后所有请求复用同一个 Chromium 实例。
    如果浏览器崩溃则自动重启。
    """
    key = "default"
    with _pool_lock:
        pool = _browser_pool.get(key)
        if pool is not None:
            try:
                # 检查浏览器是否还活着（简单 ping）
                pool["browser"].version
                return pool["browser"], pool["playwright"]
            except Exception:
                logger.warning("[browser_pool] browser crashed, restarting...")
                try:
                    pool["browser"].close()
                    pool["playwright"].stop()
                except Exception:
                    pass

        # 首次启动或重启
        from playwright.sync_api import sync_playwright
        p = sync_playwright().start()

        # 默认 Chrome 路径
        chrome_path = None
        import platform
        if platform.system() == "Windows":
            chrome_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            ]
            for path in chrome_paths:
                if os.path.exists(path):
                    chrome_path = path
                    break

        try:
            if chrome_path:
                browser = p.chromium.launch(
                    executablePath=chrome_path,
                    headless=True
                )
            else:
                browser = p.chromium.launch(headless=True)
            _browser_pool[key] = {"browser": browser, "playwright": p}
            logger.info("[browser_pool] browser started (pid=%s)", browser._impl_obj._browser_type.name)
            return browser, p
        except Exception as e:
            logger.error(f"[browser_pool] 启动浏览器失败: {e}")
            print("\n" + "=" * 60)
            print("⚠️  浏览器启动失败！")
            print(f"错误信息: {e}")
            print("-" * 60)
            print("解决方法:")
            print("  1. 运行 'playwright install chromium' 安装 Chromium")
            print("  2. 或者确保本机 Chrome 已安装并可访问")
            print("=" * 60 + "\n")
            raise


# ── Tavily Client Singleton ───────────────────────────────────────────────────
_tavily_client: Optional[Any] = None


def _get_tavily_client() -> Any:
    """获取单例 Tavily client（避免每次新建连接池开销）。"""
    global _tavily_client
    if _tavily_client is not None:
        return _tavily_client

    if not TAVILY_API_KEY:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. "
            "Copy .env.example to .env and fill in your Tavily key, "
            "or set TAVILY_API_KEY environment variable."
        )
    from tavily import TavilyClient
    _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    return _tavily_client


def _format_search_results(raw: dict) -> str:
    """Format Tavily search results into a readable string."""
    results = raw.get("results", [])
    if not results:
        answer = raw.get("answer", "")
        return answer if answer else "未找到相关结果。"

    lines = []
    if raw.get("answer"):
        lines.append(f"摘要: {raw['answer']}\n")

    lines.append("--- 搜索结果 ---")
    for i, r in enumerate(results, 1):
        title = r.get("title", "无标题")
        url = r.get("url", "")
        snippet = r.get("content", "")
        lines.append(f"{i}. {title}\n   {snippet}\n   链接: {url}")

    return "\n".join(lines)


@tool
def web_search(query: str) -> str:
    """
    搜索网页，返回与查询最相关的 Top-5 结果摘要。

    适用于：查询最新技术、库版本更新、具体 Bug 解决方案等。
    当用户询问 2025-2026 年的最新技术时，必须使用此工具。
    """
    if not query or not query.strip():
        return "[系统] 搜索关键词为空，已跳过搜索。"

    try:
        client = _get_tavily_client()
        result = client.search(
            query=query,
            max_results=5,
            include_answer=True,
            include_raw_content=False,
        )
        return _format_search_results(result)
    except Exception as e:
        logger.error(f"web_search failed: {e}")
        return f"搜索失败: {e}"


@tool
def browse_page(url: str) -> str:
    """
    抓取指定网页的正文内容，经过 Markdown 清洗后返回。

    适用于：当 web_search 的摘要无法解决复杂问题（如查找特定 Bug 报错代码）时，
    使用此工具深度抓取目标页面的完整正文。
    """
    if not url or not url.startswith(("http://", "https://")):
        return f"[系统] 无效的 URL: {url}"

    try:
        browser, _ = _get_browser()
    except ImportError:
        return (
            "[系统] Playwright 未安装。运行: playwright install chromium\n"
            "或者使用网页摘要压缩功能作为替代方案。"
        )

    try:
        page = browser.new_page()
        try:
            page.goto(url, timeout=15000, wait_until="domcontentloaded")
            content = page.inner_text("body")
        finally:
            page.close()

        cleaned = clean_markdown(content)
        compressed = compress_web_content(cleaned)
        return compressed

    except Exception as e:
        logger.error(f"browse_page failed for {url}: {e}")
        return f"页面抓取失败: {e}"
