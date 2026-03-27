"""Search Worker — Tavily 搜索 + Playwright 抓取，带置信度标记."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from src.tools.browser_tools import web_search, browse_page

logger = logging.getLogger(__name__)

# 置信度权重
_WEB_CONFIDENCE = 0.95   # 实时搜索（时效性最高）
_BROWSE_CONFIDENCE = 0.90  # 深度页面抓取


class SearchWorker:
    """Search Worker — 负责外网实时信息搜集."""

    def __init__(self):
        self.name = "search_worker"

    async def execute(self, task: dict, progress_callback=None) -> dict:
        """
        执行搜索任务。

        Args:
            task: 任务字典，包含 task_id, description, assigned_agent
            progress_callback: 进度回调函数，用于推送中间状态

        Returns:
            结果字典，包含 result, confidence, quality_score, source
        """
        task_id = task.get("task_id", "unknown")
        description = task.get("description", "")

        if progress_callback:
            await progress_callback({
                "type": "worker_progress",
                "agent": self.name,
                "task_id": task_id,
                "message": f"正在分析搜索关键词: {description[:50]}...",
            })

        # ── Step 1: 执行 Tavily 搜索 ────────────────────────────────
        try:
            search_result = await asyncio.to_thread(
                web_search.invoke, {"query": description}
            )

            if progress_callback:
                await progress_callback({
                    "type": "worker_progress",
                    "agent": self.name,
                    "task_id": task_id,
                    "message": "搜索完成，正在分析结果...",
                })

        except Exception as e:
            logger.error(f"[search_worker] web_search failed: {e}")
            search_result = f"[搜索失败] {str(e)}"

        # ── Step 2: 评估信息质量 ───────────────────────────────────
        quality_score = self._evaluate_quality(search_result)

        # ── Step 3: 时效性分析 ─────────────────────────────────────
        freshness = self._analyze_freshness(search_result, description)

        result_text = f"""[Search Worker 结果]
任务: {description}

搜索摘要:
{search_result}

时效性评估: {freshness}
置信度: {_WEB_CONFIDENCE}
质量评分: {quality_score}/10
"""

        logger.info(f"[search_worker] task {task_id} done, quality={quality_score}")

        return {
            "task_id": task_id,
            "agent": self.name,
            "result": result_text,
            "confidence": _WEB_CONFIDENCE,
            "quality_score": quality_score,
            "source": "web",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "freshness": freshness,
            "raw_data": search_result,
        }

    def _evaluate_quality(self, result: str) -> float:
        """评估搜索结果质量（0-10分）。"""
        if not result or len(result) < 20:
            return 2.0
        if "[搜索失败]" in result or "[系统]" in result:
            return 3.0

        score = 6.0
        # 包含 URL 链接加分
        if "http" in result:
            score += 1.0
        # 包含摘要内容加分
        if len(result) > 200:
            score += 1.0
        # 包含标题结构加分
        if "标题" in result or "title" in result.lower():
            score += 1.0
        # 信息量丰富
        if len(result) > 500:
            score += 1.0

        return min(score, 10.0)

    def _analyze_freshness(self, result: str, query: str) -> str:
        """分析搜索结果的时效性。"""
        now = datetime.now(timezone.utc)
        query_lower = query.lower()

        # 关键词判断时效性
        fresh_keywords = ["2025", "2026", "最新", "最新版", "新版本", "刚刚", "recent"]
        old_keywords = ["2019", "2020", "2021", "旧版", "过时", "deprecated"]

        if any(k in query_lower for k in fresh_keywords):
            if any(k in result for k in ["2025", "2026"]):
                return "最新 (2025-2026)"
            return "较新 (近期)"
        if any(k in query_lower for k in old_keywords):
            return "历史数据"

        return f"当前时间: {now.strftime('%Y-%m-%d')}，结果适用"


# 全局单例
_search_worker: SearchWorker | None = None


def get_search_worker() -> SearchWorker:
    global _search_worker
    if _search_worker is None:
        _search_worker = SearchWorker()
    return _search_worker
