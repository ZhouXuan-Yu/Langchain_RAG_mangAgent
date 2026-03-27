"""RAG Worker — ChromaDB 记忆检索，带置信度标记."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from src.tools.memory_tools import memory_search

logger = logging.getLogger(__name__)

# 置信度权重
_RAG_CONFIDENCE = 0.70   # 内部文档（低于实时搜索）


class RAGWorker:
    """RAG Worker — 负责 ChromaDB 长期记忆检索."""

    def __init__(self):
        self.name = "rag_worker"

    async def execute(self, task: dict, progress_callback=None) -> dict:
        """
        执行记忆检索任务。

        Args:
            task: 任务字典
            progress_callback: 进度回调

        Returns:
            结果字典
        """
        task_id = task.get("task_id", "unknown")
        description = task.get("description", "")

        if progress_callback:
            await progress_callback({
                "type": "worker_progress",
                "agent": self.name,
                "task_id": task_id,
                "message": f"正在检索记忆: {description[:50]}...",
            })

        # ── Step 1: ChromaDB 向量检索 ─────────────────────────────
        try:
            rag_result = await asyncio.to_thread(
                memory_search.invoke, {"query": description}
            )

            if progress_callback:
                await progress_callback({
                    "type": "worker_progress",
                    "agent": self.name,
                    "task_id": task_id,
                    "message": "记忆检索完成，正在分析...",
                })

        except Exception as e:
            logger.error(f"[rag_worker] memory_search failed: {e}")
            rag_result = f"[记忆检索失败] {str(e)}"

        # ── Step 2: 评估记忆质量 ─────────────────────────────────
        quality_score = self._evaluate_quality(rag_result)

        # ── Step 3: 提取时间信息 ─────────────────────────────────
        memory_age = self._extract_timestamp(rag_result)

        result_text = f"""[RAG Worker 结果]
任务: {description}

记忆检索结果:
{rag_result}

记忆时效: {memory_age}
置信度: {_RAG_CONFIDENCE}
质量评分: {quality_score}/10
"""

        logger.info(f"[rag_worker] task {task_id} done, quality={quality_score}")

        return {
            "task_id": task_id,
            "agent": self.name,
            "result": result_text,
            "confidence": _RAG_CONFIDENCE,
            "quality_score": quality_score,
            "source": "rag",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "memory_age": memory_age,
            "raw_data": rag_result,
        }

    def _evaluate_quality(self, result: str) -> float:
        """评估记忆检索质量（0-10分）。"""
        if not result or len(result) < 10:
            return 3.0
        if "[记忆检索失败]" in result:
            return 2.0
        if "未找到" in result or "无相关" in result:
            return 4.0

        score = 6.0
        # 有具体内容
        if len(result) > 100:
            score += 1.5
        # 包含类别信息
        if "category" in result.lower() or "类别" in result:
            score += 1.0
        # 有重要性标记
        if "importance" in result.lower() or "重要" in result:
            score += 1.5

        return min(score, 10.0)

    def _extract_timestamp(self, result: str) -> str:
        """从结果中提取或推断时间信息。"""
        import re
        # 尝试匹配 ISO 时间戳
        match = re.search(r"\d{4}-\d{2}-\d{2}", result)
        if match:
            return f"记忆时间: {match.group()}"
        if "刚才" in result or "recent" in result.lower():
            return "最近存入"
        return "时间未知"


# 全局单例
_rag_worker: RAGWorker | None = None


def get_rag_worker() -> RAGWorker:
    global _rag_worker
    if _rag_worker is None:
        _rag_worker = RAGWorker()
    return _rag_worker
