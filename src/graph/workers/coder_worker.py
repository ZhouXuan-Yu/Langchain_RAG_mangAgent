"""Coder Worker — 代码编写与审查，接收搜索结果和 RAG 结果作为输入."""

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# 置信度权重
_CODER_CONFIDENCE = 0.75   # 代码生成（基于 LLM 推理）


class CoderWorker:
    """Coder Worker — 负责代码编写，基于搜索结果和记忆上下文."""

    def __init__(self):
        self.name = "coder"

    async def execute(
        self,
        task: dict,
        context: list[dict] | None = None,
        progress_callback=None,
    ) -> dict:
        """
        执行代码编写任务。

        Args:
            task: 任务字典
            context: 来自其他 Worker 的上下文数据（如搜索结果）
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
                "message": "正在分析代码需求...",
            })

        # ── Step 1: 构建代码生成 Prompt ───────────────────────────
        context_summary = self._build_context(context)

        if progress_callback:
            await progress_callback({
                "type": "worker_progress",
                "agent": self.name,
                "task_id": task_id,
                "message": "正在生成代码...",
            })

        # ── Step 2: 调用 LLM 生成代码 ─────────────────────────────
        try:
            code_result = await self._generate_code(description, context_summary)
        except Exception as e:
            logger.error(f"[coder] code generation failed: {e}")
            code_result = f"[代码生成失败] {str(e)}"

        # ── Step 3: 评估代码质量 ─────────────────────────────────
        quality_score = self._evaluate_code_quality(code_result)

        result_text = f"""[Coder Worker 结果]
任务: {description}

代码产出:
{code_result}

上下文来源:
{context_summary}

置信度: {_CODER_CONFIDENCE}
质量评分: {quality_score}/10
"""

        logger.info(f"[coder] task {task_id} done, quality={quality_score}")

        return {
            "task_id": task_id,
            "agent": self.name,
            "result": result_text,
            "confidence": _CODER_CONFIDENCE,
            "quality_score": quality_score,
            "source": "coder",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_data": code_result,
        }

    def _build_context(self, context: list[dict] | None) -> str:
        """从其他 Worker 的结果构建上下文摘要。"""
        if not context:
            return "无外部上下文（纯代码生成）"

        parts = []
        for ctx in context:
            agent = ctx.get("agent", "unknown")
            raw = ctx.get("raw_data", "")[:300]
            parts.append(f"[{agent}]: {raw}")

        return "\n\n".join(parts) if parts else "无有效上下文"

    async def _generate_code(self, description: str, context: str) -> str:
        """调用 LLM 生成代码。"""
        from src.llm import init_deepseek_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = init_deepseek_llm(temperature=0.3, streaming=False)

        prompt = f"""你是一个专业的 Python 开发者。请根据以下需求和上下文生成代码。

需求描述:
{description}

相关上下文（来自其他 Agent）:
{context}

要求:
1. 代码必须完整可运行
2. 添加必要的注释说明
3. 考虑错误处理
4. 遵循 PEP8 风格

请直接输出代码，不要解释。"""

        response = llm.invoke([
            SystemMessage(content="你是一个专业的 Python 开发者，擅长编写高质量代码。"),
            HumanMessage(content=prompt),
        ])

        return response.content if hasattr(response, "content") else str(response)

    def _evaluate_code_quality(self, code: str) -> float:
        """评估代码质量（0-10分）。"""
        if not code or "[代码生成失败]" in code:
            return 2.0

        score = 6.0

        # 有 def/class 等函数定义
        if "def " in code or "class " in code:
            score += 1.0
        # 有注释
        if "# " in code or '"""' in code or "'''" in code:
            score += 1.0
        # 代码长度适中
        if 50 < len(code) < 500:
            score += 1.0
        # 没有明显的占位符
        if "TODO" not in code and "FIXME" not in code and "..." not in code:
            score += 1.0

        return min(score, 10.0)


# 全局单例
_coder_worker: CoderWorker | None = None


def get_coder_worker() -> CoderWorker:
    global _coder_worker
    if _coder_worker is None:
        _coder_worker = CoderWorker()
    return _coder_worker
