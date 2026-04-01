"""Coder Worker — 代码编写与审查（工部·营造），接收搜索结果和 RAG 结果作为输入，并支持文件输出."""

import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.output_manager import OutputManager

logger = logging.getLogger(__name__)

_CODER_CONFIDENCE = 0.75

# 用于从 LLM 输出中提取 FILENAME 和 DESC 注释
_FILENAME_RE = re.compile(r"<!--\s*FILENAME:\s*([^\s>]+)\s*-->", re.IGNORECASE)
_DESC_RE = re.compile(r"<!--\s*DESC:\s*([^\n]+?)\s*-->", re.IGNORECASE)


class CoderWorker:
    """Coder Worker — 负责代码编写，基于搜索结果和记忆上下文（工部·营造）."""

    def __init__(self):
        self.name = "coder"

    async def execute(
        self,
        task: dict,
        context: list[dict] | None = None,
        progress_callback=None,
        output_manager: "OutputManager | None" = None,
    ) -> dict:
        task_id = task.get("task_id", "unknown")
        description = task.get("description", "")
        output_type = task.get("output_type", "code")
        suggested_filename = task.get("suggested_filename", "")

        if progress_callback:
            await progress_callback({
                "type": "worker_progress",
                "agent": self.name,
                "task_id": task_id,
                "message": "【工部·营造】正在分析代码需求...",
            })

        context_summary = self._build_context(context)

        if progress_callback:
            await progress_callback({
                "type": "worker_progress",
                "agent": self.name,
                "task_id": task_id,
                "message": "【工部·营造】正在生成代码...",
            })

        files_saved = []
        code_result = ""
        raw_content = ""

        try:
            raw_content = await self._generate_code(description, context_summary)
            code_result = raw_content

            # 解析 LLM 输出中的 FILENAME/DESC 注释
            filename, desc = self._parse_annotations(raw_content)
            if not filename:
                filename = suggested_filename or f"{task_id}_code.py"

            # 如果有 OutputManager，写入文件
            if output_manager and raw_content and not raw_content.startswith("[代码生成失败]"):
                # 提取纯代码部分（去掉注释标记）
                pure_code = self._strip_annotations(raw_content)

                file_info = output_manager.save_file(
                    content=pure_code,
                    output_type=output_type,
                    agent_id=self.name,
                    task_id=task_id,
                    filename=filename,
                    description=desc or f"代码: {description[:80]}",
                )
                if file_info:
                    files_saved.append(file_info.to_dict())
                    code_result = f"[已保存到文件: {filename}]\n\n{raw_content}"

        except Exception as e:
            logger.error(f"[coder] code generation failed: {e}")
            code_result = f"[代码生成失败] {str(e)}"
            raw_content = code_result

        quality_score = self._evaluate_code_quality(raw_content)

        result_text = f"""[Coder Worker | 工部·营造 结果]
任务: {description}

代码产出:
{code_result}

上下文来源:
{context_summary}

置信度: {_CODER_CONFIDENCE}
质量评分: {quality_score}/10
"""

        logger.info(f"[coder] task {task_id} done, quality={quality_score}, files={len(files_saved)}")

        return {
            "task_id": task_id,
            "agent": self.name,
            "result": result_text,
            "confidence": _CODER_CONFIDENCE,
            "quality_score": quality_score,
            "source": "coder",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_data": raw_content,
            "files": files_saved,
        }

    def _build_context(self, context: list[dict] | None) -> str:
        """从其他 Worker 的结果构建上下文摘要。"""
        if not context:
            return "无外部上下文（纯代码生成）"

        parts = []
        for ctx in context:
            agent = ctx.get("agent", "unknown")
            raw = ctx.get("raw_data", "")[:500]
            parts.append(f"[{agent}]: {raw}")

        return "\n\n".join(parts) if parts else "无有效上下文"

    async def _generate_code(self, description: str, context: str) -> str:
        """调用 LLM 生成代码，使用增强版 Prompt。"""
        from src.llm import init_deepseek_llm
        from src.graph.prompt import CODER_WORKER_SYSTEM_PROMPT
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = init_deepseek_llm(temperature=0.3, streaming=False)

        user_prompt = f"""# 当前任务
{description}

# 相关上下文（来自其他 Agent）
{context}

请按上述要求生成代码，输出格式：
1. `<!-- FILENAME: xxx.py -->` 标注文件名
2. `<!-- DESC: xxx -->` 标注文件功能描述
3. `## 代码` + 代码块（带语言标注）
4. `## 实现说明`
5. `## 置信度`

重要：代码必须完整，标注文件名以便系统保存！"""

        try:
            response = await llm.ainvoke([
                SystemMessage(content=CODER_WORKER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"[coder] _generate_code failed: {e}")
            return f"[代码生成失败] {str(e)}"

    @staticmethod
    def _parse_annotations(content: str) -> tuple[str, str]:
        """
        从 LLM 输出中解析 <!-- FILENAME: xxx --> 和 <!-- DESC: xxx --> 注释。

        Returns:
            (filename, description) 元组，未找到时返回 ("", "")
        """
        fname_match = _FILENAME_RE.search(content)
        fname = fname_match.group(1).strip() if fname_match else ""

        desc_match = _DESC_RE.search(content)
        desc = desc_match.group(1).strip() if desc_match else ""

        return fname, desc

    @staticmethod
    def _strip_annotations(content: str) -> str:
        """移除 LLM 输出中的 <!-- FILENAME: ... --> 和 <!-- DESC: ... --> 注释行。"""
        lines = content.split("\n")
        cleaned = [line for line in lines if not _FILENAME_RE.match(line.strip()) and not _DESC_RE.match(line.strip())]
        return "\n".join(cleaned)

    def _evaluate_code_quality(self, code: str) -> float:
        """评估代码质量（0-10分）。"""
        if not code or "[代码生成失败]" in code:
            return 2.0

        score = 6.0

        if "def " in code or "class " in code:
            score += 1.0
        if "# " in code or '"""' in code or "'''" in code:
            score += 1.0
        if 50 < len(code) < 500:
            score += 1.0
        if "TODO" not in code and "FIXME" not in code and "..." not in code:
            score += 1.0

        return min(score, 10.0)


_coder_worker: "CoderWorker | None" = None


def get_coder_worker() -> CoderWorker:
    global _coder_worker
    if _coder_worker is None:
        _coder_worker = CoderWorker()
    return _coder_worker
