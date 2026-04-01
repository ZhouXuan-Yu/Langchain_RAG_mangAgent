"""RAG Worker — LLM 驱动的记忆检索专家（吏部·典藏）.

采用 Plan-Retrieve-Synthesize 模式：
1. Plan：LLM 分析任务，确定检索类别和关键词
2. Retrieve：执行 memory_search（可跨类别检索）
3. Synthesize：LLM 整合多条记忆，生成连贯知识上下文
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.output_manager import OutputManager

from src.tools.memory_tools import memory_search

logger = logging.getLogger(__name__)

_RAG_CONFIDENCE = 0.70

# 用于从 LLM 输出中提取 FILENAME 和 DESC 注释
_FILENAME_RE = re.compile(r"<!--\s*FILENAME:\s*([^\s>]+)\s*-->", re.IGNORECASE)
_DESC_RE = re.compile(r"<!--\s*DESC:\s*([^\n]+?)\s*-->", re.IGNORECASE)


class RAGWorker:
    """RAG Worker — LLM 驱动的本地知识库与记忆检索专家（吏部·典藏）."""

    def __init__(self):
        self.name = "rag_worker"

    async def execute(
        self,
        task: dict,
        progress_callback=None,
        output_manager: "OutputManager | None" = None,
    ) -> dict:
        task_id = task.get("task_id", "unknown")
        description = task.get("description", "")
        output_type = task.get("output_type", "search_only")
        suggested_filename = task.get("suggested_filename", "")

        if progress_callback:
            await progress_callback({
                "type": "worker_progress",
                "agent": self.name,
                "task_id": task_id,
                "message": "【吏部·典藏】正在分析检索策略...",
            })

        # ── Step 1: LLM 分析任务，制定检索计划 ───────────────────────────────
        retrieval_plan = await self._plan_retrieval(description)
        queries = retrieval_plan.get("queries", [description])
        target_categories = retrieval_plan.get("categories", [])

        if progress_callback:
            await progress_callback({
                "type": "worker_progress",
                "agent": self.name,
                "task_id": task_id,
                "message": f"【吏部·典藏】检索类别: {', '.join(target_categories) or '全部'}...",
            })

        # ── Step 2: 执行多轮检索 ────────────────────────────────────────────
        all_raw_results = []
        for i, query in enumerate(queries[:3]):
            try:
                raw = await asyncio.to_thread(memory_search.invoke, {"query": query})
                all_raw_results.append({
                    "query": query,
                    "result": raw,
                    "order": i,
                })
                if progress_callback:
                    await progress_callback({
                        "type": "worker_progress",
                        "agent": self.name,
                        "task_id": task_id,
                        "message": f"【吏部·典藏】检索 [{i+1}/{min(len(queries), 3)}]: {query[:30]}...",
                    })
            except Exception as e:
                logger.warning(f"[rag_worker] query '{query}' failed: {e}")

        merged_raw = "\n\n".join(
            f"[检索词 {r['order']+1}: {r['query']}]\n{r['result']}"
            for r in all_raw_results
        ) if all_raw_results else "[检索失败]"

        # ── Step 3: LLM 综合整合 ────────────────────────────────────────────
        if progress_callback:
            await progress_callback({
                "type": "worker_progress",
                "agent": self.name,
                "task_id": task_id,
                "message": "【吏部·典藏】正在整合知识上下文...",
            })

        synthesized = await self._synthesize(
            original_task=description,
            raw_results=merged_raw,
            categories=target_categories,
        )

        quality_score = self._evaluate_quality(synthesized)
        memory_age = self._extract_timestamp(synthesized)

        files_saved = []
        result_text = synthesized

        # 如果有 OutputManager 且需要写文件
        if output_manager and output_type not in ("search_only",):
            if synthesized and not synthesized.startswith("[检索失败]") and not synthesized.startswith("[知识整合失败]"):
                fname, desc = self._parse_annotations(synthesized)
                if not fname:
                    fname = suggested_filename or f"{task_id}_knowledge.md"

                pure_content = self._strip_annotations(synthesized)
                file_info = output_manager.save_file(
                    content=pure_content,
                    output_type=output_type,
                    agent_id=self.name,
                    task_id=task_id,
                    filename=fname,
                    description=desc or f"知识检索: {description[:80]}",
                )
                if file_info:
                    files_saved.append(file_info.to_dict())
                    result_text = f"[已保存到文件: {fname}]\n\n{synthesized}"

        result_text_full = f"""[RAG Worker | 吏部·典藏 结果]
任务: {description}

{result_text}

记忆时效: {memory_age}
置信度: {_RAG_CONFIDENCE}
质量评分: {quality_score}/10
"""

        logger.info(f"[rag_worker] task {task_id} done, quality={quality_score}, files={len(files_saved)}")

        return {
            "task_id": task_id,
            "agent": self.name,
            "result": result_text_full,
            "confidence": _RAG_CONFIDENCE,
            "quality_score": quality_score,
            "source": "rag",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "memory_age": memory_age,
            "raw_data": synthesized,
            "files": files_saved,
        }

    async def _plan_retrieval(self, task_description: str) -> dict:
        """LLM 分析任务，制定检索策略。"""
        from src.llm import init_deepseek_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = init_deepseek_llm(temperature=0.3, streaming=False)
        prompt = f"""你是一个专业知识库管理员。你的任务是为用户需求制定检索策略。

用户需求：{task_description}

可用记忆类别：
- project: 项目配置和代码结构
- conversation: 历史对话和决策
- technical: 技术方案和架构设计
- general: 通用知识和用户偏好

请输出 JSON 格式的检索计划：
{{
  "queries": ["检索词1", "检索词2"],
  "categories": ["project", "conversation"],
  "priority": "high/medium/low"
}}

要求：
- queries 应覆盖不同角度的同义词
- categories 至少选 1 个，按相关性排序
- 如涉及项目细节，优先选 project；如涉及历史决策，选 conversation

只输出 JSON，不要其他内容。"""

        try:
            response = await llm.ainvoke([
                SystemMessage(content="你是一个专业知识库管理员，擅长将用户需求转化为精准的检索策略。"),
                HumanMessage(content=prompt),
            ])
            content = response.content if hasattr(response, "content") else str(response)
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                import json
                return json.loads(match.group(0))
        except Exception as e:
            logger.warning(f"[rag_worker] _plan_retrieval failed: {e}")
        return {"queries": [task_description], "categories": [], "priority": "medium"}

    async def _synthesize(
        self,
        original_task: str,
        raw_results: str,
        categories: list[str],
    ) -> str:
        """LLM 整合多条记忆，生成连贯知识上下文。"""
        from src.llm import init_deepseek_llm
        from src.graph.prompt import RAG_WORKER_SYSTEM_PROMPT
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = init_deepseek_llm(temperature=0.2, streaming=False)

        try:
            response = await llm.ainvoke([
                SystemMessage(content=RAG_WORKER_SYSTEM_PROMPT),
                HumanMessage(content=(
                    f"# 当前任务\n{original_task}\n\n"
                    f"# 原始检索结果\n{raw_results}\n\n"
                    f"# 目标类别\n{', '.join(categories) if categories else '全部'}\n\n"
                    "请按输出格式整合记忆，生成完整报告。"
                )),
            ])
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"[rag_worker] _synthesize failed: {e}")
            return f"[知识整合失败] {raw_results}"

    def _evaluate_quality(self, result: str) -> float:
        if not result or len(result) < 10:
            return 3.0
        if "[检索失败]" in result or "[知识整合失败]" in result:
            return 2.0
        if "未找到" in result or "无相关" in result:
            return 4.0

        score = 6.0
        if len(result) > 100:
            score += 1.5
        if "category" in result.lower() or "类别" in result:
            score += 1.0
        if "importance" in result.lower() or "重要" in result:
            score += 1.5
        if "## " in result or "知识整合" in result:
            score += 1.0

        return min(score, 10.0)

    def _extract_timestamp(self, result: str) -> str:
        match = re.search(r"\d{4}-\d{2}-\d{2}", result)
        if match:
            return f"记忆时间: {match.group()}"
        if "刚才" in result or "recent" in result.lower():
            return "最近存入"
        return "时间未知"

    @staticmethod
    def _parse_annotations(content: str) -> tuple[str, str]:
        fname_match = _FILENAME_RE.search(content)
        fname = fname_match.group(1).strip() if fname_match else ""
        desc_match = _DESC_RE.search(content)
        desc = desc_match.group(1).strip() if desc_match else ""
        return fname, desc

    @staticmethod
    def _strip_annotations(content: str) -> str:
        lines = content.split("\n")
        cleaned = [
            line for line in lines
            if not _FILENAME_RE.match(line.strip()) and not _DESC_RE.match(line.strip())
        ]
        return "\n".join(cleaned)


_rag_worker: "RAGWorker | None" = None


def get_rag_worker() -> RAGWorker:
    global _rag_worker
    if _rag_worker is None:
        _rag_worker = RAGWorker()
    return _rag_worker
