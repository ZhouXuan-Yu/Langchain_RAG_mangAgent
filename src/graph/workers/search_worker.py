"""Search Worker — LLM 驱动的搜索专家（户部·探事）.

采用 Plan-Execute-Synthesize 模式：
1. Plan：LLM 分析任务，确定搜索策略和关键词
2. Execute：执行 web_search（可多次迭代优化）
3. Synthesize：LLM 整合结果，生成结构化报告
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.output_manager import OutputManager

from src.tools.browser_tools import web_search, browse_page

logger = logging.getLogger(__name__)

_WEB_CONFIDENCE = 0.95
_BROWSE_CONFIDENCE = 0.90

# 用于从 LLM 输出中提取 FILENAME 和 DESC 注释
_FILENAME_RE = re.compile(r"<!--\s*FILENAME:\s*([^\s>]+)\s*-->", re.IGNORECASE)
_DESC_RE = re.compile(r"<!--\s*DESC:\s*([^\n]+?)\s*-->", re.IGNORECASE)


class SearchWorker:
    """Search Worker — LLM 驱动的外网实时信息搜集专家（户部·探事）."""

    def __init__(self):
        self.name = "search_worker"

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
                "message": "【户部·探事】正在分析搜索策略...",
            })

        # ── Step 1: LLM 分析任务，确定搜索策略 ──────────────────────────────
        search_plan = await self._plan_search(description)
        queries = search_plan.get("queries", [description])
        focus_areas = search_plan.get("focus_areas", [])

        if progress_callback:
            await progress_callback({
                "type": "worker_progress",
                "agent": self.name,
                "task_id": task_id,
                "message": f"【户部·探事】确定搜索词: {', '.join(queries[:3])}...",
            })

        # ── Step 2: 执行多轮搜索（迭代优化）─────────────────────────────────
        all_raw_results = []
        for i, query in enumerate(queries[:5]):  # 最多 5 个搜索词
            try:
                raw = await asyncio.to_thread(web_search.invoke, {"query": query})
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
                        "message": f"【户部·探事】搜索 [{i+1}/{min(len(queries), 5)}]: {query[:30]}...",
                    })
            except Exception as e:
                logger.warning(f"[search_worker] query '{query}' failed: {e}")

        # 合并原始结果
        merged_raw = "\n\n".join(
            f"[搜索词 {r['order']+1}: {r['query']}]\n{r['result']}"
            for r in all_raw_results
        ) if all_raw_results else "[搜索失败]"

        # ── Step 3: LLM 综合分析，生成结构化报告 ─────────────────────────────
        if progress_callback:
            await progress_callback({
                "type": "worker_progress",
                "agent": self.name,
                "task_id": task_id,
                "message": "【户部·探事】正在综合分析搜索结果...",
            })

        synthesized = await self._synthesize(
            original_task=description,
            raw_results=merged_raw,
            focus_areas=focus_areas,
        )

        quality_score = self._evaluate_quality(synthesized)
        freshness = self._analyze_freshness(synthesized, description)

        files_saved = []
        result_text = synthesized

        # 如果有 OutputManager 且 output_type 不是纯搜索，写入文件
        if output_manager and output_type not in ("search_only",):
            if synthesized and not synthesized.startswith("[搜索失败]") and not synthesized.startswith("[综合分析失败]"):
                fname, desc = self._parse_annotations(synthesized)
                if not fname:
                    fname = suggested_filename or f"{task_id}_research.md"

                pure_content = self._strip_annotations(synthesized)
                file_info = output_manager.save_file(
                    content=pure_content,
                    output_type=output_type,
                    agent_id=self.name,
                    task_id=task_id,
                    filename=fname,
                    description=desc or f"搜索调研: {description[:80]}",
                )
                if file_info:
                    files_saved.append(file_info.to_dict())
                    result_text = f"[已保存到文件: {fname}]\n\n{synthesized}"

        result_text_full = f"""[Search Worker | 户部·探事 结果]
任务: {description}

{result_text}

时效性评估: {freshness}
置信度: {_WEB_CONFIDENCE}
质量评分: {quality_score}/10
"""

        logger.info(f"[search_worker] task {task_id} done, quality={quality_score}, files={len(files_saved)}")

        return {
            "task_id": task_id,
            "agent": self.name,
            "result": result_text_full,
            "confidence": _WEB_CONFIDENCE,
            "quality_score": quality_score,
            "source": "web",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "freshness": freshness,
            "raw_data": synthesized,
            "files": files_saved,
        }

    async def _plan_search(self, task_description: str) -> dict:
        """LLM 分析任务，制定搜索策略。"""
        from src.llm import init_deepseek_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = init_deepseek_llm(temperature=0.3, streaming=False)
        prompt = f"""你是一个资深网络研究员。你的任务是根据用户需求制定搜索策略。

用户需求：{task_description}

请分析需求，输出 JSON 格式的搜索计划：
{{
  "queries": ["搜索词1", "搜索词2", "..."],
  "focus_areas": ["重点关注领域1", "重点关注领域2"],
  "priority": "high/medium/low",
  "iterations": 2
}}

要求：
- queries 至少 1 个，最多 3 个（覆盖中英文、不同角度）
- 优先使用 2025-2026 年相关的搜索词
- iterations 表示预期需要的搜索轮数（1-3）

只输出 JSON，不要其他内容。"""

        try:
            response = await llm.ainvoke([
                SystemMessage(content="你是一个专业的搜索策略分析师，擅长将复杂需求转化为精准搜索词。"),
                HumanMessage(content=prompt),
            ])
            content = response.content if hasattr(response, "content") else str(response)
            import json, re
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            logger.warning(f"[search_worker] _plan_search failed: {e}")
        return {"queries": [task_description], "focus_areas": [], "priority": "medium", "iterations": 1}

    async def _synthesize(
        self,
        original_task: str,
        raw_results: str,
        focus_areas: list[str],
    ) -> str:
        """LLM 整合搜索结果，生成结构化报告。"""
        from src.llm import init_deepseek_llm
        from src.graph.prompt import SEARCH_WORKER_SYSTEM_PROMPT
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = init_deepseek_llm(temperature=0.2, streaming=False)

        prompt = f"""{SEARCH_WORKER_SYSTEM_PROMPT}

# 当前任务
{original_task}

# 原始搜索结果
{raw_results}

# 重点关注领域（供你参考）
{', '.join(focus_areas) if focus_areas else '（未指定）'}

请按上述输出格式整理搜索结果，生成完整报告。"""

        try:
            response = await llm.ainvoke([
                SystemMessage(content=SEARCH_WORKER_SYSTEM_PROMPT),
                HumanMessage(content=(
                    f"# 当前任务\n{original_task}\n\n# 原始搜索结果\n{raw_results}\n\n"
                    f"# 重点关注领域\n{', '.join(focus_areas) if focus_areas else '（未指定）'}\n\n"
                    "请按输出格式整理搜索结果，生成完整报告。"
                )),
            ])
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"[search_worker] _synthesize failed: {e}")
            return f"[综合分析失败] {raw_results}"

    def _evaluate_quality(self, result: str) -> float:
        if not result or len(result) < 20:
            return 2.0
        if "[搜索失败]" in result or "[综合分析失败]" in result:
            return 3.0

        score = 6.0
        if "http" in result:
            score += 1.0
        if len(result) > 200:
            score += 1.0
        if "## " in result or "关键发现" in result:
            score += 1.5
        if len(result) > 500:
            score += 0.5

        return min(score, 10.0)

    def _analyze_freshness(self, result: str, query: str) -> str:
        now = datetime.now(timezone.utc)
        query_lower = query.lower()

        fresh_keywords = ["2025", "2026", "最新", "最新版", "新版本", "刚刚", "recent"]
        old_keywords = ["2019", "2020", "2021", "旧版", "过时", "deprecated"]

        if any(k in query_lower for k in fresh_keywords):
            if any(k in result for k in ["2025", "2026"]):
                return "最新 (2025-2026)"
            return "较新 (近期)"
        if any(k in query_lower for k in old_keywords):
            return "历史数据"

        return f"当前时间: {now.strftime('%Y-%m-%d')}，结果适用"

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


_search_worker: "SearchWorker | None" = None


def get_search_worker() -> SearchWorker:
    global _search_worker
    if _search_worker is None:
        _search_worker = SearchWorker()
    return _search_worker
