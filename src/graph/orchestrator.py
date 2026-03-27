"""Supervisor 核心编排器 — Crayfish Multi-Agent Plan-then-Execute."""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# ── 最大循环次数（防止死亡循环）────────────────────────────────────────────
MAX_LOOP_COUNT = 15

# ── 任务状态枚举 ─────────────────────────────────────────────────────────
TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_REJECTED = "rejected"


class TaskItem(dict):
    """单个任务项。继承 dict 但提供类型安全的构造器。"""

    def __init__(self, task_id: str, description: str, assigned_agent: str):
        dict.__init__(self, {
            "task_id": task_id,
            "description": description,
            "assigned_agent": assigned_agent,
            "status": TASK_STATUS_PENDING,
            "result": None,
            "quality_score": 0.0,
        })


class CrayfishOrchestrator:
    """
    Supervisor 核心编排器 — 实现 Plan-then-Execute 模式。

    工作流程：
    1. Planning: Supervisor 分析需求，生成 JSON Plan（最多 3 个子任务）
    2. Dispatch: 并行/顺序分发任务给各 Worker
    3. Evaluate: Reviewer 评估整体质量
    4. Self-Heal: 质量不达标则触发自修复
    5. Summary: 汇总最终结果
    """

    def __init__(self):
        self.loop_count = 0
        self.task_results: list[dict] = []
        self.plan_id = ""

    async def orchestrate(
        self,
        requirement: str,
        enabled_agents: list[str],
        quality_threshold: float = 8.0,
        progress_callback: Callable[[dict], Awaitable[None]] | None = None,
    ) -> dict:
        """
        执行完整的编排流程。

        Args:
            requirement: 用户需求描述
            enabled_agents: 启用的 Agent 列表 ["search", "rag", "coder"]
            quality_threshold: 质量阈值（0-10）
            progress_callback: SSE 事件推送回调

        Yields:
            dict: 每个步骤的事件数据
        """
        self.loop_count = 0
        self.task_results = []
        self.plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        # ── Step 1: Supervisor Planning ────────────────────────────────────────
        plan_tasks = await self._create_plan(requirement, enabled_agents)

        # 推送计划生成事件
        await self._emit(progress_callback, {
            "type": "supervisor_plan",
            "data": {
                "plan_id": self.plan_id,
                "tasks": [dict(t) for t in plan_tasks],
            },
        })

        # ── Step 2: 执行任务 ────────────────────────────────────────────────
        all_results = []

        # 找出可以并行的任务（search + rag 互不干扰）
        search_tasks = [t for t in plan_tasks if t["assigned_agent"] == "search_worker"]
        rag_tasks = [t for t in plan_tasks if t["assigned_agent"] == "rag_worker"]
        coder_tasks = [t for t in plan_tasks if t["assigned_agent"] == "coder"]

        # 并行执行 search + rag（如果有的话）
        if search_tasks or rag_tasks:
            parallel_tasks = search_tasks + rag_tasks
            await self._emit(progress_callback, {
                "type": "worker_start",
                "agent": "supervisor",
                "task_id": "parallel_dispatch",
                "task_description": f"并行执行 {len(parallel_tasks)} 个任务",
            })

            # 使用 asyncio.gather 并行执行
            coroutines = [self._execute_single_task(t, progress_callback) for t in parallel_tasks]
            parallel_results = await asyncio.gather(*coroutines, return_exceptions=True)

            for result in parallel_results:
                if isinstance(result, Exception):
                    logger.error(f"[orchestrator] parallel task failed: {result}")
                else:
                    all_results.append(result)

        # 顺序执行 coder（依赖前两个任务的结果）
        for coder_task in coder_tasks:
            context = all_results if all_results else None
            result = await self._execute_coder_task(coder_task, context, progress_callback)
            if not isinstance(result, Exception):
                all_results.append(result)

        # ── Step 3: 循环计数检查 ────────────────────────────────────────────
        self.loop_count += 1

        if self.loop_count >= MAX_LOOP_COUNT - 3:
            await self._emit(progress_callback, {
                "type": "loop_warning",
                "loop_count": self.loop_count,
            })

        if self.loop_count >= MAX_LOOP_COUNT:
            await self._emit(progress_callback, {
                "type": "error",
                "message": f"达到最大循环次数 ({MAX_LOOP_COUNT})，强制终止",
            })
            return self._build_final_result(all_results, quality_threshold, False)

        # ── Step 4: 质量评估 ────────────────────────────────────────────────
        overall_quality = self._evaluate_overall_quality(all_results)

        await self._emit(progress_callback, {
            "type": "quality_score",
            "score": overall_quality,
            "threshold": quality_threshold,
        })

        # ── Step 5: 自修复循环 ────────────────────────────────────────────
        healing_attempts = 0
        while overall_quality < quality_threshold and healing_attempts < 2:
            healing_attempts += 1

            await self._emit(progress_callback, {
                "type": "self_healing",
                "attempt": healing_attempts,
                "reason": f"质量评分 {overall_quality:.1f} 低于阈值 {quality_threshold}",
            })

            # 对低质量任务进行自修复
            from src.graph.self_healer import self_heal

            for result in all_results:
                if result.get("quality_score", 0) < quality_threshold:
                    task = {
                        "task_id": result["task_id"],
                        "description": result.get("description", requirement),
                        "assigned_agent": result.get("agent", "unknown"),
                    }
                    healed = await self_heal(
                        task,
                        f"Quality {result.get('quality_score', 0):.1f} < threshold {quality_threshold}",
                        context=all_results,
                        progress_callback=progress_callback,
                    )
                    # 替换原结果
                    idx = next((i for i, r in enumerate(all_results) if r["task_id"] == result["task_id"]), -1)
                    if idx >= 0:
                        all_results[idx] = healed

            # 重新评估
            overall_quality = self._evaluate_overall_quality(all_results)

            await self._emit(progress_callback, {
                "type": "quality_score",
                "score": overall_quality,
                "threshold": quality_threshold,
                "note": f"自修复后重新评估 (尝试 #{healing_attempts})",
            })

        # ── Step 6: 汇总最终结果 ───────────────────────────────────────────
        passed = overall_quality >= quality_threshold

        # 构建汇总报告
        final_result = self._build_final_result(all_results, overall_quality, passed)

        await self._emit(progress_callback, {
            "type": "final_result",
            "summary": final_result["summary"],
            "quality_score": overall_quality,
            "passed": passed,
            "loop_count": self.loop_count,
            "healing_attempts": healing_attempts,
        })

        return final_result

    async def _create_plan(self, requirement: str, enabled_agents: list[str]) -> list[TaskItem]:
        """
        Supervisor Planning Node — 分析需求，生成 JSON Plan。

        使用"最小完备原则"：只拆解当前最紧迫的 2-3 个子任务。
        """
        from src.llm import init_deepseek_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        # 映射简写 -> 完整名称
        _name_map = {
            "search": "search_worker",
            "rag": "rag_worker",
            "coder": "coder",
        }
        enabled = [_name_map.get(a, a) for a in enabled_agents]

        prompt = f"""你是一个任务规划专家（Supervisor）。请分析用户需求，将其拆解为最多 3 个可执行的子任务。

用户需求:
{requirement}

可用的 Agent:
- search_worker: 负责外网搜索和实时信息检索
- rag_worker: 负责本地知识库和记忆检索
- coder: 负责代码编写和生成

拆分原则（最小完备原则）:
1. 只拆分确实需要的子任务，不要过度拆分
2. 互不干扰的任务（如搜索和记忆检索）可以并行
3. 代码生成任务依赖搜索/记忆结果，应放在最后
4. 每个任务描述要清晰、具体

请输出 JSON 格式的 Plan:
{{
  "plan_id": "plan_xxx",
  "tasks": [
    {{
      "task_id": "task_1",
      "description": "具体任务描述",
      "assigned_agent": "search_worker" 或 "rag_worker" 或 "coder"
    }}
  ]
}}

只输出 JSON，不要其他内容。"""

        try:
            llm = init_deepseek_llm(temperature=0.3, streaming=False)
            response = await llm.invoke([
                SystemMessage(content="你是一个专业的任务规划专家，擅长将复杂需求拆解为可执行的子任务。"),
                HumanMessage(content=prompt),
            ])

            content = response.content if hasattr(response, "content") else str(response)

            # 提取 JSON
            json_str = self._extract_json(content)
            if json_str:
                data = json.loads(json_str)
                tasks = data.get("tasks", [])
                result = []
                for t in tasks[:3]:  # 最多 3 个任务
                    task_id = t.get("task_id", f"task_{uuid.uuid4().hex[:6]}")
                    agent = t.get("assigned_agent", "")
                    # 映射简写
                    if agent == "search":
                        agent = "search_worker"
                    elif agent == "rag":
                        agent = "rag_worker"
                    elif agent == "code" or agent == "coder":
                        agent = "coder"
                    result.append(TaskItem(task_id, t.get("description", ""), agent))
                logger.info(f"[orchestrator] plan created: {len(result)} tasks")
                return result

        except Exception as e:
            logger.error(f"[orchestrator] plan creation failed: {e}")

        # 回退：基于关键词自动拆解
        return self._fallback_plan(requirement, enabled)

    def _fallback_plan(self, requirement: str, enabled_agents: list[str]) -> list[TaskItem]:
        """回退计划：基于关键词自动拆解。"""
        _name_map = {
            "search": "search_worker",
            "rag": "rag_worker",
            "coder": "coder",
        }
        # 确保使用完整名称
        enabled = [_name_map.get(a, a) for a in enabled_agents]
        tasks = []
        req_lower = requirement.lower()

        # 搜索相关关键词
        search_keywords = ["search", "搜索", "查找", "调研", "latest", "2025", "2026", "bug", "版本", "用法", "如何"]
        needs_search = any(k in req_lower for k in search_keywords)

        if needs_search and "search_worker" in enabled_agents:
            tasks.append(TaskItem(
                f"task_{uuid.uuid4().hex[:6]}",
                requirement,
                "search_worker"
            ))

        # RAG 相关关键词
        rag_keywords = ["我的", "项目", "配置", "之前", "记忆", "历史", "智程"]
        needs_rag = any(k in req_lower for k in rag_keywords)

        if needs_rag and "rag_worker" in enabled_agents:
            tasks.append(TaskItem(
                f"task_{uuid.uuid4().hex[:6]}",
                requirement,
                "rag_worker"
            ))

        # 代码相关关键词
        code_keywords = ["代码", "生成", "写", "实现", "函数", "class", "def "]
        needs_coder = any(k in req_lower for k in code_keywords)

        if needs_coder and "coder" in enabled_agents:
            tasks.append(TaskItem(
                f"task_{uuid.uuid4().hex[:6]}",
                requirement,
                "coder"
            ))

        # 如果什么都没匹配，至少执行一个
        if not tasks and enabled_agents:
            agent = enabled_agents[0] if enabled_agents else "search_worker"
            tasks.append(TaskItem(
                f"task_{uuid.uuid4().hex[:6]}",
                requirement,
                agent
            ))

        return tasks

    async def _execute_single_task(
        self,
        task: TaskItem,
        progress_callback: Callable[[dict], Awaitable[None]] | None = None,
    ) -> dict:
        """执行单个 Worker 任务。"""
        from src.graph.workers import SearchWorker, RAGWorker, CoderWorker

        task_id = task["task_id"]
        agent = task["assigned_agent"]

        await self._emit(progress_callback, {
            "type": "worker_start",
            "agent": agent,
            "task_id": task_id,
            "task_description": task["description"],
        })

        try:
            if agent == "search_worker":
                worker = SearchWorker()
                result = await worker.execute(task, progress_callback)
            elif agent == "rag_worker":
                worker = RAGWorker()
                result = await worker.execute(task, progress_callback)
            elif agent == "coder":
                worker = CoderWorker()
                result = await worker.execute(task, None, progress_callback)
            else:
                result = {
                    "task_id": task_id,
                    "agent": agent,
                    "result": f"[未知 Agent 类型] {agent}",
                    "confidence": 0.0,
                    "quality_score": 0.0,
                    "source": "unknown",
                    "raw_data": "",
                }

            await self._emit(progress_callback, {
                "type": "worker_done",
                "agent": agent,
                "task_id": task_id,
                "quality_score": result.get("quality_score", 0.0),
                "result": result.get("result", ""),
            })

            return result

        except Exception as e:
            logger.error(f"[orchestrator] task {task_id} failed: {e}")

            await self._emit(progress_callback, {
                "type": "worker_rejected",
                "agent": agent,
                "task_id": task_id,
                "reason": str(e),
            })

            return {
                "task_id": task_id,
                "agent": agent,
                "result": f"[执行失败] {str(e)}",
                "confidence": 0.0,
                "quality_score": 0.0,
                "source": "error",
                "raw_data": str(e),
            }

    async def _execute_coder_task(
        self,
        task: TaskItem,
        context: list[dict] | None,
        progress_callback: Callable[[dict], Awaitable[None]] | None = None,
    ) -> dict:
        """执行 Coder 任务（可以接收其他 Worker 的结果作为上下文）。"""
        from src.graph.workers import CoderWorker

        task_id = task["task_id"]

        await self._emit(progress_callback, {
            "type": "worker_start",
            "agent": "coder",
            "task_id": task_id,
            "task_description": task["description"],
        })

        try:
            worker = CoderWorker()
            result = await worker.execute(task, context, progress_callback)

            await self._emit(progress_callback, {
                "type": "worker_done",
                "agent": "coder",
                "task_id": task_id,
                "quality_score": result.get("quality_score", 0.0),
                "result": result.get("result", ""),
            })

            return result

        except Exception as e:
            logger.error(f"[orchestrator] coder task {task_id} failed: {e}")
            return {
                "task_id": task_id,
                "agent": "coder",
                "result": f"[代码生成失败] {str(e)}",
                "confidence": 0.0,
                "quality_score": 0.0,
                "source": "error",
                "raw_data": str(e),
            }

    def _evaluate_overall_quality(self, results: list[dict]) -> float:
        """评估整体质量分数（所有 Worker 结果的平均分）。"""
        if not results:
            return 0.0

        scores = [r.get("quality_score", 0.0) for r in results]
        return round(sum(scores) / len(scores), 1)

    def _build_final_result(
        self,
        results: list[dict],
        quality_score: float,
        passed: bool,
    ) -> dict:
        """构建最终汇总报告。"""
        lines = [
            "=== Crayfish 多 Agent 编排汇总报告 ===",
            f"计划ID: {self.plan_id}",
            f"执行时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"质量评分: {quality_score:.1f}/10",
            f"质量状态: {'通过' if passed else '未达标'}",
            f"循环次数: {self.loop_count}",
            "",
            "--- 各 Worker 结果 ---",
        ]

        for result in results:
            agent = result.get("agent", "unknown")
            score = result.get("quality_score", 0.0)
            raw = result.get("raw_data", result.get("result", ""))[:500]
            lines.append(f"\n[{agent}] 质量: {score:.1f}")
            lines.append(f"内容: {raw}")

        # 冲突检测
        from src.graph.conflict_resolver import detect_conflict, DataEntry

        entries: list[DataEntry] = []
        for r in results:
            entries.append(DataEntry(
                content=r.get("raw_data", ""),
                source=r.get("source", "unknown"),
                confidence=r.get("confidence", 0.0),
            ))

        if detect_conflict(entries):
            lines.append("\n\n⚠️ 检测到数据冲突！已使用置信度权重自动决策。")

        summary = "\n".join(lines)

        return {
            "summary": summary,
            "quality_score": quality_score,
            "passed": passed,
            "loop_count": self.loop_count,
            "results": results,
            "plan_id": self.plan_id,
        }

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """从文本中提取 JSON 字符串。"""
        import re
        # 尝试匹配 ```json ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            return match.group(1).strip()
        # 尝试直接解析整个文本
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return match.group(0)
        return None

    @staticmethod
    async def _emit(
        callback,
        data: dict,
    ) -> None:
        """安全地推送事件。callback 可以是 async generator 或普通 async callable。"""
        if not callback:
            return
        try:
            result = callback(data)
            # callback 返回的是 async generator（用于 SSE）
            if hasattr(result, "__anext__"):
                async for chunk in result:
                    pass  # generator yields SSE-formatted strings
            else:
                # 普通 awaitable
                await result
        except Exception as e:
            logger.error(f"[orchestrator] emit failed: {e}")


# ── 全局单例 ──────────────────────────────────────────────────────────────
_orchestrator: CrayfishOrchestrator | None = None


def get_orchestrator() -> CrayfishOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CrayfishOrchestrator()
    return _orchestrator
