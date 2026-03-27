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

    def __init__(self, task_id: str, description: str, assigned_agent: str, worker_kind: str | None = None):
        dict.__init__(self, {
            "task_id": task_id,
            "description": description,
            "assigned_agent": assigned_agent,
            "worker_kind": worker_kind,
            "status": TASK_STATUS_PENDING,
            "result": None,
            "quality_score": 0.0,
        })


class CrayfishOrchestrator:
    """
    Supervisor 核心编排器 — 实现 Plan-then-Execute 模式。

    工作流程（前端以「三省六部 + 台阁」呈现；对齐现代多角色产品/安全/工程分工）：
    1. 中书制敕: Supervisor 分析需求，生成 JSON Plan（最多 3 个子任务）
    2. 尚书牒发: 并行/顺序分发任务给各 Worker（六曹执司）
    3. 刑曹质勘: Reviewer 评估整体质量
    4. 吏曹纠偏: 质量不达标则触发自修复
    5. 台阁进呈: 汇总最终结果
    """

    def __init__(self):
        self.loop_count = 0
        self.task_results: list[dict] = []
        self.plan_id = ""
        self._participants: dict[str, dict] = {}

    async def orchestrate(
        self,
        requirement: str,
        enabled_agents: list[str],
        quality_threshold: float = 8.0,
        progress_callback: Callable[[dict], Awaitable[None]] | None = None,
        participants: list[dict] | None = None,
    ) -> dict:
        """
        执行完整的编排流程。

        Args:
            requirement: 用户需求描述
            enabled_agents: 启用的 Agent ID 列表（向后兼容也接受简写 search/rag/coder）
            quality_threshold: 质量阈值（0-10）
            progress_callback: SSE 事件推送回调
            participants: Agent 档案列表，含 worker_kind 用于路由决策
        """
        self.loop_count = 0
        self.task_results = []
        self.plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        self._participants = {p["id"]: p for p in (participants or [])}

        # ── Step 1: Supervisor Planning ────────────────────────────────────────
        await self._emit(progress_callback, {
            "type": "worker_progress",
            "agent": "supervisor",
            "message": "Supervisor 正在调用 LLM 生成执行计划…",
        })
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

        # 按 worker_kind 分组：内置并行的（search_worker / rag_worker）vs 顺序的（coder）
        # 通用 Agent（worker_kind 为空/null）走并行（dispatch_task）
        builtin_parallel = {"search_worker", "rag_worker"}
        sequential = {"coder"}

        parallel_tasks = [
            t for t in plan_tasks
            if (t.get("worker_kind") or "") in builtin_parallel
        ]
        coder_tasks = [t for t in plan_tasks if (t.get("worker_kind") or "") in sequential]
        # 通用 Agent（没有 worker_kind 或 worker_kind 不属于内置）
        generic_tasks = [
            t for t in plan_tasks
            if (t.get("worker_kind") or "") not in builtin_parallel
            and (t.get("worker_kind") or "") not in sequential
        ]

        # 并行执行 search + rag + 通用 Agent
        if parallel_tasks or generic_tasks:
            to_run = parallel_tasks + generic_tasks
            await self._emit(progress_callback, {
                "type": "worker_start",
                "agent": "supervisor",
                "task_id": "parallel_dispatch",
                "task_description": f"并行执行 {len(to_run)} 个任务",
            })
            coroutines = [self._execute_single_task(t, progress_callback) for t in to_run]
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

        enabled_agents 可以是简写 (search/rag/coder) 也可以是 agent_id，
        participants 字典 {id: {id, name, role, worker_kind, ...}} 提供完整档案。
        """
        from src.llm import init_deepseek_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        # 简写 -> worker_kind 映射（向后兼容旧前端传 search/rag/coder）
        _short_map = {
            "search": "search_worker",
            "rag":    "rag_worker",
            "coder":  "coder",
        }
        # 构建 participants 索引：id -> profile，优先用 self._participants
        _all_participants = dict(self._participants)  # copy
        # 若前端传了简写但没有对应的 participant，补上内置档案
        for short, wk in _short_map.items():
            if short in enabled_agents or wk in enabled_agents:
                aid = f"agent_worker_{short}" if short != "coder" else "agent_worker_coder"
                if aid not in _all_participants:
                    from src.multi_agent.orchestrator import get_agent_registry
                    reg = get_agent_registry()
                    p = reg.get(aid)
                    if p:
                        _all_participants[aid] = p
                    else:
                        # 兜底
                        _all_participants[wk] = {
                            "id": wk, "name": wk.replace("_", " ").title(),
                            "role": wk, "worker_kind": wk,
                            "description": "",
                        }

        # 归一化为 worker_kind 集合（用于校验和 fallback）；agent_main 仅总协调，不参与执行类 kind
        enabled_kinds: set[str] = set()
        for a in enabled_agents:
            if a == "agent_main":
                continue
            if a in _short_map:
                enabled_kinds.add(_short_map[a])
            else:
                # a 是 agent_id，查档案
                p = _all_participants.get(a)
                if p and p.get("worker_kind"):
                    enabled_kinds.add(p["worker_kind"])
                else:
                    enabled_kinds.add(a)  # 直接当 worker_kind 用

        # 构建动态可用 Agent 描述段落（不含 agent_main，避免被指派执行子任务）
        agent_lines = []
        for pid, p in _all_participants.items():
            if pid == "agent_main":
                continue
            wk = p.get("worker_kind")
            if wk and wk not in enabled_kinds:
                continue
            name = p.get("name", pid)
            desc = p.get("description") or (
                {"search_worker": "外网实时信息搜索，使用 Tavily",
                 "rag_worker": "本地知识库与记忆检索",
                 "coder": "代码生成与编写"}.get(wk, "")
            )
            # assigned_agent 填 agent_id（稳定）
            agent_lines.append(f"- {pid}: {desc}")

        if not agent_lines:
            agent_lines = [
                "- agent_worker_search: 外网搜索 (Tavily)",
                "- agent_worker_rag: 本地知识库检索",
                "- agent_worker_coder: 代码生成",
            ]
            enabled_kinds = {"search_worker", "rag_worker", "coder"}

        agents_desc = "\n".join(agent_lines)
        chief_note = ""
        if "agent_main" in _all_participants:
            cp = _all_participants["agent_main"]
            chief_note = (
                f"\n\n总协调者: agent_main（{cp.get('name', 'Chief Coordinator')}）— "
                "由系统指定为最高管理者；**禁止**在 JSON 的 assigned_agent 中使用 agent_main。"
                "子任务仅能分配给下列执行型 Agent id。"
            )

        prompt = f"""你是一个任务规划专家（Supervisor）。请分析用户需求，将其拆解为最多 3 个可执行的子任务。

用户需求:
{requirement}

可用的 Agent（必须严格使用下面的 agent id 作为 assigned_agent）:
{agents_desc}{chief_note}

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
      "assigned_agent": "上面列表中的 agent id"
    }}
  ]
}}

只输出 JSON，不要其他内容。"""

        try:
            llm = init_deepseek_llm(temperature=0.3, streaming=False)
            response = await llm.ainvoke([
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
                for t in tasks[:3]:
                    task_id = t.get("task_id", f"task_{uuid.uuid4().hex[:6]}")
                    agent_id = t.get("assigned_agent", "")
                    wk = None
                    # 归一化 assigned_agent：简写优先映射
                    norm_id = _short_map.get(agent_id, agent_id)
                    # 查找 worker_kind
                    p = _all_participants.get(agent_id) or _all_participants.get(norm_id)
                    if p:
                        wk = p.get("worker_kind")
                    else:
                        # unknown id，尝试用简写映射
                        wk = _short_map.get(agent_id)
                        if wk:
                            norm_id = wk
                        p = _all_participants.get(wk) if wk else None
                    # 校验
                    if wk and wk not in enabled_kinds:
                        logger.warning(f"[orchestrator] plan assigned unknown agent {agent_id} ({wk}), skipping")
                        continue
                    result.append(TaskItem(task_id, t.get("description", ""), norm_id, wk))
                logger.info(f"[orchestrator] plan created: {len(result)} tasks")
                return result

        except Exception as e:
            logger.error(f"[orchestrator] plan creation failed: {e}")

        # 回退：基于关键词自动拆解
        return self._fallback_plan(requirement, enabled_kinds)

    def _fallback_plan(self, requirement: str, enabled_kinds: set[str]) -> list[TaskItem]:
        """回退计划：基于关键词自动拆解。"""
        tasks = []
        req_lower = requirement.lower()

        if "search_worker" in enabled_kinds:
            search_keywords = ["search", "搜索", "查找", "调研", "latest", "2025", "2026", "bug", "版本", "用法", "如何"]
            if any(k in req_lower for k in search_keywords):
                tasks.append(TaskItem(
                    f"task_{uuid.uuid4().hex[:6]}",
                    requirement,
                    "agent_worker_search",
                    "search_worker",
                ))

        if "rag_worker" in enabled_kinds:
            rag_keywords = ["我的", "项目", "配置", "之前", "记忆", "历史", "智程"]
            if any(k in req_lower for k in rag_keywords):
                tasks.append(TaskItem(
                    f"task_{uuid.uuid4().hex[:6]}",
                    requirement,
                    "agent_worker_rag",
                    "rag_worker",
                ))

        if "coder" in enabled_kinds:
            code_keywords = ["代码", "生成", "写", "实现", "函数", "class", "def "]
            if any(k in req_lower for k in code_keywords):
                tasks.append(TaskItem(
                    f"task_{uuid.uuid4().hex[:6]}",
                    requirement,
                    "agent_worker_coder",
                    "coder",
                ))

        if not tasks and enabled_kinds:
            first_kind = next(iter(enabled_kinds))
            kind_to_id = {
                "search_worker": "agent_worker_search",
                "rag_worker": "agent_worker_rag",
                "coder": "agent_worker_coder",
            }
            tasks.append(TaskItem(
                f"task_{uuid.uuid4().hex[:6]}",
                requirement,
                kind_to_id.get(first_kind, first_kind),
                first_kind,
            ))

        return tasks

    async def _execute_single_task(
        self,
        task: TaskItem,
        progress_callback: Callable[[dict], Awaitable[None]] | None = None,
    ) -> dict:
        """执行单个 Worker 任务。路由依据 worker_kind 字段。"""
        from src.graph.workers import SearchWorker, RAGWorker, CoderWorker

        task_id = task["task_id"]
        agent_id = task["assigned_agent"]
        wk = task.get("worker_kind") or ""

        await self._emit(progress_callback, {
            "type": "worker_start",
            "agent": agent_id,
            "task_id": task_id,
            "task_description": task["description"],
        })

        try:
            if wk == "search_worker":
                worker = SearchWorker()
                result = await worker.execute(task, progress_callback)
            elif wk == "rag_worker":
                worker = RAGWorker()
                result = await worker.execute(task, progress_callback)
            elif wk == "coder":
                worker = CoderWorker()
                result = await worker.execute(task, None, progress_callback)
            else:
                # 通用 Agent（无 worker_kind）：走 AgentRegistry.dispatch_task
                result = await self._dispatch_generic(agent_id, task["description"], task_id)

            await self._emit(progress_callback, {
                "type": "worker_done",
                "agent": agent_id,
                "task_id": task_id,
                "quality_score": result.get("quality_score", 0.0),
                "result": result.get("result", ""),
            })

            return result

        except Exception as e:
            logger.error(f"[orchestrator] task {task_id} failed: {e}")

            await self._emit(progress_callback, {
                "type": "worker_rejected",
                "agent": agent_id,
                "task_id": task_id,
                "reason": str(e),
            })

            return {
                "task_id": task_id,
                "agent": agent_id,
                "result": f"[执行失败] {str(e)}",
                "confidence": 0.0,
                "quality_score": 0.0,
                "source": "error",
                "raw_data": str(e),
            }

    async def _dispatch_generic(
        self,
        agent_id: str,
        task_description: str,
        thread_id: str,
    ) -> dict:
        """
        对通用 Agent（非内置 worker_kind）调用 AgentRegistry 分发任务。
        返回与 Worker result 同构的字典。
        """
        from src.multi_agent.orchestrator import get_agent_registry

        reg = get_agent_registry()
        try:
            dispatch_result = await reg.dispatch_task(agent_id, task_description, thread_id=thread_id)
            if dispatch_result.get("status") == "ok":
                return {
                    "task_id": thread_id,
                    "agent": agent_id,
                    "result": dispatch_result.get("result", ""),
                    "confidence": 0.8,
                    "quality_score": 8.0,
                    "source": "generic_agent",
                    "raw_data": dispatch_result.get("result", ""),
                }
            else:
                return {
                    "task_id": thread_id,
                    "agent": agent_id,
                    "result": f"[调度失败] {dispatch_result.get('message', '未知错误')}",
                    "confidence": 0.0,
                    "quality_score": 0.0,
                    "source": "dispatch_error",
                    "raw_data": dispatch_result.get("message", ""),
                }
        except Exception as e:
            logger.error(f"[orchestrator] dispatch_generic failed for {agent_id}: {e}")
            return {
                "task_id": thread_id,
                "agent": agent_id,
                "result": f"[调度异常] {str(e)}",
                "confidence": 0.0,
                "quality_score": 0.0,
                "source": "dispatch_error",
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
            if hasattr(result, "__anext__"):
                async for chunk in result:
                    pass
            else:
                await result
        except asyncio.CancelledError:
            raise  # 取消必须传播
        except Exception as e:
            logger.warning(f"[orchestrator] _emit: callback raised {e}", exc_info=True)


# ── 全局单例 ──────────────────────────────────────────────────────────────
_orchestrator: CrayfishOrchestrator | None = None


def get_orchestrator() -> CrayfishOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CrayfishOrchestrator()
    return _orchestrator
