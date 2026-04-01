"""Supervisor 核心编排器 — Crayfish Multi-Agent Plan-then-Execute."""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from src.utils.output_manager import get_output_manager

logger = logging.getLogger(__name__)

# ── 最大循环次数（防止死亡循环）────────────────────────────────────────────
MAX_LOOP_COUNT = 15

# ── 任务状态枚举 ─────────────────────────────────────────────────────────
TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_REJECTED = "rejected"


class AgentMessageBus:
    """
    Agent 间结构化消息总线 — 负责结果共享与依赖注入。

    工作方式：
    1. Worker 执行完毕后调用 bus.store_result(task_id, result)
    2. 下游 Worker 执行前调用 bus.get_context(task_id) 注入前置结果
    3. 环形依赖检测：每次 add_dependency 时检测是否存在环
    """
    _instance: "AgentMessageBus | None" = None

    def __init__(self):
        self._results: dict[str, dict] = {}
        self._in_degree: dict[str, int] = {}
        self._dependents: dict[str, list[str]] = {}

    @classmethod
    def get_instance(cls) -> "AgentMessageBus":
        if cls._instance is None:
            cls._instance = AgentMessageBus()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        if cls._instance:
            cls._instance._results.clear()
            cls._instance._in_degree.clear()
            cls._instance._dependents.clear()

    def add_dependency(self, from_task_id: str, to_task_id: str) -> None:
        self._dependents.setdefault(from_task_id, []).append(to_task_id)
        self._in_degree[to_task_id] = self._in_degree.get(to_task_id, 0) + 1
        if self._has_cycle(from_task_id, to_task_id):
            raise ValueError(f"环形依赖检测：{from_task_id} -> {to_task_id} 形成环")

    def _has_cycle(self, from_id: str, to_id: str) -> bool:
        visited: set[str] = set()

        def dfs(node: str) -> bool:
            if node == from_id:
                return True
            if node in visited:
                return False
            visited.add(node)
            for dep in self._dependents.get(node, []):
                if dfs(dep):
                    return True
            return False

        return dfs(to_id)

    def store_result(self, task_id: str, result: dict) -> None:
        self._results[task_id] = result

    def get_context(self, task_id: str) -> list[dict]:
        return [v for k, v in self._results.items() if k != task_id]

    def mark_done(self, task_id: str) -> None:
        for dep_task_id in self._dependents.get(task_id, []):
            self._in_degree[dep_task_id] = max(0, self._in_degree.get(dep_task_id, 1) - 1)


class TaskItem(dict):
    """单个任务项。继承 dict，支持 DAG 依赖和执行模式。"""

    def __init__(
        self,
        task_id: str,
        description: str,
        assigned_agent: str,
        worker_kind: str | None = None,
        depends_on: list[str] | None = None,
        execution_mode: str = "parallel",
    ):
        dict.__init__(self, {
            "task_id": task_id,
            "description": description,
            "assigned_agent": assigned_agent,
            "worker_kind": worker_kind,
            "depends_on": depends_on or [],       # 前置任务 ID 列表
            "execution_mode": execution_mode,       # parallel | sequential
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

        # ── 创建输出管理器 ──────────────────────────────────────────────────
        output_mgr = get_output_manager(self.plan_id)
        output_mgr.ensure_dir()

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

        # ── Step 2: DAG 拓扑排序执行 ──────────────────────────────────────────
        bus = AgentMessageBus.get_instance()
        bus.reset()

        task_map: dict[str, TaskItem] = {t["task_id"]: t for t in plan_tasks}

        # 构建 DAG：注册所有 depends_on 依赖
        for t in plan_tasks:
            for dep in t.get("depends_on") or []:
                if dep in task_map:
                    bus.add_dependency(dep, t["task_id"])

        all_results: list[dict] = []
        completed: set[str] = set()

        while len(completed) < len(task_map):
            ready = [
                tid for tid, deg in bus._in_degree.items()
                if tid not in completed and deg == 0
            ]
            # 无依赖的任务（入度从未被加过）也在第一轮执行
            if not ready:
                ready = [tid for tid in task_map if tid not in completed]

            if not ready:
                logger.warning("[orchestrator] DAG deadlock, breaking")
                break

            await self._emit(progress_callback, {
                "type": "worker_start",
                "agent": "supervisor",
                "task_id": "parallel_batch",
                "task_description": f"并行执行 {len(ready)} 个任务: {ready}",
            })

            coroutines = [
                self._execute_single_task(task_map[tid], bus.get_context(tid), progress_callback, output_mgr)
                for tid in ready
            ]
            batch_results = await asyncio.gather(*coroutines, return_exceptions=True)

            for tid, result in zip(ready, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"[orchestrator] task {tid} failed: {result}")
                    result = {
                        "task_id": tid,
                        "agent": task_map[tid]["assigned_agent"],
                        "result": f"[执行失败] {str(result)}",
                        "confidence": 0.0,
                        "quality_score": 0.0,
                        "source": "error",
                        "raw_data": str(result),
                    }
                bus.store_result(tid, result)
                all_results.append(result)
                completed.add(tid)
                bus.mark_done(tid)

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
            return self._build_final_result(all_results, quality_threshold, False, output_mgr=output_mgr, requirement=requirement)

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
        final_result = self._build_final_result(all_results, overall_quality, passed, output_mgr=output_mgr, requirement=requirement)

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

        # 构建动态可用 Agent 描述段落（含 capabilities，供 Supervisor 路由匹配）
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
            caps = p.get("capabilities") or []
            caps_str = (", ".join(caps)) if caps else "无"
            agent_lines.append(f"- {pid}: {desc} [能力: {caps_str}]")

        if not agent_lines:
            agent_lines = [
                "- agent_worker_search: 外网搜索 (Tavily) [能力: web_search, browse_page]",
                "- agent_worker_rag: 本地知识库检索 [能力: memory_search, knowledge_base_search]",
                "- agent_worker_coder: 代码生成 [能力: code_generation, code_review]",
            ]
            enabled_kinds = {"search_worker", "rag_worker", "coder"}

        agents_desc = "\n".join(agent_lines)
        chief_note = ""
        if "agent_main" in _all_participants:
            cp = _all_participants["agent_main"]
            chief_note = (
                f"总协调者: agent_main（{cp.get('name', 'Chief Coordinator')}）— "
                "由系统指定为最高管理者；**禁止**在 JSON 的 assigned_agent 中使用 agent_main。"
                "子任务仅能分配给下列执行型 Agent id。"
            )

        from src.graph.prompt import build_supervisor_prompt

        supervisor_prompt = build_supervisor_prompt(
            requirement=requirement,
            agents_desc=agents_desc,
            chief_note=chief_note,
            max_tasks=5,
        )

        try:
            llm = init_deepseek_llm(temperature=0.3, streaming=False)
            response = await llm.ainvoke([
                SystemMessage(content="你是一个专业的任务规划专家，擅长将复杂需求拆解为可执行的子任务。你也是 Crayfish 系统的首席协调官，只负责规划，禁止执行子任务。"),
                HumanMessage(content=supervisor_prompt),
            ])

            content = response.content if hasattr(response, "content") else str(response)

            # 提取 JSON
            json_str = self._extract_json(content)
            if json_str:
                data = json.loads(json_str)
                tasks = data.get("tasks", [])
                result = []
                builtin_kinds = {"search_worker", "rag_worker", "coder"}
                for t in tasks:
                    task_id = t.get("task_id", f"task_{uuid.uuid4().hex[:6]}")
                    agent_id = t.get("assigned_agent", "")
                    wk = None
                    norm_id = _short_map.get(agent_id, agent_id)
                    p = _all_participants.get(agent_id) or _all_participants.get(norm_id)
                    if p:
                        wk = p.get("worker_kind")
                    else:
                        wk = _short_map.get(agent_id)
                        if wk:
                            norm_id = wk
                        p = _all_participants.get(wk) if wk else None
                    builtin_kinds = {"search_worker", "rag_worker", "coder"}
                    if wk and wk not in enabled_kinds:
                        if wk in builtin_kinds:
                            logger.warning(f"[orchestrator] plan assigned unknown builtin {agent_id} ({wk}), skipping")
                            continue
                    depends_on = t.get("depends_on") or []
                    execution_mode = t.get("execution_mode") or "parallel"
                    result.append(TaskItem(
                        task_id,
                        t.get("description", ""),
                        norm_id,
                        wk,
                        depends_on=depends_on,
                        execution_mode=execution_mode,
                    ))
                logger.info(f"[orchestrator] plan created: {len(result)} tasks")
                return result

        except Exception as e:
            logger.error(f"[orchestrator] plan creation failed: {e}")

        # 回退：基于关键词自动拆解（感知所有已注册 agent）
        return self._fallback_plan(requirement, enabled_kinds, self._participants)

    def _fallback_plan(
        self,
        requirement: str,
        enabled_kinds: set[str],
        all_participants: dict[str, dict] | None = None,
    ) -> list[TaskItem]:
        """
        回退计划：基于关键词自动拆解。

        动态感知所有已注册的 agent（包含自定义），不再局限于内置 3 个。
        优先匹配内置 agent（search/rag/coder），其余落入 generic_tasks。
        """
        tasks = []
        req_lower = requirement.lower()

        # 内置关键词匹配
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

        # 若无匹配，遍历所有已注册的自定义 agent（排除 agent_main 和内置）
        if not tasks and enabled_kinds:
            builtin_kinds = {"search_worker", "rag_worker", "coder"}
            participants = all_participants or {}

            for pid, p in participants.items():
                if pid == "agent_main":
                    continue
                wk = p.get("worker_kind")
                if wk and wk in enabled_kinds and wk not in builtin_kinds:
                    tasks.append(TaskItem(
                        f"task_{uuid.uuid4().hex[:6]}",
                        requirement,
                        pid,
                        wk,
                    ))
                    break  # 选第一个匹配的自定义 agent

            # 再兜底：选 enabled_kinds 中的第一个
            if not tasks:
                first_kind = next((k for k in enabled_kinds if k in builtin_kinds), next(iter(enabled_kinds), None))
                if first_kind:
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
        context: list[dict] | None = None,
        progress_callback: Callable[[dict], Awaitable[None]] | None = None,
    ) -> dict:
        """执行单个 Worker 任务。路由依据 worker_kind 字段，支持 context 注入。"""
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
                result = await worker.execute(task, context, progress_callback)
            else:
                result = await self._execute_generic_task(task, context, progress_callback)

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

    async def _execute_generic_task(
        self,
        task: TaskItem,
        context: list[dict] | None = None,
        progress_callback: Callable[[dict], Awaitable[None]] | None = None,
    ) -> dict:
        """
        通用 Agent 执行器 — 处理所有非内置 worker_kind。

        替代原 _dispatch_generic()，新增：
        - 实时 SSE worker_progress 推送（初始化 → 分发 → 完成）
        - 动态质量评分（基于结果长度、结构化程度、关键词覆盖）
        """
        task_id = task["task_id"]
        agent_id = task["assigned_agent"]
        profile = self._participants.get(agent_id) or {}
        agent_name = profile.get("name", agent_id)

        # 推送进度：初始化
        await self._emit(progress_callback, {
            "type": "worker_progress",
            "agent": agent_id,
            "task_id": task_id,
            "message": f"[{agent_name}] 正在初始化 AgentGraph...",
        })

        from src.multi_agent.orchestrator import get_agent_registry
        reg = get_agent_registry()

        # 推送进度：任务分发
        await self._emit(progress_callback, {
            "type": "worker_progress",
            "agent": agent_id,
            "task_id": task_id,
            "message": f"[{agent_name}] 正在执行任务 \"{task['description'][:40]}...\"",
        })

        try:
            dispatch_result = await reg.dispatch_task(
                agent_id,
                task["description"],
                thread_id=task_id,
                context=context,
            )

            if dispatch_result.get("status") == "ok":
                raw_result = dispatch_result.get("result", "")
                quality_score = self._evaluate_generic_quality(raw_result, task["description"])

                await self._emit(progress_callback, {
                    "type": "worker_progress",
                    "agent": agent_id,
                    "task_id": task_id,
                    "message": f"[{agent_name}] 执行完成，质量评分: {quality_score:.1f}",
                })

                return {
                    "task_id": task_id,
                    "agent": agent_id,
                    "result": raw_result,
                    "confidence": 0.75,
                    "quality_score": quality_score,
                    "source": "generic_agent",
                    "raw_data": raw_result,
                }
            else:
                return {
                    "task_id": task_id,
                    "agent": agent_id,
                    "result": f"[调度失败] {dispatch_result.get('message', '未知错误')}",
                    "confidence": 0.0,
                    "quality_score": 0.0,
                    "source": "dispatch_error",
                    "raw_data": dispatch_result.get("message", ""),
                }

        except Exception as e:
            logger.error(f"[orchestrator] generic task {task_id} failed: {e}")
            return {
                "task_id": task_id,
                "agent": agent_id,
                "result": f"[执行异常] {str(e)}",
                "confidence": 0.0,
                "quality_score": 0.0,
                "source": "generic_error",
                "raw_data": str(e),
            }

    def _evaluate_generic_quality(self, result: str, original_task: str) -> float:
        """
        评估通用 Agent 结果质量（0-10 分）。
        策略：结果长度 + 结构化程度 + 关键词覆盖 + 错误检测。
        """
        if not result or len(result) < 10:
            return 2.0
        if any(k in result for k in ["[执行失败]", "[调度失败]", "[异常]", "Error:", "Traceback"]):
            return 3.0

        score = 5.0
        if len(result) > 200:
            score += 1.0
        if "```" in result or ("\n- " in result) or ("\n1." in result):
            score += 1.5
        task_words = set(original_task.lower().split())
        result_words = set(result.lower().split())
        overlap = len(task_words & result_words)
        if overlap >= 3:
            score += 1.5
        if len(result) > 5000:
            score -= 0.5

        return min(max(score, 0.0), 10.0)

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
