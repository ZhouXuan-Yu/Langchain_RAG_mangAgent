"""Supervisor 核心编排器 -- Crayfish Multi-Agent Plan-then-Execute."""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "debug-d85885.log")

def _dlog(session_id: str, run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    """Write one NDJSON line to the session debug log (synchronous, fire-and-forget)."""
    import datetime as _dt
    entry = {
        "sessionId": session_id,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(_dt.datetime.now().timestamp() * 1000),
    }
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

from src.config import ENABLE_LANGSMITH, ENABLE_EPISODIC_MEMORY, ENABLE_HIERARCHICAL_ORCHESTRATION, DEFAULT_MODEL
from src.utils.output_manager import get_output_manager

# ── LangSmith 条件导入 ──────────────────────────────────────────────────────
if ENABLE_LANGSMITH:
    from langsmith import traceable
else:
    def traceable(**kwargs):
        def decorator(fn):
            return fn
        return decorator

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
    Agent 间结构化消息总线 -- 负责结果共享与依赖注入。

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
        output_type: str = "search_only",
        suggested_filename: str = "",
    ):
        dict.__init__(self, {
            "task_id": task_id,
            "description": description,
            "assigned_agent": assigned_agent,
            "worker_kind": worker_kind,
            "depends_on": depends_on or [],       # 前置任务 ID 列表
            "execution_mode": execution_mode,       # parallel | sequential
            "output_type": output_type,             # code | html | markdown | doc | data | report | search_only | mixed
            "suggested_filename": suggested_filename,
            "status": TASK_STATUS_PENDING,
            "result": None,
            "quality_score": 0.0,
        })


class CrayfishOrchestrator:
    """
    Supervisor 核心编排器 -- 实现 Plan-then-Execute 模式。

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

    @traceable(project_name="crayfish-orchestration", tags=["supervisor", "multi-agent"])
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
        self._start_time = datetime.now()

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

        # ── P2-D: 多级编排分支 ─────────────────────────────────────────────
        if ENABLE_HIERARCHICAL_ORCHESTRATION:
            _dlog("d85885", "debug-run", "B1-orch",
                  "orchestrator.py:orchestrate",
                  "calling _orchestrate_recursive with initial_plan",
                  {"n_tasks": len(plan_tasks), "task_ids": [t.get("task_id") for t in plan_tasks]})
            final_result = await self._orchestrate_recursive(
                requirement=requirement,
                enabled_agents=enabled_agents,
                participants=participants,
                depth=0,
                progress_callback=progress_callback,
                initial_plan=[dict(t) for t in plan_tasks],
                root_output_mgr=output_mgr,
            )
            # 递归路径也需要发出 final_result 事件（供 SSE 前端和 orch_jobs 消费）
            await self._emit(progress_callback, {
                "type": "final_result",
                "summary": final_result.get("summary", ""),
                "quality_score": final_result.get("quality_score", 0.0),
                "passed": final_result.get("passed", False),
                "loop_count": self.loop_count,
                "plan_id": final_result.get("plan_id", self.plan_id),
                "output_dir": final_result.get("output_dir", ""),
                "files": final_result.get("files", []),
                "summary_file": final_result.get("summary_file"),
            })
            return final_result

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

            for idx, result in enumerate(batch_results):
                if idx >= len(ready):
                    continue
                tid = ready[idx]
                if isinstance(result, Exception):
                    logger.error(f"[orchestrator] task {tid} failed: {result}")
                    _agent = task_map[tid]["assigned_agent"] if tid in task_map else "unknown"
                    result = {
                        "task_id": tid,
                        "agent": _agent,
                        "result": f"[执行失败] {str(result)}",
                        "confidence": 0.0,
                        "quality_score": 0.0,
                        "source": "error",
                        "raw_data": str(result),
                    }
                elif tid not in task_map:
                    logger.warning(f"[orchestrator] tid {tid} not in task_map, skipping")
                    continue
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

        # ── P2-C: 记录 token 使用量到 DB ────────────────────────────────
        duration_ms = int((datetime.now() - self._start_time).total_seconds() * 1000)
        try:
            from src.utils.token_tracker import TokenTracker
            tracker = TokenTracker(model=DEFAULT_MODEL)
            tracker.record(
                prompt_tokens=len(requirement) // 4,  # 粗略估算，实际以 Worker 层记录为准
                completion_tokens=len(str(final_result.get("summary", ""))) // 4,
                label="orchestrate_finalize",
            )
            tracker.save_to_db(
                job_id=self.plan_id,
                agent_id="supervisor",
                duration_ms=duration_ms,
                label="orchestrate_total",
            )
        except Exception as e:
            logging.getLogger(__name__).warning(f"[orchestrator] token record failed: {e}")

        # ── P2-A: 写入 Episodic Memory ─────────────────────────────────
        if ENABLE_EPISODIC_MEMORY:
            try:
                from src.memory.episode_store import get_episode_store
                store = get_episode_store()
                agents_used = list({r.get("agent", "unknown") for r in all_results})
                store.save(
                    job_id=self.plan_id,
                    requirement=requirement,
                    tasks=[dict(t) for t in plan_tasks],
                    results=all_results,
                    quality_score=overall_quality,
                    duration_ms=duration_ms,
                    agents_used=agents_used,
                    max_depth=1,
                    healing_attempts=healing_attempts,
                    passed=passed,
                )
                logging.getLogger(__name__).info(
                    f"[orchestrator] episode saved: job={self.plan_id} quality={overall_quality:.1f}"
                )
            except Exception as e:
                logging.getLogger(__name__).warning(f"[orchestrator] episode save failed: {e}")

        await self._emit(progress_callback, {
            "type": "final_result",
            "summary": final_result["summary"],
            "quality_score": overall_quality,
            "passed": passed,
            "loop_count": self.loop_count,
            "healing_attempts": healing_attempts,
            # 前端产出文件夹 UI 依赖以下字段（须与 _build_final_result 返回一致）
            "plan_id": final_result.get("plan_id", self.plan_id),
            "output_dir": final_result.get("output_dir", ""),
            "files": final_result.get("files", []),
            "summary_file": final_result.get("summary_file"),
        })

        return final_result

    async def _create_plan(self, requirement: str, enabled_agents: list[str]) -> list[TaskItem]:
        """
        Supervisor Planning Node -- 分析需求，生成 JSON Plan。

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
                f"总协调者: agent_main（{cp.get('name', 'Chief Coordinator')}）-- "
                "由系统指定为最高管理者；**禁止**在 JSON 的 assigned_agent 中使用 agent_main。"
                "子任务仅能分配给下列执行型 Agent id。"
            )

        from src.graph.prompt import build_supervisor_prompt

        # ── P2-A: 查询相似历史经验 ──────────────────────────────────
        experience_hint = ""
        if ENABLE_EPISODIC_MEMORY:
            try:
                from src.memory.episode_store import get_episode_store
                store = get_episode_store()
                similar = store.query_similar(requirement, top_k=2)
                if similar:
                    lines = ["【历史经验参考】:"]
                    for e in similar:
                        score = e.get("quality_score", 0)
                        req = e.get("requirement", "")[:60]
                        agents = e.get("agents_used", [])
                        tasks_raw = e.get("tasks", [])
                        task_kinds = [t.get("assigned_agent") or t.get("worker_kind", "?") for t in tasks_raw]
                        lines.append(
                            f"- 质量 {score:.1f}: \"{req}...\"\n  -> Agent组合: {agents or task_kinds}"
                        )
                    experience_hint = "\n".join(lines)
                    logger.info(f"[orchestrator] found {len(similar)} similar episodes for requirement: {requirement[:50]}")
            except Exception as e:
                logger.warning(f"[orchestrator] episode query failed: {e}")

        supervisor_prompt = build_supervisor_prompt(
            requirement=requirement,
            agents_desc=agents_desc,
            chief_note=chief_note,
            max_tasks=5,
            experience_hint=experience_hint,
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
                    wk = t.get("worker_kind")  # 优先使用 LLM 直接输出的 worker_kind
                    norm_id = agent_id

                    if not wk:
                        # LLM 未指定 worker_kind，走原有推断逻辑
                        norm_id = _short_map.get(agent_id, agent_id)
                        p = _all_participants.get(agent_id) or _all_participants.get(norm_id)
                        if p:
                            wk = p.get("worker_kind")
                        else:
                            wk = _short_map.get(agent_id)
                            if wk:
                                norm_id = wk
                            p = _all_participants.get(wk) if wk else None
                    # 验证：wk 必须属于 enabled_kinds（排除 LLM 发明的不匹配 agent）
                    if wk and wk not in enabled_kinds:
                        if wk in builtin_kinds:
                            logger.warning("[orchestrator] plan assigned unknown builtin %s (%s), skipping", agent_id, wk)
                            continue
                    depends_on = t.get("depends_on") or []
                    execution_mode = t.get("execution_mode") or "parallel"
                    output_type = t.get("output_type") or "search_only"
                    suggested_filename = t.get("suggested_filename") or ""
                    result.append(TaskItem(
                        task_id,
                        t.get("description", ""),
                        norm_id,
                        wk,
                        depends_on=depends_on,
                        execution_mode=execution_mode,
                        output_type=output_type,
                        suggested_filename=suggested_filename,
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
                output_type = "markdown" if any(k in req_lower for k in ["报告", "调研", "文档", "分析"]) else "search_only"
                tasks.append(TaskItem(
                    f"task_{uuid.uuid4().hex[:6]}",
                    requirement,
                    "agent_worker_search",
                    "search_worker",
                    output_type=output_type,
                    suggested_filename="research_report.md" if output_type == "markdown" else "",
                ))

        if "rag_worker" in enabled_kinds:
            rag_keywords = ["我的", "项目", "配置", "之前", "记忆", "历史", "智程"]
            if any(k in req_lower for k in rag_keywords):
                tasks.append(TaskItem(
                    f"task_{uuid.uuid4().hex[:6]}",
                    requirement,
                    "agent_worker_rag",
                    "rag_worker",
                    output_type="markdown",
                    suggested_filename="knowledge_summary.md",
                ))

        if "coder" in enabled_kinds:
            code_keywords = ["代码", "生成", "写", "实现", "函数", "class", "def "]
            if any(k in req_lower for k in code_keywords):
                ext = ".py"
                if "html" in req_lower or "网页" in req_lower:
                    ext = ".html"
                elif "javascript" in req_lower or "js" in req_lower:
                    ext = ".js"
                output_type = "code"
                tasks.append(TaskItem(
                    f"task_{uuid.uuid4().hex[:6]}",
                    requirement,
                    "agent_worker_coder",
                    "coder",
                    output_type=output_type,
                    suggested_filename=f"generated_code{ext}",
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

    @traceable(name="worker-execute", tags=["worker", "execute"])
    async def _execute_single_task(
        self,
        task: TaskItem,
        context: list[dict] | None = None,
        progress_callback: Callable[[dict], Awaitable[None]] | None = None,
        output_mgr=None,
    ) -> dict:
        """执行单个 Worker 任务。路由依据 worker_kind 字段，支持 context 和文件输出。"""
        from src.graph.workers import SearchWorker, RAGWorker, CoderWorker

        task_id = task["task_id"]
        agent_id = task["assigned_agent"]
        wk = task.get("worker_kind") or ""

        # 兜底：当 LLM 生成的 assigned_agent 不存在于 registry 但任务类型明确时，
        # 通过 registry 的 worker_kind 查找来推断正确的 worker_kind 并直接路由到内置 Worker。
        if not wk:
            from src.multi_agent.orchestrator import get_agent_registry
            reg = get_agent_registry()
            _short_map = {
                "agent_worker_search": "search_worker",
                "agent_worker_rag":    "rag_worker",
                "agent_worker_coder": "coder",
            }
            if agent_id in _short_map:
                wk = _short_map[agent_id]
                logger.info("[orchestrator] task %s: inferred worker_kind=%s from short_map (agent_id=%s)", task_id, wk, agent_id)
            else:
                profile = reg.get(agent_id)
                if profile and profile.get("worker_kind"):
                    wk = profile["worker_kind"]
                    logger.info("[orchestrator] task %s: inferred worker_kind=%s from registry (agent_id=%s)", task_id, wk, agent_id)
                else:
                    p = self._participants.get(agent_id)
                    if p and p.get("worker_kind"):
                        wk = p["worker_kind"]
                        logger.info("[orchestrator] task %s: inferred worker_kind=%s from participants (agent_id=%s)", task_id, wk, agent_id)
                    else:
                        # LLM 发明了不存在的 agent_id（如 UUID 串），
                        # 通过任务描述关键词推断 worker_kind
                        desc_lower = task.get("description", "").lower()
                        if any(k in desc_lower for k in ["搜索", "search", "查找", "调研", "web", "最新", "2025", "2026"]):
                            wk = "search_worker"
                        elif any(k in desc_lower for k in ["记忆", "rag", "知识库", "历史", "项目", "之前", "检索"]):
                            wk = "rag_worker"
                        elif any(k in desc_lower for k in ["代码", "code", "生成", "实现", "写", "python", "html", "javascript"]):
                            wk = "coder"
                        logger.warning(
                            "[orchestrator] task %s: invented agent_id=%s not in registry/participants, "
                            "inferred worker_kind=%s from description keywords",
                            task_id, agent_id, wk
                        )

        await self._emit(progress_callback, {
            "type": "worker_start",
            "agent": agent_id,
            "task_id": task_id,
            "task_description": task["description"],
        })

        try:
            if wk == "search_worker":
                worker = SearchWorker()
                result = await worker.execute(task, progress_callback, output_manager=output_mgr)
            elif wk == "rag_worker":
                worker = RAGWorker()
                result = await worker.execute(task, progress_callback, output_manager=output_mgr)
            elif wk == "coder":
                worker = CoderWorker()
                result = await worker.execute(task, context, progress_callback, output_manager=output_mgr)
            else:
                result = await self._execute_generic_task(task, context, progress_callback, output_mgr=output_mgr)

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
                "files": [],
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
        output_mgr=None,
    ) -> dict:
        """
        通用 Agent 执行器 -- 处理所有非内置 worker_kind。

        替代原 _dispatch_generic()，新增：
        - 实时 SSE worker_progress 推送（初始化 -> 分发 -> 完成）
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

            logger.info(
                "[orchestrator] generic task %s dispatch: status=%s, result_len=%d",
                task_id,
                dispatch_result.get("status"),
                len(dispatch_result.get("result", "")),
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
                error_msg = dispatch_result.get("message", "未知错误")
                quality_score = self._evaluate_generic_quality(error_msg, task["description"])

                await self._emit(progress_callback, {
                    "type": "worker_progress",
                    "agent": agent_id,
                    "task_id": task_id,
                    "message": f"[{agent_name}] 执行失败: {error_msg[:80]}",
                })

                return {
                    "task_id": task_id,
                    "agent": agent_id,
                    "result": f"[调度失败] {error_msg}",
                    "confidence": 0.0,
                    "quality_score": quality_score,
                    "source": "dispatch_error",
                    "raw_data": error_msg,
                }

        except Exception as e:
            logger.error(f"[orchestrator] generic task {task_id} failed: {e}")
            quality_score = self._evaluate_generic_quality(str(e), task["description"])
            return {
                "task_id": task_id,
                "agent": agent_id,
                "result": f"[执行异常] {str(e)}",
                "confidence": 0.0,
                "quality_score": quality_score,
                "source": "generic_error",
                "raw_data": str(e),
            }

    def _evaluate_generic_quality(self, result: str, original_task: str) -> float:
        """
        评估通用 Agent 结果质量（0-10 分）。
        策略：结果长度 + 结构化程度 + 关键词覆盖 + 错误检测。
        """
        if not result:
            return 2.0
        if len(result) < 10:
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

    # ══════════════════════════════════════════════════════════════════════════════
    #  P2-D 多级编排（Hierarchical Supervisor）
    # ══════════════════════════════════════════════════════════════════════════════

    MAX_HIERARCHY_DEPTH = 3

    def _is_complex_task(self, task: TaskItem) -> bool:
        """
        启发式判断：任务是否足够复杂，需要递归拆分。

        判断依据：
        - 任务描述包含复合关键词（多个、综合、完整、全面、端到端等）
        - 描述长度超过 300 字符
        - 标记为顺序执行模式
        """
        desc = task.get("description", "")
        complex_markers = [
            "多个", "综合", "完整", "全面", "端到端",
            "以及", "同时", "既", "又", "不但", "而且",
            "从头到尾", "整个系统", "一整套",
        ]
        return (
            any(m in desc for m in complex_markers)
            or len(desc) > 300
            or task.get("execution_mode") == "sequential"
        )

    async def _orchestrate_recursive(
        self,
        requirement: str,
        enabled_agents: list[str],
        participants: dict | None = None,
        depth: int = 0,
        progress_callback: Callable[[dict], Awaitable[None]] | None = None,
        initial_plan: list[dict] | None = None,
        root_output_mgr=None,
    ) -> dict:
        """
        递归编排入口（支持多层级树形执行）。

        depth=0 时，initial_plan 由 orchestrate() 直接传入（避免顶层重复规划两次）。
        root_output_mgr 由顶层 orchestrate() 创建并传入，所有递归层级和 worker 共用，
        确保所有文件写入 outputs/<plan_id>/ 而非各层级的 outputs/<plan_id>_L<depth>_xxx/。
        """

        if participants is None:
            participants = self._participants

        # 层级标识（仅用于日志/事件名，不参与目录路径）
        hierarchy_label = f"{self.plan_id}_L{depth}"
        sub_plan_id = f"{hierarchy_label}_{uuid.uuid4().hex[:6]}"

        # output_mgr 由顶层传入（统一目录），各层级复用；仅在顶层创建新实例
        if root_output_mgr is not None:
            output_mgr = root_output_mgr
        else:
            output_mgr = get_output_manager(sub_plan_id)
            output_mgr.ensure_dir()

        await self._emit(progress_callback, {
            "type": "hierarchy_start",
            "depth": depth,
            "plan_id": sub_plan_id,
            "message": f"[层级 {depth}] 正在规划: {requirement[:60]}...",
        })

        # Supervisor 规划（depth=0 且已有 initial_plan 时直接复用，避免重复调用 LLM）
        if depth == 0 and initial_plan is not None:
            plan = initial_plan
        else:
            plan = await self._create_plan(requirement, enabled_agents)

        _dlog("d85885", "debug-run", "B1-plan",
              "orchestrator.py:_orchestrate_recursive",
              f"[L{depth}] plan created",
              {"depth": depth, "n_tasks": len(plan), "task_ids": [t.get("task_id") for t in plan], "source": "initial_plan" if (depth == 0 and initial_plan is not None) else "create_plan"})

        await self._emit(progress_callback, {
            "type": "supervisor_plan",
            "depth": depth,
            "plan_id": sub_plan_id,
            "data": {
                "plan_id": sub_plan_id,
                "tasks": [dict(t) for t in plan],
            },
        })

        if not plan:
            return {
                "task_id": sub_plan_id,
                "agent": "supervisor",
                "result": f"[层级 {depth}] 无法规划任务: {requirement[:100]}",
                "quality_score": 0.0,
                "source": "hierarchy_recursive",
                "depth": depth,
            }

        # 构建 DAG 调度（复用现有 AgentMessageBus）
        bus = AgentMessageBus.get_instance()
        task_map: dict[str, TaskItem] = {t["task_id"]: t for t in plan}

        # 注册依赖关系
        for t in plan:
            for dep in t.get("depends_on") or []:
                if dep in task_map:
                    bus.add_dependency(dep, t["task_id"])

        all_results: list[dict] = []
        completed: set[str] = set()

        while len(completed) < len(task_map):
            # 找所有入度为0的就绪任务
            ready = [
                tid for tid, deg in bus._in_degree.items()
                if tid not in completed and deg == 0
            ]
            if not ready:
                ready = [tid for tid in task_map if tid not in completed]
            if not ready:
                break

            await self._emit(progress_callback, {
                "type": "worker_start",
                "agent": "supervisor",
                "depth": depth,
                "task_id": "parallel_batch",
                "task_description": f"[层级 {depth}] 并行执行 {len(ready)} 个任务: {ready}",
            })

            _dlog("d85885", "debug-run", "B1a",
                "orchestrator.py:ready-check",
                f"[L{depth}] ready tasks before gather",
                {"ready": ready, "task_map_keys": list(task_map.keys()), "completed": list(completed)})

            coroutines = []
            for tid in ready:
                task = task_map[tid]
                context = bus.get_context(tid)

                # 递归判断
                if depth < self.MAX_HIERARCHY_DEPTH and self._is_complex_task(task):
                    # 递归拆分
                    async def _recursive_sub(task_item: TaskItem):
                        await self._emit(progress_callback, {
                            "type": "hierarchy_expand",
                            "depth": depth,
                            "task_id": task_item["task_id"],
                            "message": f"[层级 {depth}] 展开复杂任务: {task_item['description'][:50]}...",
                        })
                        return await self._orchestrate_recursive(
                            requirement=task_item["description"],
                            enabled_agents=enabled_agents,
                            participants=participants,
                            depth=depth + 1,
                            progress_callback=progress_callback,
                            root_output_mgr=output_mgr,
                        )
                    coroutines.append(_recursive_sub(task))
                else:
                    # 直接执行
                    coroutines.append(
                        self._execute_single_task(task, context, progress_callback, output_mgr)
                    )

            batch_results = await asyncio.gather(*coroutines, return_exceptions=True)

            _dlog("d85885", "debug-run", "B1b",
                "orchestrator.py:after-gather",
                f"[L{depth}] batch_results after gather",
                {"ready": ready, "result_types": [type(r).__name__ for r in batch_results]})

            for idx, result in enumerate(batch_results):
                if idx >= len(ready):
                    _dlog("d85885", "debug-run", "B1-idx",
                          "orchestrator.py:extra-result",
                          f"[L{depth}] extra result at idx (ready/result mismatch)",
                          {"idx": idx, "ready_len": len(ready), "result_type": type(result).__name__})
                    continue
                tid = ready[idx]
                if isinstance(result, Exception):
                    _dlog("d85885", "debug-run", "B1c",
                        "orchestrator.py:exception-caught",
                        f"[L{depth}] task exception",
                        {"tid": tid, "exc_type": type(result).__name__, "exc": str(result),
                         "task_map_keys": list(task_map.keys()), "in_task_map": tid in task_map})
                    _agent = task_map[tid]["assigned_agent"] if tid in task_map else "unknown"
                    result = {
                        "task_id": tid,
                        "agent": _agent,
                        "result": f"[执行失败] {str(result)}",
                        "quality_score": 0.0,
                        "source": "error",
                    }
                elif tid not in task_map:
                    _dlog("d85885", "debug-run", "B1-unknown",
                          "orchestrator.py:unknown-tid",
                          f"[L{depth}] tid not in task_map",
                          {"tid": tid})
                    continue
                bus.store_result(tid, result)
                all_results.append(result)
                completed.add(tid)
                bus.mark_done(tid)

        # 聚合结果
        overall_quality = self._evaluate_overall_quality(all_results)
        passed = overall_quality >= 8.0

        await self._emit(progress_callback, {
            "type": "hierarchy_done",
            "depth": depth,
            "plan_id": sub_plan_id,
            "quality_score": overall_quality,
            "passed": passed,
            "task_count": len(all_results),
        })

        return self._aggregate_sub_results(all_results, sub_plan_id, overall_quality, depth, output_mgr=output_mgr, requirement=requirement)

    def _aggregate_sub_results(
        self,
        results: list[dict],
        plan_id: str,
        overall_quality: float,
        depth: int = 0,
        output_mgr=None,
        requirement: str = "",
    ) -> dict:
        """聚合多层级执行结果，并附加 output_dir / files（由 output_mgr 提供）。"""
        all_text_parts = []
        all_files = []
        for r in results:
            txt = r.get("result", "")
            if isinstance(txt, str):
                all_text_parts.append(txt)
            files = r.get("files", [])
            if files:
                all_files.extend(files)

        # 构建 summary（与 _build_final_result 保持一致，供 SSE final_result 事件使用）
        passed = overall_quality >= 8.0
        summary_parts = [
            f"=== 多层级编排汇总报告 ===",
            f"计划ID: {plan_id}",
            f"执行时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"质量评分: {overall_quality:.1f}/10",
            f"质量状态: {'通过' if passed else '未达标'}",
            "",
            "--- 各层级结果 ---",
        ]
        for r in results:
            summary_parts.append(f"\n[{r.get('agent','supervisor')}] 任务: {r.get('task_id','')} 质量: {r.get('quality_score',0.0):.1f}")
            txt = r.get("result", "")
            if txt:
                summary_parts.append(f"内容摘要: {txt[:300]}...")
        if all_text_parts:
            summary_parts.append("")
            summary_parts.append("--- 完整文本 ---")
            summary_parts.extend(all_text_parts)

        output_dir = ""
        summary_file_info = None
        if output_mgr is not None:
            output_dir = str(output_mgr.output_dir)
            summary_file_info = output_mgr.generate_summary_md(results, requirement)

        ret = {
            "task_id": plan_id,
            "agent": "supervisor",
            "summary": "\n".join(summary_parts),
            "result": "\n\n".join(all_text_parts),
            "quality_score": overall_quality,
            "passed": passed,
            "source": "hierarchy_recursive",
            "depth": depth,
            "output_dir": output_dir,
            "files": all_files,
            "sub_results": results,
        }
        if summary_file_info:
            ret["summary_file"] = summary_file_info.to_dict()
        return ret

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
        output_mgr=None,
        requirement: str = "",
    ) -> dict:
        """
        构建最终汇总报告。

        关键改进：
        - 不再截断 raw_data，保留完整内容（或引用文件路径）
        - 通过 output_mgr 生成 SUMMARY.md
        - 在返回结果中包含 output_dir 和 files 列表
        """
        lines = [
            "=== Crayfish 多 Agent 编排汇总报告 ===",
            f"计划ID: {self.plan_id}",
            f"执行时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"质量评分: {quality_score:.1f}/10",
            f"质量状态: {'通过' if passed else '未达标'}",
            f"循环次数: {self.loop_count}",
            "",
        ]

        # ── 收集所有文件信息 ───────────────────────────────────────────────
        all_files: list[dict] = []
        for result in results:
            files = result.get("files", [])
            if files:
                all_files.extend(files)

        # ── 汇总产出文件 ─────────────────────────────────────────────────
        if all_files:
            lines.append("--- 产出文件 ---")
            for f in all_files:
                fname = f.get("filename", "?")
                ftype = f.get("file_type", "?")
                agent = f.get("agent_id", "?")
                lines.append(f"  - [{ftype}] {fname} (by {agent})")
            lines.append("")

        # ── 各 Worker 结果摘要（不再截断 raw_data）────────────────────────
        lines.append("--- 各 Worker 结果 ---")
        for result in results:
            agent = result.get("agent", "unknown")
            score = result.get("quality_score", 0.0)
            task_id = result.get("task_id", "")
            files = result.get("files", [])

            lines.append(f"\n[{agent}] 任务: {task_id} 质量: {score:.1f}")

            # 如果有文件输出，优先引用文件路径
            if files:
                for f in files:
                    lines.append(f"  文件: {f.get('filename', '?')} ({f.get('file_type', '?')})")
                    lines.append(f"  路径: {f.get('file_path', '?')}")
            else:
                # 无文件时，显示 raw_data 完整内容（不截断）
                raw = result.get("raw_data", result.get("result", ""))
                # 但为防止 summary 本身过长，对 raw_data 做一个宽松截断（10k字符）
                if len(raw) > 10240:
                    raw = raw[:10240] + "\n\n[内容过长，已截断至10KB，完整内容见产出文件]"
                lines.append(f"内容: {raw}")

        # ── 冲突检测 ──────────────────────────────────────────────────────
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

        # ── 生成 SUMMARY.md ───────────────────────────────────────────────
        output_dir = ""
        files_info = all_files
        summary_file = None

        if output_mgr is not None:
            output_dir = str(output_mgr.output_dir)
            # 生成 SUMMARY.md
            summary_file = output_mgr.generate_summary_md(results, requirement)
            if summary_file:
                files_info = output_mgr.list_files()

        result_dict: dict = {
            "summary": summary,
            "quality_score": quality_score,
            "passed": passed,
            "loop_count": self.loop_count,
            "results": results,
            "plan_id": self.plan_id,
            "output_dir": output_dir,
            "files": files_info,
        }

        if summary_file:
            result_dict["summary_file"] = summary_file.to_dict()

        return result_dict

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
