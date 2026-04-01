"""编排后台任务管理器 — 支持任务在后台运行，前端实时订阅 SSE 事件流."""

import asyncio
import logging
import os
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from src.graph.orchestrator import CrayfishOrchestrator


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    CANCELLED = "cancelled"
    FAILED   = "failed"


@dataclass
class BackgroundJob:
    """单个编排任务的后台状态."""
    job_id:          str
    requirement:     str
    enabled_agents: list[str]
    quality_threshold: float
    created_at:     str
    status:          JobStatus = JobStatus.PENDING
    result:          Optional[dict] = None          # 最终结果
    error_message:   Optional[str] = None
    # Agent 档案（由 enabled_agents 从 AgentRegistry 加载，用于 orchestrator 路由）
    participants:    list[dict] = field(default_factory=list)
    # 关联的任务协调看板行（SQLite task_jobs）
    kanban_task_id: Optional[str] = None

    # SSE 事件缓冲（双端队列，append 时追加）
    events:          deque = field(default_factory=deque)
    # SSE 流读者引用（用于取消）
    stream_reader:    Optional[object] = field(default=None, repr=False)
    # 取消标志（asyncio.Event，set() 即取消）
    cancel_event:    asyncio.Event = field(default_factory=asyncio.Event)

    # 实时进度快照（由前端轮询或 SSE 推送使用；动态 key = agent_id）
    worker_state: dict = field(default_factory=lambda: {})

    def add_event(self, ev_type: str, data: dict) -> None:
        """追加事件到缓冲."""
        self.events.append({"type": ev_type, "data": data, "ts": datetime.now().isoformat()})

    def set_result(self, result: dict) -> None:
        self.result = result
        self.status = JobStatus.DONE

    def set_failed(self, message: str) -> None:
        self.error_message = message
        self.status = JobStatus.FAILED

    def set_cancelled(self) -> None:
        self.cancel_event.set()
        self.status = JobStatus.CANCELLED

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "requirement": self.requirement,
            "enabled_agents": self.enabled_agents,
            "quality_threshold": self.quality_threshold,
            "created_at": self.created_at,
            "status": self.status.value,
            "result": self.result,
            "error_message": self.error_message,
            "participants": self.participants,
            "kanban_task_id": self.kanban_task_id,
            "worker_state": self.worker_state,
        }


class BackgroundJobManager:
    """全局后台任务管理器（进程内单例）."""

    def __init__(self) -> None:
        self._jobs: dict[str, BackgroundJob] = {}

    def create_job(
        self,
        requirement: str,
        enabled_agents: list[str],
        quality_threshold: float,
        participants: list[dict] | None = None,
        kanban_task_id: str | None = None,
    ) -> BackgroundJob:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = BackgroundJob(
            job_id=job_id,
            requirement=requirement,
            enabled_agents=enabled_agents,
            quality_threshold=quality_threshold,
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            participants=participants or [],
            kanban_task_id=kanban_task_id,
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[BackgroundJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[BackgroundJob]:
        return list(self._jobs.values())

    def remove_job(self, job_id: str) -> bool:
        return self._jobs.pop(job_id, None) is not None

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status in (JobStatus.DONE, JobStatus.CANCELLED, JobStatus.FAILED):
            return False
        job.set_cancelled()
        return True


# ── 全局单例 ─────────────────────────────────────────────────────────────────
_job_manager: Optional[BackgroundJobManager] = None


def get_job_manager() -> BackgroundJobManager:
    global _job_manager
    if _job_manager is None:
        _job_manager = BackgroundJobManager()
    return _job_manager


# ── 后台任务运行器 ───────────────────────────────────────────────────────────


def _sync_orchestrate_wrapper(orchestrator, job) -> None:
    """
    Synchronous wrapper that runs the async orchestrate() coroutine in a thread.
    This lets asyncio.run() manage the event loop for this thread.
    """
    import asyncio as _asyncio

    async def _async_inner():
        _last_kanban_update: float = 0.0

        async def _sync_kanban(status: str, result: str | None = None, error: str | None = None) -> None:
            nonlocal _last_kanban_update
            if not job.kanban_task_id:
                return
            import time as _time
            now = _time.monotonic()
            if now - _last_kanban_update < 1.0 and status not in ("done", "failed", "cancelled"):
                return
            _last_kanban_update = now
            from src.server.task_scheduler import get_scheduler
            scheduler = get_scheduler()
            scheduler.update(
                job.kanban_task_id,
                status=status,
                result=result,
                error=error,
            )

        async def progress_callback(event: dict) -> None:
            if job.cancel_event.is_set():
                raise _asyncio.CancelledError("Job cancelled by user")
            ev_type = event.get("type", "message")
            job.add_event(ev_type, event)
            _update_worker_state(job, ev_type, event)
            # CRITICAL: update BackgroundJob object state so SSE endpoint can read it
            if ev_type == "final_result":
                job.result = {
                    "summary":          event.get("summary", ""),
                    "quality_score":    event.get("quality_score", 0),
                    "passed":           event.get("passed", False),
                    "loop_count":       event.get("loop_count", 0),
                    "healing_attempts": event.get("healing_attempts", 0),
                    "plan_id":          event.get("plan_id", ""),
                    "output_dir":       event.get("output_dir", ""),
                    "files":            event.get("files", []),
                    "summary_file":     event.get("summary_file"),
                }
                # B6: log final_result to debug file
                try:
                    import json as _json, datetime as _dt
                    _LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "debug-d85885.log")
                    _entry = {
                        "sessionId": "d85885", "runId": "debug-run",
                        "hypothesisId": "B6-files",
                        "location": "orch_jobs.py:progress_callback",
                        "message": "B6: final_result received",
                        "data": {
                            "output_dir": event.get("output_dir", ""),
                            "nFiles": len(event.get("files") or []),
                            "fileNames": [f.get("filename","?") for f in (event.get("files") or [])],
                            "plan_id": event.get("plan_id", ""),
                        },
                        "timestamp": int(_dt.datetime.now().timestamp() * 1000),
                    }
                    with open(_LOG_PATH, "a", encoding="utf-8") as _f:
                        _f.write(_json.dumps(_entry, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            if job.status != JobStatus.RUNNING:
                job.status = JobStatus.RUNNING
            await _sync_kanban("running", result=_format_progress(job.worker_state))

        try:
            result = await orchestrator.orchestrate(
                requirement=job.requirement,
                enabled_agents=job.enabled_agents,
                quality_threshold=job.quality_threshold,
                progress_callback=progress_callback,
                participants=job.participants,
            )
            if not job.cancel_event.is_set():
                job.result = result or {}
                job.status = JobStatus.DONE
        except _asyncio.CancelledError:
            job.set_cancelled()
        except Exception as e:
            tb = traceback.format_exc()
            job.set_failed(str(e))
            job.add_event("error", {
                "type": "error",
                "message": f"[编排失败] {e}",
            })
            # Write full traceback to debug log
            _LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "debug-d85885.log")
            try:
                import json as _json
                import datetime as _dt
                entry = {
                    "sessionId": "d85885",
                    "runId": "debug-run",
                    "hypothesisId": "B1-orchjobs",
                    "location": "orch_jobs.py:_sync_orchestrate_wrapper",
                    "message": "[编排失败] exception traceback",
                    "data": {"error_type": type(e).__name__, "error": str(e), "traceback": tb},
                    "timestamp": int(_dt.datetime.now().timestamp() * 1000),
                }
                with open(_LOG_PATH, "a", encoding="utf-8") as _f:
                    _f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

    try:
        _asyncio.run(_async_inner())
    except Exception as _e:
        import traceback as _tb
        logger.error("orch_jobs:_sync_orchestrate_wrapper exception: %s", _e)
        try:
            job.set_failed(str(_e))
        except Exception:
            pass

    # (exit implicit on return)


async def _run_orchestrator_job(job: BackgroundJob) -> None:
    """
    启动编排任务 — 委托给 _sync_orchestrate_wrapper 在线程池中执行。

    _sync_orchestrate_wrapper 在其内部的 asyncio.run() 中：
      1. 运行 orchestrator.orchestrate()
      2. 通过 progress_callback 更新 job.result / job.status（供 SSE 读取）
      3. 通过 _sync_kanban 更新 SQLite 看板
    """
    orchestrator = CrayfishOrchestrator()

    try:
        # 使用 asyncio.to_thread 运行同步包装器
        # 注意：这是 Python 的默认线程池，不是 llm_pool，
        # 但它只是承载 asyncio.run()，LLM 调用仍在 orchestrator 内部进行。
        await asyncio.to_thread(_sync_orchestrate_wrapper, orchestrator, job)
    except Exception as e:
        logger.error("orch_jobs:_run_orchestrator_job failed: %s", e)
        try:
            job.set_failed(str(e))
        except Exception:
            pass


def _update_worker_state(job: BackgroundJob, ev_type: str, event: dict) -> None:
    """根据事件类型动态更新 worker_state（key = agent_id）。"""
    ws = job.worker_state

    if ev_type == "supervisor_plan":
        # plan 生成后，初始化所有 participant 的状态
        ws.clear()
        for p in job.participants:
            ws[p["id"]] = {"status": "pending", "quality": 0, "name": p.get("name", p["id"])}
        # 对内置 worker（没有 participant 但已在 enabled_agents）也初始化
        for aid in job.enabled_agents:
            if aid not in ws:
                ws[aid] = {"status": "pending", "quality": 0, "name": aid}

    elif ev_type == "worker_start":
        agent = event.get("agent", "")
        if agent not in ws:
            ws[agent] = {"status": "pending", "quality": 0}
        ws[agent]["status"] = "running"

    elif ev_type == "worker_done":
        agent = event.get("agent", "")
        if agent not in ws:
            ws[agent] = {"status": "pending", "quality": 0}
        ws[agent]["status"] = "done"
        ws[agent]["quality"] = float(event.get("quality_score", 0))

    elif ev_type == "worker_rejected":
        agent = event.get("agent", "")
        if agent not in ws:
            ws[agent] = {"status": "pending", "quality": 0}
        ws[agent]["status"] = "rejected"

    elif ev_type == "self_healing":
        agent = event.get("agent", "")
        if agent in ws:
            ws[agent]["status"] = "healing"

    elif ev_type == "final_result":
        job.result = {
            "summary":          event.get("summary", ""),
            "quality_score":    event.get("quality_score", 0),
            "passed":           event.get("passed", False),
            "loop_count":       event.get("loop_count", 0),
            "healing_attempts": event.get("healing_attempts", 0),
            "plan_id":          event.get("plan_id", ""),
            "output_dir":       event.get("output_dir", ""),
            "files":            event.get("files", []),
            "summary_file":     event.get("summary_file"),
        }


def _format_progress(worker_state: dict) -> str:
    """从 worker_state 格式化进度摘要，用于看板 result 字段。"""
    if not worker_state:
        return "执行中..."
    lines = []
    for aid, state in worker_state.items():
        name = state.get("name", aid)
        status = state.get("status", "pending")
        quality = state.get("quality", 0)
        status_emoji = {
            "pending": "⏳", "running": "🔄",
            "done": "✅", "rejected": "❌", "healing": "🔧",
        }.get(status, "❓")
        lines.append(f"{status_emoji} {name}: {status}" + (f" ({quality:.1f})" if quality else ""))
    return "\n".join(lines) if lines else "执行中..."


def _format_result(result: dict | None) -> str:
    """从 final_result dict 格式化摘要。"""
    if not result:
        return ""
    score = result.get("quality_score", 0)
    passed = result.get("passed", False)
    summary = result.get("summary", "")
    if summary:
        return summary[:500] + ("..." if len(summary) > 500 else "")
    return f"质量评分: {score:.1f}/10 {'通过' if passed else '未达标'}"


def _agent_to_key(agent: str) -> str:
    if agent in ("search_worker", "search"):   return "search"
    if agent in ("rag_worker",    "rag"):      return "rag"
    if agent == "coder":                       return "coder"
    return "system"
