"""编排后台任务管理器 — 支持任务在后台运行，前端实时订阅 SSE 事件流."""

import asyncio
import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


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

    # SSE 事件缓冲（双端队列，append 时追加）
    events:          deque = field(default_factory=deque)
    # SSE 流读者引用（用于取消）
    stream_reader:    Optional[object] = field(default=None, repr=False)
    # 取消标志（asyncio.Event，set() 即取消）
    cancel_event:    asyncio.Event = field(default_factory=asyncio.Event)

    # 实时进度快照（由前端轮询或 SSE 推送使用）
    worker_state: dict = field(default_factory=lambda: {
        "search": {"status": "pending", "quality": 0},
        "rag":    {"status": "pending", "quality": 0},
        "coder":  {"status": "pending", "quality": 0},
    })

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
    ) -> BackgroundJob:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = BackgroundJob(
            job_id=job_id,
            requirement=requirement,
            enabled_agents=enabled_agents,
            quality_threshold=quality_threshold,
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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

async def _run_orchestrator_job(job: BackgroundJob) -> None:
    """
    在后台 asyncio.Task 中运行编排器。
    运行完成后将最终结果写入 job.result，并在每个事件上更新 worker_state。
    """
    from src.graph.orchestrator import CrayfishOrchestrator

    try:
        job.status = JobStatus.RUNNING

        async def progress_callback(event: dict) -> None:
            # 检查取消标志
            if job.cancel_event.is_set():
                raise asyncio.CancelledError("Job cancelled by user")

            ev_type = event.get("type", "message")
            job.add_event(ev_type, event)

            # 更新 worker_state 快照
            _update_worker_state(job, ev_type, event)

        # 每个 job 用独立 orchestrator 实例（避免并发状态污染）
        orchestrator = CrayfishOrchestrator()

        await orchestrator.orchestrate(
            requirement=job.requirement,
            enabled_agents=job.enabled_agents,
            quality_threshold=job.quality_threshold,
            progress_callback=progress_callback,
        )

        # 如果未被取消，标记完成
        if not job.cancel_event.is_set():
            job.status = JobStatus.DONE

    except asyncio.CancelledError:
        job.set_cancelled()
        logger.info(f"[orch_jobs] job {job.job_id} cancelled")

    except Exception as e:
        job.set_failed(str(e))
        job.add_event("error", {
            "type": "error",
            "message": f"[编排失败] {e}",
        })
        logger.error(f"[orch_jobs] job {job.job_id} failed: {e}", exc_info=True)


def _update_worker_state(job: BackgroundJob, ev_type: str, event: dict) -> None:
    """根据事件类型更新 worker_state 快照."""
    ws = job.worker_state

    if ev_type == "worker_start":
        agent = event.get("agent", "")
        key = _agent_to_key(agent)
        if key in ws:
            ws[key]["status"] = "running"

    elif ev_type == "worker_done":
        agent = event.get("agent", "")
        key = _agent_to_key(agent)
        if key in ws:
            ws[key]["status"] = "done"
            ws[key]["quality"] = float(event.get("quality_score", 0))

    elif ev_type == "worker_rejected":
        agent = event.get("agent", "")
        key = _agent_to_key(agent)
        if key in ws:
            ws[key]["status"] = "rejected"

    elif ev_type == "self_healing":
        agent = event.get("agent", "")
        key = _agent_to_key(agent)
        if key in ws:
            ws[key]["status"] = "healing"

    elif ev_type == "supervisor_plan":
        # 所有节点先回到 pending
        for k in ws:
            ws[k]["status"] = "pending"
            ws[k]["quality"] = 0

    elif ev_type == "final_result":
        job.result = {
            "summary":          event.get("summary", ""),
            "quality_score":    event.get("quality_score", 0),
            "passed":           event.get("passed", False),
            "loop_count":       event.get("loop_count", 0),
            "healing_attempts": event.get("healing_attempts", 0),
        }


def _agent_to_key(agent: str) -> str:
    if agent in ("search_worker", "search"):   return "search"
    if agent in ("rag_worker",    "rag"):      return "rag"
    if agent == "coder":                       return "coder"
    return "system"
