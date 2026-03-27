"""任务调度器 — SQLite 持久化 + asyncio 异步执行."""

import asyncio
import json
import logging
import os
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.config import CHECKPOINT_PATH

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
TASKS_DB = DATA_DIR / "task_scheduler.db"


def _uuid() -> str:
    return "task_" + uuid.uuid4().hex[:12]


# ═══════════════════════════════════════════════════════════════════════════════
#  SQLite 连接管理
# ═══════════════════════════════════════════════════════════════════════════════

_scheduler_lock = threading.Lock()


@contextmanager
def _conn():
    conn = sqlite3.connect(str(TASKS_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS task_jobs (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                description TEXT DEFAULT '',
                status      TEXT DEFAULT 'pending',
                priority    INTEGER DEFAULT 5,
                agent_id    TEXT,
                depends_on  TEXT DEFAULT '[]',
                result      TEXT,
                error       TEXT,
                created_at  TEXT,
                updated_at  TEXT
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON task_jobs(status)
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_priority ON task_jobs(priority DESC)
        """)


_init_db()


# ═══════════════════════════════════════════════════════════════════════════════
#  TaskScheduler 核心类
# ═══════════════════════════════════════════════════════════════════════════════

class TaskScheduler:
    """基于 SQLite 的任务队列调度器."""

    def __init__(self):
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._lock = asyncio.Lock()

    # ── CRUD ────────────────────────────────────────────────────────────────

    def create(
        self,
        title: str,
        description: str = "",
        priority: int = 5,
        agent_id: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
    ) -> dict:
        """同步创建任务（供 API 路由调用）."""
        task_id = _uuid()
        now = datetime.now().isoformat()
        with _conn() as c:
            c.execute(
                "INSERT INTO task_jobs (id,title,description,status,priority,agent_id,depends_on,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (task_id, title, description, "pending", priority, agent_id,
                 json.dumps(depends_on or []), now, now),
            )
        return self.get(task_id)

    def get(self, task_id: str) -> Optional[dict]:
        """获取单个任务."""
        with _conn() as c:
            row = c.execute("SELECT * FROM task_jobs WHERE id=?", (task_id,)).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def list(
        self,
        status: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """列出任务（可过滤）."""
        sql = "SELECT * FROM task_jobs"
        params: list = []
        where_clauses = []
        if status:
            where_clauses.append("status=?")
            params.append(status)
        if agent_id:
            where_clauses.append("agent_id=?")
            params.append(agent_id)
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += " ORDER BY priority DESC, created_at DESC LIMIT ?"
        params.append(limit)

        with _conn() as c:
            rows = c.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update(
        self,
        task_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        priority: Optional[int] = None,
        agent_id: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[dict]:
        """更新任务字段."""
        task = self.get(task_id)
        if not task:
            return None

        updates: list[str] = []
        params: list = []

        if title is not None:
            updates.append("title=?"); params.append(title)
        if description is not None:
            updates.append("description=?"); params.append(description)
        if status is not None:
            updates.append("status=?"); params.append(status)
        if priority is not None:
            updates.append("priority=?"); params.append(priority)
        if agent_id is not None:
            updates.append("agent_id=?"); params.append(agent_id)
        if depends_on is not None:
            updates.append("depends_on=?"); params.append(json.dumps(depends_on))
        if result is not None:
            updates.append("result=?"); params.append(result)
        if error is not None:
            updates.append("error=?"); params.append(error)

        updates.append("updated_at=?"); params.append(datetime.now().isoformat())
        params.append(task_id)

        with _conn() as c:
            c.execute(f"UPDATE task_jobs SET {','.join(updates)} WHERE id=?", params)
        return self.get(task_id)

    def delete(self, task_id: str) -> bool:
        """删除任务."""
        with _conn() as c:
            c.execute("DELETE FROM task_jobs WHERE id=?", (task_id,))
            return c.rowcount > 0

    def counts(self) -> dict:
        """各状态任务数量."""
        with _conn() as c:
            rows = c.execute(
                "SELECT status, COUNT(*) as cnt FROM task_jobs GROUP BY status"
            ).fetchall()
        counts = {"pending": 0, "running": 0, "done": 0, "failed": 0, "cancelled": 0}
        for r in rows:
            if r["status"] in counts:
                counts[r["status"]] = r["cnt"]
        return counts

    # ── 依赖解析 ─────────────────────────────────────────────────────────────

    def _deps_satisfied(self, task_id: str) -> bool:
        """检查任务的依赖是否全部完成（状态为 done）."""
        task = self.get(task_id)
        if not task:
            return False
        deps: list[str] = json.loads(task.get("depends_on", "[]"))
        if not deps:
            return True
        for dep_id in deps:
            dep = self.get(dep_id)
            if not dep or dep["status"] != "done":
                return False
        return True

    def get_next_runnable(self) -> Optional[dict]:
        """获取优先级最高的可运行任务（依赖已满足且状态为 pending）."""
        with _conn() as c:
            rows = c.execute(
                "SELECT * FROM task_jobs WHERE status='pending' ORDER BY priority DESC, created_at ASC"
            ).fetchall()
        for row in rows:
            d = self._row_to_dict(row)
            if self._deps_satisfied(d["id"]):
                return d
        return None

    # ── 异步执行 ─────────────────────────────────────────────────────────────

    async def enqueue_and_run(
        self,
        title: str,
        description: str = "",
        priority: int = 5,
        agent_id: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        executor_fn=None,
    ) -> dict:
        """
        创建任务并立即异步执行。

        executor_fn: async function(task_id, description) -> str
            返回执行结果字符串，抛出异常则标记为 failed。
        """
        task = self.create(title, description, priority, agent_id, depends_on)
        task_id = task["id"]

        async with self._lock:
            if task_id in self._running_tasks:
                return task
            t = asyncio.create_task(self._run_task(task_id, executor_fn))
            self._running_tasks[task_id] = t

        return task

    async def _run_task(self, task_id: str, executor_fn):
        """内部：执行单个任务."""
        try:
            self.update(task_id, status="running")
            task = self.get(task_id)
            description = task["description"] if task else ""

            if executor_fn:
                result = await executor_fn(task_id, description)
            else:
                # 默认：模拟执行（3秒后完成）
                await asyncio.sleep(3)
                result = "Task completed successfully."

            self.update(task_id, status="done", result=result)
            logger.info(f"[scheduler] task {task_id} done")
        except asyncio.CancelledError:
            self.update(task_id, status="cancelled")
            logger.info(f"[scheduler] task {task_id} cancelled")
        except Exception as e:
            logger.error(f"[scheduler] task {task_id} failed: {e}")
            self.update(task_id, status="failed", error=str(e))
        finally:
            async with self._lock:
                self._running_tasks.pop(task_id, None)

    async def cancel(self, task_id: str) -> bool:
        """取消运行中的任务."""
        async with self._lock:
            t = self._running_tasks.get(task_id)
            if t:
                t.cancel()
                return True
        # 未运行的任务直接标记为 cancelled
        task = self.get(task_id)
        if task and task["status"] == "pending":
            self.update(task_id, status="cancelled")
            return True
        return False

    def is_running(self, task_id: str) -> bool:
        return task_id in self._running_tasks

    # ── 批量操作 ─────────────────────────────────────────────────────────────

    def create_batch(self, tasks: list[dict]) -> list[dict]:
        """批量创建任务."""
        results = []
        for t in tasks:
            results.append(self.create(
                title=t.get("title", "Untitled"),
                description=t.get("description", ""),
                priority=t.get("priority", 5),
                agent_id=t.get("agent_id"),
                depends_on=t.get("depends_on", []),
            ))
        return results

    # ── 辅助 ────────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "title": row["title"],
            "description": row["description"],
            "status": row["status"],
            "priority": row["priority"],
            "agent_id": row["agent_id"],
            "depends_on": json.loads(row["depends_on"] or "[]"),
            "result": row["result"],
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


# ── 全局单例 ──────────────────────────────────────────────────────────────
_scheduler: Optional[TaskScheduler] = None


def get_scheduler() -> TaskScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler
