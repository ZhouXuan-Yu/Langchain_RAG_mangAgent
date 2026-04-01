"""Episode Store — 跨会话长期记忆（Episodic Memory）.

L3 记忆层：每次编排完成后自动记录 episode（任务清单、执行结果、质量评分、耗时），
支持 Supervisor 查询相似历史经验来辅助规划。
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config import PROJECT_ROOT

# ── DB 路径（复用 agent_usage.db）────────────────────────────────────────────
_DB_PATH = PROJECT_ROOT / "data" / "agent_usage.db"


def _conn():
    """线程安全的 SQLite 连接."""
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── EpisodeStore ──────────────────────────────────────────────────────────────


class EpisodeStore:
    """
    Episodic Memory Store — 跨会话长期记忆。

    每次编排完成后写入一条 episode，后续 Supervisor 可以查询相似历史经验。

    episode 包含：
    - 原始需求
    - 任务清单及执行结果
    - 质量评分、耗时、通过状态
    - 涉及的 agent 列表
    """

    def save(
        self,
        job_id: str,
        requirement: str,
        tasks: list[dict],
        results: list[dict],
        quality_score: float,
        duration_ms: int,
        agents_used: list[str],
        max_depth: int = 1,
        healing_attempts: int = 0,
        passed: bool = False,
    ) -> str:
        """
        保存一个编排 episode。

        Returns:
            episode_id: 新建记录的 id
        """
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()
        tasks_json = json.dumps(tasks, ensure_ascii=False)
        results_json = json.dumps(results, ensure_ascii=False)
        agents_json = json.dumps(agents_used, ensure_ascii=False)

        with _conn() as c:
            c.execute(
                """INSERT INTO agent_episodes
                (id, job_id, requirement, tasks, results,
                 quality_score, duration_ms, agents_used,
                 max_depth, healing_attempts, passed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (episode_id, job_id, requirement, tasks_json, results_json,
                 quality_score, duration_ms, agents_json,
                 max_depth, healing_attempts, 1 if passed else 0, now),
            )

        return episode_id

    def get_recent(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """获取最近的 N 条 episode."""
        with _conn() as c:
            rows = c.execute(
                """SELECT * FROM agent_episodes
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
        return [self._row_to_episode(dict(r)) for r in rows]

    def get_by_job(self, job_id: str) -> Optional[dict]:
        """按 job_id 查询 episode."""
        with _conn() as c:
            row = c.execute(
                "SELECT * FROM agent_episodes WHERE job_id=? LIMIT 1",
                (job_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_episode(dict(row))

    def query_similar(self, requirement: str, top_k: int = 3) -> list[dict]:
        """
        模糊匹配相似的历史 episode。

        使用 requirement 文本的前 50 字符做 LIKE 匹配，
        按 quality_score 降序、created_at 降序排序。
        """
        # 用关键词分词做多词匹配，提升召回率
        keywords = [w for w in requirement[:100].split() if len(w) >= 2]
        if not keywords:
            # 无关键词时直接用前50字模糊匹配
            pattern = f"%{requirement[:50]}%"
            order = "quality_score DESC, created_at DESC"
        else:
            # OR 匹配多个关键词
            pattern_parts = " OR requirement LIKE ".join(["?"] * len(keywords))
            pattern = pattern_parts
            keywords = [f"%{kw}%" for kw in keywords]
            order = "quality_score DESC, created_at DESC"

        with _conn() as c:
            if keywords:
                rows = c.execute(
                    f"""SELECT * FROM agent_episodes
                       WHERE requirement LIKE {pattern_parts}
                       ORDER BY {order}
                       LIMIT ?""",
                    keywords + [top_k],
                ).fetchall()
            else:
                rows = c.execute(
                    """SELECT * FROM agent_episodes
                       WHERE requirement LIKE ?
                       ORDER BY quality_score DESC, created_at DESC
                       LIMIT ?""",
                    (pattern, top_k),
                ).fetchall()
        return [self._row_to_episode(dict(r)) for r in rows]

    def get_stats(self) -> dict:
        """获取全局 episode 统计."""
        with _conn() as c:
            row = c.execute(
                """SELECT
                    COUNT(*) as total_episodes,
                    COALESCE(AVG(quality_score), 0) as avg_quality_score,
                    COALESCE(AVG(duration_ms), 0) as avg_duration_ms,
                    COALESCE(SUM(CASE WHEN passed=1 THEN 1 ELSE 0 END), 0) as passed_count,
                    COUNT(DISTINCT job_id) as total_jobs
                FROM agent_episodes"""
            ).fetchone()
        return dict(row) if row else {}

    def get_agent_usage_ranking(self, limit: int = 10) -> list[dict]:
        """获取使用频率最高的 agent 排名（跨 episode 聚合）。"""
        with _conn() as c:
            rows = c.execute(
                """SELECT agents_used FROM agent_episodes"""
            ).fetchall()

        from collections import Counter
        counter: Counter[str] = Counter()
        for r in rows:
            try:
                agents = json.loads(r["agents_used"] or "[]")
                counter.update(agents)
            except Exception:
                pass

        return [
            {"agent_id": agent, "episode_count": count}
            for agent, count in counter.most_common(limit)
        ]

    @staticmethod
    def _row_to_episode(row: dict) -> dict:
        """将 DB Row 转换为 episode dict（解析 JSON 字段）。"""
        try:
            tasks = json.loads(row.get("tasks") or "[]")
        except Exception:
            tasks = []
        try:
            results = json.loads(row.get("results") or "[]")
        except Exception:
            results = []
        try:
            agents = json.loads(row.get("agents_used") or "[]")
        except Exception:
            agents = []

        return {
            "id": row.get("id", ""),
            "job_id": row.get("job_id", ""),
            "requirement": row.get("requirement", ""),
            "tasks": tasks,
            "results": results,
            "quality_score": row.get("quality_score") or 0.0,
            "duration_ms": row.get("duration_ms") or 0,
            "agents_used": agents,
            "max_depth": row.get("max_depth") or 1,
            "healing_attempts": row.get("healing_attempts") or 0,
            "passed": bool(row.get("passed")),
            "created_at": row.get("created_at") or "",
        }


# ── 全局单例 ──────────────────────────────────────────────────────────────────
_episode_store: Optional[EpisodeStore] = None


def get_episode_store() -> EpisodeStore:
    global _episode_store
    if _episode_store is None:
        _episode_store = EpisodeStore()
    return _episode_store
