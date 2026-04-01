"""Token tracker — 22-24: Token 使用量追踪与成本统计."""

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import tiktoken

from src.config import PROJECT_ROOT

# ── agent_usage 表路径 ────────────────────────────────────────────────────────
_AGENTS_DB_DIR = PROJECT_ROOT / "data"
_AGENTS_DB_DIR.mkdir(exist_ok=True)
_AGENTS_DB = _AGENTS_DB_DIR / "agent_usage.db"

_usage_lock = threading.Lock()


@contextmanager
def _conn():
    """线程安全的 SQLite 连接上下文管理器."""
    conn = sqlite3.connect(str(_AGENTS_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_usage_db():
    """初始化 agent_usage 表."""
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS agent_usage (
                id              TEXT PRIMARY KEY,
                job_id          TEXT,
                agent_id        TEXT,
                provider        TEXT DEFAULT 'deepseek',
                model           TEXT DEFAULT 'deepseek-chat',
                input_tokens    INTEGER DEFAULT 0,
                output_tokens   INTEGER DEFAULT 0,
                cost_usd        REAL DEFAULT 0.0,
                duration_ms     INTEGER DEFAULT 0,
                label           TEXT DEFAULT '',
                created_at      TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS agent_episodes (
                id               TEXT PRIMARY KEY,
                job_id           TEXT,
                requirement      TEXT,
                tasks            TEXT,
                results          TEXT,
                quality_score    REAL DEFAULT 0.0,
                duration_ms      INTEGER DEFAULT 0,
                agents_used      TEXT,
                max_depth        INTEGER DEFAULT 1,
                healing_attempts INTEGER DEFAULT 0,
                passed           INTEGER DEFAULT 0,
                created_at       TEXT
            )
        """)


_init_usage_db()


# ── AgentUsageStore ──────────────────────────────────────────────────────────


class AgentUsageStore:
    """agent_usage 表的读写接口，按 job_id 聚合统计."""

    @staticmethod
    def save(
        job_id: str,
        agent_id: str,
        input_tokens: int,
        output_tokens: int,
        model: str = "deepseek-chat",
        provider: str = "deepseek",
        cost_usd: float = 0.0,
        duration_ms: int = 0,
        label: str = "",
    ) -> str:
        """写入一条 usage 记录，返回记录 id."""
        record_id = f"usage_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()
        with _conn() as c:
            c.execute(
                """INSERT INTO agent_usage
                (id, job_id, agent_id, provider, model,
                 input_tokens, output_tokens, cost_usd, duration_ms, label, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record_id, job_id, agent_id, provider, model,
                 input_tokens, output_tokens, cost_usd, duration_ms, label, now),
            )
        return record_id

    @staticmethod
    def get_by_job(job_id: str) -> list[dict]:
        """获取指定 job 的所有 usage 记录."""
        with _conn() as c:
            rows = c.execute(
                "SELECT * FROM agent_usage WHERE job_id=? ORDER BY created_at",
                (job_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def get_overview(days: int = 30) -> dict:
        """获取全局统计概览（最近 N 天）."""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with _conn() as c:
            row = c.execute(
                """SELECT
                    COUNT(*) as total_calls,
                    COALESCE(SUM(input_tokens), 0)  as total_input_tokens,
                    COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                    COALESCE(SUM(input_tokens + output_tokens), 0) as total_tokens,
                    COALESCE(SUM(cost_usd), 0.0) as total_cost_usd,
                    COALESCE(AVG(cost_usd), 0.0) as avg_cost_per_call,
                    COUNT(DISTINCT job_id) as total_jobs,
                    COUNT(DISTINCT agent_id) as total_agents
                FROM agent_usage
                WHERE created_at >= ?""",
                (cutoff,),
            ).fetchone()

        return dict(row) if row else {}

    @staticmethod
    def get_by_agent(agent_id: str, days: int = 30) -> dict:
        """获取指定 agent 的统计."""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with _conn() as c:
            row = c.execute(
                """SELECT
                    agent_id,
                    COUNT(*) as total_calls,
                    COALESCE(SUM(input_tokens), 0)  as total_input_tokens,
                    COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                    COALESCE(SUM(input_tokens + output_tokens), 0) as total_tokens,
                    COALESCE(SUM(cost_usd), 0.0) as total_cost_usd,
                    COALESCE(AVG(cost_usd), 0.0) as avg_cost_per_call,
                    COUNT(DISTINCT job_id) as total_jobs
                FROM agent_usage
                WHERE agent_id=? AND created_at >= ?
                GROUP BY agent_id""",
                (agent_id, cutoff),
            ).fetchone()

        return dict(row) if row else {}

    @staticmethod
    def get_all_agents(days: int = 30) -> list[dict]:
        """获取所有 agent 的统计列表."""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with _conn() as c:
            rows = c.execute(
                """SELECT
                    agent_id,
                    COUNT(*) as total_calls,
                    COALESCE(SUM(input_tokens), 0)  as total_input_tokens,
                    COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                    COALESCE(SUM(input_tokens + output_tokens), 0) as total_tokens,
                    COALESCE(SUM(cost_usd), 0.0) as total_cost_usd,
                    COALESCE(AVG(cost_usd), 0.0) as avg_cost_per_call,
                    COUNT(DISTINCT job_id) as total_jobs
                FROM agent_usage
                WHERE created_at >= ?
                GROUP BY agent_id
                ORDER BY total_tokens DESC""",
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def get_job_history(limit: int = 50, offset: int = 0) -> list[dict]:
        """获取编排任务历史（按 job_id 聚合）."""
        with _conn() as c:
            rows = c.execute(
                """SELECT
                    job_id,
                    agent_id,
                    SUM(input_tokens)  as input_tokens,
                    SUM(output_tokens) as output_tokens,
                    SUM(input_tokens + output_tokens) as total_tokens,
                    SUM(cost_usd) as cost_usd,
                    MIN(created_at) as first_call,
                    MAX(created_at) as last_call,
                    COUNT(*) as call_count
                FROM agent_usage
                GROUP BY job_id, agent_id
                ORDER BY first_call DESC
                LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()

        # 按 job_id 聚合
        job_map: dict[str, dict] = {}
        for r in rows:
            jid = r["job_id"]
            if jid not in job_map:
                job_map[jid] = {
                    "job_id": jid,
                    "first_call": r["first_call"],
                    "last_call": r["last_call"],
                    "total_calls": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "agents": [],
                }
            job_map[jid]["total_calls"] += r["call_count"]
            job_map[jid]["total_input_tokens"] += r["input_tokens"]
            job_map[jid]["total_output_tokens"] += r["output_tokens"]
            job_map[jid]["total_tokens"] += r["total_tokens"]
            job_map[jid]["total_cost_usd"] += r["cost_usd"]
            job_map[jid]["agents"].append({
                "agent_id": r["agent_id"],
                "call_count": r["call_count"],
                "tokens": r["total_tokens"],
                "cost_usd": r["cost_usd"],
            })

        result = list(job_map.values())
        result.sort(key=lambda x: x["first_call"], reverse=True)
        return result


class TokenTracker:
    """
    追踪 DeepSeek API 的 Token 消耗与成本.

    使用 cl100k_base 编码器（与 DeepSeek 兼容）计算 token 数，
    记录每轮调用的 prompt/completion tokens 及估算成本。
    """

    def __init__(self, model: str = "deepseek-chat"):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.model = model
        self.total_tokens: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.history: list[dict] = []

    def count(self, text: str) -> int:
        """计算文本的 token 数量."""
        return len(self.enc.encode(text))

    def save_to_db(
        self,
        job_id: str = "",
        agent_id: str = "",
        provider: str = "deepseek",
        duration_ms: int = 0,
        label: str = "",
    ) -> str:
        """
        将当前累计的 token 消耗写入 agent_usage 表。

        适用于 orchestrator 等场景：先批量累加，最后一次性入库。
        """
        return AgentUsageStore.save(
            job_id=job_id,
            agent_id=agent_id,
            provider=provider,
            model=self.model,
            input_tokens=self.total_prompt_tokens,
            output_tokens=self.total_completion_tokens,
            cost_usd=self.total_cost_usd,
            duration_ms=duration_ms,
            label=label,
        )

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str | None = None,
        label: str = "",
    ) -> None:
        """记录一次 API 调用的 token 消耗.

        Args:
            prompt_tokens: prompt 部分的 token 数
            completion_tokens: completion 部分的 token 数
            model: 模型名称（用于计算单价）
            label: 调用标签（如 "search", "reason"）
        """
        model = model or self.model
        cost = self._calc_cost(prompt_tokens, completion_tokens, model)

        self.total_tokens += prompt_tokens + completion_tokens
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd += cost

        self.history.append({
            "time": datetime.now().isoformat(),
            "label": label,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": round(cost, 6),
        })

    def _calc_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> float:
        """根据模型计算 USD 成本."""
        # DeepSeek Chat (DeepSeek-V3) pricing (approximate)
        pricing = {
            "deepseek-chat": (0.00027, 0.0011),  # (prompt_per_1k, completion_per_1k)
            "deepseek-reasoner": (0.0011, 0.0027),
        }
        p_rate, c_rate = pricing.get(model, (0.001, 0.002))
        return (prompt_tokens / 1000 * p_rate) + (completion_tokens / 1000 * c_rate)

    def summary(self) -> dict:
        """返回使用摘要."""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "num_calls": len(self.history),
        }

    def get_history(self, days: int = 7) -> list[dict]:
        """返回最近 N 天的历史记录（按天聚合）."""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        # 按天聚合
        by_date: dict[str, dict] = {}
        for h in self.history:
            ts = h.get("time", "")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            if dt < cutoff:
                continue
            date_key = dt.strftime("%Y-%m-%d")
            if date_key not in by_date:
                by_date[date_key] = {
                    "timestamp": ts,
                    "date": date_key,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "num_calls": 0,
                    "cost_usd": 0.0,
                    "model": h.get("model", ""),
                }
            d = by_date[date_key]
            d["prompt_tokens"] += h["prompt_tokens"]
            d["completion_tokens"] += h["completion_tokens"]
            d["total_tokens"] += h["total_tokens"]
            d["num_calls"] += 1
            d["cost_usd"] += h["cost_usd"]
            d["timestamp"] = max(d["timestamp"], ts)
        # 返回按日期降序排列
        result = list(by_date.values())
        result.sort(key=lambda x: x["date"], reverse=True)
        return result

    def print_report(self) -> None:
        """打印成本报告到控制台."""
        s = self.summary()
        print("\n========== Token 成本报告 ==========")
        print(f"  API 调用次数: {s['num_calls']}")
        print(f"  Prompt Tokens:  {s['prompt_tokens']:,}")
        print(f"  Completion Tokens: {s['completion_tokens']:,}")
        print(f"  总 Token:     {s['total_tokens']:,}")
        print(f"  总成本:       ${s['total_cost_usd']:.6f}")
        print("=====================================\n")
