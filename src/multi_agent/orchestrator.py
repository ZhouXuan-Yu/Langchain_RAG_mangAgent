"""多 Agent 编排器 — Agent 注册表、任务分发、Agent 间通信."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config import CHECKPOINT_PATH, DEFAULT_MODEL

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
AGENTS_DB = DATA_DIR / "agents.db"

# 默认 Agent 颜色池（供 UI 使用）
AGENT_COLORS = [
    "#8b0000",  # 深红
    "#555555",  # 炭灰
    "#999999",  # 银灰
    "#c0392b",  # 暗红
    "#4a4a4a",  # 深灰
    "#7f8c8d",  # 雾灰
    "#2c3e50",  # 藏青
    "#bdc3c7",  # 浅银
]


def _uuid() -> str:
    return "agent_" + uuid.uuid4().hex[:12]


# ═══════════════════════════════════════════════════════════════════════════════
#  SQLite 连接
# ═══════════════════════════════════════════════════════════════════════════════

_agents_lock = threading.Lock()


@contextmanager
def _conn():
    conn = sqlite3.connect(str(AGENTS_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS agent_profiles (
                id               TEXT PRIMARY KEY,
                name             TEXT NOT NULL,
                role             TEXT NOT NULL,
                description      TEXT DEFAULT '',
                model            TEXT DEFAULT 'deepseek-chat',
                color            TEXT DEFAULT '#888888',
                is_active        INTEGER DEFAULT 1,
                tasks_completed  INTEGER DEFAULT 0,
                parent_id        TEXT,
                worker_kind      TEXT,
                created_at       TEXT
            )
        """)
        # 兼容升级：若表中无 worker_kind 列则添加
        cols = [r[1] for r in c.execute("PRAGMA table_info(agent_profiles)").fetchall()]
        if "worker_kind" not in cols:
            c.execute("ALTER TABLE agent_profiles ADD COLUMN worker_kind TEXT")
        if "capabilities" not in cols:
            c.execute("ALTER TABLE agent_profiles ADD COLUMN capabilities TEXT DEFAULT '[]'")

        row = c.execute("SELECT COUNT(*) as cnt FROM agent_profiles").fetchone()
        if row["cnt"] == 0:
            now = datetime.now().isoformat()
            c.execute(
                "INSERT INTO agent_profiles (id,name,role,description,model,color,is_active,parent_id,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                ("agent_main", "Chief Coordinator", "chief",
                 "主控 Agent，负责协调所有子 Agent 的工作",
                 DEFAULT_MODEL, "#8b0000", 1, None, now),
            )

        # 幂等种子：三条内置 Worker（稳定 id，方便前端关联）
        _seed_workers(c)


def _seed_workers(c: sqlite3.Cursor) -> None:
    """确保三条内置 Worker 档案存在（幂等）。"""
    now = datetime.now().isoformat()
    workers = [
        (
            "agent_worker_search",
            "Search Worker",
            "search_worker",
            "外网实时信息搜索，使用 Tavily 搜索工具获取最新资料",
            DEFAULT_MODEL,
            "#4ade80",
            '["web_search", "browse_page"]',
        ),
        (
            "agent_worker_rag",
            "RAG Worker",
            "rag_worker",
            "本地知识库与记忆检索，从向量数据库中检索相关上下文",
            DEFAULT_MODEL,
            "#60a5fa",
            '["memory_search", "knowledge_base_search"]',
        ),
        (
            "agent_worker_coder",
            "Coder Worker",
            "coder",
            "代码生成与编写，基于上下文生成高质量代码实现",
            DEFAULT_MODEL,
            "#facc15",
            '["code_generation", "code_review"]',
        ),
    ]
    for (wk_id, wk_name, wk_kind, wk_desc, wk_model, wk_color, wk_caps) in workers:
        existing = c.execute(
            "SELECT id FROM agent_profiles WHERE id=?", (wk_id,)
        ).fetchone()
        if not existing:
            c.execute(
                "INSERT INTO agent_profiles "
                "(id,name,role,worker_kind,description,model,color,is_active,capabilities,created_at) "
                "VALUES (?,?,?,?,?,?,?,1,?,?)",
                (wk_id, wk_name, wk_kind, wk_kind, wk_desc, wk_model, wk_color, wk_caps, now),
            )


_init_db()


# ═══════════════════════════════════════════════════════════════════════════════
#  AgentRegistry 核心类
# ═══════════════════════════════════════════════════════════════════════════════

class AgentRegistry:
    """多 Agent 注册表 — 管理所有 Agent 实例及其元数据."""

    def __init__(self):
        self._instances: dict[str, Any] = {}  # agent_id -> compiled agent graph
        self._color_idx = 0

    # ── Agent CRUD ──────────────────────────────────────────────────────────

    def list_agents(self, is_active: Optional[bool] = None) -> list[dict]:
        """列出所有 Agent."""
        with _conn() as c:
            if is_active is not None:
                rows = c.execute(
                    "SELECT * FROM agent_profiles WHERE is_active=? ORDER BY created_at",
                    (1 if is_active else 0,),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM agent_profiles ORDER BY created_at"
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get(self, agent_id: str) -> Optional[dict]:
        """获取单个 Agent 信息."""
        with _conn() as c:
            row = c.execute("SELECT * FROM agent_profiles WHERE id=?", (agent_id,)).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def get_by_worker_kind(self, worker_kind: str) -> Optional[dict]:
        """根据 worker_kind 查找 Agent（用于编排路由）。"""
        with _conn() as c:
            row = c.execute(
                "SELECT * FROM agent_profiles WHERE worker_kind=? LIMIT 1",
                (worker_kind,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def create(
        self,
        name: str,
        role: str,
        description: str = "",
        model: str = DEFAULT_MODEL,
        color: Optional[str] = None,
        parent_id: Optional[str] = None,
        capabilities: list[str] | None = None,
    ) -> dict:
        """注册一个新 Agent."""
        agent_id = _uuid()
        now = datetime.now().isoformat()

        if color is None:
            color = AGENT_COLORS[self._color_idx % len(AGENT_COLORS)]
            self._color_idx += 1

        cap_json = json.dumps(capabilities or []) if capabilities is not None else "[]"
        with _conn() as c:
            c.execute(
                "INSERT INTO agent_profiles (id,name,role,description,model,color,parent_id,capabilities,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (agent_id, name, role, description, model, color, parent_id, cap_json, now),
            )

        logger.info(f"[agent_registry] created agent {agent_id}: {name} ({role})")
        return self.get(agent_id)

    def update(
        self,
        agent_id: str,
        name: Optional[str] = None,
        role: Optional[str] = None,
        description: Optional[str] = None,
        model: Optional[str] = None,
        color: Optional[str] = None,
        is_active: Optional[bool] = None,
        worker_kind: Optional[str] = None,
        capabilities: list[str] | None = None,
    ) -> Optional[dict]:
        """更新 Agent 配置."""
        updates: list[str] = []
        params: list = []
        if name is not None:
            updates.append("name=?"); params.append(name)
        if role is not None:
            updates.append("role=?"); params.append(role)
        if description is not None:
            updates.append("description=?"); params.append(description)
        if model is not None:
            updates.append("model=?"); params.append(model)
        if color is not None:
            updates.append("color=?"); params.append(color)
        if is_active is not None:
            updates.append("is_active=?"); params.append(1 if is_active else 0)
        if worker_kind is not None:
            updates.append("worker_kind=?"); params.append(worker_kind)
        if capabilities is not None:
            updates.append("capabilities=?"); params.append(json.dumps(capabilities))

        if not updates:
            return self.get(agent_id)

        params.append(agent_id)
        with _conn() as c:
            c.execute(f"UPDATE agent_profiles SET {','.join(updates)} WHERE id=?", params)

        # 清除实例缓存（下次请求时重建）
        self._instances.pop(agent_id, None)
        logger.info(f"[agent_registry] updated agent {agent_id}")
        return self.get(agent_id)

    def delete(self, agent_id: str) -> bool:
        """删除 Agent（软删除：标记为非活跃）."""
        # 不允许删除主控 Agent
        if agent_id == "agent_main":
            return False
        with _conn() as c:
            c.execute(
                "UPDATE agent_profiles SET is_active=0 WHERE id=?",
                (agent_id,),
            )
            deleted = c.rowcount > 0
        if deleted:
            self._instances.pop(agent_id, None)
            logger.info(f"[agent_registry] deactivated agent {agent_id}")
        return deleted

    def increment_completed(self, agent_id: str) -> None:
        """完成任务计数 +1."""
        with _conn() as c:
            c.execute(
                "UPDATE agent_profiles SET tasks_completed=tasks_completed+1 WHERE id=?",
                (agent_id,),
            )

    # ── Agent 实例（按需构建）───────────────────────────────────────────────

    def get_or_create_instance(self, agent_id: str) -> Optional[Any]:
        """
        获取编译好的 Agent Graph 实例（按 agent_id 缓存）。

        每个 Agent 有独立的 thread_id 和 session。
        """
        if agent_id in self._instances:
            return self._instances[agent_id]

        profile = self.get(agent_id)
        if not profile or not profile["is_active"]:
            return None

        try:
            from src.llm import init_deepseek_llm
            from src.graph import build_agent_graph
            from src.memory.sqlite_store import get_async_sqlite_checkpointer

            import asyncio

            async def _build():
                llm = init_deepseek_llm(model=profile["model"], streaming=True)
                checkpointer = await get_async_sqlite_checkpointer(str(CHECKPOINT_PATH))
                return build_agent_graph(llm, checkpointer=checkpointer)

            # 同步获取 event loop（在 FastAPI 的 async 上下文中应该有 loop）
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            agent = loop.run_until_complete(_build())
            self._instances[agent_id] = agent
            logger.info(f"[agent_registry] built and cached agent instance: {agent_id}")
            return agent
        except Exception as e:
            logger.error(f"[agent_registry] failed to build agent {agent_id}: {e}")
            return None

    # ── 任务分发 ─────────────────────────────────────────────────────────────

    async def dispatch_task(
        self,
        agent_id: str,
        task_description: str,
        thread_id: Optional[str] = None,
        context: list[dict] | None = None,
    ) -> dict:
        """
        分发任务给指定 Agent，执行并返回结果。

        Args:
            context: 前置任务的执行结果列表，注入到 prompt 中供 Agent 参考。
        """
        agent = self.get_or_create_instance(agent_id)
        if not agent:
            return {"status": "error", "message": f"Agent {agent_id} not found or inactive"}

        from langchain_core.messages import HumanMessage
        from src.server.dependencies import get_checkpointer

        thread = thread_id or f"agent_{agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        config = {"configurable": {"thread_id": thread}}

        # 构建带 context 的消息
        if context:
            ctx_lines = [f"【上下文（来自前置任务）】:"]
            for ci, cr in enumerate(context, 1):
                src = cr.get("agent", "unknown")
                res = cr.get("result", "")
                score = cr.get("quality_score", 0.0)
                ctx_lines.append(f"[{ci}] 来源: {src} (质量: {score:.1f})")
                ctx_lines.append(f"    {str(res)[:300]}...")
            ctx_block = "\n".join(ctx_lines)
            prompt_with_ctx = (
                f"{ctx_block}\n\n"
                f"【本次任务】:\n{task_description}"
            )
        else:
            prompt_with_ctx = task_description

        accumulated = ""

        try:
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=prompt_with_ctx)], "thread_id": thread},
                config=config,
                version="v2",
            ):
                ev_type = event.get("event", "")
                if ev_type == "chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        content = chunk.content
                        if isinstance(content, list):
                            for part in content:
                                if isinstance(part, str):
                                    accumulated += part
                                elif isinstance(part, dict) and part.get("type") == "text":
                                    accumulated += part.get("text") or ""

            if not accumulated or len(accumulated.strip()) < 5:
                logger.warning(
                    "[agent_registry] dispatch %s returned empty/too-short result (len=%d). "
                    "Prompt was: %.200s...",
                    agent_id, len(accumulated), prompt_with_ctx[:200]
                )
                return {
                    "status": "error",
                    "message": f"Agent '{agent_id}' produced empty result. "
                               f"This may indicate a configuration issue or the agent "
                               f"failed to produce meaningful output.",
                    "result": accumulated,
                    "agent_id": agent_id,
                }

            self.increment_completed(agent_id)
            return {"status": "ok", "result": accumulated, "agent_id": agent_id}

        except Exception as e:
            logger.error(f"[agent_registry] dispatch failed for {agent_id}: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "agent_id": agent_id}

    # ── Agent 组织架构 ───────────────────────────────────────────────────────

    def get_org_tree(self) -> list[dict]:
        """获取 Agent 组织树（扁平结构，带 parent_id 用于前端渲染）."""
        agents = self.list_agents()
        return agents

    def get_children(self, parent_id: str) -> list[dict]:
        """获取指定 Agent 的子节点."""
        with _conn() as c:
            rows = c.execute(
                "SELECT * FROM agent_profiles WHERE parent_id=? ORDER BY created_at",
                (parent_id,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ── 辅助 ────────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "name": row["name"],
            "role": row["role"],
            "description": row["description"],
            "model": row["model"],
            "color": row["color"],
            "is_active": bool(row["is_active"]),
            "tasks_completed": row["tasks_completed"],
            "parent_id": row["parent_id"],
            "worker_kind": row["worker_kind"],
            "created_at": row["created_at"],
            "capabilities": json.loads(row["capabilities"] or "[]"),
        }


# ── 全局单例 ──────────────────────────────────────────────────────────────
_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry
