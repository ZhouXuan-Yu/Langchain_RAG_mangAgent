"""SQLite checkpointer — 07: 对话状态持久化与断电恢复."""

import asyncio
import logging
import sqlite3
from typing import Any, Optional

from src.config import CHECKPOINT_PATH

logger = logging.getLogger(__name__)


def get_sqlite_checkpointer(db_path: str | None = None) -> Any:
    """
    获取同步 SQLite checkpointer — 用于同步 Agent（src/main.py）。
    """
    from langgraph.checkpoint.sqlite import SqliteSaver
    from pathlib import Path

    path = db_path or str(CHECKPOINT_PATH)

    # 确保父目录存在
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)

    # 同步版本同样需要显式 setup()
    saver.setup()
    return saver


async def get_async_sqlite_checkpointer(db_path: str | None = None) -> Any:
    """
    获取异步 SQLite checkpointer — 用于异步 Agent（FastAPI / src/server/）。

    Args:
        db_path: SQLite 数据库文件路径，默认使用 CHECKPOINT_PATH

    Returns:
        AsyncSqliteSaver checkpointer instance（已初始化数据库表）
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    path = db_path or str(CHECKPOINT_PATH)

    # 确保父目录存在
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    conn = await asyncio.wait_for(
        aiosqlite.connect(path),
        timeout=10.0,
    )
    saver = AsyncSqliteSaver(conn)

    # 关键修复：显式调用 setup() 创建数据库表
    # AsyncSqliteSaver 不会自动初始化，首次写入前必须调用
    await saver.setup()
    logger.info(f"[checkpointer] initialized DB at {path}")

    # 关键修复：禁用 WAL 模式
    # AsyncSqliteSaver 内部会创建独立连接运行 WAL 模式，
    # 导致通过独立同步连接（如 list_threads）无法读到最新写入的数据
    # 必须将内部连接的 journal_mode 也改为 DELETE
    await saver.conn.execute("PRAGMA journal_mode=DELETE")
    await saver.conn.commit()

    return saver


def get_thread_config(thread_id: str) -> dict:
    """
    生成标准的 thread_config — 用于隔离不同用户的对话状态.

    Args:
        thread_id: 会话 ID，每个用户/项目应有独立的 thread_id

    Example:
        config = get_thread_config("zhouxuan_session_001")
        for event in agent.stream({"messages": [HumanMessage(content="...")]}, config=config):
            ...
    """
    return {"configurable": {"thread_id": thread_id}}


def recover_or_new(
    checkpointer: Any,
    thread_id: str,
) -> dict:
    """
    尝试从 checkpointer 恢复对话状态，若无记录则返回空状态.

    Args:
        checkpointer: SqliteSaver instance
        thread_id: 要恢复的会话 ID

    Returns:
        包含 messages 列表的字典，若无记录则返回空字典
    """
    config = get_thread_config(thread_id)
    try:
        saved = checkpointer.get(config)
        if saved and "messages" in saved.get("channel_values", {}):
            messages = saved["channel_values"]["messages"]
            logger.info(f"恢复对话: thread={thread_id}, 共 {len(messages)} 条消息")
            return {"messages": messages}
    except Exception as e:
        logger.warning(f"无法恢复 thread={thread_id}: {e}")
    return {}


# ── 工具事件记录 ──────────────────────────────────────────────────────────────
_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    thread_id     TEXT PRIMARY KEY,
    title         TEXT    NOT NULL DEFAULT '',
    message_count INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_TOOL_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS tool_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT    NOT NULL,
    turn_index  INTEGER NOT NULL DEFAULT 0,
    step_index  INTEGER NOT NULL DEFAULT 0,
    seq         INTEGER NOT NULL DEFAULT 0,
    event_type  TEXT    NOT NULL,          -- 'tool_start' | 'tool_result'
    tool_name   TEXT    NOT NULL,
    tool_input  TEXT,
    tool_result TEXT,
    ts          TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_tool_events_thread ON tool_events(thread_id, turn_index);
"""


def _ensure_db_schema(db_path: str | None = None) -> None:
    """确保所有表都存在（sessions + tool_events）。"""
    db_path = db_path or str(CHECKPOINT_PATH)
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SESSIONS_DDL)
        conn.executescript(_TOOL_EVENTS_DDL)
        conn.commit()
    finally:
        conn.close()


# ── 为了向后兼容，保留旧函数名 ──────────────────────────────────
def _ensure_tool_events_table(db_path: str | None = None) -> None:
    _ensure_db_schema(db_path)


def save_tool_event(
    thread_id: str,
    turn_index: int,
    step_index: int,
    seq: int,
    event_type: str,
    tool_name: str,
    tool_input: str = "",
    tool_result: str = "",
) -> None:
    """同步写入一条工具事件（对话页 SSE 推流时调用）。"""
    db_path = str(CHECKPOINT_PATH)
    try:
        _ensure_db_schema(db_path)
        import sqlite3
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute(
            """
            INSERT INTO tool_events
                (thread_id, turn_index, step_index, seq, event_type, tool_name, tool_input, tool_result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (thread_id, turn_index, step_index, seq, event_type, tool_name, tool_input, tool_result),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"[tool_events] save failed: {e}")


def upsert_session(thread_id: str, title: str = "") -> None:
    """创建或更新 session 元信息（标题、首条消息时调用）。"""
    db_path = str(CHECKPOINT_PATH)
    try:
        _ensure_db_schema(db_path)
        import sqlite3
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute(
            """
            INSERT INTO sessions (thread_id, title, message_count, created_at, updated_at)
            VALUES (?, ?, 0, datetime('now'), datetime('now'))
            ON CONFLICT(thread_id) DO UPDATE SET
                title = CASE WHEN title = '' THEN excluded.title ELSE title END,
                updated_at = datetime('now')
            """,
            (thread_id, title),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"[sessions] upsert failed: {e}")


def update_session_message_count(thread_id: str) -> None:
    """对话结束时更新消息计数（从 tool_events 估算）。"""
    db_path = str(CHECKPOINT_PATH)
    try:
        import sqlite3
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.execute(
            "SELECT COUNT(DISTINCT turn_index) FROM tool_events WHERE thread_id = ?",
            (thread_id,),
        )
        row = cursor.fetchone()
        count = (row[0] * 2) if row and row[0] else 0  # 每个 turn 至少 1 user + 1 assistant
        conn.execute(
            "UPDATE sessions SET message_count = ?, updated_at = datetime('now') WHERE thread_id = ?",
            (count, thread_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"[sessions] update count failed: {e}")


def get_session_detail(thread_id: str, db_path: str | None = None) -> dict:
    """
    读取某 thread_id 的完整对话详情：
      - messages：从 checkpointer 读取（HumanMessage / AIMessage / ToolMessage）
      - tool_events：工具调用记录
    返回结构：{ messages: [...], tool_events: {...} }
    """
    import uuid as _uuid
    _run_id = str(_uuid.uuid4())[:8]
    _log = lambda msg, data: _write_log({
        "id": f"log_{int(__import__('time').time()*1000)}_{_run_id}",
        "sessionId": "4eeec4",
        "location": "sqlite_store.py:get_session_detail",
        "message": msg,
        "data": data,
        "timestamp": int(__import__('time').time()*1000),
        "runId": "debug",
        "hypothesisId": "H2",
    })
    _write_log = lambda p: (
        __import__('pathlib').Path(__import__('sys').prefix).parent / "debug-4eeec4.log"
    ).write_text("", encoding="utf-8") or True  # noop placeholder
    import time as _time
    import json as _json
    _LOG_PATH = __import__('pathlib').Path(__file__).resolve().parent.parent.parent / "debug-4eeec4.log"
    def _dbg(msg, data):
        try:
            with open(_LOG_PATH, "a", encoding="utf-8") as _f:
                _f.write(_json.dumps({
                    "id": f"log_{int(_time.time()*1000)}_{_run_id}",
                    "sessionId": "4eeec4",
                    "location": "sqlite_store.py:get_session_detail",
                    "message": msg,
                    "data": data,
                    "timestamp": int(_time.time()*1000),
                    "runId": "debug",
                    "hypothesisId": "H2",
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    _dbg("ENTER", {"thread_id": thread_id})
    db_path = db_path or str(CHECKPOINT_PATH)
    import sqlite3
    conn = sqlite3.connect(db_path, check_same_thread=False)

    # 0) 确保 sessions 表存在（新建数据库首次访问时）

    # 1) 从 checkpointer 取消息
    messages = []
    try:
        cursor = conn.execute(
            """
            SELECT data
              FROM checkpoints
             WHERE thread_id = ?
             ORDER BY updated_at DESC
             LIMIT 1
            """,
            (thread_id,),
        )
        row = cursor.fetchone()
        if row:
            import pickle
            saved = pickle.loads(row[0])
            raw_msgs = saved.get("channel_values", {}).get("messages", []) if isinstance(saved, dict) else []
            _dbg("CHECKPOINT_DATA", {"saved_keys": list(saved.keys()) if isinstance(saved, dict) else type(saved).__name__, "raw_msgs_len": len(raw_msgs)})
            for m in raw_msgs:
                if hasattr(m, "type"):
                    msg_type = m.type
                else:
                    msg_type = type(m).__name__
                content = m.content if hasattr(m, "content") else str(m)
                _dbg("MSG_EXTRACT", {"type": msg_type, "content_len": len(content), "content_preview": content[:80]})
                tool_calls = None
                if hasattr(m, "tool_calls") and m.tool_calls:
                    tool_calls = [
                        {"id": tc.id, "name": tc.name, "args": tc.args}
                        for tc in m.tool_calls
                    ]
                messages.append({"type": msg_type, "content": content, "tool_calls": tool_calls})
    except Exception as e:
        _dbg("CHECKPOINT_ERROR", {"error": str(e)})
        logger.warning(f"[session_detail] messages: {e}")

    # 2) 取工具事件
    tool_events: dict[int, dict] = {}
    try:
        cursor = conn.execute(
            """
            SELECT turn_index, step_index, seq, event_type, tool_name, tool_input, tool_result, ts
              FROM tool_events
             WHERE thread_id = ?
             ORDER BY turn_index, step_index, seq
            """,
            (thread_id,),
        )
        _dbg("TOOL_EVENTS_QUERY", {"row_count": cursor.fetchone().__len__() if False else -1})
        # re-execute since fetchone above consumed
        cursor = conn.execute(
            """
            SELECT turn_index, step_index, seq, event_type, tool_name, tool_input, tool_result, ts
              FROM tool_events
             WHERE thread_id = ?
             ORDER BY turn_index, step_index, seq
            """,
            (thread_id,),
        )
        for row in cursor:
            turn_i, step_i, seq_i, ev_type, tname, tinput, tresult, ts = row
            _dbg("TOOL_ROW", {"turn_i": turn_i, "step_i": step_i, "seq_i": seq_i, "ev_type": ev_type, "tname": tname})
            key = (turn_i, step_i)
            if key not in tool_events:
                tool_events[key] = {"turn": turn_i, "step": step_i, "tools": [], "ts": ts}
            tool_events[key]["tools"].append({
                "type": ev_type,
                "name": tname,
                "input": tinput,
                "result": tresult,
            })
        _dbg("TOOL_EVENTS_DONE", {"distinct_keys": len(tool_events), "total_tools": sum(len(v["tools"]) for v in tool_events.values())})
    except Exception as e:
        _dbg("TOOL_EVENTS_ERROR", {"error": str(e)})
        logger.warning(f"[session_detail] tool_events: {e}")

    conn.close()
    _dbg("EXIT", {"messages_count": len(messages), "tool_events_count": len(tool_events), "H1_check": "turn_i values above tell us if H1 is true"})
    return {"messages": messages, "tool_events": list(tool_events.values())}


def list_threads(checkpointer: Any = None) -> list[dict]:
    """
    列出所有会话（优先读 sessions 表，fallback 到 checkpoints）。

    Returns:
        [{thread_id, title, message_count, created_at, updated_at}, ...]
    """
    db_path = str(CHECKPOINT_PATH)
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        # 优先从 sessions 表读
        try:
            cursor = conn.execute(
                """
                SELECT thread_id, title, message_count, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT 50
                """
            )
            rows = cursor.fetchall()
            conn.close()
            if rows:
                return [
                    {
                        "thread_id": r[0],
                        "title": r[1] or "",
                        "message_count": r[2] or 0,
                        "created_at": r[3],
                        "updated_at": r[4],
                    }
                    for r in rows
                ]
        except Exception:
            pass
        # Fallback：读 checkpoints 表
        cursor = conn.execute(
            """
            SELECT thread_id, MIN(created_at) as created_at, MAX(updated_at) as updated_at
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY updated_at DESC
            LIMIT 50
            """
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "thread_id": r[0],
                "title": "",
                "message_count": 0,
                "created_at": r[1],
                "updated_at": r[2],
            }
            for r in rows
        ]
    except Exception as e:
        logger.warning(f"无法列出 threads: {e}")
        return []
