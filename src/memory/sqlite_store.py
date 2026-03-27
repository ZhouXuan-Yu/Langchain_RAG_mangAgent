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


def list_threads(checkpointer: Any) -> list[dict]:
    """
    列出 checkpointer 中所有保存的会话 thread.

    Returns:
        [{thread_id, updated_at, message_count}, ...]
    """
    try:
        # LangGraph SqliteSaver stores in a `checkpoints` table
        # Access via the internal session
        import sqlite3
        db_path = str(CHECKPOINT_PATH)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            """
            SELECT thread_id, created_at, updated_at
            FROM (SELECT thread_id, MIN(created_at) as created_at, MAX(updated_at) as updated_at
                  FROM checkpoints
                  GROUP BY thread_id)
            ORDER BY updated_at DESC
            LIMIT 50
            """
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "thread_id": r[0],
                "created_at": r[1],
                "updated_at": r[2],
            }
            for r in rows
        ]
    except Exception as e:
        logger.warning(f"无法列出 threads: {e}")
        return []
