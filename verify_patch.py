"""Monkey-patch put to verify if LangGraph actually calls it."""
import os, sqlite3, asyncio

# Patch BEFORE any other imports
_original_put = None
_put_calls = []

def patched_put(self, config, checkpoint, metadata, new_versions):
    _put_calls.append({
        'thread_id': config.get('configurable', {}).get('thread_id', '?'),
        'checkpoint_keys': list(checkpoint.get('channel_values', {}).keys()) if isinstance(checkpoint, dict) else '?',
    })
    print(f">>> [PATCHED put] thread_id={config.get('configurable', {}).get('thread_id', '?')}, "
          f"keys={list(checkpoint.get('channel_values', {}).keys()) if isinstance(checkpoint, dict) else '?'}")
    return _original_put(self, config, checkpoint, metadata, new_versions)

# Patch AsyncSqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
_original_put = AsyncSqliteSaver.put
AsyncSqliteSaver.put = patched_put

# Now run the test
async def test():
    from langgraph.graph import StateGraph
    from langchain_core.messages import HumanMessage
    from typing import TypedDict, Annotated
    from operator import add as operator_add
    from langchain_core.messages import BaseMessage
    import aiosqlite

    class TestState(TypedDict):
        messages: Annotated[list[BaseMessage], add]

    print("=== Monkey-patch test: Does LangGraph call put()? ===\n")

    db_file = "data/checkpointer/test_patch.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    conn = await aiosqlite.connect(db_file)
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    saver = AsyncSqliteSaver(conn)

    print(f"Before compile: put_calls={len(_put_calls)}")

    builder = StateGraph(TestState)
    builder.add_node("test", lambda s: {"messages": [HumanMessage(content="response")]})
    builder.add_edge("__start__", "test")
    graph = builder.compile(checkpointer=saver)

    print(f"After compile: put_calls={len(_put_calls)}")

    # Also patch the instance's class (since put is a method)
    instance_class = type(saver)
    instance_class.put = patched_put

    config = {"configurable": {"thread_id": "patch_test_001"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hello")]},
        config=config
    )

    print(f"After ainvoke: put_calls={len(_put_calls)}")
    print(f"Result messages: {len(result.get('messages', []))}")

    # Check DB
    conn2 = sqlite3.connect(db_file)
    cur = conn2.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print(f"\nDB tables: {tables}")
    for tbl in tables:
        cur = conn2.execute(f"SELECT COUNT(*) FROM {tbl}")
        count = cur.fetchone()[0]
        print(f"  {tbl}: {count} rows")
    conn2.close()
    await conn.close()

    if os.path.exists(db_file):
        os.remove(db_file)

asyncio.run(test())
