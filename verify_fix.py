"""Verify fix works."""
import asyncio, os, sqlite3
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
from operator import add as operator_add
from langchain_core.messages import BaseMessage

class TestState(TypedDict):
    messages: Annotated[list[BaseMessage], add]

async def test():
    print("=== Test: With fix (setup called) - do checkpoints persist? ===")

    print("1. Getting async checkpointer with fix...")
    # Import from the fixed module
    from src.memory.sqlite_store import get_async_sqlite_checkpointer
    saver = await get_async_sqlite_checkpointer()
    print(f"   is_setup={saver.is_setup}")

    print("2. Building minimal graph...")
    builder = StateGraph(TestState)
    builder.add_node("test", lambda s: {"messages": [HumanMessage(content="response")]})
    builder.add_edge("__start__", "test")
    graph = builder.compile(checkpointer=saver)

    print("3. Running graph...")
    config = {"configurable": {"thread_id": "fix_test_001"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hello")]},
        config=config
    )
    print(f"   Result messages: {len(result.get('messages', []))}")

    print("4. Checking DB...")
    db_path = "data/checkpointer/checkpoints.db"
    size = os.path.getsize(db_path)
    print(f"   Size: {size} bytes")
    conn2 = sqlite3.connect(db_path)
    cur = conn2.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print(f"   Tables: {tables}")
    for tbl in tables:
        cur = conn2.execute(f"SELECT COUNT(*) FROM {tbl}")
        count = cur.fetchone()[0]
        print(f"   {tbl}: {count} rows")
        if count > 0:
            cur = conn2.execute(f"SELECT thread_id, checkpoint_id FROM {tbl} LIMIT 3")
            for row in cur.fetchall():
                print(f"     thread_id={row[0]}, checkpoint_id={row[1]}")
    conn2.close()

    if size > 0:
        print("\nSUCCESS: Checkpoints are being saved!")
    else:
        print("\nFAILURE: DB still empty!")

asyncio.run(test())
