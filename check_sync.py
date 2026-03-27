"""Check if SqliteSaver.put auto-calls setup, and if AsyncSqliteSaver.put does too."""
import inspect
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

print("=== SqliteSaver.put source ===")
src = inspect.getsource(SqliteSaver.put)
print(src)

print("\n=== AsyncSqliteSaver.put source ===")
src = inspect.getsource(AsyncSqliteSaver.put)
print(src)
