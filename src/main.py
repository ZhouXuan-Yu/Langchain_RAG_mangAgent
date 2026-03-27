"""Main entry script — 主入口：对话循环 + 状态恢复."""

import logging
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 必须先加载配置（验证 API key）
from src.config import CHECKPOINT_PATH, CHROMA_PATH, USER_NAME

# LangSmith 追踪（可选）
try:
    from src.supervision.langsmith_client import setup_langsmith
    setup_langsmith()
except Exception:
    pass

# 基础设施
from src.llm import init_deepseek_llm
from src.graph.prompt import build_system_prompt
from src.memory import get_sqlite_checkpointer
from src.graph import build_agent_graph, build_react_agent
from src.utils import TokenTracker

# 确保数据目录存在
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def print_welcome() -> None:
    print(f"""
╔══════════════════════════════════════════════╗
║   LangGraph RAG Agent — DeepSeek + ChromaDB  ║
║   用户: {USER_NAME:<33s}║
║   输入 'quit' 或 'exit' 退出                   ║
║   输入 'reset' 重置当前会话                    ║
║   输入 'cost' 查看 Token 成本报告              ║
║   输入 'memory' 查看所有长期记忆               ║
╚══════════════════════════════════════════════╝
""")


def main() -> None:
    """主循环：初始化组件 -> 对话循环 -> 事件流处理."""
    print_welcome()

    # 1. 初始化 LLM
    logger.info("初始化 DeepSeek LLM...")
    llm = init_deepseek_llm(streaming=True)

    # 2. 初始化 SqliteSaver（状态持久化 + 断电恢复）
    logger.info("初始化 SqliteSaver checkpointer...")
    checkpointer = get_sqlite_checkpointer()

    # 3. 构建 Agent Graph
    logger.info("构建 LangGraph Agent...")
    agent = build_agent_graph(llm, checkpointer=checkpointer)

    # 4. 初始化 Token 追踪器
    tracker = TokenTracker()

    # 5. 对话循环
    thread_id = "default_session"
    config = {"configurable": {"thread_id": thread_id}}

    # 检查是否有恢复的会话
    try:
        saved = checkpointer.get(config)
        if saved and "messages" in saved.get("channel_values", {}):
            msg_count = len(saved["channel_values"]["messages"])
            print(f"[系统] 检测到已保存的会话（{msg_count} 条消息），已恢复。\n")
    except Exception:
        pass

    print("Assistant: 你好！有什么我可以帮助你的吗？\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            tracker.print_report()
            print("再见！")
            break

        if user_input.lower() == "reset":
            print("Assistant: 已重置会话。\n")
            thread_id = f"session_{os.urandom(4).hex()}"
            config = {"configurable": {"thread_id": thread_id}}
            continue

        if user_input.lower() == "cost":
            tracker.print_report()
            continue

        if user_input.lower() == "memory":
            from src.memory.chroma_store import ChromaMemoryStore
            store = ChromaMemoryStore()
            all_memories = store.get_all()
            print(f"\n--- 长期记忆 ({len(all_memories)} 条) ---")
            for m in all_memories:
                print(f"  [{m['metadata'].get('category', '?')}] {m['content'][:80]}")
            print()
            continue

        # 6. 执行 Agent
        print("Assistant: ", end="", flush=True)
        try:
            response_content = ""
            for event in agent.stream(
                {"messages": [{"role": "user", "content": user_input}], "thread_id": thread_id},
                config=config,
                stream_mode="values",
            ):
                if "messages" in event:
                    last_msg = event["messages"][-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        content = last_msg.content
                        # 流式输出（逐字打印）
                        if hasattr(last_msg, "type") and last_msg.type == "ai":
                            print(content, end="", flush=True)
                            response_content = content
            print("\n")

            # 7. 记录 Token 使用
            try:
                tracker.record(
                    prompt_tokens=tracker.count(build_system_prompt()),
                    completion_tokens=tracker.count(response_content),
                    label="agent_response",
                )
            except Exception:
                pass

        except KeyboardInterrupt:
            print("\n[中断] 已暂停。输入 'quit' 退出。\n")
            continue
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            print(f"\n[系统错误] {e}\n")


if __name__ == "__main__":
    main()
