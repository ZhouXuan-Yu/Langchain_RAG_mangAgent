"""Graph state definition — 05-16+27: LangGraph AgentState (Extended for Crayfish Multi-Agent)."""

from typing import TypedDict, Annotated, Any, Sequence
from operator import add as operator_add

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.memory.memory_schema import MemoryRecord


def add_messages(left: list[BaseMessage], right: list[BaseMessage]) -> list[BaseMessage]:
    """
    自定义消息合并函数，替代 langchain_core.messages.add_messages。
    将右侧消息追加到左侧列表。
    """
    return left + right


# ── Data Context Entry (带置信度标记) ────────────────────────────────────────
class DataEntry(TypedDict):
    """带置信度标记的数据条目 — 用于解决数据冲突。"""
    content: str              # 数据内容
    source: str               # "web" | "rag" | "memory" | "reasoning"
    timestamp: str            # ISO 格式时间戳
    confidence: float         # 0.0 - 1.0 置信度
    task_id: str | None       # 关联的任务 ID


# ── Task Item (JSON Plan 中的任务项) ──────────────────────────────────────────
class TaskItem(TypedDict):
    """Supervisor 生成的单个任务项。"""
    task_id: str
    description: str
    assigned_agent: str        # "search_worker" | "rag_worker" | "coder"
    status: str                # "pending" | "running" | "completed" | "rejected"
    result: str | None
    quality_score: float       # 0-10 质量评分


class AgentState(TypedDict):
    """
    LangGraph Agent 完整状态定义 (Crayfish Multi-Agent Extended).

    包含：
    - messages: 对话历史（通过 add_messages 合并）
    - thread_id: 会话隔离 ID
    - memory_context: ChromaDB 检索到的记忆上下文
    - web_context: 网页搜索结果
    - pending_memory: 待评估存入的新事实列表
    - memory_updated: 本轮是否更新了记忆
    - last_tool_result: 上次工具调用结果
    - turn_count: 回合计数器（防止无限循环）
    - route_decision: 当前路由决策（debug 用）
    ─────────────────────────────────────────────────────────────────
    Crayfish Multi-Agent 新增字段：
    - plan_list: 动态生成的任务清单 (TaskItem list)
    - data_context: 存储搜索到的中间数据 + 来源标签 + 时间戳
    - next_step: 指示下一个该由谁执行
    - loop_counter: 递归计数器（超过 15 强制中断）
    - quality_score: 当前产出的质量评分 (0-10)
    - self_healing_attempts: 自修复重试次数（超过 2 次则放弃）
    - current_worker: 当前执行的 worker 名称（前端展示用）
    - error_log: 最近一次的错误日志
    - human_decision: 人工介入决策（approve/reject/modify）
    """

    # 对话消息历史，自动追加合并
    messages: Annotated[list[BaseMessage], add_messages]

    # 会话隔离
    thread_id: str

    # 外部知识上下文
    memory_context: Annotated[list[str], operator_add]
    web_context: Annotated[list[str], operator_add]

    # 主动记忆
    pending_memory: list[MemoryRecord]

    # 状态标志
    memory_updated: bool
    last_tool_result: str | None
    turn_count: int
    route_decision: str | None

    # ── Crayfish Multi-Agent 新增字段 ────────────────────────────────────

    # 动态任务清单（Supervisor 生成的 JSON Plan）
    plan_list: Annotated[list[TaskItem], operator_add]

    # 中间数据上下文（带置信度标记，解决数据冲突）
    data_context: Annotated[list[DataEntry], operator_add]

    # 下一步路由指示
    next_step: str | None

    # 递归计数器（防止死亡循环，超过 15 强制中断）
    loop_counter: int

    # 质量评分（Reviewer 评估，≥8 分通过）
    quality_score: float

    # 自修复重试次数（超过 2 次放弃）
    self_healing_attempts: int

    # 当前执行的 Worker 名称（前端展示用）
    current_worker: str | None

    # 错误日志（用于 self-healing）
    error_log: str | None

    # 人工介入决策
    human_decision: str | None
