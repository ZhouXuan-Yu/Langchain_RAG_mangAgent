"""Pydantic models — HTTP 请求与响应数据结构."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
#  对话 / 记忆
# ═══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """POST /api/chat/stream 的请求体."""
    message: str = Field(..., description="用户输入消息")
    thread_id: str = Field(default="default", description="会话 ID")
    image_url: Optional[str] = Field(default=None, description="图片 URL（可选）")
    model: Optional[str] = Field(default=None, description="模型名称（可选，None 使用默认）")


class MemoryItem(BaseModel):
    """记忆条目（用于展示 / 手动保存）."""
    id: str
    content: str
    category: str
    importance: int
    timestamp: str


class MemorySaveRequest(BaseModel):
    """POST /api/memory 的请求体."""
    content: str
    category: str = "general"
    importance: int = Field(default=5, ge=1, le=10)


class ConfigInfo(BaseModel):
    """GET /api/config 返回的当前配置."""
    available_models: list[str]
    current_model: str
    user_name: str
    user_tech_stack: list[str]
    user_hardware: str
    temperature: float
    max_tokens: int


class ModelSwitchRequest(BaseModel):
    """POST /api/model/switch 的请求体."""
    model: str = Field(..., description="要切换到的模型名称")


class CostReport(BaseModel):
    """GET /api/cost 返回的 Token 使用报告."""
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float
    num_calls: int


# ═══════════════════════════════════════════════════════════════════════════════
#  会话管理
# ═══════════════════════════════════════════════════════════════════════════════

class SessionSummary(BaseModel):
    """会话摘要（用于列表展示）."""
    thread_id: str
    title: Optional[str] = None
    message_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    is_archived: bool = False


class SessionUpdateRequest(BaseModel):
    """PATCH /api/sessions/{id} 的请求体."""
    title: Optional[str] = None
    is_archived: Optional[bool] = None


class CostHistoryEntry(BaseModel):
    """成本历史条目."""
    timestamp: str
    date: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    num_calls: int
    cost_usd: float
    model: str


class CostHistoryReport(BaseModel):
    """GET /api/cost/history 返回的成本历史."""
    entries: list[CostHistoryEntry]
    total_cost_usd: float
    total_tokens: int
    total_calls: int
    period_start: str
    period_end: str


# ═══════════════════════════════════════════════════════════════════════════════
#  多 Agent 管理
# ═══════════════════════════════════════════════════════════════════════════════

class AgentProfile(BaseModel):
    """Agent 配置档案."""
    id: str
    name: str
    role: str
    description: str
    model: str
    color: str = "#888888"
    is_active: bool = True
    tasks_completed: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class AgentCreateRequest(BaseModel):
    """POST /api/agents 的请求体."""
    name: str
    role: str
    description: str = ""
    model: str = "deepseek-chat"
    color: str = "#888888"


class AgentUpdateRequest(BaseModel):
    """PATCH /api/agents/{id} 的请求体."""
    name: Optional[str] = None
    role: Optional[str] = None
    description: Optional[str] = None
    model: Optional[str] = None
    color: Optional[str] = None
    is_active: Optional[bool] = None


class AgentTaskHistory(BaseModel):
    """Agent 的任务历史."""
    agent_id: str
    agent_name: str
    total_tasks: int
    done: int
    failed: int
    pending: int
    recent_tasks: list["TaskJob"]


# ═══════════════════════════════════════════════════════════════════════════════
#  任务队列
# ═══════════════════════════════════════════════════════════════════════════════

class TaskJob(BaseModel):
    """任务条目."""
    id: str
    title: str
    description: str = ""
    status: str = "pending"  # pending | running | done | failed | cancelled
    priority: int = 5
    agent_id: Optional[str] = None
    depends_on: list[str] = []
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class TaskCreateRequest(BaseModel):
    """POST /api/tasks 的请求体."""
    title: str
    description: str = ""
    agent_id: Optional[str] = None
    priority: int = Field(default=5, ge=1, le=10)
    depends_on: list[str] = []


class TaskBatchCreateRequest(BaseModel):
    """POST /api/tasks/batch 的请求体 — 批量创建任务."""
    tasks: list[TaskCreateRequest]


class TaskUpdateRequest(BaseModel):
    """PATCH /api/tasks/{id} 的请求体."""
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[int] = Field(default=None, ge=1, le=10)
    agent_id: Optional[str] = None
    depends_on: Optional[list[str]] = None


class TaskListResponse(BaseModel):
    """任务列表响应."""
    tasks: list[TaskJob]
    total: int
    pending: int
    running: int
    done: int
    failed: int


# ═══════════════════════════════════════════════════════════════════════════════
#  知识库 / 文档上传
# ═══════════════════════════════════════════════════════════════════════════════

class Document(BaseModel):
    """文档条目."""
    id: str
    filename: str
    doc_type: str  # pdf | image | docx | txt | markdown | code | csv
    size_bytes: int = 0
    chunk_count: int = 0
    status: str = "uploading"  # uploading | processing | ready | failed
    error: Optional[str] = None
    uploaded_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class DocumentUploadResponse(BaseModel):
    """POST /api/documents/upload 的响应体."""
    document: Document
    message: str


class DocumentListResponse(BaseModel):
    """文档列表响应."""
    documents: list[Document]
    total: int


# 解决 forward reference
AgentTaskHistory.model_rebuild()
