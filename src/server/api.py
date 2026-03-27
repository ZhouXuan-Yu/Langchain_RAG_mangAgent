"""API 路由 — 所有 HTTP 端点，包括 SSE 流式对话."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, File, Form, HTTPException, Header, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from src.server.models import (
    AgentCreateRequest,
    AgentProfile,
    AgentTaskHistory,
    AgentUpdateRequest,
    ChatRequest,
    ConfigInfo,
    CostHistoryEntry,
    CostHistoryReport,
    CostReport,
    Document,
    DocumentListResponse,
    DocumentUploadResponse,
    MemoryItem,
    MemorySaveRequest,
    ModelSwitchRequest,
    SessionSummary,
    SessionUpdateRequest,
    TaskBatchCreateRequest,
    TaskCreateRequest,
    TaskJob,
    TaskListResponse,
    TaskUpdateRequest,
)
from src.server.dependencies import get_agent, switch_model, get_current_model, get_checkpointer
from src.tools.memory_tools import get_memory_store
from src.utils.token_tracker import TokenTracker
from src.config import (
    DEFAULT_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    USER_NAME,
    USER_TECH_STACK,
    USER_HARDWARE,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# ── 全局 Token 追踪器 ──────────────────────────────────────────────────────────
_token_tracker = TokenTracker(model=DEFAULT_MODEL)

# ── 可用模型列表 ───────────────────────────────────────────────────────────────
AVAILABLE_MODELS = [
    "deepseek-chat",       # DeepSeek V3
    "deepseek-reasoner",   # DeepSeek R1
]


# ═══════════════════════════════════════════════════════════════════════════════
#  对话端点（核心）
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE 流式对话 — 核心端点."""

    async def event_generator() -> AsyncGenerator[str, None]:
        model = req.model or get_current_model()
        agent = await get_agent(model)
        config = {"configurable": {"thread_id": req.thread_id}}

        accumulated_content = ""

        try:
            async for event in agent.astream_events(
                {
                    "messages": [HumanMessage(content=req.message)],
                    "thread_id": req.thread_id,
                },
                config=config,
                version="v2",
            ):
                event_type = event.get("event", "")

                if event_type == "chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        accumulated_content += chunk.content
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

                elif event_type == "tool":
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})
                    yield f"data: {json.dumps({'type': 'tool', 'name': tool_name, 'input': tool_input})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'content': accumulated_content})}\n\n"

            try:
                _token_tracker.record(
                    prompt_tokens=0,
                    completion_tokens=_token_tracker.count(accumulated_content),
                    model=model,
                    label="chat_stream",
                )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"chat_stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  记忆端点
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/memory", response_model=list[MemoryItem])
async def list_memory() -> list[MemoryItem]:
    """获取所有 ChromaDB 记忆条目."""
    try:
        store = get_memory_store()
        memories = store.get_all()
        return [
            MemoryItem(
                id=m["id"],
                content=m["metadata"].get("content", ""),
                category=m["metadata"].get("category", "general"),
                importance=m["metadata"].get("importance", 5),
                timestamp=m["metadata"].get("timestamp", ""),
            )
            for m in memories
        ]
    except Exception as e:
        logger.error(f"list_memory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory")
async def save_memory(req: MemorySaveRequest) -> dict:
    """手动保存一条记忆."""
    try:
        store = get_memory_store()
        record_id = store.upsert(
            content=req.content,
            metadata={
                "category": req.category,
                "importance": req.importance,
                "timestamp": datetime.now().isoformat(),
            },
        )
        return {"status": "ok", "result": record_id}
    except Exception as e:
        logger.error(f"save_memory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str) -> dict:
    """删除指定 ID 的记忆."""
    try:
        store = get_memory_store()
        success = store.delete(memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"delete_memory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  会话管理端点
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/chat/reset")
async def reset_session(req: ChatRequest) -> dict:
    """重置会话：删除 SQLite checkpoint 数据."""
    try:
        checkpointer = await get_checkpointer()
        config = {"configurable": {"thread_id": req.thread_id}}
        await checkpointer.adelete(config)
        return {"status": "ok", "new_thread_id": req.thread_id}
    except Exception as e:
        logger.error(f"reset_session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{thread_id}")
async def get_session(thread_id: str) -> dict:
    """获取某会话的完整消息历史."""
    try:
        checkpointer = await get_checkpointer()
        config = {"configurable": {"thread_id": thread_id}}
        saved = await checkpointer.aget(config)
        if not saved:
            return {"messages": []}
        messages = saved.get("channel_values", {}).get("messages", [])
        return {
            "messages": [
                {
                    "type": type(m).__name__,
                    "content": m.content if hasattr(m, "content") else str(m),
                }
                for m in messages
            ]
        }
    except Exception as e:
        logger.error(f"get_session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions() -> list[SessionSummary]:
    """列出所有会话（基于 SQLite threads 表）."""
    try:
        from src.memory.sqlite_store import get_sqlite_checkpointer
        checkpointer = get_sqlite_checkpointer()
        threads = checkpointer.list_threads(limit=50)
        return [
            SessionSummary(
                thread_id=t.get("thread_id", ""),
                title=t.get("title"),
                message_count=t.get("message_count", 0),
                created_at=t.get("created_at"),
                updated_at=t.get("updated_at"),
                is_archived=False,
            )
            for t in threads
        ]
    except Exception as e:
        logger.error(f"list_sessions error: {e}")
        return []


@router.patch("/sessions/{thread_id}")
async def update_session(thread_id: str, req: SessionUpdateRequest) -> dict:
    """更新会话标题或归档状态."""
    # 当前 SQLite store 不支持更新，这里预留接口
    return {"status": "ok", "thread_id": thread_id}


# ═══════════════════════════════════════════════════════════════════════════════
#  配置与模型切换端点
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/config", response_model=ConfigInfo)
async def get_config() -> ConfigInfo:
    """获取当前配置."""
    return ConfigInfo(
        available_models=AVAILABLE_MODELS,
        current_model=get_current_model(),
        user_name=USER_NAME,
        user_tech_stack=USER_TECH_STACK,
        user_hardware=USER_HARDWARE,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )


@router.post("/model/switch")
async def model_switch(req: ModelSwitchRequest) -> dict:
    """动态切换 LLM 模型."""
    if req.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")
    result = switch_model(req.model)
    return result


@router.get("/cost", response_model=CostReport)
async def get_cost() -> CostReport:
    """获取 Token 使用量报告."""
    s = _token_tracker.summary()
    return CostReport(**s)


@router.get("/cost/history", response_model=CostHistoryReport)
async def get_cost_history(days: int = 7) -> CostHistoryReport:
    """获取 Token 成本历史."""
    entries = _token_tracker.get_history(days=days)
    return CostHistoryReport(
        entries=[
            CostHistoryEntry(
                timestamp=e.get("timestamp", ""),
                date=e.get("date", ""),
                prompt_tokens=e.get("prompt_tokens", 0),
                completion_tokens=e.get("completion_tokens", 0),
                total_tokens=e.get("total_tokens", 0),
                num_calls=e.get("num_calls", 0),
                cost_usd=e.get("cost_usd", 0.0),
                model=e.get("model", ""),
            )
            for e in entries
        ],
        total_cost_usd=sum(e.cost_usd for e in entries),
        total_tokens=sum(e.total_tokens for e in entries),
        total_calls=sum(e.num_calls for e in entries),
        period_start=entries[-1].date if entries else "",
        period_end=entries[0].date if entries else "",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  多 Agent 管理端点
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/agents", response_model=list[AgentProfile])
async def list_agents(is_active: Optional[bool] = None) -> list[AgentProfile]:
    """列出所有 Agent."""
    from src.multi_agent.orchestrator import get_agent_registry
    reg = get_agent_registry()
    agents = reg.list(is_active=is_active)
    return [AgentProfile(**a) for a in agents]


@router.post("/agents", response_model=AgentProfile)
async def create_agent(req: AgentCreateRequest) -> AgentProfile:
    """注册新 Agent."""
    from src.multi_agent.orchestrator import get_agent_registry
    reg = get_agent_registry()
    agent = reg.create(
        name=req.name,
        role=req.role,
        description=req.description,
        model=req.model,
        color=req.color,
    )
    return AgentProfile(**agent)


@router.get("/agents/{agent_id}", response_model=AgentProfile)
async def get_agent_detail(agent_id: str) -> AgentProfile:
    """获取 Agent 详情 + 任务历史."""
    from src.multi_agent.orchestrator import get_agent_registry
    from src.server.task_scheduler import get_scheduler
    reg = get_agent_registry()
    scheduler = get_scheduler()

    agent = reg.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    tasks = scheduler.list(agent_id=agent_id, limit=20)
    done = sum(1 for t in tasks if t["status"] == "done")
    failed = sum(1 for t in tasks if t["status"] == "failed")

    return AgentTaskHistory(
        agent_id=agent["id"],
        agent_name=agent["name"],
        total_tasks=len(tasks),
        done=done,
        failed=failed,
        pending=len(tasks) - done - failed,
        recent_tasks=[TaskJob(**t) for t in tasks[:10]],
    )


@router.patch("/agents/{agent_id}", response_model=AgentProfile)
async def update_agent(agent_id: str, req: AgentUpdateRequest) -> AgentProfile:
    """更新 Agent 配置."""
    from src.multi_agent.orchestrator import get_agent_registry
    reg = get_agent_registry()
    agent = reg.update(
        agent_id,
        name=req.name,
        role=req.role,
        description=req.description,
        model=req.model,
        color=req.color,
        is_active=req.is_active,
    )
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentProfile(**agent)


@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str) -> dict:
    """删除 Agent（软删除）."""
    from src.multi_agent.orchestrator import get_agent_registry
    reg = get_agent_registry()
    ok = reg.delete(agent_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot delete this agent")
    return {"status": "ok"}


@router.post("/agents/{agent_id}/dispatch")
async def dispatch_to_agent(agent_id: str, message: str, thread_id: Optional[str] = None) -> dict:
    """分发给 Agent 执行任务（同步等待结果）."""
    from src.multi_agent.orchestrator import get_agent_registry
    reg = get_agent_registry()
    result = await reg.dispatch_task(agent_id, message, thread_id)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  任务队列端点
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> TaskListResponse:
    """列出所有任务（支持过滤）."""
    from src.server.task_scheduler import get_scheduler
    scheduler = get_scheduler()
    tasks = scheduler.list(status=status, agent_id=agent_id)
    counts = scheduler.counts()
    return TaskListResponse(
        tasks=[TaskJob(**t) for t in tasks],
        total=len(tasks),
        **counts,
    )


@router.post("/tasks", response_model=TaskJob)
async def create_task(req: TaskCreateRequest) -> TaskJob:
    """创建新任务."""
    from src.server.task_scheduler import get_scheduler
    scheduler = get_scheduler()

    async def _execute(task_id: str, description: str):
        from src.multi_agent.orchestrator import get_agent_registry
        reg = get_agent_registry()
        if req.agent_id:
            result = await reg.dispatch_task(req.agent_id, description, thread_id=task_id)
            return result.get("result", str(result))
        else:
            # 默认：模拟执行
            await asyncio.sleep(2)
            return f"Task completed: {description}"

    task = await scheduler.enqueue_and_run(
        title=req.title,
        description=req.description,
        priority=req.priority,
        agent_id=req.agent_id,
        depends_on=req.depends_on,
        executor_fn=_execute,
    )
    return TaskJob(**task)


@router.post("/tasks/batch")
async def create_tasks_batch(req: TaskBatchCreateRequest) -> list[TaskJob]:
    """批量创建任务（多 Agent 并行执行）."""
    from src.server.task_scheduler import get_scheduler
    scheduler = get_scheduler()

    results = []
    for task_req in req.tasks:
        async def _exec(tid, desc):
            from src.multi_agent.orchestrator import get_agent_registry
            reg = get_agent_registry()
            if task_req.agent_id:
                r = await reg.dispatch_task(task_req.agent_id, desc, thread_id=tid)
                return r.get("result", str(r))
            await asyncio.sleep(2)
            return f"Batch task done: {desc}"

        t = await scheduler.enqueue_and_run(
            title=task_req.title,
            description=task_req.description,
            priority=task_req.priority,
            agent_id=task_req.agent_id,
            depends_on=task_req.depends_on,
            executor_fn=_exec,
        )
        results.append(TaskJob(**t))

    return results


@router.get("/tasks/{task_id}", response_model=TaskJob)
async def get_task(task_id: str) -> TaskJob:
    """获取任务详情."""
    from src.server.task_scheduler import get_scheduler
    scheduler = get_scheduler()
    task = scheduler.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskJob(**task)


@router.patch("/tasks/{task_id}", response_model=TaskJob)
async def update_task(task_id: str, req: TaskUpdateRequest) -> TaskJob:
    """更新任务状态/优先级."""
    from src.server.task_scheduler import get_scheduler
    scheduler = get_scheduler()
    task = scheduler.update(
        task_id,
        title=req.title,
        description=req.description,
        status=req.status,
        priority=req.priority,
        agent_id=req.agent_id,
        depends_on=req.depends_on,
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskJob(**task)


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str) -> dict:
    """删除任务."""
    from src.server.task_scheduler import get_scheduler
    scheduler = get_scheduler()
    ok = scheduler.delete(task_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "ok"}


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> dict:
    """取消运行中的任务."""
    from src.server.task_scheduler import get_scheduler
    scheduler = get_scheduler()
    ok = await scheduler.cancel(task_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot cancel this task")
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════════════
#  知识库 / 文档上传端点
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(doc_type: Optional[str] = None) -> DocumentListResponse:
    """列出所有已上传文档."""
    from src.document_processor import get_document_store
    store = get_document_store()
    docs = store.list(doc_type=doc_type)
    return DocumentListResponse(
        documents=[Document(**d) for d in docs],
        total=len(docs),
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    """上传文档：提取文本、分块、注入 ChromaDB."""
    from src.document_processor import (
        get_document_store,
        detect_doc_type,
        extract_text,
        chunk_text,
    )

    try:
        content = await file.read()
        filename = file.filename or "unknown"
        doc_type = detect_doc_type(filename)
        size = len(content)

        store = get_document_store()
        doc = store.save(filename=filename, doc_type=doc_type, size_bytes=size, content=content)

        # 后台处理文本提取和 ChromaDB 注入
        try:
            text = extract_text(content, doc_type)
            chunks = chunk_text(text)
            store.add_chunks(doc["id"], chunks)
            count = store.ingest_to_chroma(doc["id"])
            doc = store.get(doc["id"])
        except Exception as e:
            logger.error(f"[upload] processing error: {e}")
            store.update_status(doc["id"], "failed", error=str(e))
            doc = store.get(doc["id"])

        return DocumentUploadResponse(
            document=Document(**doc),
            message=f"Uploaded and processed: {len(chunks) if 'chunks' in dir() else 0} chunks",
        )
    except Exception as e:
        logger.error(f"[upload] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}", response_model=Document)
async def get_document(doc_id: str) -> Document:
    """获取文档详情."""
    from src.document_processor import get_document_store
    store = get_document_store()
    doc = store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return Document(**doc)


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    """删除文档."""
    from src.document_processor import get_document_store
    store = get_document_store()
    ok = store.delete(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "ok"}
