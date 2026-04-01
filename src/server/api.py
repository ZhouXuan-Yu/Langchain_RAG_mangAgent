"""API 路由 — 所有 HTTP 端点，包括 SSE 流式对话."""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncGenerator, Optional


from fastapi import APIRouter, File, Form, HTTPException, Header, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage

from src.server.models import (
    AgentCreateRequest,
    AgentProfile,
    AgentTaskHistory,
    AgentUpdateRequest,
    ApiKeysUpdateRequest,
    ChatRequest,
    ConfigInfo,
    ConfigUpdateRequest,
    CostHistoryEntry,
    CostHistoryReport,
    CostReport,
    Document,
    DocumentChunkItem,
    DocumentListResponse,
    DocumentPreviewResponse,
    DocumentUploadResponse,
    KbSearchRequest,
    KbSearchResponse,
    KbSearchChunk,
    MemoryItem,
    MemorySaveRequest,
    ModelConfigRequest,
    ModelSwitchRequest,
    OrchestrateRequest,
    SessionSearchRequest,
    SessionSearchResponse,
    SessionSearchResult,
    SessionSummary,
    SessionUpdateRequest,
    TaskBatchCreateRequest,
    TaskCreateRequest,
    TaskJob,
    TaskListResponse,
    TaskUpdateRequest,
)
from src.server.dependencies import (
    get_agent,
    switch_model,
    get_current_model,
    get_checkpointer,
    _current_temperature,
    _current_max_tokens,
)
from src.tools.memory_tools import get_memory_store
from src.memory.sqlite_store import get_session_detail, save_tool_event, upsert_session, update_session_message_count
from src.utils.token_tracker import TokenTracker
from src.config import (
    DEFAULT_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    USER_NAME,
    USER_TECH_STACK,
    USER_HARDWARE,
    MAX_CONTEXT_TOKENS,
    MODEL_PROVIDERS,
    is_provider_configured,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

_SHUTDOWN_TRACKER: dict = {"pending_tasks": 0, "max_pending": 0}

# ── 全局 Token 追踪器 ──────────────────────────────────────────────────────────
_token_tracker = TokenTracker(model=DEFAULT_MODEL)

# ═══════════════════════════════════════════════════════════════════════════════
#  可用模型列表（从 MODEL_PROVIDERS 动态生成）───────────────────────────────────
def _build_available_models_response() -> list[dict]:
    """
    构建前端可用的模型列表，按 provider 分组。
    返回: [{
        provider_id: "deepseek",
        provider_name: "DeepSeek",
        models: [{id, name, description}, ...],
        configured: True/False
    }, ...]
    """
    result = []
    for pid, pcfg in MODEL_PROVIDERS.items():
        result.append({
            "provider_id": pid,
            "provider_name": pcfg["name"],
            "models": pcfg.get("default_models", []),
            "configured": is_provider_configured(pid),
        })
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  对话端点（核心）
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE 流式对话 — 实时推送 token 和工具调用事件。"""

    async def event_generator() -> AsyncGenerator[str, None]:
        model = req.model or get_current_model()
        agent = await get_agent(model)
        config = {"configurable": {"thread_id": req.thread_id}}

        # 首次发消息 → 创建/更新 session 记录
        thread_id = req.thread_id
        title_hint = req.message[:60] if req.message else ""
        await asyncio.to_thread(upsert_session, thread_id, title_hint)
        logger.info("[H1] upsert_session called thread_id=%s title_hint=%s", thread_id, title_hint[:30] if title_hint else "")

        # on_tool_start / on_tool_end 对同一 run_id 各应推送一次；此前共用 dedup 导致
        # tool_result 永远被跳过，前端卡片一直停在「等待结果」。
        seen_tool_starts: set[str] = set()
        seen_tool_ends: set[str] = set()

        # turn/step 计数器：用于记录会话页展开所需的工具调用次序
        cur_turn: int = 0
        cur_step: int = 0

        def _event_run_id(ev: dict) -> str:
            rid = ev.get("run_id")
            return str(rid) if rid else ""

        try:
            # ── 使用 astream_events (version="v2") 捕获所有流式事件 ─────────────
            # 每个事件: {"event": str, "run_id": str, "data": {...}, ...}
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=req.message)], "thread_id": req.thread_id},
                config=config,
                version="v2",
            ):
                ev_type = event.get("event", "")

                # ── LLM 开始生成 ───────────────────────────────────────────────────
                if ev_type in ("on_chat_model_start", "chat_model_start"):
                    pass  # noop

                # ── 流式 token ───────────────────────────────────────────────────
                elif ev_type in ("on_chat_model_stream", "chat_model_stream"):
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        if isinstance(content, list):
                            text_parts = []
                            for part in content:
                                if isinstance(part, str):
                                    text_parts.append(part)
                                elif isinstance(part, dict) and part.get("type") == "text":
                                    text_parts.append(part.get("text") or "")
                            content = "".join(text_parts)
                        if content.strip().startswith("Human:"):
                            cur_turn += 1
                            cur_step = 0
                        skip = (
                            content.strip() in ("Tool", "Tool/use", "Invoking tool:", "=" * 20)
                            or content.strip().startswith("=") and len(content.strip()) < 5
                        )
                        if not skip:
                            yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

                # ── 工具调用开始 ────────────────────────────────────────────────
                elif ev_type == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})
                    rid = _event_run_id(event)
                    dedup_key = rid or f"{tool_name}:{json.dumps(tool_input, ensure_ascii=False)[:120]}"
                    if dedup_key not in seen_tool_starts:
                        seen_tool_starts.add(dedup_key)
                        logger.info("[chat_stream] tool_start: %s", tool_name)
                        yield f"data: {json.dumps({'type': 'tool', 'name': tool_name, 'input': tool_input, 'result': ''})}\n\n"
                    # 持久化到 DB
                    cur_step += 1
                    await asyncio.to_thread(
                        save_tool_event,
                        thread_id, cur_turn, cur_step, 1,
                        "tool_start", tool_name,
                        json.dumps(tool_input, ensure_ascii=False)[:2000],
                        "",
                    )

                # ── 工具调用结束 ────────────────────────────────────────────────
                elif ev_type == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    raw_output = event.get("data", {}).get("output", "")
                    if isinstance(raw_output, dict):
                        tool_result = json.dumps(raw_output, ensure_ascii=False)
                    else:
                        tool_result = str(raw_output) if raw_output else ""
                    rid = _event_run_id(event)
                    dedup_key = rid or f"end:{tool_name}:{tool_result[:80]}"
                    if dedup_key not in seen_tool_ends:
                        seen_tool_ends.add(dedup_key)
                        yield f"data: {json.dumps({'type': 'tool_result', 'name': tool_name, 'result': tool_result})}\n\n"
                    # 持久化到 DB（seq=2 表示结果）
                    await asyncio.to_thread(
                        save_tool_event,
                        thread_id, cur_turn, cur_step, 2,
                        "tool_result", tool_name,
                        "", tool_result[:5000],
                    )

                # ── LLM 完整回复结束（捕获 token 用量并写入 tracker）───────────────
                elif ev_type in ("on_chat_model_end", "chat_model_end"):
                    output = event.get("data", {}).get("output")
                    if output is not None:
                        um = getattr(output, "usage_metadata", None) or {}
                        if isinstance(um, dict):
                            prompt_tokens = max(0, (um.get("input_tokens", 0) or 0) - ((um.get("input_token_details", {}) or {}).get("cache_read", 0) or 0))
                            completion_tokens = um.get("output_tokens", 0) or 0
                            rm = getattr(output, "response_metadata", None) or {}
                            model_name = (rm.get("model_name", "") if isinstance(rm, dict) else "") or req.model or get_current_model()
                            _token_tracker.record(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, model=model_name, label="llm_call")

            # ── 推送完成信号 ─────────────────────────────────────────────────
            yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
            # 更新会话消息计数
            await asyncio.to_thread(update_session_message_count, thread_id)

        except Exception as e:
            logger.error(f"chat_stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
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
    """获取某会话的完整消息历史（兼容旧接口）。"""
    detail = get_session_detail(thread_id)
    return {"messages": detail["messages"], "tool_events": detail["tool_events"]}


@router.get("/session/{thread_id}/detail")
async def get_session_detail_api(thread_id: str) -> dict:
    """获取某会话的完整详情：消息历史 + 工具调用事件（可折叠展示用）。"""
    try:
        detail = get_session_detail(thread_id)
        if not detail["messages"]:
            return {"messages": [], "tool_events": [], "turns": []}
        # 按用户轮次分 turn（兼容 LangChain：type 为 human/ai/tool 或类名）
        def _is_human(t: str | None) -> bool:
            return t in ("HumanMessage", "human")

        def _is_ai(t: str | None) -> bool:
            return t in ("AIMessage", "ai")

        def _is_tool(t: str | None) -> bool:
            return t in ("ToolMessage", "tool")

        turns: list[dict] = []
        current_turn: dict | None = None
        for msg in detail["messages"]:
            mt = msg.get("type")
            if _is_human(mt):
                if current_turn:
                    turns.append(current_turn)
                current_turn = {
                    "user": msg.get("content", ""),
                    "assistant": "",
                    "tools": [],
                    "msg_index": len(turns),
                }
            elif _is_ai(mt) and current_turn is not None:
                current_turn["assistant"] = msg.get("content", "")
            elif _is_tool(mt) and current_turn is not None:
                current_turn["tools"].append({
                    "name": "",
                    "result": msg.get("content", ""),
                })
        if current_turn:
            turns.append(current_turn)
        # 把 tool_events 合并进去（追加到已从 messages 提取的 tools 列表）
        for te in detail["tool_events"]:
            ti = te.get("turn", 0)
            if ti < len(turns):
                existing = turns[ti].get("tools") or []
                new_tools = te.get("tools") or []
                # 避免重复追加
                seen = set(t.get("name") or "" for t in existing)
                for nt in new_tools:
                    if (nt.get("name") or "") not in seen:
                        existing.append(nt)
                        seen.add(nt.get("name") or "")
                turns[ti]["tools"] = existing
        return {"messages": detail["messages"], "tool_events": detail["tool_events"], "turns": turns}
    except Exception as e:
        logger.error(f"get_session_detail error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions() -> list[SessionSummary]:
    """列出所有会话（基于 sessions 表）。"""
    try:
        from src.memory.sqlite_store import list_threads as list_threads_from_db
        threads = list_threads_from_db()
        logger.info("[H3] list_sessions API called, returning %d sessions", len(threads))
        return [
            SessionSummary(
                thread_id=t.get("thread_id", ""),
                title=t.get("title") or "",
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


@router.post("/sessions/search", response_model=SessionSearchResponse)
async def session_search(req: SessionSearchRequest) -> SessionSearchResponse:
    """
    对历史会话内容进行关键词 + 语义混合检索（RAG 搜索）。
    优先从 SQLite checkpoints 提取消息文本做关键词匹配，
    同时查询 ChromaDB conversation category 做向量相似度补充。
    """
    import time
    t0 = time.monotonic()
    kw = req.query.lower().strip()

    from src.memory.sqlite_store import list_threads as list_all_threads
    from src.memory.sqlite_store import get_session_detail as get_sess_detail
    from src.memory.chroma_store import ChromaMemoryStore

    # ── 1. SQLite 关键词搜索所有会话 ───────────────────────────
    all_threads = list_all_threads()
    scored: list[dict] = []
    for t in all_threads:
        tid = t.get("thread_id", "")
        if not tid:
            continue
        try:
            detail = get_sess_detail(tid)
        except Exception:
            continue

        msgs = detail.get("messages") or []
        # 拼接所有消息内容用于搜索
        msg_texts = []
        for m in msgs:
            c = m.get("content", "") or ""
            if isinstance(c, str) and c.strip():
                msg_texts.append(c.strip())
        full_text = "\n".join(msg_texts)
        title = t.get("title") or ("会话 " + tid[:8])

        if not kw:
            score = 1.0
            snippet = msg_texts[0][:200] if msg_texts else ""
        else:
            kw_lower = kw
            title_matches = kw_lower in title.lower()
            body_matches = kw_lower in full_text.lower()
            if not body_matches and not title_matches:
                continue
            # 优先标题命中，其次正文命中次数
            snippet_candidates = [ln for ln in msg_texts if kw_lower in ln.lower()]
            snippet = snippet_candidates[0][:200] if snippet_candidates else (msg_texts[0][:200] if msg_texts else "")
            score = 2.0 if title_matches else (1.0 + 0.1 * full_text.lower().count(kw_lower))

        scored.append({
            "thread_id": tid,
            "title": title,
            "snippet": snippet,
            "score": score,
            "message_count": t.get("message_count", 0) or 0,
            "updated_at": t.get("updated_at"),
        })

    # ── 2. ChromaDB 向量搜索（category=conversation）────────────
    if kw:
        try:
            store = ChromaMemoryStore()
            raw_vec = store.search(query=kw, top_k=req.top_k, category="conversation")
            for r in raw_vec:
                sim = float(r.get("similarity", 0))
                if sim < req.min_score:
                    continue
                meta = r.get("metadata") or {}
                tid = meta.get("thread_id", "")
                if not tid:
                    continue
                # 避免重复（已有关键词结果的覆盖分数）
                existing = next((x for x in scored if x["thread_id"] == tid), None)
                if existing:
                    existing["score"] = max(existing["score"], sim * 10)
                    if not existing["snippet"]:
                        existing["snippet"] = r.get("content", "")[:200]
                else:
                    scored.append({
                        "thread_id": tid,
                        "title": meta.get("title", "") or ("会话 " + tid[:8]),
                        "snippet": r.get("content", "")[:200],
                        "score": sim * 10,
                        "message_count": meta.get("message_count", 0) or 0,
                        "updated_at": meta.get("updated_at"),
                    })
        except Exception:
            pass  # ChromaDB 不可用时仅用 SQLite 结果

    # ── 3. 排序并返回 Top K ───────────────────────────────────
    scored.sort(key=lambda x: x["score"], reverse=True)
    results = [
        SessionSearchResult(
            thread_id=x["thread_id"],
            title=x["title"],
            snippet=x["snippet"],
            score_percent=min(round(x["score"] * 10, 1), 100.0),
            message_count=x["message_count"],
            updated_at=x["updated_at"],
        )
        for x in scored[: req.top_k]
    ]

    elapsed = int((time.monotonic() - t0) * 1000)
    return SessionSearchResponse(
        query=req.query,
        results=results,
        total=len(results),
        elapsed_ms=elapsed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  配置与模型切换端点
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/config", response_model=ConfigInfo)
async def get_config() -> ConfigInfo:
    """获取当前配置（包含运行时动态值和动态模型列表）."""
    return ConfigInfo(
        available_models=_build_available_models_response(),
        current_model=get_current_model(),
        user_name=USER_NAME,
        user_tech_stack=USER_TECH_STACK,
        user_hardware=USER_HARDWARE,
        temperature=_current_temperature,
        max_tokens=_current_max_tokens,
        max_context_tokens=MAX_CONTEXT_TOKENS,
    )


@router.post("/model/switch")
async def model_switch(req: ModelSwitchRequest) -> dict:
    """动态切换 LLM 模型。支持纯模型名（如 deepseek-chat）或 provider/model 格式（如 claude/claude-3-5-sonnet-20241022）。"""
    from src.server.dependencies import switch_model as _switch

    # Validate: must exist in some provider
    all_model_ids = []
    for pcfg in MODEL_PROVIDERS.values():
        for m in pcfg.get("default_models", []):
            all_model_ids.append(m["id"])
            # Also accept provider/model format
            all_model_ids.append(f"{p_id}/{m['id']}" for p_id in MODEL_PROVIDERS)

    # Build set of valid model identifiers
    valid_ids = set()
    for p_id, pcfg in MODEL_PROVIDERS.items():
        for m in pcfg.get("default_models", []):
            valid_ids.add(m["id"])           # deepseek-chat
            valid_ids.add(f"{p_id}/{m['id']}")  # claude/claude-3-5-sonnet-20241022

    if req.model not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")
    result = _switch(req.model)
    return result


@router.post("/config/model")
async def update_model_config(req: ModelConfigRequest) -> dict:
    """更新模型参数（temperature、max_tokens），不影响默认模型."""
    if req.temperature is not None:
        from src.server.dependencies import set_temperature

        set_temperature(req.temperature)
    if req.max_tokens is not None:
        from src.server.dependencies import set_max_tokens

        set_max_tokens(req.max_tokens)
    logger.info(f"[config] model params updated: temp={req.temperature}, max_tokens={req.max_tokens}")
    return {"status": "ok"}


@router.post("/config/keys")
async def update_api_keys(req: ApiKeysUpdateRequest) -> dict:
    """
    更新运行时 API keys（立即生效，清空旧 agent 缓存）。

    注意：API keys 仅存储在内存中，服务重启后会恢复为 .env 中的值。
    如需持久化，请直接编辑 .env 文件。
    """
    from src.config import set_runtime_api_key
    from src.server.dependencies import _agent_registry

    updates = []
    if req.deepseek_api_key is not None:
        set_runtime_api_key("deepseek", req.deepseek_api_key)
        updates.append("deepseek")
    if req.anthropic_api_key is not None:
        set_runtime_api_key("claude", req.anthropic_api_key)
        updates.append("claude")
    if req.openai_api_key is not None:
        set_runtime_api_key("openai", req.openai_api_key)
        updates.append("openai")
    if req.openai_base_url is not None:
        from src.config import RUNTIME_API_KEYS
        # openai_base_url is not stored in RUNTIME_API_KEYS, store it in a dedicated global
        import src.config as _cfg
        _cfg.OPENAI_BASE_URL = req.openai_base_url
        updates.append("openai_base_url")
    if req.google_api_key is not None:
        set_runtime_api_key("gemini", req.google_api_key)
        updates.append("gemini")

    # Clear agent cache so new agents use the updated keys
    _agent_registry.clear()
    logger.info(f"[config] API keys updated for: {updates}")
    return {"status": "ok", "updated": updates}


@router.post("/cost/reset")
async def reset_cost() -> dict:
    """重置 Token 统计计数器."""
    global _token_tracker
    _token_tracker = TokenTracker(model=get_current_model())
    logger.info("[config] token stats reset")
    return {"status": "ok"}


@router.delete("/memory/all")
async def delete_all_memory() -> dict:
    """清空所有长期记忆（ChromaDB）."""
    try:
        store = get_memory_store()
        store.clear()
        logger.info("[config] all memories cleared")
        return {"status": "ok", "message": "All memories cleared"}
    except Exception as e:
        logger.error(f"delete_all_memory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chroma/reset")
async def reset_chroma() -> dict:
    """重置整个向量数据库（删除所有集合，包括文档 chunks）."""
    try:
        from src.document_processor import get_document_store

        # 清空 memory store
        store = get_memory_store()
        store.clear()

        # 清空文档 store（重置目录）
        doc_store = get_document_store()
        doc_store.reset()

        logger.info("[config] chroma vector DB reset")
        return {"status": "ok", "message": "Vector database reset complete"}
    except Exception as e:
        logger.error(f"reset_chroma error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cost", response_model=CostReport)
async def get_cost() -> CostReport:
    """获取 Token 使用量报告."""
    s = _token_tracker.summary()
    return CostReport(**s)


@router.get("/cost/history", response_model=CostHistoryReport)
async def get_cost_history(days: int = 7) -> CostHistoryReport:
    """获取 Token 成本历史."""
    entries = _token_tracker.get_history(days=days)
    model_entries = [
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
    ]
    return CostHistoryReport(
        entries=model_entries,
        total_cost_usd=sum(e.cost_usd for e in model_entries),
        total_tokens=sum(e.total_tokens for e in model_entries),
        total_calls=sum(e.num_calls for e in model_entries),
        period_start=model_entries[-1].date if model_entries else "",
        period_end=model_entries[0].date if model_entries else "",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  多 Agent 管理端点
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/agents", response_model=list[AgentProfile])
async def list_agents(is_active: Optional[bool] = None) -> list[AgentProfile]:
    """列出所有 Agent."""
    from src.multi_agent.orchestrator import get_agent_registry
    reg = get_agent_registry()
    agents = reg.list_agents(is_active=is_active)
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
        worker_kind=req.worker_kind,
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
    """取消运行中的任务（含编排任务联动）。"""
    from src.server.orch_jobs import get_job_manager
    from src.server.task_scheduler import get_scheduler

    scheduler = get_scheduler()
    task = scheduler.get(task_id)

    # 编排任务：同时取消后台 orch job
    if task and task.get("task_kind") == "orchestrate":
        orch_job_id = task.get("orchestrate_job_id")
        if orch_job_id:
            job_mgr = get_job_manager()
            job_mgr.cancel_job(orch_job_id)
            logger.info(f"[api] cancelled orch job {orch_job_id} via kanban task {task_id}")

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
    docs = store.list_docs(doc_type=doc_type)
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
        _log,
        chunk_text,
        detect_doc_type,
        extract_text,
        get_document_store,
        is_docx_zip_blob,
        is_legacy_word_doc,
        normalize_zip_upload_bytes,
        pdf_upload_failure_hint,
    )

    try:
        content = await file.read()
        filename = file.filename or "unknown"
        doc_type = detect_doc_type(filename)
        content = normalize_zip_upload_bytes(content, doc_type)
        size = len(content)

        logger.info(f"[upload] received file: {filename}, type: {doc_type}, size: {size} bytes")
        _log("H_UPLOAD", "api.upload_document:enter", "enter", {"filename": filename, "doc_type": doc_type, "size": size})

        store = get_document_store()
        doc = store.save(filename=filename, doc_type=doc_type, size_bytes=size, content=content)
        doc_id = doc["id"]
        chunks: list[str] = []

        try:
            text = extract_text(content, doc_type)
            _log("H_EXTRACT", "api.upload_document:extract_text", "exit", {"doc_type": doc_type, "text_len": len(text or ""), "empty": not bool((text or "").strip())})
            logger.info(f"[upload] extracted text length: {len(text) if text else 0} chars")
            if not (text or "").strip():
                err = "未能从文件中提取到文本（空文件、加密文档或缺少解析依赖）。"
                if size == 0:
                    err = (
                        "上传内容为空（0 字节），服务器未收到文件数据。"
                        "请重新选择本地文件后再试；若文件在网盘同步目录，请确认已同步完成。"
                    )
                elif doc_type == "pdf":
                    err = "未能从 PDF 中提取到文本。" + pdf_upload_failure_hint(content)
                elif doc_type == "docx":
                    if is_legacy_word_doc(content):
                        err = (
                            "该文件为旧版 Word 二进制格式（.doc），不是 .docx。"
                            "请在 Word 或 WPS 中使用「另存为」选择「Word 文档 (*.docx)」后重新上传。"
                        )
                    elif not is_docx_zip_blob(content):
                        err = (
                            "文件内容不是有效的 Word 文档（.docx 应为 ZIP 压缩包），可能已损坏或扩展名错误。"
                            "请重新导出为 .docx 后再试。"
                        )
                    else:
                        err += " 若已安装依赖，可执行：pip install python-docx"
                store.update_status(
                    doc_id,
                    "failed",
                    chunk_count=0,
                    error=err,
                )
            else:
                chunks = chunk_text(text)
                store.add_chunks(doc_id, chunks)
                store.ingest_to_chroma(doc_id)
            doc = store.get(doc_id)
        except Exception as e:
            logger.error(f"[upload] processing error: {e}", exc_info=True)
            store.update_status(doc_id, "failed", error=str(e))
            doc = store.get(doc_id)

        msg = f"已处理 {len(chunks)} 个文本块"
        if doc and doc.get("status") == "failed":
            msg = doc.get("error") or "处理失败"
        return DocumentUploadResponse(
            document=Document(**doc),
            message=msg,
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


@router.get("/documents/{doc_id}/preview", response_model=DocumentPreviewResponse)
async def get_document_preview(doc_id: str) -> DocumentPreviewResponse:
    """返回文档已入库的文本块，供知识库右侧预览（与向量分块一致）."""
    from src.document_processor import get_document_store
    store = get_document_store()
    doc = store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    raw = store.get_chunks(doc_id)
    chunks = [DocumentChunkItem(index=i, text=t) for i, t in enumerate(raw)]
    return DocumentPreviewResponse(document=Document(**doc), chunks=chunks)


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    """删除文档."""
    from src.document_processor import get_document_store
    store = get_document_store()
    ok = store.delete(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════════════
#  知识库 RAG 检索
# ═══════════════════════════════════════════════════════════════════════════════

def _count_tokens(text: str) -> int:
    """粗估 token 数（中文 ≈ 1.5 chars/token，英文按空格分词）。"""
    chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    english_words = len(text) - chinese
    return int(chinese * 0.75 + english_words * 0.25)


@router.post("/kb/search", response_model=KbSearchResponse)
async def kb_search(req: KbSearchRequest) -> KbSearchResponse:
    """
    对知识库文档执行向量语义检索，返回带百分制相似度分数的相关文本块。
    采用 cosine 相似度（ChromaDB hnsw:space=cosine），分数 = (1 - distance) × 100。
    支持按 min_score 过滤，并展示来源文件名、块长度等信息。
    """
    import time

    t0 = time.monotonic()
    from src.memory.chroma_store import ChromaMemoryStore

    store = ChromaMemoryStore()
    raw = store.search(query=req.query, top_k=req.top_k, category="document")

    chunks: list[KbSearchChunk] = []
    for r in raw:
        sim = float(r.get("similarity", 0))
        if sim < req.min_score:
            continue
        meta = r.get("metadata") or {}
        content = r.get("content") or ""
        chunks.append(KbSearchChunk(
            chunk_id=r.get("id", ""),
            content=content,
            source_filename=meta.get("filename", "未知"),
            doc_type=meta.get("doc_type", "txt"),
            distance=float(r.get("distance", 0)),
            similarity=round(sim, 4),
            score_percent=round(sim * 100, 1),
            token_count=_count_tokens(content),
        ))

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return KbSearchResponse(
        query=req.query,
        chunks=chunks,
        total=len(chunks),
        elapsed_ms=elapsed_ms,
        search_params={"top_k": req.top_k, "min_score": req.min_score},
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Crayfish 多 Agent 编排端点 (SSE 流式)
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
#  POST /orchestrate        → 创建任务，立即返回 job_id
#  GET  /orchestrate/jobs   → 列表
#  GET  /orchestrate/jobs/{id}             → 状态 + 结果
#  GET  /orchestrate/jobs/{id}/events      → SSE 实时事件流
#  DELETE /orchestrate/jobs/{id}            → 取消任务
# ═══════════════════════════════════════════════════════════════════════════════


@router.post("/orchestrate")
async def orchestrate(req: OrchestrateRequest) -> dict:
    """
    创建后台编排任务，立即返回 job_id。
    同时在任务协调看板中创建编排任务行（task_kind=orchestrate）。
    """
    from src.multi_agent.orchestrator import get_agent_registry
    from src.server.orch_jobs import get_job_manager, _run_orchestrator_job
    from src.server.task_scheduler import get_scheduler

    reg = get_agent_registry()
    scheduler = get_scheduler()

    # Chief Coordinator (agent_main) 必选：总协调，与子 Agent 一并参与编排上下文
    enabled = list(dict.fromkeys(req.enabled_agents))
    chief_profile = reg.get("agent_main")
    if chief_profile and chief_profile.get("is_active") and "agent_main" not in enabled:
        enabled.insert(0, "agent_main")

    # 加载 participants（Agent 档案）
    participants = []
    for aid in enabled:
        p = reg.get(aid)
        if p:
            participants.append(p)

    job_mgr = get_job_manager()
    # 先创建 job 获取 job_id，再建看板行（需要 job_id）
    job = job_mgr.create_job(
        requirement=req.requirement,
        enabled_agents=enabled,
        quality_threshold=req.quality_threshold,
        participants=participants,
        kanban_task_id=None,  # 暂填 None，创建看板行后再更新
    )

    # 创建编排看板任务（关联 job_id）
    kanban_task = scheduler.create_orchestrate_shell(
        requirement=req.requirement,
        job_id=job.job_id,
        enabled_agents=enabled,
        participants=participants,
        quality_threshold=req.quality_threshold,
    )
    # 将 kanban_task_id 写回 job（只读属性 dataclass，直接替换）
    job.kanban_task_id = kanban_task["id"]

    # 立即返回 job_id，不阻塞
    _SHUTDOWN_TRACKER["pending_tasks"] += 1
    if _SHUTDOWN_TRACKER["pending_tasks"] > _SHUTDOWN_TRACKER["max_pending"]:
        _SHUTDOWN_TRACKER["max_pending"] = _SHUTDOWN_TRACKER["pending_tasks"]
    logger.info(f"[orchestrate] +1 pending tasks={_SHUTDOWN_TRACKER['pending_tasks']}, max={_SHUTDOWN_TRACKER['max_pending']}")
    asyncio.create_task(_run_orchestrator_job(job))

    # 任务完成后自动减少计数
    async def _track_done():
        await asyncio.sleep(0.1)  # 让任务有时间启动
        # 检查任务是否完成
        import time
        deadline = time.monotonic() + 600  # 10分钟超时
        while time.monotonic() < deadline:
            j = job_mgr.get_job(job.job_id)
            if j and j.status.value in ("done", "failed", "cancelled"):
                _SHUTDOWN_TRACKER["pending_tasks"] -= 1
                logger.info(f"[orchestrate] -1 pending tasks={_SHUTDOWN_TRACKER['pending_tasks']}, job_id={job.job_id}")
                return
            await asyncio.sleep(1)
        _SHUTDOWN_TRACKER["pending_tasks"] -= 1
        logger.warning(f"[orchestrate] timeout cleanup, pending tasks={_SHUTDOWN_TRACKER['pending_tasks']}")

    asyncio.create_task(_track_done())

    logger.info(f"[orch_jobs] created job {job.job_id} -> kanban {kanban_task['id']}: {req.requirement[:60]}...")
    return {
        "job_id": job.job_id,
        "kanban_task_id": kanban_task["id"],
        "status": job.status.value,
        "created_at": job.created_at,
    }


@router.get("/orchestrate/jobs")
async def list_orch_jobs() -> dict:
    """列出所有编排任务（最新在前，最多 50 条）。"""
    from src.server.orch_jobs import get_job_manager

    jobs = get_job_manager().list_jobs()
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return {
        "jobs": [j.to_dict() for j in jobs[:50]],
        "total": len(jobs),
    }


@router.get("/orchestrate/jobs/{job_id}")
async def get_orch_job(job_id: str) -> dict:
    """查询单个任务的状态 + 结果。"""
    from src.server.orch_jobs import get_job_manager, JobStatus

    job = get_job_manager().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    resp = job.to_dict()
    # 运行中也需要返回 events，否则前端轮询 /jobs/{id} 时执行日志始终为空
    resp["events"] = [dict(e) for e in job.events]
    return resp


@router.get("/orchestrate/jobs/{job_id}/events")
async def stream_orch_job_events(job_id: str) -> StreamingResponse:
    """
    SSE 流式推送任务的历史事件 + 新事件（long-polling 轮询，每 0.5s 检查一次）。
    前端保持此连接即可持续接收实时更新。
    """
    from src.server.orch_jobs import get_job_manager, JobStatus

    job = get_job_manager().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        import json as _j

        # 1. 先推送已有的历史事件
        for ev in job.events:
            ev_dict = dict(ev)
            ev_type = ev_dict.get("type", "?")
            # 把 type 嵌入 data 内部 — 防止前端 SSE parser 丢失 event: type 行时无法路由
            if "type" not in ev_dict.get("data", {}):
                ev_dict["data"] = dict(ev_dict.get("data", {}), type=ev_type)
            payload = _j.dumps(ev_dict, ensure_ascii=False)
            yield f"data: {payload}\n\n"

        # 2. 如果任务还在跑，继续推送新事件
        if job.status not in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
            last_idx = len(job.events)
            waited = 0
            max_wait = 300

            while job.status not in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
                if waited >= max_wait:
                    yield f"data: { _j.dumps({'type':'error','message':'SSE 连接超时（5分钟无活动）'}, ensure_ascii=False) }\n\n"
                    break
                await asyncio.sleep(0.5)
                waited += 0.5
                if len(job.events) > last_idx:
                    waited = 0
                    for ev in list(job.events)[last_idx:]:
                        last_idx = len(job.events)
                        ev_dict = dict(ev)
                        ev_type = ev_dict.get("type", "?")
                        if "type" not in ev_dict.get("data", {}):
                            ev_dict["data"] = dict(ev_dict.get("data", {}), type=ev_type)
                        payload = _j.dumps(ev_dict, ensure_ascii=False)
                        yield f"data: {payload}\n\n"

        # 3. 最终状态
        yield f"data: {_j.dumps({'type': 'done', 'status': job.status.value}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/orchestrate/jobs/{job_id}")
async def cancel_orch_job(job_id: str) -> dict:
    """取消正在运行的后台编排任务。"""
    from src.server.orch_jobs import get_job_manager, JobStatus

    job = get_job_manager().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in (JobStatus.DONE, JobStatus.CANCELLED, JobStatus.FAILED):
        raise HTTPException(status_code=409, detail=f"Job already {job.status.value}")

    ok = get_job_manager().cancel_job(job_id)
    return {"job_id": job_id, "status": job.status.value, "cancelled": ok}


# ═══════════════════════════════════════════════════════════════════════════════
#  文件输出 API — 编排任务的产出文件
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/orchestrate/jobs/{job_id}/files")
async def list_orch_files(job_id: str) -> dict:
    """列出编排任务的产出文件列表。"""
    from src.server.orch_jobs import get_job_manager

    job = get_job_manager().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.result:
        raise HTTPException(status_code=404, detail="Job result not ready")

    files = job.result.get("files", [])
    output_dir = job.result.get("output_dir", "")

    return {
        "job_id": job_id,
        "output_dir": output_dir,
        "files": files,
        "file_count": len(files),
    }


@router.get("/orchestrate/jobs/{job_id}/files/{filename}")
async def get_orch_file(job_id: str, filename: str) -> dict:
    """读取指定文件的内容（用于前端预览）。"""
    from src.server.orch_jobs import get_job_manager

    job = get_job_manager().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.result:
        raise HTTPException(status_code=404, detail="Job result not ready")

    output_dir = job.result.get("output_dir", "")
    if not output_dir:
        raise HTTPException(status_code=404, detail="No output directory")

    from pathlib import Path
    filepath = Path(output_dir) / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    try:
        content = filepath.read_text(encoding="utf-8")
        file_type = _detect_file_type(filename)
        return {
            "filename": filename,
            "file_type": file_type,
            "size": filepath.stat().st_size,
            "content": content,
            "previewable": file_type in ("code", "html", "markdown", "data"),
        }
    except UnicodeDecodeError:
        raise HTTPException(status_code=415, detail="Binary file, use download endpoint")


@router.get("/orchestrate/jobs/{job_id}/download/{filename}")
async def download_orch_file(job_id: str, filename: str) -> StreamingResponse:
    """下载指定的产出文件。"""
    from src.server.orch_jobs import get_job_manager

    job = get_job_manager().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = job.result.get("output_dir", "") if job.result else ""
    if not output_dir:
        raise HTTPException(status_code=404, detail="No output directory")

    from pathlib import Path
    filepath = Path(output_dir) / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    from fastapi.responses import FileResponse
    return FileResponse(
        path=str(filepath),
        filename=filename,
        media_type="application/octet-stream",
    )


@router.get("/orchestrate/jobs/{job_id}/download-all")
async def download_all_orch_files(job_id: str) -> StreamingResponse:
    """打包下载所有产出文件（.zip）。"""
    from src.server.orch_jobs import get_job_manager
    from src.utils.output_manager import get_output_manager

    job = get_job_manager().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    plan_id = ""
    if job.result:
        # 直接从 result 字典读取（最可靠）
        plan_id = job.result.get("plan_id", "")
        if not plan_id:
            # 向后兼容：从 summary 文本正则提取
            summary = job.result.get("summary", "")
            import re
            match = re.search(r"计划ID[:：]\s*([^\s\n]+)", summary)
            if match:
                plan_id = match.group(1)

    if not plan_id:
        raise HTTPException(status_code=404, detail="No plan_id found")

    om = get_output_manager(plan_id)
    archive_path = om.create_archive()

    if not archive_path or not archive_path.exists():
        raise HTTPException(status_code=500, detail="Failed to create archive")

    from fastapi.responses import FileResponse
    return FileResponse(
        path=str(archive_path),
        filename=f"{plan_id}.zip",
        media_type="application/zip",
    )


@router.get("/orchestrate/outputs/{plan_id}")
async def get_output_info(plan_id: str) -> dict:
    """获取指定 plan 的输出目录信息。"""
    from src.utils.output_manager import get_output_manager

    om = get_output_manager(plan_id)
    if not om.is_created:
        raise HTTPException(status_code=404, detail="Output not found")

    return om.get_output_info()


@router.get("/orchestrate/outputs/{plan_id}/{filename}")
async def get_output_file(plan_id: str, filename: str) -> dict:
    """读取指定 plan 输出目录中的文件内容。"""
    from src.utils.output_manager import get_output_manager

    om = get_output_manager(plan_id)
    if not om.is_created:
        raise HTTPException(status_code=404, detail="Output not found")

    content = om.read_file(filename)
    if content is None:
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    file_type = _detect_file_type(filename)
    return {
        "filename": filename,
        "file_type": file_type,
        "content": content,
        "previewable": file_type in ("code", "html", "markdown", "data"),
    }


def _detect_file_type(filename: str) -> str:
    """根据扩展名检测文件类型。"""
    from pathlib import Path
    ext = Path(filename).suffix.lower()
    type_map = {
        ".py": "code", ".js": "code", ".ts": "code", ".jsx": "code",
        ".tsx": "code", ".rs": "code", ".go": "code", ".java": "code",
        ".cpp": "code", ".c": "code", ".h": "code", ".sh": "code",
        ".bat": "code", ".html": "html", ".htm": "html", ".css": "html",
        ".scss": "html", ".sass": "html",
        ".md": "markdown", ".mdx": "markdown", ".txt": "markdown",
        ".docx": "doc", ".json": "data", ".csv": "data",
        ".yaml": "data", ".yml": "data", ".xml": "data", ".toml": "data",
        ".pdf": "report",
        ".png": "image", ".jpg": "image", ".jpeg": "image",
        ".gif": "image", ".svg": "image", ".webp": "image",
    }
    return type_map.get(ext, "other")
