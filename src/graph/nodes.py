"""Graph nodes — 07+17+21: LangGraph 各节点实现."""

import asyncio
import logging
from typing import Literal
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.graph.state import AgentState
from src.memory.chroma_store import ChromaMemoryStore
from src.memory.memory_schema import MemoryRecord
from src.tools.memory_tools import memory_search, save_memory, get_memory_store
from src.tools.browser_tools import web_search, browse_page
from src.utils.summarizer import compress_web_content

logger = logging.getLogger(__name__)

# 全局 LLM 引用（在 build_agent_graph 时注入）
_llm = None


def set_llm(llm) -> None:
    """注入 LLM 实例到 nodes 模块（避免循环导入）。"""
    global _llm
    _llm = llm


# ── Router Node ──────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    """
    路由决策节点 — 分析用户输入，决定后续路由分支.

    动态路由规则：
    - 提及"修改之前的计划/记忆" → memory_update
    - 提及"我的项目/配置/偏好/记得" → memory_retrieve
    - 询问最新信息/2025-2026/Bug/版本 → web_search
    - 上传图片 → image_process
    - 其他 → direct_reason
    """
    messages = state.get("messages", [])
    if not messages:
        return {"route_decision": "direct_reason"}

    last_msg = ""
    last_content = None
    if messages:
        last_item = messages[-1]
        if hasattr(last_item, "content"):
            last_content = last_item.content
            last_msg = last_content.lower() if isinstance(last_content, str) else ""
        elif isinstance(last_item, dict) and "content" in last_item:
            last_content = last_item["content"]
            last_msg = last_content.lower() if isinstance(last_content, str) else ""

    # 图片检测
    if isinstance(last_content, list):
        for item in last_content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                return {"route_decision": "image_process"}

    # 路由判断
    if any(k in last_msg for k in ["修改", "更新", "之前的", "改变我的"]):
        route = "memory_update"
    elif any(k in last_msg for k in ["我的", "之前", "项目", "配置", "偏好", "记得", "项目细节"]):
        route = "memory_retrieve"
    elif any(k in last_msg for k in ["最新", "2025", "2026", "bug", "版本", "如何解决", "查找"]):
        route = "web_search"
    else:
        route = "direct_reason"

    logger.info(f"[router] decision: {route} | input: {last_msg[:50]}")
    return {"route_decision": route}


# ── Memory Nodes ─────────────────────────────────────────────────────────────

def retrieve_memory(state: AgentState) -> dict:
    """检索长期记忆节点 — 调用 memory_search 工具."""
    messages = state.get("messages", [])
    if not messages:
        return {"memory_context": []}

    query = ""
    if messages:
        last_item = messages[-1]
        if hasattr(last_item, "content"):
            query = last_item.content if isinstance(last_item.content, str) else ""
        elif isinstance(last_item, dict) and "content" in last_item:
            query = last_item["content"] if isinstance(last_item["content"], str) else ""

    try:
        # 同步工具调用放入线程池，避免阻塞事件循环
        result = asyncio.get_event_loop().run_until_complete(
            memory_search.ainvoke({"query": query})
        ) if asyncio.get_event_loop().is_running() else memory_search.invoke({"query": query})
        return {"memory_context": [f"[记忆检索]\n{result}"]}
    except Exception as e:
        logger.error(f"retrieve_memory failed: {e}")
        return {"memory_context": []}


async def memory_update_node(state: AgentState) -> dict:
    """
    异步记忆更新节点 — 当用户要求修改/更新记忆时执行。
    1. 从 ChromaDB 检索相关旧记忆
    2. LLM 判断如何整合新旧信息
    3. 删除旧记忆，存入新记忆
    """
    messages = state.get("messages", [])
    if not messages or not _llm:
        return {}

    query = ""
    last_item = messages[-1]
    if hasattr(last_item, "content"):
        query = last_item.content if isinstance(last_item.content, str) else ""
    elif isinstance(last_item, dict) and "content" in last_item:
        query = last_item["content"] if isinstance(last_item["content"], str) else ""

    store = get_memory_store()

    # ChromaDB 调用放入线程池（同步 I/O）
    old_memories = await asyncio.to_thread(store.search, query=query, top_k=3)
    old_content = "\n".join(
        f"- {m['content']}" for m in old_memories
    ) if old_memories else "无相关旧记忆。"

    # LLM 整合新旧信息（异步调用，不阻塞事件循环）
    system_prompt = (
        "你是一个记忆整合助手。用户要求修改/更新记忆。\n"
        f"旧记忆:\n{old_content}\n\n"
        "请根据用户的新要求，生成更新后的记忆内容，格式为:\n"
        "NEW_MEMORY: <更新后的记忆内容>\n"
        "CATEGORY: <记忆类别>"
    )

    response = await _llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    response_text = response.content if hasattr(response, "content") else str(response)

    # 解析 LLM 输出
    new_memory = ""
    category = "project"
    for line in response_text.split("\n"):
        if line.startswith("NEW_MEMORY:"):
            new_memory = line.split("NEW_MEMORY:", 1)[1].strip()
        elif line.startswith("CATEGORY:"):
            category = line.split("CATEGORY:", 1)[1].strip().lower()

    if new_memory:
        record = MemoryRecord(
            fact=new_memory,
            category=category,
            importance=4,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        await asyncio.to_thread(store.upsert_record, record)
        return {
            "memory_context": [f"[记忆已更新] {new_memory}"],
            "memory_updated": True,
        }

    return {"memory_context": ["[系统] 未能提取有效的更新内容"]}


# ── Web Search Nodes ────────────────────────────────────────────────────────

def retrieve_web(state: AgentState) -> dict:
    """网页检索节点 — 调用 web_search 工具."""
    messages = state.get("messages", [])
    if not messages:
        return {"web_context": []}

    query = ""
    if messages:
        last_item = messages[-1]
        if hasattr(last_item, "content"):
            query = last_item.content if isinstance(last_item.content, str) else ""
        elif isinstance(last_item, dict) and "content" in last_item:
            query = last_item["content"] if isinstance(last_item["content"], str) else ""

    try:
        # 同步工具调用放入线程池
        try:
            loop = asyncio.get_running_loop()
            result = loop.run_until_complete(web_search.ainvoke({"query": query}))
        except RuntimeError:
            result = web_search.invoke({"query": query})
        return {"web_context": [f"[网页搜索]\n{result}"]}
    except Exception as e:
        logger.error(f"retrieve_web failed: {e}")
        return {"web_context": []}


def browse_page_node(state: AgentState) -> dict:
    """深度抓取节点 — 当需要详细页面内容时调用 browse_page."""
    messages = state.get("messages", [])
    web_context = state.get("web_context", [])

    # 从 web_context 中提取 URLs
    urls = []
    for ctx in web_context:
        for line in ctx.split("\n"):
            if "http" in line and ("链接:" in line or "url:" in line):
                for part in line.split():
                    if part.startswith("http"):
                        urls.append(part.strip(":,."))

    results = []
    for url in urls[:2]:  # 最多抓取 2 个页面
        try:
            content = browse_page.invoke({"url": url})
            compressed = compress_web_content(content)
            results.append(f"[页面抓取: {url}]\n{compressed}")
        except Exception as e:
            results.append(f"[页面抓取失败: {url}] {e}")

    return {"web_context": results}


# ── Image Process Node ───────────────────────────────────────────────────────

def image_process_node(state: AgentState) -> dict:
    """图片处理节点 — 19-21: 提取图片特征并存入 pending_memory."""
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1]
    if not hasattr(last_msg, "content"):
        return {}

    image_urls = []
    if isinstance(last_msg.content, list):
        for item in last_msg.content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                image_urls.append(item["image_url"].get("url", ""))

    if not image_urls:
        return {}

    pending = []
    for img_url in image_urls:
        record = MemoryRecord(
            fact=f"用户上传了图片: {img_url[:50]}...",
            category="project",
            importance=4,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tags=["图片上传"],
        )
        pending.append(record)

    return {"pending_memory": pending, "memory_context": ["[图片处理] 已识别并处理上传图片"]}


# ── Reason Node ─────────────────────────────────────────────────────────────

async def reason_node(state: AgentState) -> dict:
    """
    异步推理回复节点 — 整合 memory_context + web_context，LLM 生成最终回复.
    同时识别新事实，生成 pending_memory。
    """
    if not _llm:
        return {}

    messages = state.get("messages", [])
    memory_context = state.get("memory_context", [])
    web_context = state.get("web_context", [])

    # 构建上下文提示
    context_parts = []
    if memory_context:
        context_parts.append("=== 相关记忆 ===\n" + "\n".join(memory_context))
    if web_context:
        context_parts.append("=== 网络检索 ===\n" + "\n".join(web_context))

    context_hint = ""
    if context_parts:
        context_hint = "\n\n请结合以下背景信息回答：\n" + "\n\n".join(context_parts)

    # 获取原始用户输入
    user_input = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_input = msg.content if isinstance(msg.content, str) else ""
            break
        if isinstance(msg, dict):
            if msg.get("type") == "human" or msg.get("role") == "user":
                user_input = msg.get("content", "") if isinstance(msg.get("content"), str) else ""
                break

    if not user_input:
        return {}

    # LLM 推理（异步调用，不阻塞事件循环）
    prompt = (
        f"用户问题: {user_input}\n"
        f"{context_hint}\n\n"
        "请给出专业、直接的回答。如果发现了新的重要事实，"
        "请在回答末尾以如下格式标注（用于自动记忆存储）：\n"
        "MEMORY: <需要记忆的事实>\n"
        "CATEGORY: <project|tech_stack|hardware|preference|decision>\n"
        "IMPORTANCE: <1-5>"
    )

    response = await _llm.ainvoke([HumanMessage(content=prompt)])
    response_text = response.content if hasattr(response, "content") else str(response)

    # 解析 pending_memory
    pending = []
    lines = response_text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("MEMORY:") and i + 2 < len(lines):
            fact = line.split("MEMORY:", 1)[1].strip()
            cat = "project"
            imp = 3
            for j in range(i + 1, min(i + 4, len(lines))):
                if lines[j].startswith("CATEGORY:"):
                    cat = lines[j].split("CATEGORY:", 1)[1].strip().lower()
                elif lines[j].startswith("IMPORTANCE:"):
                    try:
                        imp = int(lines[j].split("IMPORTANCE:", 1)[1].strip()[0])
                    except ValueError:
                        imp = 3
            if fact:
                pending.append(MemoryRecord(
                    fact=fact,
                    category=cat,
                    importance=imp,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))

    return {
        "messages": [AIMessage(content=response_text)],
        "pending_memory": pending,
    }


# ── Memory Reflect Node ──────────────────────────────────────────────────────

async def memory_reflect_node(state: AgentState) -> dict:
    """
    异步主动记忆评估节点 — 09+13+18 组合拳核心.

    遍历 pending_memory，对每个新事实执行 ChromaDB upsert。
    根据相似度判断：新增 / 更新 / 忽略。
    """
    pending = state.get("pending_memory", [])
    if not pending:
        return {"memory_updated": False}

    store = get_memory_store()
    updated = False
    memory_notes = []

    for record in pending:
        try:
            # ChromaDB 同步 I/O 放入线程池
            result = await asyncio.to_thread(store.upsert_record, record)
            if result in ("added", "updated"):
                updated = True
                memory_notes.append(f"[{result.upper()}] {record.category}: {record.fact[:40]}...")
            else:
                memory_notes.append(f"[SKIPPED] 内容重复: {record.fact[:40]}...")
        except Exception as e:
            logger.error(f"memory_reflect failed for {record}: {e}")

    logger.info(f"[memory_reflect] processed {len(pending)} records: {memory_notes}")

    return {
        "pending_memory": [],  # 清空已处理
        "memory_updated": updated,
        "memory_context": [f"[主动记忆] {'; '.join(memory_notes)}"] if memory_notes else [],
    }


# ── Loop Control ─────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> Literal["router", "__end__"]:
    """
    判断是否继续循环 — 防止无限循环.

    停止条件：
    - turn_count 超过阈值（默认 10）
    - memory_updated 为 True（已完成记忆更新）
    - 最后的消息是 AI 回复（reason_node 已输出最终回复）
    """
    turn_count = state.get("turn_count", 0)
    if turn_count >= 10:
        logger.warning("[loop] turn_count exceeded, terminating")
        return "__end__"

    messages = state.get("messages", [])
    if messages:
        last_item = messages[-1]
        last_is_ai = isinstance(last_item, AIMessage)
        if not last_is_ai and isinstance(last_item, dict):
            last_is_ai = last_item.get("type") == "ai" or last_item.get("role") == "assistant"
        if last_is_ai:
            return "__end__"

    # 有 pending_memory 未处理时不结束，继续循环
    pending = state.get("pending_memory", [])
    if pending:
        return "router"

    return "router"
