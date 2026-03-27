"""Memory tools — 09+13: memory_search 和 save_memory LangChain 工具."""

import logging
from typing import Any, Optional
from datetime import datetime, timezone

from langchain_core.tools import tool

from src.memory.chroma_store import ChromaMemoryStore
from src.memory.memory_schema import MemoryRecord

logger = logging.getLogger(__name__)

# 全局单例，避免重复创建 ChromaDB 连接
_memory_store: Optional[ChromaMemoryStore] = None


def get_memory_store() -> ChromaMemoryStore:
    """获取全局 ChromaMemoryStore 单例."""
    global _memory_store
    if _memory_store is None:
        _memory_store = ChromaMemoryStore()
    return _memory_store


def set_memory_store(store: ChromaMemoryStore) -> None:
    """设置全局 ChromaMemoryStore 单例（用于测试注入）."""
    global _memory_store
    _memory_store = store


@tool
def memory_search(query: str, category: str | None = None, top_k: int = 5) -> str:
    """从长期记忆（向量库）中检索与查询最相关的内容。

    当用户询问「我之前的项目」「我的配置」「我的偏好」或**与本人相关的历史事实**时，应调用此工具。
    若用户明确问「知识库里上传的文档」「我上传的 PDF」等，请改用 knowledge_base_search。
    严禁在未检索的情况下对用户的项目细节进行假设。

    Args:
        query: 搜索查询（如"智程导航的架构"、"我的硬件配置"）
        category: 可选，限定记忆类别（project/tech_stack/hardware/preference/decision 等）
        top_k: 返回的最相关记忆数量，默认 5

    Returns:
        格式化的记忆列表，包含内容、分类、重要性评分和时间戳
    """
    store = get_memory_store()
    results = store.search(query=query, top_k=top_k, category=category)

    if not results:
        return "在长期记忆中未找到相关内容。"

    lines = ["--- 相关记忆检索结果 ---"]
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        sim = r.get("similarity", 0)
        lines.append(
            f"\n{i}. [{meta.get('category', 'unknown')}] "
            f"(相似度: {sim:.2%})\n"
            f"   事实: {meta.get('fact', r.get('content', ''))}\n"
            f"   重要性: {meta.get('importance', 0)}/5"
            + (f" | 项目: {meta.get('project_ref', '')}" if meta.get("project_ref") else "")
        )
    return "\n".join(lines)


@tool
def knowledge_base_search(query: str, top_k: int = 8) -> str:
    """从**知识库**（用户在网页端上传的 PDF、Word、TXT 等文档切块）中语义检索相关内容。

    当用户引用「上传的论文」「知识库里的文档」「我放在 KB 里的资料」或需要根据**已导入文档**回答时，必须调用本工具。
    与 personal memory（memory_search）不同：本工具只查 category=document 的文档块。

    Args:
        query: 检索问句或关键词（可与用户问题同义改写）
        top_k: 返回片段数量，默认 8

    Returns:
        带文件名与相似度的文档片段列表；若无命中则说明知识库中暂无相关内容
    """
    store = get_memory_store()
    results = store.search(query=query, top_k=top_k, category="document")

    if not results:
        return "知识库中未找到与查询相关的文档片段（可能尚未上传文档，或需换关键词重试）。"

    lines = ["--- 知识库文档检索结果 ---"]
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        sim = r.get("similarity", 0)
        fn = meta.get("filename", meta.get("source", "未知文件"))
        body = meta.get("fact", r.get("content", ""))
        lines.append(
            f"\n{i}. 文件: {fn} (相似度: {sim:.2%})\n   片段: {body}"
        )
    return "\n".join(lines)


@tool
def save_memory(
    fact: str,
    category: str,
    importance: int = 3,
    project_ref: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """将新的事实/决策存入长期记忆（ChromaDB）。

    当检测到用户提供新的事实、决策变更或项目更新时，主动调用此工具。
    系统会自动处理重复内容（相似内容不会被重复存储）。

    Args:
        fact: 记忆内容（核心事实描述）
        category: 记忆类别，必须是以下之一：
                  project（项目）/ tech_stack（技术栈）/ hardware（硬件）/
                  preference（偏好）/ decision（决策）
        importance: 重要性评分 1-5，默认 3
        project_ref: 关联的项目名称（可选）
        tags: 自定义标签列表（可选）

    Returns:
        操作结果：added/updated/skipped + 记忆 ID
    """
    from src.middleware.input_guard import validate_memory_fact

    guard_err = validate_memory_fact(fact)
    if guard_err:
        return guard_err

    record = MemoryRecord(
        fact=fact,
        category=category,
        importance=importance,
        project_ref=project_ref,
        timestamp=datetime.now(timezone.utc).isoformat(),
        tags=tags or [],
    )

    store = get_memory_store()
    result = store.upsert_record(record)

    if result == "added":
        return f"[记忆已添加] 类别: {category} | 内容: {fact[:50]}..."
    elif result == "updated":
        return f"[记忆已更新] 类别: {category} | 内容: {fact[:50]}..."
    else:
        return f"[记忆已忽略] 内容重复，相似记忆未更新。"
