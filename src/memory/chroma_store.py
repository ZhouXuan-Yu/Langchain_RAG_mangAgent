"""ChromaDB memory store — 09+18: 主动记忆 Upsert 核心逻辑."""

import uuid
import logging
from typing import Any, Optional

import chromadb
from chromadb.utils import embedding_functions

from src.config import (
    CHROMA_PATH,
    CHROMA_COLLECTION_NAME,
    UPSERT_SIMILARITY_THRESHOLD,
)
from src.memory.memory_schema import MemoryRecord

logger = logging.getLogger(__name__)


class ChromaMemoryStore:
    """
    ChromaDB 长期记忆封装 — 09+13+18 主动记忆 Upsert 核心.

    核心逻辑：
    1. search() — 向量相似度检索，返回最相关的记忆片段
    2. upsert() — 智能 Upsert：
       - 相似度 < threshold → 新增
       - 相似度 > threshold + 新内容更全 → 删除旧 + 写入新
       - 相似度 > threshold + 内容重复 → 忽略
    """

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION_NAME,
        path: str | None = None,
        threshold: float = UPSERT_SIMILARITY_THRESHOLD,
    ):
        self.client = chromadb.PersistentClient(
            path=str(path or CHROMA_PATH)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.threshold = threshold

    # ── Core Operations ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[dict]:
        """向量相似度检索，返回记忆片段列表."""
        where_filter = {"category": category} if category else None

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
        )

        memories = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                memories.append({
                    "id": results["ids"][0][i],
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1 - results["distances"][0][i],
                })
        return memories

    def upsert(
        self,
        content: str,
        metadata: dict,
        record_id: str | None = None,
        threshold: float | None = None,
    ) -> str:
        """
        主动记忆 Upsert — 09+13+18 组合拳核心.

        Args:
            content: 记忆内容文本
            metadata: 记忆元数据（category, importance, project_ref 等）
            record_id: 可选，强制指定 ID（用于强制更新）
            threshold: 相似度阈值，默认 0.85

        Returns:
            操作描述: "added" | "updated" | "skipped"
        """
        threshold = threshold or self.threshold

        results = self.collection.query(
            query_texts=[content],
            n_results=1,
            where={"category": metadata.get("category")} if metadata.get("category") else None,
        )

        new_id = record_id or str(uuid.uuid4())
        content_in_meta = content  # 保存原始内容用于对比

        if (
            results["documents"]
            and results["documents"][0]
            and results["distances"]
            and results["distances"][0]
        ):
            similarity = 1 - results["distances"][0][0]
            old_id = results["ids"][0][0]
            old_meta = results["metadatas"][0][0]
            old_content = old_meta.get("content", "")

            if similarity > threshold:
                # 相似度高：判断是否需要更新
                new_len = len(content)
                old_len = len(old_content)

                if new_len > old_len * 1.2:
                    # 新内容更全面：删除旧记录，插入新记录
                    self.collection.delete(ids=[old_id])
                    self.collection.add(
                        ids=[new_id],
                        documents=[content],
                        metadatas=[{**metadata, "content": content_in_meta}],
                    )
                    logger.info(f"[upsert] updated memory {old_id} -> {new_id}")
                    return "updated"
                else:
                    # 内容重复：忽略
                    logger.debug(f"[upsert] skipped duplicate (similarity={similarity:.3f})")
                    return "skipped"

        # 相似度低或无历史记录：新增
        self.collection.add(
            ids=[new_id],
            documents=[content],
            metadatas=[{**metadata, "content": content_in_meta}],
        )
        logger.info(f"[upsert] added new memory {new_id}")
        return "added"

    def upsert_record(self, record: MemoryRecord) -> str:
        """直接使用 MemoryRecord 对象执行 upsert."""
        return self.upsert(
            content=record.to_document(),
            metadata=record.to_metadata(),
        )

    def delete(self, record_id: str) -> bool:
        """根据 ID 删除记忆."""
        try:
            self.collection.delete(ids=[record_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete record {record_id}: {e}")
            return False

    def get_all(self, category: str | None = None) -> list[dict]:
        """获取所有记忆（用于调试）."""
        where_filter = {"category": category} if category else None
        results = self.collection.get(where=where_filter)

        memories = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"]):
                memories.append({
                    "id": results["ids"][i],
                    "content": doc,
                    "metadata": results["metadatas"][i],
                })
        return memories

    def clear(self) -> None:
        """清空所有记忆（危险操作）."""
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
