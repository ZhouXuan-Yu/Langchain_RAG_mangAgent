"""Pydantic memory schema — 13: 结构化记忆存储格式."""

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    """存入 ChromaDB 的统一结构化格式 — 09+13+18 主动记忆组合拳核心."""

    fact: str = Field(description="记忆内容（核心事实描述）")
    category: Literal["project", "tech_stack", "hardware", "preference", "decision"] = Field(
        description="记忆分类：project=项目, tech_stack=技术栈, hardware=硬件, preference=偏好, decision=决策"
    )
    importance: int = Field(ge=1, le=5, default=3, description="重要性评分 1-5")
    project_ref: Optional[str] = Field(default=None, description="关联项目名称")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 时间戳"
    )
    tags: list[str] = Field(default_factory=list, description="自定义标签")

    def to_document(self) -> str:
        """转为纯文本存入 ChromaDB（用于向量检索）."""
        parts = [f"[{self.category}]", self.fact]
        if self.project_ref:
            parts.append(f"项目: {self.project_ref}")
        if self.tags:
            parts.append(f"标签: {', '.join(self.tags)}")
        return " | ".join(parts)

    def to_metadata(self) -> dict:
        """转为 ChromaDB metadata 格式."""
        return {
            "fact": self.fact,
            "category": self.category,
            "importance": self.importance,
            "project_ref": self.project_ref or "",
            "timestamp": self.timestamp,
            "tags": ", ".join(self.tags) if self.tags else "",
        }

    def is_more_comprehensive_than(self, other: "MemoryRecord") -> bool:
        """判断新记忆是否比旧记忆更全面（用于 Upsert 冲突处理）."""
        if self.category != other.category:
            return False
        # 新记忆更长（内容更丰富）
        return len(self.fact) > len(other.fact) * 1.2
