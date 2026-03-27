"""冲突决策器 — 当多个 Worker 返回结果冲突时，使用置信度权重决策."""

import logging
from typing import Annotated, TypedDict

logger = logging.getLogger(__name__)

# ── 置信度权重配置 ────────────────────────────────────────────────────────
WEIGHT_WEB = 0.95     # 实时搜索（最新、最权威）
WEIGHT_RAG = 0.70     # 内部文档（项目配置等）
WEIGHT_MEMORY = 0.50   # 历史对话（最旧）


class DataEntry(TypedDict):
    """带置信度标记的数据条目。"""
    content: str
    source: str        # "web" | "rag" | "memory" | "coder"
    confidence: float  # 0.0 - 1.0


def resolve_conflict(entries: list[DataEntry]) -> DataEntry:
    """
    冲突决策：返回最高置信度的数据条目。

    权重规则：
    - 实时搜索 (web): 0.95 — 最新信息、权威来源
    - 内部文档 (rag): 0.70 — 项目特定配置
    - 历史对话 (memory): 0.50 — 上下文记忆

    Args:
        entries: 多个 Worker 返回的数据条目列表

    Returns:
        置信度最高的 DataEntry
    """
    if not entries:
        return {"content": "", "source": "none", "confidence": 0.0}

    if len(entries) == 1:
        return entries[0]

    # 按置信度降序排序
    sorted_entries = sorted(entries, key=lambda e: e.get("confidence", 0.0), reverse=True)

    winner = sorted_entries[0]
    logger.info(
        f"[conflict_resolver] resolved conflict: "
        f"winner={winner.get('source')} (conf={winner.get('confidence')}) "
        f"from {len(entries)} candidates"
    )

    return winner


def detect_conflict(entries: list[DataEntry]) -> bool:
    """
    检测是否存在数据冲突。

    冲突条件：
    - 同一主题有多个不同来源的结果
    - 时间戳差异超过 1 年

    Args:
        entries: 数据条目列表

    Returns:
        True 如果检测到冲突
    """
    if len(entries) < 2:
        return False

    sources = set(e.get("source", "") for e in entries)

    # 多个不同来源 = 可能冲突
    if len(sources) >= 2:
        # 检查时间戳差异
        import re
        timestamps = []
        for entry in entries:
            content = entry.get("content", "")
            match = re.search(r"\d{4}", content)
            if match:
                try:
                    timestamps.append(int(match.group()))
                except ValueError:
                    pass

        if len(timestamps) >= 2:
            year_diff = max(timestamps) - min(timestamps)
            if year_diff >= 1:
                logger.info(f"[conflict_resolver] conflict detected: year_diff={year_diff}, sources={sources}")
                return True

    return False


def build_context_summary(entries: list[DataEntry]) -> str:
    """
    将多个数据条目合并为一个上下文摘要。

    按置信度从高到低排列，每条附带来历标签。

    Args:
        entries: 数据条目列表

    Returns:
        合并后的上下文字符串
    """
    if not entries:
        return "无上下文信息"

    sorted_entries = sorted(entries, key=lambda e: e.get("confidence", 0.0), reverse=True)

    lines = ["=== 多源上下文摘要 ==="]
    for i, entry in enumerate(sorted_entries, 1):
        source = entry.get("source", "unknown")
        confidence = entry.get("confidence", 0.0)
        content = entry.get("content", "")[:500]  # 截断避免过长

        source_label = {
            "web": "[WEB-最新]",
            "rag": "[RAG-内部]",
            "memory": "[MEMORY-历史]",
            "coder": "[CODER-生成]",
        }.get(source, f"[{source}]")

        lines.append(f"\n{i}. {source_label} (置信度: {confidence:.2f})")
        lines.append(f"   {content}")

    return "\n".join(lines)
