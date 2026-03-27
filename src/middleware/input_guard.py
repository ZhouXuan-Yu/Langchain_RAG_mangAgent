"""Input guard — 10-12: 工具输入验证（在工具实现内调用）."""

from typing import Optional


def validate_search_query(query: str) -> Optional[str]:
    """
    验证 web_search 工具的输入 — 防止 Agent 生成空或异常关键词.

    Returns:
        None 如果验证通过
        str 错误消息 如果验证失败
    """
    if not query or not query.strip():
        return "[系统] 搜索关键词不能为空，已跳过搜索步骤。"
    if len(query) > 500:
        return f"[系统] 搜索关键词过长（{len(query)}字符 > 500），已跳过。"
    if any(char in query for char in ["--", "UNION", "DROP ", "DELETE ", "<script"]):
        return "[系统] 搜索关键词包含可疑字符，已跳过。"
    return None


def validate_memory_fact(fact: str) -> Optional[str]:
    """验证 save_memory 工具的输入."""
    if not fact or not fact.strip():
        return "[系统] 记忆内容不能为空。"
    if len(fact) > 5000:
        return f"[系统] 记忆内容过长（{len(fact)}字符），已截断。"
    return None
