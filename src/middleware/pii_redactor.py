"""PII redactor — 10: 输入脱敏中间件."""

import re
import copy
from typing import Any

_PII_PATTERNS = None


def _get_patterns() -> dict:
    """延迟编译 PII 正则，提升首次匹配性能."""
    global _PII_PATTERNS
    if _PII_PATTERNS is None:
        _PII_PATTERNS = {
            "phone": (re.compile(r"\b1[3-9]\d{9}\b"), "[手机号]"),
            "email": (re.compile(r"\b[\w.-]+@[\w.-]+\.\w{2,}\b"), "[邮箱]"),
            "id_card": (re.compile(r"\b\d{17}[\dXx]\b"), "[身份证号]"),
            "bank_card": (re.compile(r"\b\d{16,19}\b"), "[银行卡号]"),
        }
    return _PII_PATTERNS


def redact_pii(text: str) -> str:
    """检测并替换文本中的 PII（个人信息）."""
    if not text:
        return text
    for pattern, replacement in _get_patterns().values():
        text = pattern.sub(replacement, text)
    return text


def before_model(state: dict, config: dict | None = None) -> dict:
    """
    LangGraph before_model 中间件 — 在 LLM 调用前自动脱敏。

    只对最新消息做 PII 脱敏（O(1)），避免 O(n) 遍历整个历史。
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_item = messages[-1]
    if not hasattr(last_item, "content") or not isinstance(last_item.content, str):
        return {}

    redacted = redact_pii(last_item.content)
    if redacted == last_item.content:
        return {}

    new_msg = copy.copy(last_item)
    new_msg.content = redacted
    return {"messages": messages[:-1] + [new_msg]}
