"""PII redactor — 10: 输入脱敏中间件."""

import re
import copy
import logging

logger = logging.getLogger(__name__)

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


def pii_pre_model_hook(state: dict, config: dict | None = None) -> dict:
    """
    LangGraph pre_model_hook — 仅在 LLM 调用前对用户消息做 PII 脱敏。

    注意：图结构为 agent → tools → pre_model_hook → agent，工具在 pre_model 之前执行，
    因此不能在此处拦截 tool_calls；参数校验应在各工具实现内完成。
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        mtype = getattr(msg, "type", None)
        if mtype != "human":
            continue
        if not hasattr(msg, "content") or not isinstance(msg.content, str):
            return {}
        redacted = redact_pii(msg.content)
        if redacted == msg.content:
            return {}
        new_msg = copy.copy(msg)
        new_msg.content = redacted
        new_list = list(messages)
        new_list[idx] = new_msg
        return {"messages": new_list}

    return {}
