"""Web content summarizer — 22-24: 网页摘要压缩，节省 40%+ Token."""

from src.config import WEB_CONTENT_MAX_CHARS


def compress_web_content(raw_text: str, max_chars: int | None = None) -> str:
    """
    在喂给 LLM 前压缩网页内容，节省 Token 消耗.

    策略：优先保留首段（通常为摘要/导语）和末段（通常为总结），
    中间内容按顺序填入，超出部分截断。

    Args:
        raw_text: 原始网页文本
        max_chars: 最大字符数，默认 3000（约 750 tokens）

    Returns:
        压缩后的文本（若原文已 <= max_chars 则直接返回）
    """
    max_chars = max_chars or WEB_CONTENT_MAX_CHARS

    if not raw_text:
        return ""
    if len(raw_text) <= max_chars:
        return raw_text

    paragraphs = raw_text.split("\n\n")
    if not paragraphs:
        return raw_text[:max_chars] + "\n\n[内容已截断]"

    # 策略：保留前 N 段 + 最后一段（总结）
    # 优先保留首段（通常包含最重要的信息）
    compressed: list[str] = []
    current_len = 0
    last_idx = len(paragraphs) - 1

    for i, para in enumerate(paragraphs):
        is_last = (i == last_idx)
        para_len = len(para) + 2  # +2 for \n\n separator

        # 强制保留首段和末段
        if i == 0 or is_last:
            if current_len + para_len <= max_chars:
                compressed.append(para)
                current_len += para_len
            elif current_len == 0:
                # 即使首段超长也要截断保留
                compressed.append(para[:max_chars])
                current_len = max_chars
            continue

        # 中间段落：填充剩余空间
        if current_len + para_len <= max_chars:
            compressed.append(para)
            current_len += para_len
        else:
            # 空间不足，追加截断提示
            break

    result = "\n\n".join(compressed)
    if len(raw_text) > len(result):
        result += f"\n\n[内容已压缩，原长度 {len(raw_text)} 字符]"
    return result
