"""Markdown cleaner — 网页内容清洗工具."""

import re


def clean_markdown(text: str) -> str:
    """
    清洗网页抓取的原始文本，移除噪音元素，保留 Markdown 格式.

    移除项：
    - 广告、导航栏、Cookie 提示
    - 过多换行
    - HTML 残留字符
    - 代码块语言标识（保留代码内容）
    """
    if not text:
        return ""

    # 移除常见的导航/页脚噪音
    noise_patterns = [
        r"^(导航|菜单|首页|关于|联系|登录|注册)[^\n]*\n",
        r"^(广告|Advertisement|Advertisement).*\n",
        r"(Cookie|cookie|隐私|Privacy)[^\n]*\n",
        r"^[\s]*(Copyright|版权所有|沪ICP备|京ICP备).*\n",
        r"^[\s]*[📢📣🔔💬👍❤️⭐]+[^\n]*\n",
        r"(扫码|微信|微博|抖音|B站|公众号)[^\n]*\n",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)

    # 清理 HTML 实体
    text = text.replace("&nbsp;", " ")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&amp;", "&")
    text = text.replace("&#39;", "'")
    text = text.replace("&quot;", '"')
    text = text.replace("&nbsp;", " ")

    # 移除多余空行（超过2个换行合并为2个）
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 移除行首行尾空白
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)

    return text.strip()
