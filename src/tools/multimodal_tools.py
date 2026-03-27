"""Multimodal tools — 19-21: 图片处理与特征提取."""

import base64
import io
import logging
from typing import Any, Union

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from src.memory.memory_schema import MemoryRecord
from src.memory.chroma_store import ChromaMemoryStore

logger = logging.getLogger(__name__)


def encode_image_path(image_path: str) -> str:
    """读取本地图片文件并转为 base64 编码字符串."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_image_features(image_data: str, llm: Any) -> str:
    """
    使用 LLM 提取图像描述文本 — 作为记忆存储的中间件.

    Args:
        image_data: base64 编码的图片数据（带或不带 data URI 前缀）
        llm: LangChain LLM 实例（用于多模态推理）

    Returns:
        LLM 生成的图像描述文本
    """
    # 移除 data URI 前缀（如果存在）
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    prompt = (
        "描述这张图片的关键内容，包括：\n"
        "1. 图像类型（截图/照片/图表/代码/文档）\n"
        "2. 主要视觉元素和信息\n"
        "3. 任何技术相关内容（如代码片段、架构图、错误信息）\n"
        "4. 整体意图或上下文\n"
        "用简洁、准确的文本描述。"
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )

    response = llm.invoke([message])
    return response.content if hasattr(response, "content") else str(response)


def _infer_category_from_text(text: str) -> str:
    """根据图片描述文本推断记忆类别."""
    text_lower = text.lower()
    if any(k in text_lower for k in ["代码", "code", "bug", "error", "terminal"]):
        return "tech_stack"
    if any(k in text_lower for k in ["架构", "diagram", "chart", "graph"]):
        return "project"
    if any(k in text_lower for k in ["硬件", "hardware", "gpu", "cpu", "设备"]):
        return "hardware"
    return "project"


@tool
def process_image(image_path: str, description_hint: str = "") -> str:
    """处理用户上传的图片，提取特征并可选存入记忆。

    当用户上传截图、照片或图表时，使用此工具提取信息。
    适用于：代码截图、架构图、错误信息截图、项目文档等。

    Args:
        image_path: 图片文件路径（绝对路径或相对于当前目录）
        description_hint: 可选的描述提示（帮助 LLM 更准确理解图片）

    Returns:
        提取的图片描述内容
    """
    try:
        from src.llm.deepseek_client import init_deepseek_llm
    except ImportError:
        return "[系统] LLM 客户端未安装，无法处理图片。"

    try:
        img_data = encode_image_path(image_path)
    except FileNotFoundError:
        return f"[系统] 图片文件未找到: {image_path}"
    except Exception as e:
        return f"[系统] 读取图片失败: {e}"

    try:
        llm = init_deepseek_llm(streaming=False)
        description = extract_image_features(img_data, llm)

        # 如果用户提供了描述提示，追加到描述中
        if description_hint:
            description = f"{description_hint}\n\n附加信息: {description}"

        # 自动推断类别并存入记忆
        category = _infer_category_from_text(description)
        try:
            store = ChromaMemoryStore()
            record = MemoryRecord(
                fact=description,
                category=category,
                importance=4,
                timestamp=__import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                tags=["图片上传", "multimodal"],
            )
            result = store.upsert_record(record)
            save_note = f"\n[图片特征已存入记忆，类别: {category}，结果: {result}]"
        except Exception as e:
            logger.warning(f"Failed to save image to memory: {e}")
            save_note = ""

        return description + save_note

    except Exception as e:
        logger.error(f"process_image failed: {e}")
        return f"[系统] 图片处理失败: {e}"
