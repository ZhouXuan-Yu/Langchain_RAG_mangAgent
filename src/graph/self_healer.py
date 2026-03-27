"""自修复逻辑 — 捕获错误并触发 LLM 驱动的自我修复."""

import logging
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

MAX_HEAL_ATTEMPTS = 2  # 最多自修复 2 次


async def self_heal(
    failed_task: dict,
    error_log: str,
    context: list[dict] | None = None,
    max_attempts: int = MAX_HEAL_ATTEMPTS,
    progress_callback: Callable[[dict], Awaitable[None]] | None = None,
) -> dict:
    """
    自修复循环：

    1. 捕获错误（JSON 解析失败 / 工具调用失败）
    2. 将错误日志回传给 LLM
    3. 要求修复后重试
    4. 超过 max_attempts 则放弃

    Args:
        failed_task: 失败的任务字典
        error_log: 错误日志字符串
        context: 之前成功执行的 Worker 结果
        max_attempts: 最大重试次数
        progress_callback: 进度回调

    Returns:
        修复后的结果字典，或包含错误信息的字典
    """
    task_id = failed_task.get("task_id", "unknown")
    description = failed_task.get("description", "")
    agent = failed_task.get("assigned_agent", "unknown")
    attempt = 0

    while attempt < max_attempts:
        attempt += 1

        logger.info(f"[self_healer] healing attempt #{attempt} for task {task_id}")

        if progress_callback:
            await progress_callback({
                "type": "self_healing",
                "agent": "self_healer",
                "task_id": task_id,
                "attempt": attempt,
                "reason": error_log[:200],
            })

        # ── 构建修复 Prompt ───────────────────────────────────────
        context_summary = _build_error_context(context, error_log, attempt)

        try:
            fixed_result = await _llm_self_heal(description, agent, error_log, context_summary, attempt)

            # 检查修复是否有效
            if not _is_repair_valid(fixed_result):
                error_log = f"[修复 #{attempt}] 结果无效: {fixed_result[:100]}"
                logger.warning(f"[self_healer] repair #{attempt} invalid: {fixed_result[:100]}")
                continue

            logger.info(f"[self_healer] healing succeeded on attempt #{attempt}")

            return {
                "task_id": task_id,
                "agent": agent,
                "result": fixed_result,
                "confidence": 0.6,  # 自修复结果置信度略低
                "quality_score": 6.0,
                "source": "self_healed",
                "healing_attempts": attempt,
                "raw_data": fixed_result,
            }

        except Exception as e:
            error_log = f"[修复 #{attempt}] 异常: {str(e)}"
            logger.error(f"[self_healer] healing attempt #{attempt} failed: {e}")

    # 超过最大尝试次数，触发人工介入
    logger.warning(f"[self_healer] max attempts ({max_attempts}) exceeded for task {task_id}")

    if progress_callback:
        await progress_callback({
            "type": "human_intervention",
            "agent": "self_healer",
            "task_id": task_id,
            "message": f"自修复失败（已尝试 {max_attempts} 次），建议人工介入",
        })

    return {
        "task_id": task_id,
        "agent": agent,
        "result": f"[自修复失败] 已尝试 {max_attempts} 次仍无法完成，建议人工介入。\n\n原始错误:\n{error_log}",
        "confidence": 0.0,
        "quality_score": 0.0,
        "source": "failed",
        "healing_attempts": max_attempts,
        "requires_human": True,
    }


async def _llm_self_heal(
    description: str,
    agent: str,
    error_log: str,
    context_summary: str,
    attempt: int,
) -> str:
    """调用 LLM 进行自我修复。"""
    from src.llm import init_deepseek_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = init_deepseek_llm(temperature=0.2, streaming=False)

    agent_hints = {
        "search_worker": "搜索任务",
        "rag_worker": "记忆检索任务",
        "coder": "代码生成任务",
    }

    prompt = f"""你是一个任务自修复助手。以下任务执行失败，请分析错误并尝试修复。

任务类型: {agent_hints.get(agent, '未知任务')}
任务描述: {description}

错误信息:
{error_log}

相关上下文:
{context_summary}

要求:
1. 分析错误原因
2. 提出修复方案
3. 重新执行任务
4. 输出修复后的结果

修复尝试 #{attempt}:
"""

    response = await llm.ainvoke([
        SystemMessage(content="你是一个专业的任务自修复助手，擅长分析错误并提供修复方案。"),
        HumanMessage(content=prompt),
    ])

    return response.content if hasattr(response, "content") else str(response)


def _build_error_context(context: list[dict] | None, error_log: str, attempt: int) -> str:
    """构建错误上下文字符串。"""
    if not context:
        return "无相关上下文（首次执行失败）"

    lines = []
    for ctx in context:
        agent = ctx.get("agent", "unknown")
        raw = ctx.get("raw_data", "")[:200]
        lines.append(f"[{agent}]: {raw}")

    return "\n".join(lines)


def _is_repair_valid(result: str) -> bool:
    """检查修复结果是否有效。"""
    if not result or len(result) < 20:
        return False

    # 排除明显的错误标记
    invalid_markers = [
        "[修复失败]",
        "[无法修复]",
        "抱歉，我无法",
        "I cannot",
        "[ERROR]",
        "修复失败",
    ]

    for marker in invalid_markers:
        if marker in result:
            return False

    return True
