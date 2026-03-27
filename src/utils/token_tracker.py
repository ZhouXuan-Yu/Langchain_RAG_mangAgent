"""Token tracker — 22-24: Token 使用量追踪与成本统计."""

from datetime import datetime
from typing import Literal

import tiktoken


class TokenTracker:
    """
    追踪 DeepSeek API 的 Token 消耗与成本.

    使用 cl100k_base 编码器（与 DeepSeek 兼容）计算 token 数，
    记录每轮调用的 prompt/completion tokens 及估算成本。
    """

    def __init__(self, model: str = "deepseek-chat"):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.model = model
        self.total_tokens: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.history: list[dict] = []

    def count(self, text: str) -> int:
        """计算文本的 token 数量."""
        return len(self.enc.encode(text))

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str | None = None,
        label: str = "",
    ) -> None:
        """记录一次 API 调用的 token 消耗.

        Args:
            prompt_tokens: prompt 部分的 token 数
            completion_tokens: completion 部分的 token 数
            model: 模型名称（用于计算单价）
            label: 调用标签（如 "search", "reason"）
        """
        model = model or self.model
        cost = self._calc_cost(prompt_tokens, completion_tokens, model)

        self.total_tokens += prompt_tokens + completion_tokens
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd += cost

        self.history.append({
            "time": datetime.now().isoformat(),
            "label": label,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": round(cost, 6),
        })

    def _calc_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> float:
        """根据模型计算 USD 成本."""
        # DeepSeek Chat (DeepSeek-V3) pricing (approximate)
        pricing = {
            "deepseek-chat": (0.00027, 0.0011),  # (prompt_per_1k, completion_per_1k)
            "deepseek-reasoner": (0.0011, 0.0027),
        }
        p_rate, c_rate = pricing.get(model, (0.001, 0.002))
        return (prompt_tokens / 1000 * p_rate) + (completion_tokens / 1000 * c_rate)

    def summary(self) -> dict:
        """返回使用摘要."""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "num_calls": len(self.history),
        }

    def get_history(self, days: int = 7) -> list[dict]:
        """返回最近 N 天的历史记录（按天聚合）."""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        # 按天聚合
        by_date: dict[str, dict] = {}
        for h in self.history:
            ts = h.get("time", "")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            if dt < cutoff:
                continue
            date_key = dt.strftime("%Y-%m-%d")
            if date_key not in by_date:
                by_date[date_key] = {
                    "timestamp": ts,
                    "date": date_key,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "num_calls": 0,
                    "cost_usd": 0.0,
                    "model": h.get("model", ""),
                }
            d = by_date[date_key]
            d["prompt_tokens"] += h["prompt_tokens"]
            d["completion_tokens"] += h["completion_tokens"]
            d["total_tokens"] += h["total_tokens"]
            d["num_calls"] += 1
            d["cost_usd"] += h["cost_usd"]
            d["timestamp"] = max(d["timestamp"], ts)
        # 返回按日期降序排列
        result = list(by_date.values())
        result.sort(key=lambda x: x["date"], reverse=True)
        return result

    def print_report(self) -> None:
        """打印成本报告到控制台."""
        s = self.summary()
        print("\n========== Token 成本报告 ==========")
        print(f"  API 调用次数: {s['num_calls']}")
        print(f"  Prompt Tokens:  {s['prompt_tokens']:,}")
        print(f"  Completion Tokens: {s['completion_tokens']:,}")
        print(f"  总 Token:     {s['total_tokens']:,}")
        print(f"  总成本:       ${s['total_cost_usd']:.6f}")
        print("=====================================\n")
