"""Calculator tool — 04-06: 安全数学计算."""

import math
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """安全计算数学表达式。

    支持的操作符和函数：
    - 基础运算：+, -, *, /, **, %, //
    - 数学函数：sqrt, sin, cos, tan, log, log10, exp, pow
    - 常量：pi, e
    - 括号分组：()
    """
    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pow": pow,
        "abs": abs,
        "floor": math.floor,
        "ceil": math.ceil,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        # Convert numpy arrays or special types to plain Python
        if hasattr(result, "item"):
            result = result.item()
        return str(result)
    except SyntaxError:
        return f"计算错误: 表达式语法不正确 — {expression}"
    except NameError as e:
        return f"计算错误: 包含未定义的函数或变量 — {e}"
    except ZeroDivisionError:
        return "计算错误: 除数不能为零"
    except Exception as e:
        return f"计算错误: {e}"
