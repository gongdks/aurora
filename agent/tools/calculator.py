"""安全数学计算器。

支持四则运算、整除、取模、乘方、括号、数学常量和常用函数。
通过 AST 白名单机制防止代码注入。
"""

import ast
import math
import operator

from langchain.tools import tool

from agent.tools.registry import register

_MAX_EXPR_LEN = 200
_MAX_POW_EXPONENT = 1000

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}

_CHINESE_MAP = {
    "＋": "+",
    "－": "-", "﹣": "-",
    "×": "*", "＊": "*",
    "÷": "/", "／": "/",
    "％": "%",
}


def _normalize_expr(expr: str) -> str:
    """预处理表达式：替换中文运算符、统一 ^ 为 **、去空格。"""
    for zh, en in _CHINESE_MAP.items():
        expr = expr.replace(zh, en)
    expr = expr.replace("^", "**")
    return expr.strip()


def _walk(node: ast.AST) -> float:
    """递归遍历 AST 节点，安全求值。"""
    if isinstance(node, ast.Expression):
        return _walk(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不允许的值类型: {type(node.value).__name__}")

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"未知标识符: '{node.id}'，仅支持 pi/e/tau")

    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        left = _walk(node.left)
        right = _walk(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_POW_EXPONENT:
                raise ValueError(f"指数过大（{abs(right)} > {_MAX_POW_EXPONENT}）")
        return _SAFE_OPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_walk(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("仅支持直接函数调用，不支持链式或属性访问")
        fname = node.func.id
        if fname not in _SAFE_FUNCTIONS:
            raise ValueError(f"不支持的函数: '{fname}'")
        if node.keywords:
            raise ValueError(f"不支持关键字参数")
        args = [_walk(a) for a in node.args]
        return _SAFE_FUNCTIONS[fname](*args)

    raise ValueError(f"不支持的语法: {ast.dump(node)}")


def _safe_eval(expr: str) -> float:
    """安全计算，仅允许白名单内的运算和函数。"""
    expr = _normalize_expr(expr)
    if len(expr) > _MAX_EXPR_LEN:
        raise ValueError(f"表达式过长（{len(expr)} 字符），最多支持 {_MAX_EXPR_LEN} 字符")
    tree = ast.parse(expr, mode="eval")
    return _walk(tree)


def _format_result(value: float) -> str:
    """格式化结果：整数显示为整数，浮点保留合理精度。"""
    if isinstance(value, int) or (isinstance(value, float) and value == int(value) and abs(value) < 1e15):
        return str(int(value))
    rounded = round(value, 10)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded}"


@register
@tool
def math_calculator(expression: str) -> str:
    """数学计算器，支持四则运算、整除、取模、乘方、括号、常量和常用函数。

    支持的运算符：+  -  *  /  //  %  ^(乘方)
    支持的常量：pi, e, tau
    支持的函数：abs, round, sqrt, sin, cos, tan, log, log2, log10, exp, floor, ceil

    Args:
        expression: 数学表达式，例如 "128 * 56 / 8 + 2^3"、"sqrt(144) + pi"
    """
    try:
        result = _safe_eval(expression)
        formatted = _format_result(result)
        return f"计算结果：{formatted}"
    except ZeroDivisionError:
        return "计算失败：除数不能为零"
    except ValueError as exc:
        return f"计算失败：{exc}"
    except (SyntaxError, TypeError) as exc:
        return f"计算失败：表达式格式错误 — {exc}"