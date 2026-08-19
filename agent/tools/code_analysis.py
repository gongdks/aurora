"""代码静态分析 — AST 语法检查、基本质量扫描。

不依赖外部 linter（pylint/flake8），使用 Python 内置 ast 模块。
对于其他语言，提供基本的语法检查能力。
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys

from langchain.tools import tool

from agent.config import settings
from agent.tools.registry import register
from agent.utils.path_guard import safe_resolve

_MAX_FILE_SIZE = 500_000


# ============================================================================
# 1. lint — Python AST 语法 + 基本质量检查
# ============================================================================

@register(tags={"code"})
@tool
def code_lint(filepath: str) -> str:
    """对 Python 文件执行静态分析，检查语法错误和常见问题。"""
    try:
        safe = safe_resolve(filepath, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"路径错误：{exc}"

    if not os.path.isfile(safe):
        return f"文件不存在：{filepath}"

    if os.path.getsize(safe) > _MAX_FILE_SIZE:
        return f"文件过大（>500KB），跳过分析：{filepath}"

    is_python = safe.endswith(".py")
    rel = os.path.relpath(safe, os.path.realpath(settings.FILE_READER_ROOT))

    parts: list[str] = [f"📋 代码分析: {rel}\n"]

    # ---- Python AST 检查 ----
    if is_python:
        ast_issues = _check_python_ast(safe)
        if ast_issues:
            parts.append("### 语法/结构问题")
            parts.extend(ast_issues)
        else:
            parts.append("✅ 语法检查通过，无结构问题。")
    else:
        parts.append("⚠️ 非 Python 文件，仅支持基本检查。")

    # ---- 尝试运行外部 linter ----
    external_results = _try_external_linter(safe)
    if external_results:
        parts.append("\n### 外部 Linter 结果")
        parts.append(external_results)

    return "\n".join(parts)


def _check_python_ast(filepath: str) -> list[str]:
    """Python AST 语法和基本质量检查。"""
    issues: list[str] = []

    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError as exc:
        return [f"❌ 无法读取文件：{exc}"]

    # 语法检查
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as exc:
        issues.append(f"❌ 语法错误 第 {exc.lineno} 行: {exc.msg}")
        return issues

    # 逐行扫描
    lines = source.split("\n")

    for node in ast.walk(tree):
        # bare except
        if isinstance(node, ast.ExceptHandler):
            if node.type is None and node.name is None:
                lineno = node.lineno
                issues.append(f"⚠️  第 {lineno} 行: bare except（建议指定异常类型）")

        # 过长的 lambda
        if isinstance(node, ast.Lambda):
            end_line = getattr(node, 'end_lineno', node.lineno)
            if end_line - node.lineno > 3:
                issues.append(f"💡 第 {node.lineno} 行: lambda 过长（{end_line - node.lineno + 1} 行），建议用 def")

        # 过深的嵌套
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
            depth = _nesting_depth(node)
            if depth > 4:
                issues.append(f"💡 第 {node.lineno} 行: 嵌套深度 {depth}，建议拆分")

    return issues


def _nesting_depth(node: ast.AST) -> int:
    """计算节点的嵌套深度。"""
    depth = 0
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            depth = max(depth, _nesting_depth(child) + 1)
    return depth


def _try_external_linter(filepath: str) -> str | None:
    """尝试运行 pylint 或 flake8。"""
    for linter_cmd, linter_args in [
        ("pylint", ["--disable=all", "--enable=E,F", filepath]),
        ("flake8", [filepath]),
    ]:
        try:
            result = subprocess.run(
                [linter_cmd] + linter_args,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = (result.stdout + result.stderr).strip()
            if output:
                return f"```\n{output[:2000]}\n```"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return None


# ============================================================================
# 2. typecheck — 运行 mypy 类型检查
# ============================================================================

@register(tags={"code"})
@tool
def code_typecheck(filepath: str = ".") -> str:
    """运行 mypy 类型检查（如果已安装）。"""
    try:
        safe = safe_resolve(filepath, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"路径错误：{exc}"

    # 检查 mypy 是否可用
    try:
        subprocess.run(
            [sys.executable, "-m", "mypy", "--version"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return (
            "⚠️ mypy 未安装。\n"
            "安装方法: pip install mypy\n"
            "安装后可进行 Python 类型检查。"
        )

    try:
        result = subprocess.run(
            [sys.executable, "-m", "mypy", safe, "--ignore-missing-imports"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        if not output:
            return f"✅ mypy 类型检查通过: {filepath}"
        return f"📋 mypy 类型检查: {filepath}\n```\n{output[:3000]}\n```"
    except subprocess.TimeoutExpired:
        return "⏰ mypy 执行超时（>120s）。"
    except OSError as exc:
        return f"❌ mypy 执行失败：{exc}"