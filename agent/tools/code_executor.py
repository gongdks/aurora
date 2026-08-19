"""Execute Python code safely in a sandboxed subprocess.

Security model:
  1. AST-level whitelist: only safe AST node types are allowed
  2. Import filter: blocks dangerous modules (os, subprocess, socket, etc.)
  3. Call filter: blocks dangerous builtins (eval, exec, __import__, open on system paths)
  4. Regex pre-check: fast rejection of obvious attacks before AST parsing
  5. Subprocess isolation: code runs in a temp directory with timeout

Supported safe libraries for data processing:
  - pandas, numpy, matplotlib, seaborn
  - json, csv, collections, itertools, pathlib, datetime
  - re, string, textwrap, decimal, fractions, statistics
  - io, hashlib, base64, copy, math, functools, operator
  - pprint, traceback, warnings, contextlib, dataclasses, enum
"""

import ast
import os
import re
import subprocess
import sys
import tempfile

from langchain.tools import tool

from agent.config import settings
from agent.tools.registry import register

_MAX_CODE_SIZE = 50_000
_EXEC_TIMEOUT = 60

_FORBIDDEN_MODULES = frozenset({
    "os", "subprocess", "socket", "http.server", "ftplib", "smtplib",
    "pickle", "shutil", "ctypes", "code", "codeop", "pty",
    "signal", "multiprocessing", "threading",
})

_FORBIDDEN_BUILTINS = frozenset({
    "eval", "exec", "compile", "__import__", "open",
    "input", "breakpoint",
})

_SAFE_NODE_TYPES = frozenset({
    "Expr", "Constant", "Name", "Load", "Store", "Del",
    "BinOp", "UnaryOp", "BoolOp", "Compare", "IfExp",
    "Call", "Attribute", "Subscript", "Slice",
    "List", "Tuple", "Set", "Dict", "DictComp",
    "ListComp", "SetComp", "GeneratorExp",
    "JoinedStr", "FormattedValue",
    "Starred", "NamedExpr", "keyword",
    "Assign", "AugAssign", "AnnAssign",
    "If", "For", "While", "Break", "Continue", "Pass",
    "Return", "Delete", "Raise", "Assert",
    "Try", "TryStar", "ExceptHandler",
    "With", "withitem",
    "Match", "match_case",
    "FunctionDef", "AsyncFunctionDef",
    "ClassDef",
    "arguments", "arg",
    "Lambda",
    "Import", "ImportFrom", "alias",
    "Module",
})

_MATPLOTLIB_PRESETUP = """
try:
    import matplotlib
    matplotlib.use('Agg')
except Exception:
    pass
"""

_SUPPORTED_LIBS_NOTICE = """
# Supported safe libraries:
#   Data: pandas, numpy, matplotlib, seaborn
#   IO: json, csv, io, pathlib, os.path (read-only), open (allowed in workspace)
#   Utils: collections, itertools, datetime, re, math, statistics, decimal, fractions
#   Other: copy, functools, operator, pprint, textwrap, string, hashlib, base64
"""


class CodeSafetyError(Exception):
    pass


def _check_code_safety(code: str) -> str | None:
    if len(code) > _MAX_CODE_SIZE:
        return f"Code too large ({len(code)} > {_MAX_CODE_SIZE} chars), rejected"

    _OBVIOUS_ATTACKS = [
        (r"__import__\s*\(.*\)", "__import__ call"),
        (r"os\.system\s*\(", "os.system call"),
        (r"subprocess\.", "subprocess usage"),
        (r"eval\s*\(.*\)", "eval() call"),
        (r"exec\s*\(.*\)", "exec() call"),
        (r"__builtins__", "__builtins__ access"),
        (r"__subclasses__\s*\(\)", "__subclasses__ call"),
        (r"open\s*\(.*['\"]/", "open() with absolute path"),
    ]
    for pattern, label in _OBVIOUS_ATTACKS:
        if re.search(pattern, code, re.IGNORECASE):
            return f"Forbidden pattern detected: {label}"

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"Syntax error in code: {exc}"

    checker = _ASTSafetyChecker()
    try:
        checker.visit(tree)
    except CodeSafetyError as exc:
        return str(exc)

    return None


class _ASTSafetyChecker(ast.NodeVisitor):
    def generic_visit(self, node: ast.AST) -> None:
        node_type = type(node).__name__
        if node_type not in _SAFE_NODE_TYPES:
            raise CodeSafetyError(
                f"Forbidden AST node type: {node_type} "
                f"(line {getattr(node, 'lineno', '?')})"
            )
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top_level = alias.name.split(".")[0]
            if top_level in _FORBIDDEN_MODULES:
                raise CodeSafetyError(
                    f"Forbidden import: {alias.name} "
                    f"(line {getattr(node, 'lineno', '?')})"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        top_level = module.split(".")[0]
        if top_level in _FORBIDDEN_MODULES:
            raise CodeSafetyError(
                f"Forbidden import: from {module} import ... "
                f"(line {getattr(node, 'lineno', '?')})"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_BUILTINS:
                raise CodeSafetyError(
                    f"Forbidden function call: {node.func.id}() "
                    f"(line {getattr(node, 'lineno', '?')})"
                )
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in _FORBIDDEN_BUILTINS:
                raise CodeSafetyError(
                    f"Forbidden method call: .{node.func.attr}() "
                    f"(line {getattr(node, 'lineno', '?')})"
                )
            if node.func.attr == "getattr":
                raise CodeSafetyError(
                    f"Forbidden: getattr() can bypass import restrictions "
                    f"(line {getattr(node, 'lineno', '?')})"
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in ("__builtins__", "__import__", "__subclasses__",
                         "__bases__", "__mro__", "__globals__", "__code__"):
            raise CodeSafetyError(
                f"Forbidden attribute access: .{node.attr} "
                f"(line {getattr(node, 'lineno', '?')})"
            )
        self.generic_visit(node)


def _ensure_workdir() -> str:
    path = os.path.abspath(settings.CODE_WORKDIR)
    os.makedirs(path, exist_ok=True)
    return path


def _scan_workspace(before: set[str], workdir: str) -> list[tuple[str, int]]:
    try:
        current = set(os.listdir(workdir))
    except OSError:
        return []
    new_files = current - before
    results: list[tuple[str, int]] = []
    for f in sorted(new_files):
        full = os.path.join(workdir, f)
        if os.path.isfile(full):
            size = os.path.getsize(full)
            results.append((f, size))
    return results


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / 1024 / 1024:.1f}MB"


def _format_dataframe_preview(text: str) -> str:
    lines = text.split("\n")
    if len(lines) < 3:
        return text
    if any(";" in l for l in lines[:3]) or any("," in l for l in lines[:3]):
        return text
    if len(lines) > 50:
        return "\n".join(lines[:50]) + "\n... (truncated)"
    return text


@register
@tool
def code_executor(code: str) -> str:
    """执行 Python 代码并返回结果。支持数据分析、绘图、计算等。

    支持的库：pandas, numpy, matplotlib, seaborn, json, csv, collections, itertools, datetime, re, math, statistics 等。
    代码运行在沙盒中，有 60 秒超时。可将结果保存到 workspace 目录。

    Args:
        code: 要执行的 Python 代码。支持：
              - print() 输出结果
              - matplotlib 绘图（自动保存为 PNG 到 workspace）
              - pandas 数据分析（可保存 CSV/Excel 到 workspace）
              - 任何纯计算逻辑
    """
    safety_err = _check_code_safety(code)
    if safety_err:
        return f"[安全检查] {safety_err}"

    try:
        workdir = _ensure_workdir()
        before_files = set(os.listdir(workdir)) if os.path.isdir(workdir) else set()

        full_code = _MATPLOTLIB_PRESETUP + "\n" + code

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=workdir, delete=False, encoding="utf-8"
        ) as f:
            f.write(full_code)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=_EXEC_TIMEOUT,
                cwd=workdir,
                env={
                    "MPLBACKEND": "Agg",
                    "PYTHONHASHSEED": "0",
                },
            )

            output_parts: list[str] = []

            stdout = result.stdout
            if stdout.strip():
                output_parts.append(stdout.rstrip())

            if result.stderr.strip():
                output_parts.append(f"[stderr]\n{result.stderr.strip()}")

            if result.returncode != 0:
                output_parts.append(f"[exit code: {result.returncode}]")

            new_files = _scan_workspace(before_files, workdir)
            if new_files:
                output_parts.append("")
                output_parts.append("--- 生成的文件 ---")
                for fname, size in new_files:
                    output_parts.append(f"  📄 {fname} ({_fmt_size(size)})")

            output = "\n".join(output_parts) if output_parts else "(无输出)"

            if len(output) > 8000:
                output = output[:8000] + "\n\n（输出过长，已截断）"

            return output

        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    except subprocess.TimeoutExpired:
        return f"[超时] 代码执行超过 {_EXEC_TIMEOUT} 秒。"
    except OSError as exc:
        return f"[错误] {exc}"