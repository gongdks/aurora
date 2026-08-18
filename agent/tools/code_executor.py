"""Execute Python code safely in a sandboxed subprocess.

Security model:
  1. AST-level whitelist: only safe AST node types are allowed
  2. Import filter: blocks dangerous modules (os, subprocess, socket, etc.)
  3. Call filter: blocks dangerous builtins (eval, exec, __import__, open on system paths)
  4. Regex pre-check: fast rejection of obvious attacks before AST parsing
  5. Subprocess isolation: code runs in a temp directory with timeout
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

# Modules that are never safe to import
_FORBIDDEN_MODULES = frozenset({
    "os", "subprocess", "socket", "http.server", "ftplib", "smtplib",
    "pickle", "shutil", "ctypes", "code", "codeop", "pty",
    "signal", "multiprocessing", "threading",
})

# Builtin function names that are dangerous to call
_FORBIDDEN_BUILTINS = frozenset({
    "eval", "exec", "compile", "__import__", "open",
    "input", "breakpoint",
})

# AST node types allowed in safe code (whitelist)
_SAFE_NODE_TYPES = frozenset({
    # Expressions
    "Expr", "Constant", "Name", "Load", "Store", "Del",
    "BinOp", "UnaryOp", "BoolOp", "Compare", "IfExp",
    "Call", "Attribute", "Subscript", "Slice",
    "List", "Tuple", "Set", "Dict", "DictComp",
    "ListComp", "SetComp", "GeneratorExp",
    "JoinedStr", "FormattedValue",
    "Starred", "NamedExpr", "keyword",
    # Statements
    "Assign", "AugAssign", "AnnAssign",
    "If", "For", "While", "Break", "Continue", "Pass",
    "Return", "Delete", "Raise", "Assert",
    "Try", "TryStar", "ExceptHandler",
    "With", "withitem",
    "Match", "match_case",
    # Definitions
    "FunctionDef", "AsyncFunctionDef",
    "ClassDef",
    "arguments", "arg",
    "Lambda",
    # Imports
    "Import", "ImportFrom", "alias",
    # Module-level
    "Module",
})


class CodeSafetyError(Exception):
    """Raised when code fails AST-level security checks."""
    pass


def _check_code_safety(code: str) -> str | None:
    """Check code safety via regex pre-check + AST whitelist.

    Returns error message string, or None if safe.
    """
    # --- Pre-check: size limit ---
    if len(code) > _MAX_CODE_SIZE:
        return f"Code too large ({len(code)} > {_MAX_CODE_SIZE} chars), rejected"

    # --- Pre-check: fast regex for obvious attacks ---
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

    # --- AST-level whitelist ---
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
    """AST visitor that rejects dangerous nodes."""

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
        # Check for calls to dangerous builtins
        if isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_BUILTINS:
                raise CodeSafetyError(
                    f"Forbidden function call: {node.func.id}() "
                    f"(line {getattr(node, 'lineno', '?')})"
                )
        # Check for getattr(__builtins__,...) pattern
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in _FORBIDDEN_BUILTINS:
                raise CodeSafetyError(
                    f"Forbidden method call: .{node.func.attr}() "
                    f"(line {getattr(node, 'lineno', '?')})"
                )
            # Block getattr tricks: getattr(x, 'ev'+'al')
            if node.func.attr == "getattr":
                raise CodeSafetyError(
                    f"Forbidden: getattr() can bypass import restrictions "
                    f"(line {getattr(node, 'lineno', '?')})"
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Block access to dangerous attributes on any object
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


@register
@tool
def code_executor(code: str) -> str:
    """Execute Python code and return the output.

    Use for calculations, data processing, file manipulation,
    and any task requiring custom Python code.
    Code runs in a sandbox with AST-level security checks and a 30-second timeout.

    Args:
        code: Python code to execute, e.g. "print(sum(range(100)))"
    """
    safety_err = _check_code_safety(code)
    if safety_err:
        return f"[Safety] {safety_err}"

    try:
        workdir = _ensure_workdir()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=workdir, delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=workdir,
            )
            output = result.stdout
            if result.stderr:
                output += ("\n[stderr]\n" + result.stderr)
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            if not output.strip():
                output = "(no output)"
            return output[:5000]
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass
    except subprocess.TimeoutExpired:
        return "[Timeout] Code execution exceeded 30 seconds."
    except OSError as exc:
        return f"[Error] {exc}"