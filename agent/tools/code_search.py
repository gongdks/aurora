"""代码搜索工具 — grep（内容搜索）、glob（文件名匹配）、outline（结构提取）。

纯 Python 实现，不依赖外部命令行工具，在项目目录内安全执行。
"""

from __future__ import annotations

import ast
import fnmatch
import os
import re
from pathlib import Path

from langchain.tools import tool

from agent.config import settings
from agent.tools.registry import register
from agent.utils.path_guard import safe_resolve

# ---- 可搜索/匹配的文件扩展名 ----
_TEXT_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".scss", ".less",
    ".json", ".yaml", ".yml", ".toml", ".xml", ".md", ".rst", ".txt",
    ".csv", ".ini", ".cfg", ".conf", ".env", ".sh", ".bat", ".ps1",
    ".sql", ".graphql", ".proto", ".rs", ".go", ".java", ".kt", ".swift",
    ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".vue", ".svelte",
}

_IGNORED_DIRS: set[str] = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".idea", ".vscode", "dist", "build", ".next", ".nuxt",
    "target", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}

_MAX_SEARCH_FILES = 500
_MAX_MATCH_PER_FILE = 200
_MAX_TOTAL_MATCHES = 2000
_MAX_GLOB_RESULTS = 200


def _is_text_file(filepath: str) -> bool:
    """判断文件是否为可搜索的文本类型。"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in _TEXT_EXTENSIONS:
        return True
    # 无扩展名但非二进制的文件也算
    if not ext:
        try:
            with open(filepath, encoding="utf-8", errors="strict") as f:
                f.read(1024)
            return True
        except (OSError, UnicodeDecodeError):
            return False
    return False


def _collect_files(root: str, glob_pattern: str | None = None) -> list[str]:
    """收集 root 下所有可搜索文本文件，可选 glob 过滤。"""
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            if glob_pattern and not fnmatch.fnmatch(fname, glob_pattern):
                continue
            if _is_text_file(full):
                files.append(full)
                if len(files) >= _MAX_SEARCH_FILES:
                    return files
    return files


# ============================================================================
# 1. grep — 正则内容搜索
# ============================================================================

@register
@tool
def grep(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    ignore_case: bool = False,
    context: int = 0,
    max_count: int = 50,
) -> str:
    """在项目文件中搜索匹配正则表达式的内容。

    类似命令行 ripgrep (rg)，在项目目录内安全执行。
    最常用于：查找函数/类定义、引用、TODO 注释、错误处理等。

    Args:
        pattern: 正则表达式，如 "def func_name"、"class MyClass"、"TODO"
        path: 搜索目录，默认 "."（项目根目录）
        glob: 文件名过滤，如 "*.py"、"*.{ts,tsx}"
        ignore_case: 是否忽略大小写
        context: 匹配行前后的上下文行数（-- 输出中标记）
        max_count: 最大返回匹配数（默认 50）
    """
    try:
        root = safe_resolve(path, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"路径错误：{exc}"

    if not os.path.isdir(root):
        return f"路径不是目录：{path}"

    # 解析 glob 过滤
    glob_pattern = None
    if glob:
        # 支持 "*.{ts,tsx}" 转成多个 fnmatch 模式
        glob_pattern = glob

    # 编译正则
    try:
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags | re.MULTILINE)
    except re.error as e:
        return f"正则表达式错误：{e}"

    # 收集文件
    files = _collect_files(root, glob_pattern)
    if not files:
        return f"未找到匹配 '{glob_pattern or '*'}' 的文本文件。"

    # 搜索
    results: list[str] = []
    total_matches = 0
    searched = 0

    for filepath in files:
        if total_matches >= max_count:
            break
        try:
            with open(filepath, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            continue

        searched += 1
        file_matches = 0

        for lineno, line in enumerate(lines, 1):
            if total_matches >= max_count:
                break
            if file_matches >= _MAX_MATCH_PER_FILE:
                break
            if not regex.search(line.rstrip("\n")):
                continue

            file_matches += 1
            total_matches += 1

            rel_path = os.path.relpath(filepath, root)
            line_str = line.rstrip("\n")

            if context > 0:
                ctx_before: list[str] = []
                ctx_after: list[str] = []
                for offset in range(1, context + 1):
                    if lineno - offset >= 1:
                        ctx_before.append(lines[lineno - offset - 1].rstrip("\n"))
                    if lineno + offset <= len(lines):
                        ctx_after.append(lines[lineno + offset - 1].rstrip("\n"))
                results.append(f"{rel_path}:{lineno}")  # file header
                for cl in ctx_before[::-1]:
                    results.append(f"  -{cl}")
                results.append(f"  >{line_str}")
                for cl in ctx_after:
                    results.append(f"  +{cl}")
                results.append("--")
            else:
                results.append(f"{rel_path}:{lineno}: {line_str}")

    if not results:
        return f"在 {searched} 个文件中未找到匹配 '{pattern}' 的内容。"

    header = f"找到 {total_matches} 个匹配 (搜索 {searched} 个文件)"
    if total_matches >= max_count:
        header += f"（已达到上限 {max_count}）"

    return header + "\n" + "\n".join(results)


# ============================================================================
# 2. glob — 文件名模式匹配
# ============================================================================

@register
@tool
def glob(
    pattern: str,
    path: str = ".",
) -> str:
    """在项目目录中查找匹配指定模式的路径。

    支持标准 glob 模式：* 匹配任意字符，** 匹配任意深度目录。

    Args:
        pattern: 匹配模式，如 "**/*.py"、"src/**/*.ts"、"*.json"
        path: 搜索根目录，默认 "."
    """
    try:
        root = safe_resolve(path, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"路径错误：{exc}"

    if not os.path.isdir(root):
        return f"路径不是目录：{path}"

    results: list[str] = []
    pattern_lower = pattern.lower()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""

        for fname in filenames:
            full_rel = os.path.join(rel_dir, fname) if rel_dir else fname
            if fnmatch.fnmatch(full_rel, pattern) or fnmatch.fnmatch(full_rel.lower(), pattern_lower):
                full_path = os.path.join(dirpath, fname)
                size = os.path.getsize(full_path)
                results.append((full_rel, size))

        # 检查目录本身是否匹配
        for dname in dirnames:
            full_rel = os.path.join(rel_dir, dname) + "/" if rel_dir else dname + "/"
            if fnmatch.fnmatch(full_rel, pattern) or fnmatch.fnmatch(full_rel.lower(), pattern_lower):
                results.append((full_rel, -1))

        if len(results) >= _MAX_GLOB_RESULTS:
            break

    if not results:
        return f"未找到匹配 '{pattern}' 的文件或目录。"

    sorted_results = sorted(results, key=lambda x: x[0])
    lines = [f"匹配 '{pattern}' ({len(sorted_results)} 个结果):"]
    for rel_path, size in sorted_results:
        if size < 0:
            lines.append(f"  📁 {rel_path}")
        else:
            lines.append(f"  📄 {rel_path} ({_fmt_size(size)})")

    if len(sorted_results) >= _MAX_GLOB_RESULTS:
        lines.append(f"  ...（已达到上限 {_MAX_GLOB_RESULTS}）")

    return "\n".join(lines)


# ============================================================================
# 3. outline — 代码结构提取
# ============================================================================

def _extract_python_outline(filepath: str) -> list[str]:
    """提取 Python 文件的顶级定义（函数、类、赋值）。"""
    entries: list[str] = []
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            entry = _describe_python_node(node)
            if entry:
                entries.append(entry)
    except (SyntaxError, OSError):
        entries.append("  (无法解析 Python 文件)")
    return entries


def _describe_python_node(node: ast.AST) -> str | None:
    """描述单个 AST 节点。"""
    if isinstance(node, ast.FunctionDef):
        decorators = [d.id for d in node.decorator_list
                      if isinstance(d, ast.Name)]
        prefix = f"@{', @'.join(decorators)} " if decorators else ""
        params = [a.arg for a in node.args.args]
        return f"  def {prefix}{node.name}({', '.join(params)})  [line {node.lineno}]"
    if isinstance(node, ast.AsyncFunctionDef):
        params = [a.arg for a in node.args.args]
        return f"  async def {node.name}({', '.join(params)})  [line {node.lineno}]"
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(b) for b in node.bases] if node.bases else []
        base_str = f"({', '.join(bases)})" if bases else ""
        methods = sum(1 for n in node.body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        return f"  class {node.name}{base_str} — {methods} methods  [line {node.lineno}]"
    if isinstance(node, ast.Assign):
        targets = [ast.unparse(t) for t in node.targets]
        names = ", ".join(targets)
        return f"  {names} = ...  [line {node.lineno}]"
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return f"  {node.target.id}: {ast.unparse(node.annotation)}  [line {node.lineno}]"
    return None


def _extract_generic_outline(filepath: str) -> list[str]:
    """泛用代码结构提取（基于正则，支持 JS/TS/Go/Java 等）。"""
    entries: list[str] = []
    patterns: list[tuple[str, str]] = [
        # JS/TS
        (r'^(export\s+)?(async\s+)?function\s+(\w+)', 'def'),
        (r'^(export\s+)?class\s+(\w+)', 'class'),
        (r'^(export\s+)?(const|let|var)\s+(\w+)\s*=', 'const'),
        # Go
        (r'^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)', 'def'),
        (r'^type\s+(\w+)\s+struct', 'class'),
        # Java/Kotlin
        (r'^\s*(public|private|protected)?\s*(static)?\s*(class|interface)\s+(\w+)', 'class'),
        (r'^\s*(public|private|protected)?\s*(static)?\s*\w+\s+(\w+)\s*\(', 'def'),
    ]
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                for pat, kind in patterns:
                    m = re.match(pat, line.rstrip("\n"))
                    if m:
                        name = m.group(m.lastindex or 1)
                        if kind == 'def':
                            entries.append(f"  func {name}()  [line {lineno}]")
                        elif kind == 'class':
                            entries.append(f"  class {name}  [line {lineno}]")
                        elif kind == 'const':
                            entries.append(f"  const {name} = ...  [line {lineno}]")
                        break
    except OSError:
        return ["  (无法读取文件)"]
    return entries or ["  (未识别到明显结构)"]


OUTLINE_MAX_LINES = 300


@register
@tool
def code_outline(filepath: str) -> str:
    """提取代码文件的结构大纲——列出函数、类、顶层定义。

    用于快速了解代码文件的整体结构，无需阅读全部内容。
    对 Python 文件使用 AST 解析（更精确），其他文件使用正则匹配。

    Args:
        filepath: 相对于项目根目录的代码文件路径，如 "agent/agent.py"
    """
    try:
        safe = safe_resolve(filepath, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"路径错误：{exc}"

    if not os.path.isfile(safe):
        return f"文件不存在：{filepath}"

    ext = os.path.splitext(safe)[1].lower()

    rel = os.path.relpath(safe, os.path.realpath(settings.FILE_READER_ROOT))
    header = f"📄 {rel}"

    if ext == ".py":
        entries = _extract_python_outline(safe)
    else:
        entries = _extract_generic_outline(safe)

    if len(entries) > OUTLINE_MAX_LINES:
        entries = entries[:OUTLINE_MAX_LINES]
        entries.append(f"  ...（已截断，超过 {OUTLINE_MAX_LINES} 条）")

    return header + "\n" + "\n".join(entries)


# ---- helpers ----

def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / 1024 / 1024:.1f}MB"
