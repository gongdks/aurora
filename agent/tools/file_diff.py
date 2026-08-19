"""文件对比工具 — 比较两个文件的差异。

支持文本文件的行级对比，输出类似 diff 的差异格式。
"""

from __future__ import annotations

import difflib
import os

from langchain.tools import tool

from agent.config import settings
from agent.tools.registry import register
from agent.utils.path_guard import safe_resolve

_MAX_DIFF_LINES = 500
_MAX_FILE_SIZE = 200_000


def _read_file_for_diff(path: str) -> list[str] | None:
    """读取文件内容用于对比，返回行列表或 None。"""
    try:
        size = os.path.getsize(path)
        if size > _MAX_FILE_SIZE:
            return None
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.readlines()
    except (OSError, UnicodeDecodeError):
        return None


@register(tags={"file"})
@tool
def file_diff(
    file_a: str,
    file_b: str,
    context: int = 3,
    ignore_whitespace: bool = False,
    ignore_case: bool = False,
) -> str:
    """对比两个文件的差异（unified diff 格式）。"""
    try:
        safe_a = safe_resolve(file_a, settings.FILE_READER_ROOT)
        safe_b = safe_resolve(file_b, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"路径错误：{exc}"

    if not os.path.isfile(safe_a):
        return f"❌ 文件不存在：{file_a}"
    if not os.path.isfile(safe_b):
        return f"❌ 文件不存在：{file_b}"

    lines_a = _read_file_for_diff(safe_a)
    if lines_a is None:
        return f"❌ 无法读取文件 {file_a}（可能过大或编码问题）"
    lines_b = _read_file_for_diff(safe_b)
    if lines_b is None:
        return f"❌ 无法读取文件 {file_b}（可能过大或编码问题）"

    rel_a = os.path.relpath(safe_a, os.path.realpath(settings.FILE_READER_ROOT))
    rel_b = os.path.relpath(safe_b, os.path.realpath(settings.FILE_READER_ROOT))

    if ignore_whitespace or ignore_case:
        normalized_a = [
            (" ".join(line.split()) if ignore_whitespace else line).lower() if ignore_case else line
            for line in lines_a
        ]
        normalized_b = [
            (" ".join(line.split()) if ignore_whitespace else line).lower() if ignore_case else line
            for line in lines_b
        ]
    else:
        normalized_a = lines_a
        normalized_b = lines_b

    differ = difflib.unified_diff(
        normalized_a,
        normalized_b,
        fromfile=rel_a,
        tofile=rel_b,
        n=context,
    )

    diff_lines = list(differ)
    if not diff_lines:
        return f"✅ 两个文件完全相同：{rel_a} 和 {rel_b}"

    additions = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
    deletions = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))

    output_lines = [f"📊 文件对比结果：", f"  {rel_a} vs {rel_b}", ""]
    output_lines.append(f"  新增 {additions} 行，删除 {deletions} 行")
    output_lines.append("")

    for i, line in enumerate(diff_lines):
        if i >= _MAX_DIFF_LINES:
            output_lines.append(f"...（差异过长，已截断，共 {len(diff_lines)} 行）")
            break
        output_lines.append(line.rstrip("\n"))

    return "\n".join(output_lines)


@register(tags={"file"})
@tool
def file_diff_summary(file_a: str, file_b: str) -> str:
    """快速对比两个文件的摘要。"""
    return file_diff(file_a=file_a, file_b=file_b, context=2)


@register(tags={"file"})
@tool
def file_patch(original: str, modified: str, output: str = "") -> str:
    """生成两个文件之间的 patch 补丁。"""
    try:
        safe_a = safe_resolve(original, settings.FILE_READER_ROOT)
        safe_b = safe_resolve(modified, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"路径错误：{exc}"

    if not os.path.isfile(safe_a):
        return f"❌ 文件不存在：{original}"
    if not os.path.isfile(safe_b):
        return f"❌ 文件不存在：{modified}"

    lines_a = _read_file_for_diff(safe_a)
    lines_b = _read_file_for_diff(safe_b)
    if lines_a is None or lines_b is None:
        return "❌ 无法读取文件（可能过大）"

    rel_a = os.path.relpath(safe_a, os.path.realpath(settings.FILE_READER_ROOT))
    rel_b = os.path.relpath(safe_b, os.path.realpath(settings.FILE_READER_ROOT))

    diff_lines = list(difflib.unified_diff(
        lines_a, lines_b,
        fromfile=rel_a, tofile=rel_b,
        n=3,
    ))

    if not diff_lines:
        return f"✅ 两个文件完全相同，无需生成 patch"

    patch_content = "".join(diff_lines)

    if output:
        try:
            safe_out = safe_resolve(output, settings.FILE_READER_ROOT)
            parent = os.path.dirname(safe_out)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(safe_out, "w", encoding="utf-8") as f:
                f.write(patch_content)
            return f"✅ Patch 文件已保存：{output}（{len(diff_lines)} 行差异）"
        except (ValueError, OSError) as exc:
            return f"❌ 保存 patch 失败：{exc}"

    return f"📋 Patch 内容（{len(diff_lines)} 行）：\n\n{patch_content[:5000]}"