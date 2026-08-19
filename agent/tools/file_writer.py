"""Write, edit, and manage local files.

Provides file writing, editing, and management (delete/copy/move/rename/mkdir).
Paths are resolved relative to the configured root directory.
"""

import os

from langchain.tools import tool

from agent.config import settings
from agent.tools.registry import register
from agent.utils.path_guard import safe_resolve


def _read_file(safe: str) -> str:
    """Read file content for editing."""
    with open(safe, encoding="utf-8") as fh:
        return fh.read()


def _write_file(safe: str, content: str) -> None:
    """Write file content, creating parent directories if needed."""
    parent = os.path.dirname(safe)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(safe, "w", encoding="utf-8") as fh:
        fh.write(content)


@register
@tool
def file_writer(filename: str, content: str) -> str:
    """创建或覆写文件。

    用于创建新文件或完全替换已有文件的内容。

    Args:
        filename: 文件路径，相对于项目工作目录，如 "output/report.txt"、"src/main.py"
        content: 要写入的完整文本内容
    """
    try:
        safe = safe_resolve(filename, settings.FILE_READER_ROOT)
        _write_file(safe, content)
        lines = content.count("\n") + 1
        return f"✅ 已写入 {filename}（{len(content)} 字符，{lines} 行）"
    except ValueError as exc:
        return f"路径错误：{exc}"
    except OSError as exc:
        return f"写入失败：{exc}"


@register
@tool
def file_editor(filename: str, old_text: str, new_text: str) -> str:
    """精确替换文件中的文本（仅替换首次出现）。

    适用场景：修改函数名、变量名、单行配置等。

    Args:
        filename: 文件路径，如 "config.py"、"src/main.py"
        old_text: 要查找并替换的文本（需精确匹配，含缩进）
        new_text: 替换后的文本
    """
    try:
        safe = safe_resolve(filename, settings.FILE_READER_ROOT)
        if not os.path.isfile(safe):
            return f"❌ 文件不存在：{filename}"
        content = _read_file(safe)
        if old_text not in content:
            # 尝试给出有用的错误信息
            preview = old_text[:80].replace("\n", "\\n")
            return f"❌ 在 {filename} 中未找到指定文本。\n查找内容预览: {preview}..."
        new_content = content.replace(old_text, new_text, 1)
        _write_file(safe, new_content)
        return f"✅ 已编辑 {filename}：替换 {len(old_text)} → {len(new_text)} 字符（1 处）"
    except ValueError as exc:
        return f"路径错误：{exc}"
    except OSError as exc:
        return f"编辑失败：{exc}"


@register
@tool
def file_editor_all(filename: str, old_text: str, new_text: str) -> str:
    """精确替换文件中所有匹配的文本（替换所有出现）。

    适用场景：全局重命名变量、统一修改配置值等。

    Args:
        filename: 文件路径，如 "config.py"
        old_text: 要查找并替换的文本（需精确匹配，含缩进）
        new_text: 替换后的文本
    """
    try:
        safe = safe_resolve(filename, settings.FILE_READER_ROOT)
        if not os.path.isfile(safe):
            return f"❌ 文件不存在：{filename}"
        content = _read_file(safe)
        count = content.count(old_text)
        if count == 0:
            preview = old_text[:80].replace("\n", "\\n")
            return f"❌ 在 {filename} 中未找到指定文本。\n查找内容预览: {preview}..."
        new_content = content.replace(old_text, new_text)
        _write_file(safe, new_content)
        return f"✅ 已编辑 {filename}：替换 {len(old_text)} → {len(new_text)} 字符（{count} 处）"
    except ValueError as exc:
        return f"路径错误：{exc}"
    except OSError as exc:
        return f"编辑失败：{exc}"


@register
@tool
def file_editor_multiline(
    filename: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
) -> str:
    """精确的多行文本替换——用于修改函数体、类定义、代码块等。

    与 file_editor 类似，但专门优化了多行文本替换的体验。
    old_text 必须精确匹配（含每行的缩进和空白）。
    当 old_text 只在文件中出现一次时不需要 replace_all。

    Args:
        filename: 文件路径，如 "agent/tools/file_writer.py"
        old_text: 要替换的原始文本（必须精确匹配，含所有缩进和空格）
        new_text: 新文本
        replace_all: 是否替换所有匹配（默认 False，仅替换第一处）
    """
    try:
        safe = safe_resolve(filename, settings.FILE_READER_ROOT)
        if not os.path.isfile(safe):
            return f"❌ 文件不存在：{filename}"
        content = _read_file(safe)

        count = content.count(old_text)
        if count == 0:
            # 帮助定位：显示 old_text 的前几行
            preview = old_text[:120].replace("\n", "\\n")
            return (
                f"❌ 在 {filename} 中未找到指定文本（{len(old_text)} 字符）。\n"
                f"请检查缩进和空白是否完全匹配。\n"
                f"查找内容预览: {preview}..."
            )
        if count > 1 and not replace_all:
            return (
                f"⚠️  文本在文件中出现了 {count} 次。\n"
                f"请使用 replace_all=True 替换全部，或提供更精确的上下文使匹配唯一。"
            )

        new_content = content.replace(old_text, new_text) if replace_all else content.replace(old_text, new_text, 1)
        replaced_count = count if replace_all else 1
        _write_file(safe, new_content)

        added_lines = new_text.count("\n") - old_text.count("\n")
        detail = f"（{'全部' if replace_all else '1'} 处"
        if added_lines > 0:
            detail += f"，+{added_lines} 行"
        elif added_lines < 0:
            detail += f"，{added_lines} 行"
        detail += "）"

        return f"✅ 已编辑 {filename}：替换 {len(old_text)} → {len(new_text)} 字符 {detail}"
    except ValueError as exc:
        return f"路径错误：{exc}"
    except OSError as exc:
        return f"编辑失败：{exc}"


@register
@tool
def file_list(directory: str = ".") -> str:
    """List files in a directory (within project root).

    Use directory_list for standard directory listing with sizes.

    Args:
        directory: Directory path relative to project root, defaults to "."
    """
    try:
        safe = safe_resolve(directory, settings.FILE_READER_ROOT)
        if not os.path.isdir(safe):
            return f"Not a directory: {directory}"
        entries = sorted(os.listdir(safe))
        if not entries:
            return f"Directory {directory} is empty."
        lines = []
        for e in entries:
            full = os.path.join(safe, e)
            if os.path.isdir(full):
                lines.append(f"  📁 {e}/")
            else:
                size = os.path.getsize(full)
                lines.append(f"  📄 {e} ({_fmt_size(size)})")
        return f"Contents of {directory}:\n" + "\n".join(lines)
    except ValueError as exc:
        return f"Path error: {exc}"
    except OSError as exc:
        return f"List failed: {exc}"


@register
@tool
def file_delete(filename: str) -> str:
    """删除项目目录内的文件。

    仅支持删除文件，不支持删除目录。

    Args:
        filename: 要删除的文件路径，如 "output/old_report.txt"
    """
    try:
        safe = safe_resolve(filename, settings.FILE_READER_ROOT)
        if not os.path.isfile(safe):
            return f"❌ 文件不存在：{filename}"
        os.remove(safe)
        return f"✅ 已删除文件：{filename}"
    except ValueError as exc:
        return f"路径错误：{exc}"
    except OSError as exc:
        return f"删除失败：{exc}"


@register
@tool
def file_copy(source: str, destination: str) -> str:
    """复制项目目录内的文件。

    Args:
        source: 源文件路径，如 "data/original.csv"
        destination: 目标路径，如 "data/backup.csv"
    """
    import shutil

    try:
        src = safe_resolve(source, settings.FILE_READER_ROOT)
        dst = safe_resolve(destination, settings.FILE_READER_ROOT)
        if not os.path.isfile(src):
            return f"❌ 源文件不存在：{source}"
        parent = os.path.dirname(dst)
        if parent:
            os.makedirs(parent, exist_ok=True)
        shutil.copy2(src, dst)
        size = os.path.getsize(dst)
        return f"✅ 已复制：{source} → {destination}（{_fmt_size(size)}）"
    except ValueError as exc:
        return f"路径错误：{exc}"
    except OSError as exc:
        return f"复制失败：{exc}"


@register
@tool
def file_move(source: str, destination: str) -> str:
    """移动项目目录内的文件（重命名或改变目录）。

    Args:
        source: 源文件路径，如 "data/old.csv"
        destination: 目标路径，如 "archive/old.csv"
    """
    import shutil

    try:
        src = safe_resolve(source, settings.FILE_READER_ROOT)
        dst = safe_resolve(destination, settings.FILE_READER_ROOT)
        if not os.path.exists(src):
            return f"❌ 源路径不存在：{source}"
        parent = os.path.dirname(dst)
        if parent:
            os.makedirs(parent, exist_ok=True)
        shutil.move(src, dst)
        return f"✅ 已移动：{source} → {destination}"
    except ValueError as exc:
        return f"路径错误：{exc}"
    except OSError as exc:
        return f"移动失败：{exc}"


@register
@tool
def dir_create(path: str) -> str:
    """创建目录（支持多级创建）。

    Args:
        path: 目录路径，如 "output/reports/2024"
    """
    try:
        safe = safe_resolve(path, settings.FILE_READER_ROOT)
        if os.path.exists(safe):
            if os.path.isdir(safe):
                return f"✅ 目录已存在：{path}"
            else:
                return f"❌ 同名文件已存在：{path}"
        os.makedirs(safe, exist_ok=True)
        return f"✅ 已创建目录：{path}"
    except ValueError as exc:
        return f"路径错误：{exc}"
    except OSError as exc:
        return f"创建目录失败：{exc}"


@register
@tool
def dir_delete(path: str) -> str:
    """删除项目目录内的空目录。

    仅支持删除空目录。如需删除含文件的目录，请先删除目录中的文件。

    Args:
        path: 要删除的目录路径，如 "output/empty_dir"
    """
    try:
        safe = safe_resolve(path, settings.FILE_READER_ROOT)
        if not os.path.isdir(safe):
            return f"❌ 目录不存在：{path}"
        entries = os.listdir(safe)
        if entries:
            return f"❌ 目录不为空（含 {len(entries)} 个条目），请先删除其中的文件"
        os.rmdir(safe)
        return f"✅ 已删除空目录：{path}"
    except ValueError as exc:
        return f"路径错误：{exc}"
    except OSError as exc:
        return f"删除目录失败：{exc}"


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / 1024 / 1024:.1f}MB"