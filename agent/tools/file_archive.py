"""文件归档工具 — 创建和解压 ZIP 归档文件。

支持在项目目录内创建 zip 压缩包和解压 zip 文件。
"""

from __future__ import annotations

import os
import zipfile

from langchain.tools import tool

from agent.config import settings
from agent.tools.registry import register
from agent.utils.path_guard import safe_resolve

_MAX_ARCHIVE_SIZE = 500 * 1024 * 1024
_MAX_FILE_SIZE_COMPRESS = 100 * 1024 * 1024
_MAX_FILES_IN_ARCHIVE = 1000

_IGNORED_DIRS = frozenset({
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".idea", ".vscode", "dist", "build",
})


def _collect_zip_files(source_path: str, base_dir: str) -> list[tuple[str, str]]:
    """收集要压缩的文件列表，返回 [(绝对路径, 归档内相对路径), ...]。"""
    files: list[tuple[str, str]] = []

    if os.path.isfile(source_path):
        files.append((source_path, os.path.basename(source_path)))
        return files

    for dirpath, dirnames, filenames in os.walk(source_path):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
        for fname in filenames:
            if len(files) >= _MAX_FILES_IN_ARCHIVE:
                return files
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, base_dir)
            files.append((full, rel))

    return files


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / 1024 / 1024:.1f}MB"


@register(tags={"file"})
@tool
def file_zip(
    source: str,
    archive_name: str = "",
) -> str:
    """将文件或文件夹压缩为 ZIP 归档。"""
    try:
        safe_source = safe_resolve(source, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"路径错误：{exc}"

    if not os.path.exists(safe_source):
        return f"❌ 源路径不存在：{source}"

    base_dir = safe_source if os.path.isdir(safe_source) else os.path.dirname(safe_source)

    files = _collect_zip_files(safe_source, base_dir)
    if not files:
        return f"❌ 没有可压缩的文件：{source}"

    total_size = sum(os.path.getsize(f) for f, _ in files if os.path.isfile(f))
    if total_size > _MAX_ARCHIVE_SIZE:
        return f"❌ 文件总大小超过 {_fmt_size(_MAX_ARCHIVE_SIZE)}，拒绝压缩（当前 {_fmt_size(total_size)}）"

    if not archive_name:
        source_base = os.path.basename(safe_source.rstrip(os.sep)) or "archive"
        archive_name = f"{source_base}.zip"
    elif not archive_name.endswith(".zip"):
        archive_name += ".zip"

    try:
        safe_dest = safe_resolve(archive_name, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"路径错误：{exc}"

    try:
        with zipfile.ZipFile(safe_dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for full_path, arc_name in files:
                if os.path.getsize(full_path) > _MAX_FILE_SIZE_COMPRESS:
                    zf.write(full_path, arc_name)
                else:
                    zf.write(full_path, arc_name)

        archive_size = os.path.getsize(safe_dest)
        return (
            f"✅ 已创建归档：{archive_name}\n"
            f"📦 包含 {len(files)} 个文件，归档大小 {_fmt_size(archive_size)}\n"
            f"📁 源路径：{source}"
        )
    except OSError as exc:
        return f"❌ 创建归档失败：{exc}"


@register(tags={"file"})
@tool
def file_unzip(
    archive: str,
    extract_to: str = "",
) -> str:
    """解压 ZIP 归档文件到项目目录。"""
    try:
        safe_archive = safe_resolve(archive, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"路径错误：{exc}"

    if not os.path.isfile(safe_archive):
        return f"❌ 归档文件不存在：{archive}"

    if not zipfile.is_zipfile(safe_archive):
        return f"❌ 不是有效的 ZIP 文件：{archive}"

    if not extract_to:
        base_name = os.path.splitext(os.path.basename(safe_archive))[0]
        extract_to = base_name

    try:
        safe_dest = safe_resolve(extract_to, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"路径错误：{exc}"

    try:
        os.makedirs(safe_dest, exist_ok=True)

        with zipfile.ZipFile(safe_archive, "r") as zf:
            file_list = zf.namelist()

            total_size = 0
            for info in zf.infolist():
                if info.file_size > _MAX_FILE_SIZE_COMPRESS:
                    return (
                        f"❌ 归档中包含过大文件（{info.filename}: {_fmt_size(info.file_size)}），"
                        f"拒绝解压"
                    )
                total_size += info.file_size
                if total_size > _MAX_ARCHIVE_SIZE:
                    return f"❌ 解压后总大小超过 {_fmt_size(_MAX_ARCHIVE_SIZE)}，拒绝解压"

            zf.extractall(safe_dest)

        extracted_files = len([f for f in os.listdir(safe_dest)])
        return (
            f"✅ 已解压到：{extract_to}/\n"
            f"📦 归档包含 {len(file_list)} 个条目\n"
            f"📁 解压后根目录有 {extracted_files} 个条目"
        )
    except zipfile.BadZipFile:
        return f"❌ 归档文件损坏：{archive}"
    except OSError as exc:
        return f"❌ 解压失败：{exc}"


@register(tags={"file"})
@tool
def file_zip_list(archive: str) -> str:
    """列出 ZIP 归档文件的内容列表。"""
    try:
        safe_archive = safe_resolve(archive, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"路径错误：{exc}"

    if not os.path.isfile(safe_archive):
        return f"❌ 归档文件不存在：{archive}"

    if not zipfile.is_zipfile(safe_archive):
        return f"❌ 不是有效的 ZIP 文件：{archive}"

    try:
        with zipfile.ZipFile(safe_archive, "r") as zf:
            infos = zf.infolist()
            lines = [f"📦 归档内容列表：{os.path.basename(safe_archive)}", f"共 {len(infos)} 个条目：", ""]

            total_uncompressed = 0
            total_compressed = 0
            for info in infos[:200]:
                is_dir = info.filename.endswith("/")
                prefix = "📁" if is_dir else "📄"
                size_str = "" if is_dir else f" ({_fmt_size(info.file_size)})"
                lines.append(f"  {prefix} {info.filename}{size_str}")
                total_uncompressed += info.file_size
                total_compressed += info.compress_size

            if len(infos) > 200:
                lines.append(f"  ...（共 {len(infos)} 个，仅显示前 200 个）")

            lines.append("")
            lines.append(f"总大小：{_fmt_size(total_uncompressed)}（压缩后 {_fmt_size(total_compressed)}）")

            return "\n".join(lines)
    except zipfile.BadZipFile:
        return f"❌ 归档文件损坏：{archive}"
    except OSError as exc:
        return f"❌ 读取归档失败：{exc}"