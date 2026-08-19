"""打开本地文件 —— 使用系统默认应用打开文件。

Windows: os.startfile()
macOS:   open
Linux:   xdg-open

安全策略：
  - 禁止 null 字节注入
  - 禁止敏感系统目录（Windows/System32 等）
  - 路径穿越防护
"""

import logging
import os
import re
import subprocess
import sys

from langchain.tools import tool

from agent.config import settings
from agent.tools.registry import register

logger = logging.getLogger(__name__)

_BLOCKED_DIRS = frozenset([
    "windows", "system32", "syswow64", "program files", "program files (x86)",
    "programdata", "$recycle.bin", "recovery",
])

_FOLDER_ALIASES = {
    "桌面": "Desktop",
    "desktop": "Desktop",
    "文档": "Documents",
    "documents": "Documents",
    "下载": "Downloads",
    "downloads": "Downloads",
    "图片": "Pictures",
    "pictures": "Pictures",
    "音乐": "Music",
    "music": "Music",
    "视频": "Videos",
    "videos": "Videos",
    "收藏": "Favorites",
    "favorites": "Favorites",
}

_HOME_DIRS = [
    "Desktop", "Documents", "Downloads", "Pictures",
    "Music", "Videos", "Favorites",
]


def _translate_drive_alias(file_path: str) -> str:
    match = re.match(r'^([A-Za-z])盘(?:[\\/](.*))?$', file_path)
    if match:
        drive = match.group(1).upper()
        rest = match.group(2)
        if rest:
            return f"{drive}:/{rest}"
        return f"{drive}:/"
    return file_path


def _translate_folder_alias(file_path: str) -> str:
    parts = file_path.replace("\\", "/").split("/")
    is_home_relative = file_path.startswith("~")
    is_absolute = os.path.isabs(file_path)

    for i, part in enumerate(parts):
        part_stripped = part.strip()
        key = part_stripped.lower()
        if key in _FOLDER_ALIASES:
            english = _FOLDER_ALIASES[key]
            if i == 0 and not is_home_relative and not is_absolute:
                parts[i] = f"~/{english}"
            else:
                parts[i] = english

    return "/".join(parts)


def _is_safe_path(abs_path: str) -> bool:
    try:
        path_lower = abs_path.lower()
        for blocked in _BLOCKED_DIRS:
            if f"\\{blocked}\\" in path_lower or path_lower.endswith(f"\\{blocked}"):
                return False
        return True
    except Exception:
        return False


def _try_resolve_path(file_path: str) -> str | None:
    """Try to resolve a path. Returns absolute path if exists, None otherwise."""
    if "\x00" in file_path:
        return None

    file_path = _translate_drive_alias(file_path)
    file_path = _translate_folder_alias(file_path)
    expanded = os.path.expanduser(file_path)

    if os.path.isabs(expanded):
        full = os.path.realpath(expanded)
    else:
        root = os.path.realpath(settings.FILE_READER_ROOT)
        full = os.path.realpath(os.path.join(root, expanded))

    if not _is_safe_path(full):
        return None

    if os.path.exists(full):
        return full
    return None


_COMMON_EXTENSIONS = [".txt", ".pdf", ".docx", ".doc", ".xlsx", ".xls",
                      ".pptx", ".ppt", ".png", ".jpg", ".jpeg", ".gif",
                      ".bmp", ".mp3", ".mp4", ".zip", ".rar", ".py",
                      ".md", ".json", ".csv", ".html", ".htm", ".xml"]


def _smart_find_file(filename: str) -> str | None:
    """Search common locations for a file by name.

    Search order:
    1. Project root (current working directory)
    2. User home directory
    3. Common home subdirectories (Desktop, Documents, etc.)
    4. All drives root and common subdirs
    5. One level of subdirectories within common dirs
    6. Try common extensions if file has no extension
    """
    basename = os.path.basename(filename)
    if not basename:
        return None

    home = os.path.expanduser("~")
    search_dirs = [os.getcwd(), home]
    for d in _HOME_DIRS:
        search_dirs.append(os.path.join(home, d))

    for drive_letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        drive_root = f"{drive_letter}:/"
        if os.path.isdir(drive_root):
            search_dirs.append(drive_root)
            for d in _HOME_DIRS:
                search_dirs.append(os.path.join(drive_root, d))

    candidates = [basename]
    name_root, ext = os.path.splitext(basename)
    if not ext:
        for e in _COMMON_EXTENSIONS:
            candidates.append(name_root + e)

    for search_dir in search_dirs:
        for candidate_name in candidates:
            candidate = os.path.join(search_dir, candidate_name)
            if os.path.isfile(candidate) and _is_safe_path(candidate):
                logger.info("[file_opener] Smart find: located '%s' at '%s'", basename, candidate)
                return candidate

    for search_dir in search_dirs:
        try:
            for entry in os.scandir(search_dir):
                if entry.is_dir():
                    for candidate_name in candidates:
                        sub = os.path.join(entry.path, candidate_name)
                        if os.path.isfile(sub) and _is_safe_path(sub):
                            logger.info("[file_opener] Smart find (sub): located '%s' at '%s'", basename, sub)
                            return sub
        except (PermissionError, OSError):
            continue

    return None


def _resolve_path(file_path: str) -> str:
    """Resolve file path with smart fallback.

    Resolution strategy:
    1. Try direct path resolution (with alias translation)
    2. If not found, try smart search for the filename
    """
    logger.info("[file_opener] Resolving: %s", file_path)

    direct = _try_resolve_path(file_path)
    if direct:
        logger.info("[file_opener] Direct match: %s", direct)
        return direct

    basename = os.path.basename(file_path) or file_path
    logger.info("[file_opener] Direct failed, smart searching for: %s", basename)
    found = _smart_find_file(basename)
    if found:
        return found

    raise FileNotFoundError(f"文件或目录不存在：{file_path}")


def _open_path(full_path: str) -> None:
    logger.info("[file_opener] Opening: %s", full_path)
    if sys.platform == "win32":
        os.startfile(full_path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", full_path])
    else:
        subprocess.Popen(["xdg-open", full_path])


@register(tags={"file"})
@tool
def file_opener(file_path: str) -> str:
    """打开本地文件或文件夹（使用系统默认应用）。"""
    try:
        full = _resolve_path(file_path)
        display = os.path.basename(full)
        if os.path.isdir(full):
            _open_path(full)
            return f"已打开文件夹：{full}"
        _open_path(full)
        return f"已打开文件：{display}（{full}）"
    except ValueError as exc:
        return f"路径错误：{exc}"
    except FileNotFoundError as exc:
        return str(exc)
    except OSError as exc:
        return f"打开失败：{exc}"