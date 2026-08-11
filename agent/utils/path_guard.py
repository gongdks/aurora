"""统一路径安全守卫 —— 防止路径穿越攻击。

所有文件读写操作都应通过 safe_resolve() 解析路径。
"""

from __future__ import annotations

import os
import re

from agent.config import settings

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


def _translate_folder_alias(rel_path: str) -> str:
    parts = rel_path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        part_stripped = part.strip()
        key = part_stripped.lower()
        if key in _FOLDER_ALIASES:
            parts[i] = _FOLDER_ALIASES[key]
    return "/".join(parts)


def _translate_drive_alias(file_path: str) -> str:
    match = re.match(r'^([A-Za-z])盘(?:[\\/](.*))?$', file_path)
    if match:
        drive = match.group(1).upper()
        rest = match.group(2)
        if rest:
            return f"{drive}:/{rest}"
        return f"{drive}:/"
    return file_path


def safe_resolve(rel_path: str, root: str | None = None) -> str:
    """解析相对路径并确保在允许的根目录内。

    安全措施：
      1. 拒绝含 null 字节的路径（防注入）
      2. 拒绝绝对路径（强制相对路径）
      3. 通过 realpath + commonpath 确保解析后仍在根目录内

    Args:
        rel_path: 相对路径
        root: 根目录，None 则使用 settings.FILE_READER_ROOT

    Returns:
        解析后的绝对安全路径

    Raises:
        ValueError: 路径非法或越界
    """
    if "\x00" in rel_path:
        raise ValueError(f"路径包含非法字符：{rel_path!r}")

    rel_path = _translate_drive_alias(rel_path)
    rel_path = _translate_folder_alias(rel_path)

    expanded = os.path.expanduser(rel_path)
    if os.path.isabs(expanded):
        full = os.path.realpath(expanded)
    else:
        root_dir = os.path.realpath(root) if root else os.path.realpath(settings.FILE_READER_ROOT)
        full = os.path.realpath(os.path.join(root_dir, expanded))

    root_dir = os.path.realpath(root) if root else os.path.realpath(settings.FILE_READER_ROOT)
    if os.path.commonpath([root_dir, full]) != root_dir:
        raise ValueError(f"路径越界：{rel_path} 不在允许的目录内（仅限项目目录）")

    return full