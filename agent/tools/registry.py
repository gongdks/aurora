"""Tool registry — centralized tool management.

Provides a decorator-based registration system and auto-discovery of
dedicated tool modules.

Tool loading order (last-wins):
  1. Auto-discover and import all dedicated modules in agent/tools/
  2. Register minimal built-in fallbacks (system_info, directory_list)
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import sys
from collections.abc import Callable
from typing import Any

from agent.config import settings
from agent.utils.path_guard import safe_resolve

logger = logging.getLogger(__name__)

_TOOLS: dict[str, Any] = {}
_DISCOVERED: set[str] = set()


def register(cls: type | None = None, *, name: str | None = None, **kwargs: Any) -> Any:
    """Decorator to register a tool class or function.

    Can be used on a Tool class or a standalone function:

        @register
        def my_tool(query: str) -> str:
            ...

        @register(name="custom_name")
        class MyTool(BaseTool):
            ...
    """
    def _wrapper(obj: Any) -> Any:
        tool_name = name or getattr(obj, "name", None) or obj.__name__
        if tool_name in _TOOLS:
            logger.debug("Overriding existing tool: %s", tool_name)
        _TOOLS[tool_name] = obj
        logger.debug("Registered tool: %s", tool_name)
        return obj

    if cls is not None:
        return _wrapper(cls)
    return _wrapper


def list_tools() -> list:
    """Return all registered tools as LangChain tool objects.

    Auto-discovers tool modules on first call so that @register
    decorators from dedicated modules take effect.
    """
    ensure_builtin_tools()
    from langchain_core.tools import tool as lc_tool

    tools_list: list = []
    for tool_name, tool_obj in _TOOLS.items():
        try:
            if isinstance(tool_obj, type):
                instance = tool_obj()
                tools_list.append(instance)
            elif callable(tool_obj):
                wrapped = lc_tool(tool_obj)
                tools_list.append(wrapped)
            else:
                tools_list.append(tool_obj)
        except Exception as exc:
            logger.warning("Failed to load tool %s: %s", tool_name, exc)

    return tools_list


def get_tool(name: str) -> Any | None:
    """Get a registered tool by name."""
    return _TOOLS.get(name)


def clear_tools() -> None:
    """Clear all registered tools (useful for testing)."""
    _TOOLS.clear()


def _discover_tool_modules() -> None:
    """Auto-discover and import all tool modules in agent/tools/.

    Each module's @register decorators will execute on import,
    automatically populating _TOOLS.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_pkg = "agent.tools"

    for finder, module_name, is_pkg in pkgutil.iter_modules([current_dir]):
        if module_name.startswith("_") or module_name == "registry":
            continue
        if module_name in _DISCOVERED:
            continue
        if is_pkg:
            continue
        try:
            importlib.import_module(f".{module_name}", parent_pkg)
            _DISCOVERED.add(module_name)
            logger.info("Auto-loaded tool module: %s", module_name)
        except Exception as exc:
            logger.warning("Failed to load tool module %s: %s", module_name, exc)


def ensure_builtin_tools() -> None:
    """Ensure all tool modules are discovered and registered.

    This function is idempotent — safe to call multiple times.
    Dedicated modules (file_reader, file_writer, web_search, browser,
    calculator, note_manager, code_executor, etc.) are auto-discovered.
    Minimal built-ins (system_info, directory_list) are always registered.
    """
    _discover_tool_modules()

    if "system_info" not in _TOOLS:
        _TOOLS["system_info"] = _system_info_tool

    if "directory_list" not in _TOOLS:
        _TOOLS["directory_list"] = _directory_list_tool


# ------------------------------------------------------------------
# Minimal built-in fallbacks (used only if dedicated modules don't
# provide an implementation — dedicated modules take precedence)
# ------------------------------------------------------------------


def _system_info_tool() -> str:
    """Get system information (OS, Python version, working directory).

    Returns:
        System information summary
    """
    import platform
    return (
        f"**System**: {platform.system()} {platform.release()}\n"
        f"**Python**: {platform.python_version()}\n"
        f"**Working directory**: {os.getcwd()}\n"
        f"**Processor**: {platform.processor() or 'N/A'}"
    )


def _directory_list_tool(path: str = ".") -> str:
    """List files and directories in a given path.

    Args:
        path: Directory path to list (relative to project root)

    Returns:
        Directory listing with file sizes
    """
    try:
        abs_path = safe_resolve(path, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"路径错误：{exc}"

    if not os.path.exists(abs_path):
        return f"Error: Path not found: {path}"

    entries = sorted(os.listdir(abs_path))
    lines = [f"**Directory**: {abs_path}\n"]
    for entry in entries[:100]:
        full = os.path.join(abs_path, entry)
        if os.path.isdir(full):
            lines.append(f"  📁 {entry}/")
        else:
            size = os.path.getsize(full)
            size_str = f"{size:,} bytes" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"
            lines.append(f"  📄 {entry} ({size_str})")
    return "\n".join(lines)