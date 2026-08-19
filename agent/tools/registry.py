"""Tool registry — centralized tool management.

Provides a decorator-based registration system with:
- Tag-based tool categorization (core, file, web, code, dev, etc.)
- Scene-based loading (load only relevant tools per scenario)
- Auto-discovery of dedicated tool modules
- ToolRouter for intelligent tool selection

Loading order (last-wins):
  1. Auto-discover and import all dedicated modules in agent/tools/
  2. Register minimal built-in fallbacks (system_info, directory_list)
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import sys
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from agent.config import settings
from agent.utils.path_guard import safe_resolve

logger = logging.getLogger(__name__)

_TOOLS: dict[str, Any] = {}
_DISCOVERED: set[str] = set()
_TAGS: dict[str, set[str]] = defaultdict(set)

CORE_TAGS = {"core"}
SCENE_TAGS: dict[str, set[str]] = {
    "general": {"core"},
    "file": {"core", "file"},
    "web": {"core", "web"},
    "code": {"core", "code"},
    "dev": {"core", "dev", "git", "shell"},
    "analysis": {"core", "code", "file"},
    "research": {"core", "web", "summarize", "translate"},
}


def register(
    cls: type | None = None,
    *,
    name: str | None = None,
    tags: str | set[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Decorator to register a tool class or function with optional tags.

    Tags enable scene-based loading — only tools matching the current
    scenario are exposed to the LLM, reducing token waste and selection
    confusion.

    Usage:
        @register(tags={"core", "file"})
        def file_reader(filename: str) -> str:
            ...

        @register(name="custom", tags="web")
        class MyTool(BaseTool):
            ...
    """
    def _wrapper(obj: Any) -> Any:
        tool_name = name or getattr(obj, "name", None) or obj.__name__
        if tool_name in _TOOLS:
            logger.debug("Overriding existing tool: %s", tool_name)
        _TOOLS[tool_name] = obj

        if tags:
            tag_set = {tags} if isinstance(tags, str) else tags
            _TAGS[tool_name].update(tag_set)
            if "core" in tag_set:
                _TAGS[tool_name].add("core")

        logger.debug("Registered tool: %s (tags: %s)", tool_name, _TAGS.get(tool_name, set()))
        return obj

    if cls is not None:
        return _wrapper(cls)
    return _wrapper


def list_tools(
    tags: set[str] | None = None,
    exclude_tags: set[str] | None = None,
) -> list:
    """Return registered tools as LangChain tool objects.

    Args:
        tags: If provided, only return tools matching ANY of these tags.
        exclude_tags: If provided, exclude tools matching ANY of these tags.

    Auto-discovers tool modules on first call.
    """
    ensure_builtin_tools()
    from langchain_core.tools import tool as lc_tool

    tools_list: list = []
    for tool_name, tool_obj in _TOOLS.items():
        tool_tags = _TAGS.get(tool_name, set())

        if tags is not None:
            if not (tool_tags & tags):
                continue

        if exclude_tags is not None:
            if tool_tags & exclude_tags:
                continue

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


def list_scene_tools(scene: str) -> list:
    """Return tools for a specific scenario.

    Scenes:
        - general: Core tools only
        - file: File operations + core
        - web: Web search/fetch + core
        - code: Code execution/analysis + core
        - dev: Dev tools (git, shell) + core
        - analysis: Data analysis + core
        - research: Research tools (web, summarize, translate) + core
    """
    ensure_builtin_tools()
    tag_set = SCENE_TAGS.get(scene, CORE_TAGS)
    return list_tools(tags=tag_set)


def get_tool(name: str) -> Any | None:
    """Get a registered tool by name."""
    return _TOOLS.get(name)


def get_tool_tags(name: str) -> set[str]:
    """Get tags for a registered tool."""
    return _TAGS.get(name, set())


def get_all_tags() -> dict[str, set[str]]:
    """Get all tools and their tags."""
    return dict(_TAGS)


def get_available_scenes() -> list[str]:
    """List available scene names."""
    return list(SCENE_TAGS.keys())


def clear_tools() -> None:
    """Clear all registered tools (useful for testing)."""
    _TOOLS.clear()
    _TAGS.clear()


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
    """
    _discover_tool_modules()

    if "system_info" not in _TOOLS:
        _TOOLS["system_info"] = _system_info_tool
        _TAGS["system_info"] = {"core", "system"}

    if "directory_list" not in _TOOLS:
        _TOOLS["directory_list"] = _directory_list_tool
        _TAGS["directory_list"] = {"core", "file"}


# ------------------------------------------------------------------
# ToolRouter — intelligent two-stage tool selection
# ------------------------------------------------------------------


class ToolRouter:
    """Two-stage tool routing: scenario → specific tools.

    Reduces LLM decision space by first selecting a scenario,
    then only exposing tools relevant to that scenario.

    Usage:
        router = ToolRouter()
        tools = router.route("research")  # returns web + summarize tools
    """

    def __init__(self) -> None:
        ensure_builtin_tools()

    def route(
        self,
        scenario: str | None = None,
        extra_tags: set[str] | None = None,
    ) -> list:
        """Get tools for a given scenario.

        Args:
            scenario: Scene name (general/file/web/code/dev/analysis/research).
                      If None, returns all core tools.
            extra_tags: Additional tags to include beyond the scene defaults.
        """
        if scenario and scenario in SCENE_TAGS:
            tags = SCENE_TAGS[scenario].copy()
        else:
            tags = CORE_TAGS.copy()

        if extra_tags:
            tags.update(extra_tags)

        return list_tools(tags=tags)

    def detect_scenario(self, user_input: str) -> str:
        """Auto-detect the most relevant scenario from user input.

        Uses simple keyword matching for fast routing.
        """
        input_lower = user_input.lower()

        scenarios: list[tuple[str, list[str]]] = [
            ("dev", ["git", "commit", "branch", "代码", "bug", "错误", "安装", "pip", "package", "依赖"]),
            ("code", ["python", "代码", "执行", "运行", "计算", "分析", "数据", "plot", "图表"]),
            ("file", ["文件", "读写", "保存", "路径", "目录", "重命名", "复制", "删除"]),
            ("web", ["搜索", "搜索一下", "查找", "google", "baidu", "网页", "网站", "抓取", "爬取"]),
            ("research", ["翻译", "总结", "摘要", "翻译一下", "总结一下", "概括"]),
        ]

        for scenario, keywords in scenarios:
            if any(kw in input_lower for kw in keywords):
                return scenario

        return "general"

    def smart_route(self, user_input: str) -> tuple[str, list]:
        """Auto-detect scenario and return matching tools.

        Returns (scenario_name, tools_list).
        """
        scenario = self.detect_scenario(user_input)
        tools = self.route(scenario)
        return scenario, tools


# ------------------------------------------------------------------
# Minimal built-in fallbacks
# ------------------------------------------------------------------


def _system_info_tool() -> str:
    """获取操作系统、Python 版本和工作目录信息。"""
    import platform
    return (
        f"System: {platform.system()} {platform.release()}\n"
        f"Python: {platform.python_version()}\n"
        f"Working directory: {os.getcwd()}\n"
        f"Processor: {platform.processor() or 'N/A'}"
    )


def _directory_list_tool(path: str = ".") -> str:
    """列出指定路径下的文件和目录（相对于项目根目录）。"""
    try:
        abs_path = safe_resolve(path, settings.FILE_READER_ROOT)
    except ValueError as exc:
        return f"Path error: {exc}"

    if not os.path.exists(abs_path):
        return f"Error: Path not found: {path}"

    entries = sorted(os.listdir(abs_path))
    lines = [f"Directory: {abs_path}"]
    for entry in entries[:100]:
        full = os.path.join(abs_path, entry)
        if os.path.isdir(full):
            lines.append(f"  [DIR]  {entry}/")
        else:
            size = os.path.getsize(full)
            size_str = f"{size}B" if size < 1024 else f"{size / 1024:.1f}KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f}MB"
            lines.append(f"  [FILE] {entry} ({size_str})")
    return "\n".join(lines)