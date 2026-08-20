"""Skill learner — extracts reusable skills from successful executions.

Core flow:
  Execution success → Analyze tool usage patterns → Generate skill definition
  → Validate against history → Store in SkillStore → Register as runtime tool

Only promotes skills that demonstrate consistent success patterns.

Key design decisions:
  - Dynamic tool introspection: parameter names and aliases are auto-
    discovered from registered tool signatures via _ToolIntrospector,
    eliminating the need to manually maintain param alias maps
  - Tool name resolution: dynamically matches inferred names against the
    actual tool registry (_TOOLS), with hardcoded aliases as fallback
  - Skill execution: SkillExecutionEngine executes learned tool sequences
    against real registered tools, with runtime parameter injection
  - Skill promotion: promoted tools are true callables that execute the
    learned sequence, not just descriptive wrappers
  - Forward-compatible: when function-calling (JSON Schema) is available,
    the resolution layer can be replaced by direct schema lookup
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import re
import time
from typing import Any

from agent.utils.skill_store import SkillDefinition, SkillStore

logger = logging.getLogger(__name__)

_MAX_TOOL_SEQUENCE_LENGTH = 10
_MIN_TOOL_SEQUENCE_LENGTH = 2
_MAX_TRIGGER_PATTERNS = 5

_TOOL_NAME_ALIASES: dict[str, list[str]] = {
    "text_reader": ["read_file", "read", "file_read", "read_text", "读取", "阅读", "file_reader"],
    "file_opener": ["open_file", "file_open", "打开文件"],
    "file_writer": ["write_file", "write", "file_write", "保存", "写入", "write_file_content"],
    "file_editor": ["edit_file", "修改文件", "编辑文件"],
    "file_editor_all": ["edit_all", "full_edit", "replace_all"],
    "file_editor_multiline": ["multiline_edit", "multi_edit"],
    "file_copy": ["copy_file", "复制"],
    "file_move": ["move_file", "rename", "移动", "重命名"],
    "file_delete": ["delete_file", "remove_file", "删除文件"],
    "file_list": ["list_files", "ls", "列出文件", "file_listing"],
    "dir_create": ["create_dir", "mkdir", "创建目录"],
    "dir_delete": ["rmdir", "delete_dir", "remove_dir", "删除目录"],
    "directory_list": ["list_dir", "list", "directory", "目录", "列表", "dir"],
    "web_search": ["search_web", "search", "搜索", "查找"],
    "web_content_fetcher": ["fetch_url", "fetch", "fetch_page", "访问", "获取"],
    "web_fetcher": ["web_fetch_tool", "fetch_web", "download", "下载"],
    "code_executor": ["run_code", "execute_code", "run", "execute", "执行", "运行"],
    "code_lint": ["analyze_code", "code_analysis", "code_analyzer", "analyze", "分析", "lint"],
    "code_typecheck": ["typecheck", "类型检查"],
    "grep": ["search_code", "代码搜索", "code_search", "rg", "ripgrep"],
    "code_outline": ["outline", "code_outline", "代码大纲"],
    "summarize_text": ["summarize", "summary", "总结", "摘要"],
    "summarize_file": ["summarize_file", "file_summary", "文件摘要"],
    "summarize_url": ["summarize_url", "url_summary", "网页摘要"],
    "translate": ["translate_text", "翻译"],
    "translate_batch": ["translate_batch", "batch_translate", "批量翻译"],
    "shell_executor": ["shell", "terminal", "command", "命令行", "终端"],
    "shell_executor_multi": ["shell_multi", "multi_shell", "批量命令"],
    "math_calculator": ["calculate", "calc", "calculator", "计算"],
    "image_analyze": ["analyze_image", "image", "图片分析", "image_analyzer", "image_analysis"],
    "image_batch_analyze": ["image_batch", "batch_analyze", "批量图片分析"],
    "file_diff": ["diff", "compare", "对比", "比较"],
    "file_diff_summary": ["diff_summary", "compare_files", "diff_file"],
    "file_patch": ["patch", "apply_patch", "补丁"],
    "file_zip": ["archive", "compress", "压缩", "归档", "file_archive"],
    "file_unzip": ["unzip", "extract", "解压"],
    "file_zip_list": ["list_archive", "archive_list"],
    "http_request": ["http", "request", "api", "接口", "http_client"],
    "http_get": ["http_get", "get_request"],
    "http_post": ["http_post", "post_request"],
    "datetime_query": ["datetime", "date", "时间", "日期", "datetime_tool"],
    "note_save": ["note", "memo", "笔记", "note_manager"],
    "note_read": ["note_read", "读取笔记", "read_note"],
    "note_list": ["note_list", "列出笔记", "list_notes"],
    "git_commit": ["git", "commit", "git_commit"],
    "git_push": ["git_push", "push"],
    "git_pull": ["git_pull", "pull"],
    "git_branch": ["git_branch", "branch"],
    "git_status": ["git_status", "status"],
    "git_log": ["git_log", "log", "git_history"],
    "git_log_since": ["git_log_since", "log_since", "recent_log"],
    "git_diff": ["git_diff_tool", "diff_staged", "git_diff_unstaged"],
    "git_diff_staged": ["git_diff_staged", "staged_diff"],
    "git_add": ["git_add", "stage", "git_stage"],
    "git_remote": ["git_remote_tool", "remote"],
    "git_show": ["git_show_tool", "show_commit"],
    "system_info": ["system", "info", "系统信息"],
    "glob": ["glob_tool", "find_files", "search_files", "文件搜索"],
    "browser_navigate": ["browser", "navigate", "浏览器"],
    "browser_click": ["click", "点击"],
    "browser_type": ["type", "输入"],
    "browser_snapshot": ["snapshot", "页面快照"],
    "browser_screenshot": ["screenshot", "截图"],
    "browser_list_interactive": ["list_interactive", "列出交互"],
    "browser_close": ["browser_close", "close_browser"],
    "browser_press_key": ["browser_press", "press_key", "按键"],
}

_TOOL_PARAM_ALIASES: dict[str, dict[str, str]] = {
    "text_reader": {"path": "filename", "file_path": "filename", "filepath": "filename"},
    "file_opener": {"path": "file_path", "file_path": "file_path", "filename": "file_path"},
    "file_writer": {"path": "filename", "file_path": "filename", "content": "content"},
    "file_editor": {"path": "filename", "file_path": "filename"},
    "file_editor_all": {"path": "filename", "file_path": "filename", "old_string": "old_string", "new_string": "new_string"},
    "file_editor_multiline": {"path": "filename", "file_path": "filename", "old_string": "old_string", "new_string": "new_string"},
    "file_copy": {"source": "source", "dest": "destination", "target": "destination"},
    "file_move": {"source": "source", "dest": "destination", "target": "destination"},
    "file_delete": {"path": "filename", "file_path": "filename"},
    "file_list": {"path": "path", "dir": "path", "directory": "path", "recursive": "recursive"},
    "dir_create": {"path": "path", "dir": "path", "directory": "path"},
    "dir_delete": {"path": "path", "dir": "path", "directory": "path", "recursive": "recursive"},
    "directory_list": {"path": "path", "dir": "path", "directory": "path"},
    "web_search": {"query": "query", "search_query": "query", "input": "query"},
    "web_content_fetcher": {"url": "url", "link": "url", "path": "url"},
    "web_fetcher": {"url": "url", "link": "url", "extract_text": "extract_text"},
    "code_executor": {"code": "code", "script": "code", "language": "language", "timeout": "timeout"},
    "code_lint": {"code": "filepath", "source": "filepath", "path": "filepath", "language": "language"},
    "code_typecheck": {"code": "filepath", "source": "filepath", "path": "filepath", "language": "language"},
    "grep": {"query": "pattern", "pattern": "pattern", "path": "path", "dir": "path", "glob": "glob", "ignore_case": "ignore_case", "context": "context"},
    "code_outline": {"path": "filepath", "file_path": "filepath"},
    "summarize_text": {"text": "text", "content": "text", "input": "text"},
    "summarize_file": {"path": "filename", "file_path": "filename", "file": "filename"},
    "summarize_url": {"url": "url", "link": "url"},
    "translate": {"text": "text", "content": "text", "input": "text", "source_lang": "source_lang", "target_lang": "target_lang"},
    "translate_batch": {"texts": "texts", "source_lang": "source_lang", "target_lang": "target_lang"},
    "shell_executor": {"command": "command", "cmd": "command", "script": "command", "timeout": "timeout"},
    "shell_executor_multi": {"commands": "commands", "timeout": "timeout"},
    "math_calculator": {"expression": "expression", "expr": "expression", "query": "expression"},
    "image_analyze": {"image_path": "filepath", "path": "filepath", "file": "filepath"},
    "image_batch_analyze": {"image_paths": "filepaths", "paths": "filepaths"},
    "file_diff": {"path": "file_a", "file_path": "file_a", "file_a": "file_a", "file_b": "file_b", "context": "context"},
    "file_diff_summary": {"path": "file_a", "file_path": "file_a", "file_a": "file_a", "file_b": "file_b"},
    "file_patch": {"path": "original", "file_path": "original", "modified": "modified", "output": "output"},
    "file_zip": {"path": "source", "file_path": "source", "source_path": "source", "archive_name": "archive_name"},
    "file_unzip": {"path": "archive", "file_path": "archive", "extract_to": "extract_to"},
    "file_zip_list": {"path": "archive", "file_path": "archive"},
    "http_request": {"url": "url", "link": "url", "method": "method", "headers": "headers", "body": "body"},
    "http_get": {"url": "url", "link": "url", "headers": "headers"},
    "http_post": {"url": "url", "link": "url", "headers": "headers", "body": "body"},
    "datetime_query": {"question": "question", "query": "question"},
    "note_save": {"content": "content", "text": "content"},
    "note_read": {"title": "title"},
    "note_list": {},
    "git_commit": {"message": "message", "msg": "message"},
    "git_push": {"remote": "remote", "branch": "branch"},
    "git_pull": {"remote": "remote", "branch": "branch"},
    "git_branch": {"list_all": "list_all"},
    "git_status": {"porcelain": "porcelain"},
    "git_log": {"max_count": "max_count", "count": "max_count"},
    "git_log_since": {"since": "since", "max_count": "max_count"},
    "git_diff": {"cached": "cached", "stat": "stat"},
    "git_diff_staged": {"stat": "stat"},
    "git_add": {"all": "all", "force": "force"},
    "git_remote": {"verbose": "verbose"},
    "git_show": {"stat": "stat"},
    "system_info": {},
    "glob": {"pattern": "pattern", "path": "path", "recursive": "recursive"},
    "browser_navigate": {"url": "url", "link": "url"},
    "browser_click": {"selector": "selector", "element": "selector"},
    "browser_type": {"selector": "selector", "element": "selector", "text": "text"},
    "browser_snapshot": {},
    "browser_screenshot": {"filename": "filename", "path": "filename"},
    "browser_list_interactive": {},
    "browser_close": {"tab_id": "tab_id"},
    "browser_press_key": {"key": "key", "modifiers": "modifiers"},
}


class _ToolIntrospector:
    """Dynamically extracts tool metadata from registered tool objects.

    Replaces hardcoded _TOOL_PARAM_ALIASES with runtime introspection
    of tool function signatures. When a tool has params like
    ``filename``, this class auto-discovers them so the skill
    engine can inject runtime args without manual alias maintenance.

    Usage:
        introspector = _ToolIntrospector()
        params = introspector.get_param_names("text_reader")
        # → {"filename"}
        aliases = introspector.get_param_aliases("text_reader")
        # → {"path": "filename", "file_path": "filename", ...}  (hardcoded + dynamic)
    """

    _instance: _ToolIntrospector | None = None

    def __new__(cls) -> _ToolIntrospector:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._param_cache: dict[str, set[str]] = {}
            cls._instance._alias_cache: dict[str, dict[str, str]] = {}
        return cls._instance

    def _extract_params(self, tool_obj: Any) -> set[str]:
        if tool_obj is None:
            return set()
        if hasattr(tool_obj, "func") and callable(tool_obj.func):
            target = tool_obj.func
        elif callable(tool_obj):
            target = tool_obj
        else:
            return set()
        try:
            sig = inspect.signature(target)
        except (ValueError, TypeError):
            return set()
        return {
            name for name, param in sig.parameters.items()
            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_ONLY,
            )
        }

    def get_param_names(self, tool_name: str) -> set[str]:
        if tool_name in self._param_cache:
            return self._param_cache[tool_name]
        registry_ok = _ensure_registry()
        if not registry_ok:
            return set()
        from agent.tools.registry import _TOOLS
        obj = _TOOLS.get(tool_name)
        params = self._extract_params(obj)
        self._param_cache[tool_name] = params
        return params

    def get_resolved_param_aliases(self, tool_name: str) -> dict[str, str]:
        """Merge hardcoded aliases with dynamically discovered params.

        Hardcoded aliases handle semantic mappings (path→filename) that
        cannot be auto-discovered. Dynamic params add any missing names.
        """
        if tool_name in self._alias_cache:
            return self._alias_cache[tool_name]
        hardcoded = _TOOL_PARAM_ALIASES.get(tool_name, {}).copy()
        dynamic_params = self.get_param_names(tool_name)
        for param_name in dynamic_params:
            if param_name not in hardcoded:
                hardcoded[param_name] = param_name
        self._alias_cache[tool_name] = hardcoded
        return hardcoded

    def clear_cache(self) -> None:
        self._param_cache.clear()
        self._alias_cache.clear()


_registry_loaded = False
_registry_failed = False


def _resolve_tool_in_registry(tool_name: str) -> tuple[str | None, Any | None]:
    """Resolve a tool name against the registry and alias maps.

    Shared by both SkillLearner and SkillExecutionEngine to ensure
    consistent resolution behavior.

    Resolution order:
      1. Exact match in _TOOLS
      2. Canonical name lookup (tool_name IS a canonical alias key)
      3. Alias lookup (tool_name is an alias of a canonical)
      4. Case-insensitive exact match
      5. Substring match (name ⊆ registered name, score >= 0.7)
      6. Docstring keyword match

    Returns (resolved_name, tool_object). Either can be None.
    """
    registry_ok = _ensure_registry()

    # 1. Exact match
    if registry_ok:
        from agent.tools.registry import _TOOLS
        if tool_name in _TOOLS:
            return (tool_name, _TOOLS[tool_name])

    # 2. Canonical name lookup
    if tool_name in _TOOL_NAME_ALIASES:
        if registry_ok:
            from agent.tools.registry import _TOOLS
            if tool_name in _TOOLS:
                return (tool_name, _TOOLS[tool_name])
        return (tool_name, None)

    # 3. Alias lookup
    for canonical, aliases in _TOOL_NAME_ALIASES.items():
        if tool_name in aliases:
            if registry_ok:
                from agent.tools.registry import _TOOLS
                if canonical in _TOOLS:
                    return (canonical, _TOOLS[canonical])
            return (canonical, None)

    if not registry_ok:
        return (None, None)

    from agent.tools.registry import _TOOLS

    # 4. Case-insensitive exact match
    name_lower = tool_name.lower()
    for registered_name in _TOOLS:
        if name_lower == registered_name.lower():
            return (registered_name, _TOOLS[registered_name])

    # 5. Substring match — tool name must be a substring of the
    #    registered name (not vice versa), with strict threshold.
    best_match: str | None = None
    best_score = 0.0
    for registered_name in _TOOLS:
        reg_lower = registered_name.lower()
        if name_lower == reg_lower:
            best_match = registered_name
            best_score = len(name_lower) + 100
            break
        if name_lower in reg_lower:
            score = len(name_lower) / max(len(reg_lower), 1)
            if score > best_score:
                best_score = score
                best_match = registered_name

    if best_match and best_score >= 0.7:
        return (best_match, _TOOLS[best_match])

    # 6. Docstring keyword match — requires strong multi-keyword
    #    matches to avoid false positives (e.g. "create" matching
    #    "dir_create" when the real intent is "create_chart").
    keywords = set(re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", name_lower))
    keywords = {k for k in keywords if len(k) >= 2}
    if keywords:
        for registered_name, obj in _TOOLS.items():
            doc = ""
            if hasattr(obj, "__doc__") and obj.__doc__:
                doc = obj.__doc__.lower()
            elif hasattr(obj, "func") and hasattr(obj.func, "__doc__") and obj.func.__doc__:
                doc = obj.func.__doc__.lower()
            elif hasattr(obj, "description"):
                doc = str(obj.description).lower()

            reg_lower = registered_name.lower()
            score = 0
            name_hits = 0
            for kw in keywords:
                if kw in reg_lower:
                    score += 2
                    name_hits += 1
                if kw in doc:
                    score += 1
            # Require at least 2 name hits (or 1 name hit + 2 doc hits)
            # to prevent single-common-word false positives.
            if score >= 4 and name_hits >= 1:
                return (registered_name, obj)

    return (None, None)


def _ensure_registry() -> bool:
    """Ensure the tool registry is loaded (idempotent).

    Returns True if registry is available, False if loading failed.
    Failures are cached — once the registry fails to load, subsequent
    calls return False immediately rather than retrying expensive imports.
    """
    global _registry_loaded, _registry_failed
    if _registry_loaded:
        return True
    if _registry_failed:
        return False
    try:
        from agent.tools.registry import ensure_builtin_tools
        ensure_builtin_tools()
        _registry_loaded = True
        return True
    except (KeyboardInterrupt, SystemExit):
        _registry_failed = True
        logger.debug(
            "[SkillLearner] Tool registry load interrupted by environment; "
            "falling back to keyword-based resolution only"
        )
        return False
    except Exception:
        _registry_failed = True
        logger.warning(
            "[SkillLearner] Tool registry failed to load; "
            "falling back to keyword-based resolution only"
        )
        return False


class SkillExecutionEngine:
    """Executes learned skill tool sequences against the actual tool registry.

    Resolves inferred tool names to real registered tools, injects runtime
    parameters into stored arg schemas, and collects results with error
    tolerance (one step failure doesn't abort the whole sequence).

    Usage:
        engine = SkillExecutionEngine()
        result = engine.execute_skill(skill, query="AI趋势", filename="data.csv")
    """

    def __init__(self) -> None:
        self._tool_cache: dict[str, Any] = {}
        self._last_results: list[dict[str, Any]] = []

    def _resolve_tool(self, tool_name: str) -> tuple[str | None, Any | None]:
        """Resolve a tool name (inferred or alias) to the actual registered tool.

        Delegates to the shared _resolve_tool_in_registry function
        and caches the result for performance.
        """
        if tool_name in self._tool_cache:
            return self._tool_cache[tool_name]

        result = _resolve_tool_in_registry(tool_name)
        self._tool_cache[tool_name] = result
        return result

    def _call_tool(
        self,
        tool_obj: Any,
        args: dict[str, Any],
        timeout: float = 30.0,
    ) -> Any:
        """Call a tool object with the given args, handling wrapper types.

        Args:
            tool_obj: The tool to call (callable, BaseTool, or function).
            args: Arguments to pass to the tool.
            timeout: Maximum time in seconds to wait for tool execution.
        """
        import concurrent.futures
        import threading

        def _execute() -> Any:
            try:
                if callable(tool_obj):
                    return tool_obj(**args)
                if hasattr(tool_obj, "invoke"):
                    return tool_obj.invoke(args)
                if hasattr(tool_obj, "func") and callable(tool_obj.func):
                    return tool_obj.func(**args)
                return str(tool_obj)
            except Exception as exc:
                return exc

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_execute)
            try:
                result = future.result(timeout=timeout)
                if isinstance(result, Exception):
                    raise result
                return result
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutError(
                    f"Tool execution timed out after {timeout}s"
                )

    def _build_step_args(
        self,
        args_schema: dict[str, Any],
        runtime_kwargs: dict[str, Any],
        resolved_tool_name: str | None = None,
    ) -> dict[str, Any]:
        """Build actual tool arguments from stored schema + runtime parameters.

        Strategy:
          1. Use _ToolIntrospector to get merged param aliases (hardcoded
             + dynamic signature introspection)
          2. For each stored arg, apply param alias mapping, then override
             with runtime value if available
          3. Inject runtime params ONLY if the resolved tool is known to
             accept them (via merged param names)
          4. Special handling: stored keys query/input/task/prompt map
             to the runtime 'query' context
        """
        introspector = _ToolIntrospector()
        param_aliases = introspector.get_resolved_param_aliases(
            resolved_tool_name or ""
        )
        allowed_param_names: set[str] = set(param_aliases.keys()) | set(
            param_aliases.values()
        )

        args: dict[str, Any] = {}
        for key, stored_value in args_schema.items():
            mapped_key = param_aliases.get(key, key)

            if key in runtime_kwargs:
                args[mapped_key] = runtime_kwargs[key]
            elif mapped_key in runtime_kwargs:
                args[mapped_key] = runtime_kwargs[mapped_key]
            elif key == "query" and "query" in runtime_kwargs:
                args[mapped_key] = runtime_kwargs["query"]
            elif key == "input" and "query" in runtime_kwargs:
                args[mapped_key] = runtime_kwargs["query"]
            elif key == "task" and "query" in runtime_kwargs:
                args[mapped_key] = runtime_kwargs["query"]
            elif key == "prompt" and "query" in runtime_kwargs:
                args[mapped_key] = runtime_kwargs["query"]
            else:
                args[mapped_key] = stored_value

        _SKIP_KEYS = {"skill_name", "skill_id", "step_index"}
        for key, value in runtime_kwargs.items():
            if key in args or key in _SKIP_KEYS:
                continue
            if key in allowed_param_names:
                mapped_key = param_aliases.get(key, key)
                args[mapped_key] = value

        return args

    def execute_skill(
        self,
        skill: SkillDefinition,
        **runtime_kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a skill's full tool sequence.

        Args:
            skill: The skill definition to execute
            **runtime_kwargs: Runtime parameters injected into tool args.
                              Common keys: query, filename, path, etc.

        Returns:
            dict with:
              - skill_name: executed skill name
              - steps: list of per-step results
              - overall_success: bool
              - combined_result: concatenated results
              - failure_count: number of failed steps
        """
        steps: list[dict[str, Any]] = []
        combined_parts: list[str] = []
        failure_count = 0

        for idx, step in enumerate(skill.tool_sequence):
            tool_name = step.get("tool", "?")
            args_schema = step.get("args_schema", {})

            resolved_name, tool_obj = self._resolve_tool(tool_name)

            if tool_obj is None:
                steps.append({
                    "step": idx,
                    "tool": tool_name,
                    "resolved": None,
                    "status": "unresolved",
                    "error": f"Tool '{tool_name}' not found in registry",
                })
                failure_count += 1
                combined_parts.append(f"[Step {idx + 1}] {tool_name}: 未找到工具")
                continue

            step_args = self._build_step_args(args_schema, runtime_kwargs, resolved_name)

            try:
                result = self._call_tool(tool_obj, step_args)
                result_str = str(result) if result is not None else ""
                steps.append({
                    "step": idx,
                    "tool": tool_name,
                    "resolved": resolved_name,
                    "status": "success",
                    "result": result_str[:500],
                })
                combined_parts.append(
                    f"[Step {idx + 1}] {resolved_name}: {result_str[:200]}"
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                steps.append({
                    "step": idx,
                    "tool": tool_name,
                    "resolved": resolved_name,
                    "status": "failed",
                    "error": str(exc),
                })
                failure_count += 1
                combined_parts.append(
                    f"[Step {idx + 1}] {resolved_name}: 执行失败 - {exc}"
                )

        self._last_results = steps
        overall_success = failure_count == 0 and len(steps) > 0

        return {
            "skill_name": skill.name,
            "skill_id": skill.skill_id,
            "steps": steps,
            "overall_success": overall_success,
            "combined_result": "\n".join(combined_parts),
            "failure_count": failure_count,
            "total_steps": len(steps),
            "executed_at": time.time(),
        }

    def get_last_results(self) -> list[dict[str, Any]]:
        """Get the results of the last skill execution."""
        return list(self._last_results)

    def clear_cache(self) -> None:
        """Clear the tool resolution cache."""
        self._tool_cache.clear()
        _ToolIntrospector().clear_cache()

    def get_tool_schema(self, tool_name: str) -> dict[str, Any] | None:
        """Get the JSON Schema for a registered tool's parameters.

        Forward-compatible: when migrating to function-calling (JSON
        Schema), this method provides the schema that would be used
        for LLM tool binding.

        Returns a dict with:
          - name: tool name
          - params: list of {name, type, required} dicts
          - description: tool description
        """
        resolved_name, tool_obj = self._resolve_tool(tool_name)
        if tool_obj is None:
            return None
        introspector = _ToolIntrospector()
        param_names = introspector.get_param_names(resolved_name or tool_name)
        params_info = []
        for pname in sorted(param_names):
            params_info.append({
                "name": pname,
                "type": "string",
                "required": True,
            })
        description = ""
        if hasattr(tool_obj, "description"):
            description = str(tool_obj.description)
        elif hasattr(tool_obj, "__doc__") and tool_obj.__doc__:
            description = tool_obj.__doc__
        elif hasattr(tool_obj, "func") and tool_obj.func.__doc__:
            description = tool_obj.func.__doc__
        return {
            "name": resolved_name or tool_name,
            "params": params_info,
            "description": description.strip(),
        }

    def list_all_schemas(self) -> list[dict[str, Any]]:
        """Get JSON Schemas for all registered tools.

        This is the forward-compatible bridge to function-calling:
        the returned list can be passed to ``llm.bind_tools(schemas)``
        when migrating away from free-form tool name inference.
        """
        registry_ok = _ensure_registry()
        if not registry_ok:
            return []
        from agent.tools.registry import _TOOLS
        schemas = []
        for name in sorted(_TOOLS.keys()):
            schema = self.get_tool_schema(name)
            if schema:
                schemas.append(schema)
        return schemas

    def execute_multiple(
        self,
        skills: list[SkillDefinition],
        **runtime_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute multiple skills in sequence, collecting all results.

        Useful when multiple skills match a user query — execute them
        all and return combined results for synthesis.
        """
        results: list[dict[str, Any]] = []
        for skill in skills:
            try:
                result = self.execute_skill(skill, **runtime_kwargs)
                results.append(result)
            except Exception as exc:
                results.append({
                    "skill_name": skill.name,
                    "skill_id": skill.skill_id,
                    "overall_success": False,
                    "error": str(exc),
                    "steps": [],
                    "combined_result": f"Skill execution failed: {exc}",
                    "failure_count": 1,
                    "total_steps": 0,
                    "executed_at": time.time(),
                })
        return results

    def execute_best_match(
        self,
        query: str,
        matcher: "SkillMatcher | None" = None,
        **runtime_kwargs: Any,
    ) -> dict[str, Any] | None:
        """Find the best matching skill for a query and execute it.

        Combines skill matching + execution in a single call,
        returning None if no reliable skill matches.
        """
        if matcher is None:
            matcher = SkillMatcher()
        skill, score = matcher.best_match(query)
        if skill is None or score < 0.1:
            return None
        result = self.execute_skill(skill, **runtime_kwargs)
        result["match_score"] = score
        result["matched_query"] = query
        return result

    def validate_skill(self, skill: SkillDefinition) -> dict[str, Any]:
        """Validate a skill's tool sequence without actually executing.

        Checks that all tools in the sequence can be resolved and
        reports any issues found.
        """
        issues: list[str] = []
        resolved_count = 0
        unresolved_count = 0

        for idx, step in enumerate(skill.tool_sequence):
            tool_name = step.get("tool", "?")
            resolved_name, tool_obj = self._resolve_tool(tool_name)
            if tool_obj is None:
                unresolved_count += 1
                issues.append(
                    f"Step {idx}: tool '{tool_name}' (resolved: {resolved_name}) "
                    f"not found in registry"
                )
            else:
                resolved_count += 1

        valid = unresolved_count == 0 and resolved_count > 0
        return {
            "valid": valid,
            "total_steps": len(skill.tool_sequence),
            "resolved_count": resolved_count,
            "unresolved_count": unresolved_count,
            "issues": issues,
            "skill_name": skill.name,
            "skill_id": skill.skill_id,
        }


class SkillLearner:
    """Learns reusable skills from successful agent executions.

    Usage:
        store = SkillStore()
        learner = SkillLearner(store)

        # After a successful execution:
        skill = learner.learn_from_execution(
            user_input="分析销售数据",
            plan=["read_file", "analyze_data", "create_chart"],
            results=["文件已读取", "分析完成", "图表已创建"],
            tool_calls=[{"name": "read_file", "args": {...}}, ...],
            success=True,
        )
        if skill:
            print(f"New skill learned: {skill.name}")

        # Promote and execute:
        engine = SkillExecutionEngine()
        learner.promote_tools(skill, engine=engine)
        result = engine.execute_skill(skill, query="Q3销售数据")
    """

    def __init__(self, store: SkillStore | None = None) -> None:
        self._store = store or SkillStore()
        self._learned_skills: dict[str, SkillDefinition] = {}
        self._matcher = SkillMatcher(self._store)
        self._introspector = _ToolIntrospector()

    def learn_from_execution(
        self,
        user_input: str,
        plan: list[str],
        results: list[str],
        tool_calls: list[dict[str, Any]] | None = None,
        success: bool = True,
        reflection_score: float = 0.0,
    ) -> SkillDefinition | None:
        if not success and reflection_score < 0.5:
            return None

        if not plan or len(plan) < _MIN_TOOL_SEQUENCE_LENGTH:
            return None

        tool_sequence = self._extract_tool_sequence(plan, tool_calls)
        if len(tool_sequence) < _MIN_TOOL_SEQUENCE_LENGTH:
            return None

        skill_name = self._generate_skill_name(user_input, plan)
        if not skill_name:
            return None

        trigger_patterns = self._extract_trigger_patterns(user_input)

        existing = self._store.get_by_name(skill_name)
        if existing:
            if success:
                self._store.update_stats(existing.skill_id, success=True)
            return existing

        description = self._generate_description(skill_name, tool_sequence, user_input)

        tool_schemas = self._build_tool_schemas(tool_sequence)

        semantic_vector: list[float] = []
        if self._matcher.semantic_available:
            try:
                from agent.llm.embeddings import get_embedding_provider
                provider = get_embedding_provider()
                embed_text = f"{skill_name} {description}"
                if trigger_patterns:
                    embed_text += " " + " ".join(trigger_patterns)
                semantic_vector = provider.embed(embed_text)
            except Exception:
                pass

        skill = SkillDefinition(
            name=skill_name,
            description=description,
            trigger_patterns=trigger_patterns[:_MAX_TRIGGER_PATTERNS],
            tool_sequence=tool_sequence[:_MAX_TOOL_SEQUENCE_LENGTH],
            tool_schemas=tool_schemas,
            confidence=0.5 if success else 0.3,
            usage_count=1,
            success_count=1 if success else 0,
            failure_count=0 if success else 1,
            semantic_vector=semantic_vector,
        )

        self._store.save(skill)
        self._learned_skills[skill.skill_id] = skill

        logger.info(
            "[SkillLearner] New skill learned: %s (tools: %d, triggers: %d)",
            skill_name, len(tool_sequence), len(trigger_patterns),
        )

        return skill

    def _extract_tool_sequence(
        self,
        plan: list[str],
        tool_calls: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        sequence: list[dict[str, Any]] = []

        if isinstance(tool_calls, list) and tool_calls:
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                name = call.get("name", "")
                args = call.get("args", {})
                if name:
                    resolved = self._resolve_tool_name(name)
                    sequence.append({
                        "tool": resolved or name,
                        "args_schema": self._summarize_args(args),
                    })
        else:
            for step in plan:
                tool_hint = self._infer_tool_from_step(step)
                if tool_hint:
                    sequence.append({"tool": tool_hint, "args_schema": {}})

        return sequence

    def _resolve_tool_name(self, name: str) -> str | None:
        """Resolve a tool name against the actual registry.

        Delegates to the shared _resolve_tool_in_registry function
        to ensure consistent resolution with SkillExecutionEngine.
        """
        resolved_name, _ = _resolve_tool_in_registry(name)
        return resolved_name

    def _summarize_args(self, args: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str):
                if len(value) > 30:
                    summary[key] = value[:30] + "..."
                else:
                    summary[key] = value
            elif isinstance(value, (int, float, bool)):
                summary[key] = value
            elif isinstance(value, list):
                summary[key] = f"list[{len(value)}]"
            else:
                summary[key] = str(type(value).__name__)
        return summary

    def _build_tool_schemas(
        self, tool_sequence: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build JSON Schema list for each tool in a skill's sequence.

        Uses _ToolIntrospector to auto-discover param names from
        registered tool signatures, producing forward-compatible
        schemas for function-calling migration.
        """
        schemas: list[dict[str, Any]] = []
        for step in tool_sequence:
            tool_name = step.get("tool", "?")
            resolved_name, tool_obj = _resolve_tool_in_registry(tool_name)
            if tool_obj is None:
                schemas.append({
                    "tool": tool_name,
                    "resolved": None,
                    "params": [],
                    "description": "",
                })
                continue
            actual_name = resolved_name or tool_name
            params = self._introspector.get_param_names(actual_name)
            param_info = [
                {"name": p, "type": "string", "required": True}
                for p in sorted(params)
            ]
            description = ""
            if hasattr(tool_obj, "description"):
                description = str(tool_obj.description)
            elif hasattr(tool_obj, "__doc__") and tool_obj.__doc__:
                description = tool_obj.__doc__
            elif hasattr(tool_obj, "func") and tool_obj.func.__doc__:
                description = tool_obj.func.__doc__
            schemas.append({
                "tool": actual_name,
                "resolved": actual_name,
                "params": param_info,
                "description": description.strip(),
            })
        return schemas

    def _infer_tool_from_step(self, step: str) -> str | None:
        """Infer which tool was likely used from a step description.

        Uses a dynamic resolution strategy:
          1. Check if any registered tool name appears in the step text
          2. Check known aliases against the step text
          3. Score candidate tools by keyword overlap with their docstrings

        When registry is unavailable, falls back to the legacy keyword
        mapping to avoid hard failure.
        """
        registry_ok = _ensure_registry()
        step_lower = step.lower()

        if registry_ok:
            from agent.tools.registry import _TOOLS

            # 1. Exact registered tool name in step
            for registered_name in _TOOLS:
                if registered_name.lower() in step_lower:
                    return registered_name

        # 2. Alias match (works without registry)
        #    Score by alias length: longer aliases are more specific
        best_alias: str | None = None
        best_score = 0
        for canonical, aliases in _TOOL_NAME_ALIASES.items():
            for alias in aliases:
                alias_lower = alias.lower()
                if alias_lower in step_lower:
                    score = len(alias_lower)
                    if score > best_score:
                        best_score = score
                        best_alias = canonical
                elif step_lower in alias_lower and len(step_lower) >= 2:
                    score = len(step_lower) * 0.5
                    if score > best_score:
                        best_score = score
                        best_alias = canonical
        if best_alias:
            return best_alias

        if not registry_ok:
            return self._legacy_keyword_infer(step)

        # 3. Keyword scoring against tool docstrings
        from agent.tools.registry import _TOOLS
        step_keywords = set(re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", step_lower))
        step_keywords = {k for k in step_keywords if len(k) >= 2}

        if not step_keywords:
            return None

        best_tool: str | None = None
        best_doc_score = 0
        for registered_name, obj in _TOOLS.items():
            doc = ""
            if hasattr(obj, "__doc__") and obj.__doc__:
                doc = obj.__doc__.lower()
            elif hasattr(obj, "func") and hasattr(obj.func, "__doc__") and obj.func.__doc__:
                doc = obj.func.__doc__.lower()
            elif hasattr(obj, "description"):
                doc = str(obj.description).lower()

            name_lower = registered_name.lower()
            score = 0
            name_hits = 0
            for kw in step_keywords:
                if kw in name_lower:
                    score += 3
                    name_hits += 1
                if kw in doc:
                    score += 1

            # Require at least one name hit to avoid false positives
            # from single common words matching unrelated tools.
            if score > best_doc_score and name_hits >= 1:
                best_doc_score = score
                best_tool = registered_name

        if best_tool and best_doc_score >= 4:
            return best_tool

        return None

    def _legacy_keyword_infer(self, step: str) -> str | None:
        """Legacy keyword-based tool inference (fallback when registry is unavailable)."""
        step_lower = step.lower()
        tool_keywords = {
            "text_reader": ["读取", "阅读", "查看", "read", "open", "file", "reader"],
            "file_writer": ["写入", "保存", "创建文件", "write", "save"],
            "file_editor": ["编辑", "修改", "edit", "modify"],
            "file_editor_all": ["全部替换", "replace all", "full edit"],
            "file_editor_multiline": ["多行编辑", "multiline", "multi edit"],
            "file_opener": ["打开", "open"],
            "file_copy": ["复制", "copy"],
            "file_move": ["移动", "重命名", "move", "rename"],
            "file_delete": ["删除", "delete", "remove"],
            "file_list": ["列出文件", "list files", "ls"],
            "dir_create": ["创建目录", "mkdir", "create dir"],
            "dir_delete": ["删除目录", "rmdir", "remove dir"],
            "directory_list": ["列表", "目录", "list", "directory", "dir"],
            "web_search": ["搜索", "查找", "查询", "search", "find", "web"],
            "web_content_fetcher": ["获取", "访问", "fetch", "url", "download"],
            "web_fetcher": ["下载", "download", "web fetch"],
            "code_executor": ["执行", "运行", "run", "execute", "code"],
            "code_lint": ["分析", "代码检查", "analyze", "lint", "code"],
            "code_typecheck": ["类型检查", "typecheck"],
            "grep": ["搜索", "查找", "search", "grep", "ripgrep", "代码搜索"],
            "code_outline": ["大纲", "outline", "结构", "structure"],
            "summarize_text": ["总结", "摘要", "summarize", "summary"],
            "summarize_file": ["文件摘要", "file summary"],
            "summarize_url": ["网页摘要", "url summary"],
            "translate": ["翻译", "translate"],
            "translate_batch": ["批量翻译", "batch translate"],
            "shell_executor": ["命令行", "终端", "shell", "terminal", "command"],
            "shell_executor_multi": ["批量命令", "multi shell"],
            "math_calculator": ["计算", "calc", "calculate", "math"],
            "image_analyze": ["图片", "图像", "image", "analyze", "分析"],
            "image_batch_analyze": ["批量图片", "batch image"],
            "file_diff": ["对比", "比较", "diff", "compare"],
            "file_diff_summary": ["差异摘要", "diff summary"],
            "file_patch": ["patch", "补丁", "apply patch"],
            "file_zip": ["压缩", "归档", "archive", "compress", "zip"],
            "file_unzip": ["解压", "extract", "unzip"],
            "file_zip_list": ["列出归档", "list archive"],
            "http_request": ["http", "request", "api", "接口"],
            "http_get": ["http get", "get request"],
            "http_post": ["http post", "post request"],
            "datetime_query": ["时间", "日期", "datetime", "date"],
            "note_save": ["笔记", "note", "memo", "保存笔记"],
            "note_read": ["读取笔记", "read note"],
            "note_list": ["列出笔记", "list notes"],
            "git_commit": ["git commit", "提交"],
            "git_push": ["git push", "推送"],
            "git_pull": ["git pull", "拉取"],
            "git_branch": ["git branch", "分支"],
            "git_status": ["git status", "状态"],
            "git_log": ["git log", "历史", "history"],
            "git_log_since": ["近期日志", "recent log"],
            "git_diff": ["git diff", "差异"],
            "git_diff_staged": ["暂存差异", "staged diff"],
            "git_add": ["git add", "暂存"],
            "git_remote": ["git remote", "远程"],
            "git_show": ["git show", "查看提交"],
            "system_info": ["系统", "system", "info"],
            "glob": ["文件搜索", "find files", "glob"],
            "browser_navigate": ["浏览器", "browser", "navigate"],
            "browser_click": ["点击", "click"],
            "browser_type": ["输入", "type", "input text"],
            "browser_snapshot": ["页面快照", "snapshot"],
            "browser_screenshot": ["截图", "screenshot"],
            "browser_list_interactive": ["列出交互", "list interactive"],
            "browser_close": ["关闭浏览器", "close browser"],
            "browser_press_key": ["按键", "press key"],
        }
        for tool_name, keywords in tool_keywords.items():
            if any(kw in step_lower for kw in keywords):
                return tool_name
        return None

    def _generate_skill_name(self, user_input: str, plan: list[str]) -> str | None:
        input_lower = user_input.lower()

        name_patterns = [
            (r"分析(.+?)数据", "analyze_{topic}_data"),
            (r"(.+?)生成.*?(图表|图|chart)", "generate_{topic}_chart"),
            (r"(.+?)总结|汇总", "summarize_{topic}"),
            (r"(.+?)翻译", "translate_{topic}"),
            (r"查找|搜索(.+)", "search_{topic}"),
            (r"创建|生成(.+?)代码", "generate_{topic}_code"),
            (r"修复|调试(.+)", "debug_{topic}"),
            (r"部署|发布(.+)", "deploy_{topic}"),
            (r"读取(.+?)文件", "read_{topic}_file"),
            (r"写入|保存(.+)", "save_{topic}"),
            (r"计算(.+)", "calculate_{topic}"),
        ]

        for pattern, template in name_patterns:
            match = re.search(pattern, input_lower)
            if match:
                topic = match.group(1).strip()
                if len(topic) > 20:
                    topic = topic[:20]
                topic = re.sub(r"[^\w\u4e00-\u9fff]+", "_", topic)
                return template.format(topic=topic)

        first_step = plan[0] if plan else ""
        step_words = re.findall(r"[\u4e00-\u9fff]+|\w+", first_step)
        if step_words:
            topic = "_".join(step_words[:3]).lower()
            if len(topic) > 30:
                topic = topic[:30]
            return f"skill_{topic}"

        input_hash = hashlib.md5(user_input.encode("utf-8")).hexdigest()[:8]
        return f"skill_{input_hash}"

    def _extract_trigger_patterns(self, user_input: str) -> list[str]:
        patterns: list[str] = []

        words = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", user_input)
        for word in words:
            if len(word) >= 2:
                patterns.append(word.lower())

        if len(patterns) < 2:
            patterns.append(user_input.lower()[:10])

        return patterns[:_MAX_TRIGGER_PATTERNS]

    def _generate_description(
        self, skill_name: str, tool_sequence: list[dict], user_input: str
    ) -> str:
        tools_str = " -> ".join(
            t.get("tool", "?") for t in tool_sequence[:5]
        )
        return (
            f"Skill '{skill_name}': {len(tool_sequence)} steps "
            f"[{tools_str}]. Triggered by: {user_input[:60]}"
        )

    def find_relevant_skill(self, user_input: str) -> SkillDefinition | None:
        skill, _ = self._matcher.best_match(user_input)
        return skill

    def find_relevant_skill_with_score(
        self, user_input: str
    ) -> tuple[SkillDefinition | None, float]:
        return self._matcher.best_match(user_input)

    def hybrid_search_skills(
        self, user_input: str, top_k: int = 5
    ) -> list[tuple[SkillDefinition, float]]:
        return self._matcher.hybrid_search(user_input, top_k=top_k)

    def rebuild_semantic_index(self) -> int:
        return self._matcher.rebuild_semantic_index()

    def get_learned_skills(self) -> list[SkillDefinition]:
        return list(self._learned_skills.values())

    def get_store_stats(self) -> dict[str, int]:
        return self._store.stats

    def promote_tools(
        self,
        skill: SkillDefinition,
        engine: SkillExecutionEngine | None = None,
        extra_tags: set[str] | None = None,
    ) -> bool:
        """Promote a skill as a runtime-available executable tool.

        The promoted tool is a real callable that:
          1. Takes runtime parameters (query, filename, etc.)
          2. Uses SkillExecutionEngine to execute the learned tool sequence
          3. Returns the combined result of all steps
          4. Auto-infers scene tags from the skill's tool sequence for
             scene-based routing optimization

        This replaces the old behavior where promoted tools only printed
        descriptive text without actually executing anything.
        """
        try:
            from agent.tools.registry import register

            exec_engine = engine or SkillExecutionEngine()

            scene_tags = self._infer_scene_tags(skill)
            if extra_tags:
                scene_tags.update(extra_tags)
            scene_tags.add("skill")

            def _skill_tool(**kwargs: Any) -> str:
                result = exec_engine.execute_skill(skill, **kwargs)
                if result["overall_success"]:
                    self._store.update_stats(skill.skill_id, success=True)
                else:
                    self._store.update_stats(skill.skill_id, success=False)
                return result.get("combined_result", "")

            _skill_tool.__name__ = f"skill_{skill.skill_id}"
            _skill_tool.__qualname__ = f"skill_{skill.skill_id}"
            _skill_tool.__doc__ = skill.description

            register(tags=scene_tags)(_skill_tool)
            logger.info(
                "[SkillLearner] Skill promoted as executable tool: %s (tags: %s)",
                skill.name, scene_tags,
            )
            return True
        except Exception as exc:
            logger.warning("[SkillLearner] Failed to promote skill: %s", exc)
            return False

    def _infer_scene_tags(self, skill: SkillDefinition) -> set[str]:
        """Infer scene tags from the tool sequence for optimal routing.

        Maps tool names to their likely scene tags based on common
        tool-name prefixes and registered tag conventions.
        """
        tags: set[str] = {"core"}
        tool_names = {step.get("tool", "") for step in skill.tool_sequence}
        tool_names.update(s.get("tool", "") for s in (skill.tool_schemas or []))

        _TAG_MAP: dict[str, set[str]] = {
            "web": {"web_search", "web_content_fetcher", "web_fetcher",
                    "browser_navigate", "browser_click", "browser_type",
                    "browser_snapshot", "browser_screenshot"},
            "file": {"text_reader", "file_opener", "file_writer", "file_editor",
                     "file_copy", "file_move", "file_delete", "file_list",
                     "dir_create", "dir_delete", "directory_list",
                     "file_zip", "file_unzip", "file_diff", "file_patch"},
            "code": {"code_executor", "code_lint", "code_typecheck",
                     "grep", "code_outline", "glob"},
            "dev": {"shell_executor", "shell_executor_multi",
                    "git_commit", "git_push", "git_pull", "git_branch",
                    "git_status", "git_log", "git_diff", "git_add",
                    "http_request", "http_get", "http_post"},
            "analysis": {"math_calculator", "image_analyze", "image_batch_analyze"},
            "summarize": {"summarize_text", "summarize_file", "summarize_url"},
            "translate": {"translate", "translate_batch"},
        }

        for tag, tool_set in _TAG_MAP.items():
            if tool_names & tool_set:
                tags.add(tag)

        return tags

    def cleanup(self) -> int:
        removed = self._store.cleanup_expired(min_usage=1, min_confidence=0.3)
        logger.info("[SkillLearner] Cleaned up %d low-quality skills", removed)
        return removed

    def execute_skill_for_query(
        self,
        user_input: str,
        engine: SkillExecutionEngine | None = None,
        **runtime_kwargs: Any,
    ) -> dict[str, Any] | None:
        """Find best matching skill for a query and execute it.

        Combines find_relevant_skill + execute_skill + promote in one call.
        Returns the execution result dict, or None if no skill matches.
        """
        skill = self.find_relevant_skill(user_input)
        if skill is None:
            logger.debug("[SkillLearner] No matching skill for: %s", user_input)
            return None

        exec_engine = engine or SkillExecutionEngine()
        result = exec_engine.execute_skill(skill, **runtime_kwargs)

        if result["overall_success"]:
            self._store.update_stats(skill.skill_id, success=True)
        else:
            self._store.update_stats(skill.skill_id, success=False)

        return result

    def list_reliable_skills(self) -> list[SkillDefinition]:
        """Return all reliable skills (is_reliable=True)."""
        return [s for s in self._store.list_all() if s.is_reliable]

    def get_skill_stats(self) -> dict[str, Any]:
        """Get detailed statistics about all skills."""
        all_skills = self._store.list_all()
        reliable = [s for s in all_skills if s.is_reliable]
        by_confidence = sorted(all_skills, key=lambda s: s.confidence, reverse=True)
        return {
            "total": len(all_skills),
            "reliable": len(reliable),
            "unreliable": len(all_skills) - len(reliable),
            "skills_by_confidence": [
                {"name": s.name, "confidence": s.confidence, "usage": s.usage_count}
                for s in by_confidence[:10]
            ],
            "avg_confidence": (
                sum(s.confidence for s in all_skills) / max(len(all_skills), 1)
            ),
        }

    def validate_all_skills(
        self, engine: SkillExecutionEngine | None = None
    ) -> list[dict[str, Any]]:
        """Validate all stored skills and report issues.

        Returns a list of validation results for each skill.
        """
        exec_engine = engine or SkillExecutionEngine()
        results: list[dict[str, Any]] = []
        for skill in self._store.list_all():
            result = exec_engine.validate_skill(skill)
            results.append(result)
            if not result["valid"]:
                logger.warning(
                    "[SkillLearner] Invalid skill '%s': %s",
                    skill.name, result["issues"],
                )
        return results


class SkillMatcher:
    """Multi-strategy skill matching with semantic + keyword fallback.

    Matching order:
      1. Semantic search (Embedding cosine similarity) — catches paraphrases
      2. Keyword search (trigger_patterns + name substring) — exact matches
      3. Combined hybrid search — merges results with weighted scoring

    Usage:
        matcher = SkillMatcher(store)
        skill, score = matcher.best_match("分析销售数据趋势")
    """

    _SEMANTIC_THRESHOLD = 0.25
    _KEYWORD_BOOST = 0.15

    def __init__(self, store: SkillStore | None = None) -> None:
        self._store = store or SkillStore()
        self._semantic_available: bool | None = None

    @property
    def semantic_available(self) -> bool:
        if self._semantic_available is not None:
            return self._semantic_available
        try:
            from agent.llm.embeddings import get_embedding_provider
            provider = get_embedding_provider()
            test_vec = provider.embed("test")
            self._semantic_available = len(test_vec) > 0
        except Exception:
            self._semantic_available = False
        return self._semantic_available

    def best_match(
        self,
        query: str,
        min_score: float = 0.0,
        require_reliable: bool = True,
    ) -> tuple[SkillDefinition | None, float]:
        """Find the best matching skill for a user query.

        Args:
            query: User input text to match against skills.
            min_score: Minimum acceptable score (0.0-1.0).
            require_reliable: If True, only consider reliable skills
                             (usage_count >= 2 and success_rate >= 0.6).

        Returns (skill, score) tuple. Score is 0.0-1.0.
        """
        results = self.hybrid_search(query)
        if require_reliable:
            results = [
                (s, sc) for s, sc in results
                if s.is_reliable
            ]
        if not results:
            return (None, 0.0)
        best_skill, best_score = results[0]
        if best_score < min_score:
            return (None, best_score)
        return (best_skill, best_score)

    def hybrid_search(
        self, query: str, top_k: int = 5
    ) -> list[tuple[SkillDefinition, float]]:
        """Combined semantic + keyword search with score fusion."""
        scored: dict[str, tuple[SkillDefinition, float]] = {}

        if self.semantic_available:
            semantic_results = self._store.semantic_search(
                query, top_k=top_k * 2, threshold=self._SEMANTIC_THRESHOLD
            )
            for skill, sem_score in semantic_results:
                if skill.skill_id not in scored:
                    scored[skill.skill_id] = (skill, sem_score)
                else:
                    existing = scored[skill.skill_id][1]
                    scored[skill.skill_id] = (skill, max(existing, sem_score))

        keyword_results = self._store.find_matching(query)
        for skill in keyword_results:
            kid = skill.skill_id
            if kid in scored:
                base = scored[kid][1]
                boosted = min(1.0, base + self._KEYWORD_BOOST)
                scored[kid] = (skill, boosted)
            else:
                scored[kid] = (skill, 0.5)

        sorted_results = sorted(scored.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def rebuild_semantic_index(self) -> int:
        """Rebuild semantic vectors for all skills in the store."""
        if not self.semantic_available:
            logger.warning("[SkillMatcher] Semantic unavailable, skipping rebuild")
            return 0
        try:
            from agent.llm.embeddings import get_embedding_provider
            provider = get_embedding_provider()
        except ImportError:
            return 0

        skills = self._store.list_all()
        updated = 0
        for skill in skills:
            text = f"{skill.name} {skill.description}"
            if skill.trigger_patterns:
                text += " " + " ".join(skill.trigger_patterns)
            vec = provider.embed(text)
            if vec:
                self._store.update_semantic_vector(skill.skill_id, vec)
                updated += 1
        logger.info("[SkillMatcher] Rebuilt semantic index for %d skills", updated)
        return updated

    def fuzzy_match(
        self, query: str, min_score: float = 0.3
    ) -> list[tuple[SkillDefinition, float]]:
        """Fuzzy match using character-level similarity (Levenshtein-like).

        Useful for catching typos or slight variations in skill names.
        """
        query_lower = query.lower()
        results: list[tuple[SkillDefinition, float]] = []

        for skill in self._store.list_all():
            if not skill.is_reliable:
                continue

            name_lower = skill.name.lower()
            desc_lower = skill.description.lower()
            patterns = [p.lower() for p in skill.trigger_patterns]

            score = 0.0
            if query_lower == name_lower:
                score = 1.0
            elif query_lower in name_lower or name_lower in query_lower:
                score = 0.8
            elif any(query_lower in p or p in query_lower for p in patterns):
                score = 0.7
            elif query_lower in desc_lower:
                score = 0.5
            else:
                query_chars = set(query_lower)
                name_chars = set(name_lower)
                overlap = len(query_chars & name_chars)
                total = max(len(query_chars | name_chars), 1)
                score = overlap / total * 0.4

            if score >= min_score:
                results.append((skill, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def regex_match(
        self, pattern: str, flags: int = 0
    ) -> list[tuple[SkillDefinition, float]]:
        """Match skills using a regex pattern against names and descriptions."""
        try:
            compiled = re.compile(pattern, flags)
        except re.error:
            return []

        results: list[tuple[SkillDefinition, float]] = []
        for skill in self._store.list_all():
            if not skill.is_reliable:
                continue

            name_match = compiled.search(skill.name)
            desc_match = compiled.search(skill.description)
            pattern_matches = [
                compiled.search(p) for p in skill.trigger_patterns
            ]

            if name_match:
                score = 0.9
            elif any(pattern_matches):
                score = 0.7
            elif desc_match:
                score = 0.5
            else:
                continue

            results.append((skill, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def explain_match(
        self, query: str, skill: SkillDefinition
    ) -> dict[str, Any]:
        """Explain why a skill matches a query (for debugging)."""
        reasons: list[str] = []
        query_lower = query.lower()

        if query_lower in skill.name.lower():
            reasons.append(f"Query substring matches skill name: '{skill.name}'")

        for pattern in skill.trigger_patterns:
            if pattern.lower() in query_lower:
                reasons.append(f"Trigger pattern '{pattern}' matches query")

        if self.semantic_available and skill.semantic_vector:
            try:
                from agent.llm.embeddings import get_embedding_provider, cosine_similarity
                provider = get_embedding_provider()
                query_vec = provider.embed(query)
                if query_vec:
                    sem_score = cosine_similarity(query_vec, skill.semantic_vector)
                    if sem_score > 0.3:
                        reasons.append(
                            f"Semantic similarity: {sem_score:.3f} "
                            f"(embedding cosine distance)"
                        )
            except Exception:
                pass

        return {
            "skill_name": skill.name,
            "query": query,
            "reasons": reasons,
            "is_reliable": skill.is_reliable,
            "usage_count": skill.usage_count,
            "confidence": skill.confidence,
        }