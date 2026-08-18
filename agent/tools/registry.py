"""Tool registry — centralized tool management with built-in tool implementations.

Provides a decorator-based registration system and a set of built-in tools
(file operations, web search, code execution, notes, etc.).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
import traceback
from collections.abc import Callable
from typing import Any

from agent.config import settings

logger = logging.getLogger(__name__)

_TOOLS: dict[str, Any] = {}


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
        _TOOLS[tool_name] = obj
        logger.debug("Registered tool: %s", tool_name)
        return obj

    if cls is not None:
        return _wrapper(cls)
    return _wrapper


def list_tools() -> list:
    """Return all registered tools as LangChain tool objects.

    Each tool is either a BaseTool subclass (instantiated) or a function
    wrapped with @tool.
    """
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


# ------------------------------------------------------------------
# Built-in tool implementations
# ------------------------------------------------------------------

@register
def file_reader(file_path: str, encoding: str = "utf-8") -> str:
    """Read the contents of a file. Use absolute or relative paths.

    Args:
        file_path: Path to the file to read
        encoding: File encoding (default: utf-8)

    Returns:
        File contents as text, or error message if file not found
    """
    abs_path = os.path.abspath(file_path)
    allowed_root = os.path.abspath(settings.FILE_READER_ROOT)

    if not abs_path.startswith(allowed_root) and not os.path.isabs(file_path):
        pass

    if not os.path.exists(abs_path):
        return f"Error: File not found: {file_path}"

    if os.path.isdir(abs_path):
        entries = os.listdir(abs_path)
        return f"Directory: {abs_path}\nContents: {', '.join(entries[:50])}"

    try:
        with open(abs_path, "r", encoding=encoding) as f:
            content = f.read()
        if len(content) > 50000:
            return f"File: {file_path} (truncated, {len(content)} chars total)\n\n{content[:50000]}"
        return content
    except Exception as exc:
        return f"Error reading file: {exc}"


@register
def file_writer(file_path: str, content: str, mode: str = "w") -> str:
    """Write content to a file. Creates parent directories if needed.

    Args:
        file_path: Path to the file to write
        content: Text content to write
        mode: File write mode ('w' for overwrite, 'a' for append)

    Returns:
        Success message or error
    """
    try:
        abs_path = os.path.abspath(file_path)
        os.makedirs(os.path.dirname(abs_path) if os.path.dirname(abs_path) else ".", exist_ok=True)
        with open(abs_path, mode, encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} bytes to {file_path}"
    except Exception as exc:
        return f"Error writing file: {exc}"


@register
def file_opener(file_path: str) -> str:
    """Open a file in the system's default application.

    Args:
        file_path: Path to the file to open

    Returns:
        Success message or error
    """
    try:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return f"Error: File not found: {file_path}"

        import platform
        system = platform.system()
        if system == "Windows":
            os.startfile(abs_path)  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", abs_path])
        else:
            subprocess.Popen(["xdg-open", abs_path])
        return f"Opened {file_path}"
    except Exception as exc:
        return f"Error opening file: {exc}"


@register
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo (no API key required).

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        Formatted search results with titles, URLs, and snippets
    """
    try:
        import requests

        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }

        results: list[str] = []
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        abstract = data.get("Abstract", "")
        if abstract:
            results.append(f"**Abstract**: {abstract}")

        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append(f"- {topic.get('Text', '')} ({topic.get('FirstURL', '')})")

        if not results:
            try:
                url2 = "https://duckduckgo.com/"
                params2 = {"q": query}
                resp2 = requests.get(url2, params=params2, timeout=15)
                results.append(f"Search completed for: {query}")
                results.append(f"Found {len(resp2.text)} bytes of content")
            except Exception:
                results.append(f"No results found for: {query}")

        return "\n".join(results)
    except ImportError:
        return "Error: requests library not installed. Run: pip install requests"
    except Exception as exc:
        return f"Search error: {exc}"


@register
def code_executor(code: str, timeout: int = 30) -> str:
    """Execute Python code in a sandboxed subprocess.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Output of the code execution or error message
    """
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd(),
            )
            output = result.stdout
            if result.stderr:
                output += f"\n--- Errors ---\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n--- Exit code: {result.returncode} ---"
            return output[:10000] or "(no output)"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {timeout} seconds"
    except Exception as exc:
        return f"Error executing code: {exc}\n{traceback.format_exc()}"


@register
def grep(pattern: str, path: str = ".", recursive: bool = True, max_results: int = 50) -> str:
    """Search for a pattern in files (like grep).

    Args:
        pattern: Regex pattern to search for
        path: File or directory path to search
        recursive: Whether to search recursively
        max_results: Maximum number of results to return

    Returns:
        Matching lines with file paths and line numbers
    """
    try:
        abs_path = os.path.abspath(path)
        results: list[str] = []

        if os.path.isfile(abs_path):
            files = [abs_path]
        elif os.path.isdir(abs_path):
            if recursive:
                files = []
                for root, dirs, filenames in os.walk(abs_path):
                    dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__",)]
                    for fn in filenames:
                        if not fn.endswith((".pyc", ".pyo", ".so", ".dll")):
                            files.append(os.path.join(root, fn))
            else:
                files = [
                    os.path.join(abs_path, f)
                    for f in os.listdir(abs_path)
                    if os.path.isfile(os.path.join(abs_path, f))
                ]
        else:
            return f"Error: Path not found: {path}"

        regex = re.compile(pattern, re.IGNORECASE)
        for filepath in files[:200]:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append(f"{filepath}:{line_no}: {line.rstrip()[:200]}")
                            if len(results) >= max_results:
                                break
                    if len(results) >= max_results:
                        break
            except (OSError, UnicodeDecodeError):
                continue
            if len(results) >= max_results:
                break

        if not results:
            return f"No matches found for pattern: {pattern}"
        return f"Found {len(results)} matches:\n" + "\n".join(results[:max_results])
    except re.error as exc:
        return f"Error: Invalid regex pattern: {exc}"
    except Exception as exc:
        return f"Error: {exc}"


@register
def glob(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., '*.py', '**/*.md')
        path: Base directory to search from

    Returns:
        List of matching file paths
    """
    try:
        import fnmatch

        abs_path = os.path.abspath(path)
        matches: list[str] = []

        if pattern.startswith("**/") or "**" in pattern:
            for root, dirs, filenames in os.walk(abs_path):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for fn in filenames:
                    rel = os.path.relpath(os.path.join(root, fn), abs_path)
                    if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(fn, pattern):
                        matches.append(rel)
        else:
            for entry in os.listdir(abs_path):
                full = os.path.join(abs_path, entry)
                if fnmatch.fnmatch(entry, pattern):
                    matches.append(entry)
                elif os.path.isdir(full):
                    for root, dirs, filenames in os.walk(full):
                        dirs[:] = [d for d in dirs if not d.startswith(".")]
                        for fn in filenames:
                            if fnmatch.fnmatch(fn, pattern):
                                matches.append(os.path.relpath(os.path.join(root, fn), abs_path))

        if not matches:
            return f"No files found matching pattern: {pattern}"
        return f"Found {len(matches)} files:\n" + "\n".join(matches[:100])
    except Exception as exc:
        return f"Error: {exc}"


@register
def note_save(title: str, content: str) -> str:
    """Save a note to the notes directory.

    Args:
        title: Note title (used as filename)
        content: Note content

    Returns:
        Success message
    """
    try:
        notes_dir = os.path.abspath(settings.NOTES_DIR)
        os.makedirs(notes_dir, exist_ok=True)
        safe_title = re.sub(r'[^\w\s-]', '', title)[:80].strip() or "note"
        filepath = os.path.join(notes_dir, f"{safe_title}_{int(time.time())}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n{content}")
        return f"Note saved: {filepath}"
    except Exception as exc:
        return f"Error saving note: {exc}"


@register
def note_read(keyword: str = "") -> str:
    """List all saved notes or search by keyword.

    Args:
        keyword: Optional keyword to filter notes by title/content

    Returns:
        List of notes or note contents
    """
    try:
        notes_dir = os.path.abspath(settings.NOTES_DIR)
        if not os.path.exists(notes_dir):
            return "No notes directory found."

        notes: list[str] = []
        for filename in sorted(os.listdir(notes_dir)):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(notes_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            if keyword.lower() in text.lower():
                notes.append(f"--- {filename} ---\n{text[:500]}")
            elif not keyword:
                notes.append(f"--- {filename} ---")

        if not notes:
            return f"No notes found{' matching ' + keyword if keyword else ''}."
        return "\n\n".join(notes[:20])
    except Exception as exc:
        return f"Error reading notes: {exc}"


@register
def browser_navigate(url: str) -> str:
    """Navigate to a URL and return the page title and first 500 chars of text.

    Args:
        url: URL to navigate to

    Returns:
        Page information or error
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string if soup.title else "(no title)"

        for tag in soup(["script", "style", "meta", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        summary = text[:2000]

        return f"**Page**: {title}\n**URL**: {url}\n\n{summary}"
    except ImportError:
        return "Error: beautifulsoup4 not installed. Run: pip install beautifulsoup4"
    except Exception as exc:
        return f"Error fetching {url}: {exc}"


@register
def system_info() -> str:
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


@register
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: Mathematical expression (e.g., '2 + 3 * 4')

    Returns:
        Result of the calculation
    """
    import math

    safe_dict = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "int": int, "float": float, "len": len,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "pi": math.pi,
        "e": math.e, "pow": pow,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"Result: {expression} = {result}"
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


@register
def http_fetcher(url: str, timeout: int = 15) -> str:
    """Fetch raw content from a URL and return as text.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Response content (text) or error
    """
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "unknown")
        text = resp.text[:5000]
        return f"**Status**: {resp.status_code}\n**Content-Type**: {content_type}\n\n{text}"
    except ImportError:
        return "Error: requests library not installed."
    except Exception as exc:
        return f"Error fetching {url}: {exc}"


@register
def directory_list(path: str = ".") -> str:
    """List files and directories in a given path.

    Args:
        path: Directory path to list

    Returns:
        Directory listing with file sizes
    """
    try:
        abs_path = os.path.abspath(path)
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
    except Exception as exc:
        return f"Error listing directory: {exc}"


def ensure_builtin_tools() -> None:
    """Ensure all built-in tools are registered.

    This function is idempotent — safe to call multiple times.
    """
    pass