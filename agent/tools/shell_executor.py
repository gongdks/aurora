"""Safe shell command execution for dev tasks.

Security:
  - Blocklist of dangerous commands (rm -rf, format, etc.)
  - Working directory restricted to project root
  - Timeout enforcement
  - Output truncation
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

from langchain.tools import tool

from agent.config import settings
from agent.tools.registry import register

_MAX_OUTPUT = 20_000
_TIMEOUT = 30
_MAX_COMMAND_LEN = 500

_BLOCKED_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+.*-rf\b", "rm -rf command"),
    (r"\brm\s+.*--no-preserve-root\b", "rm with no-preserve-root"),
    (r"\bformat\s+[A-Za-z]:", "disk format"),
    (r"\bdd\s+if=", "dd command"),
    (r"\bshutdown\b", "shutdown command"),
    (r"\breboot\b", "reboot command"),
    (r"\bpoweroff\b", "poweroff command"),
    (r"\bchmod\s+.*-R\s+/\s", "chmod on root"),
    (r"\bchown\s+.*-R\s+/\s", "chown on root"),
    (r"\beval\s*\(", "eval() call"),
    (r"\bexec\s*\(", "exec() call"),
    (r"\b__import__", "__import__ access"),
    (r"\bimportlib\b", "importlib usage"),
    (r"\bsubprocess\b", "subprocess module"),
    (r"\bos\.system\b", "os.system call"),
    (r"\bos\.popen\b", "os.popen call"),
    (r"\bpathlib\b.*Path\b.*write", "pathlib file write"),
]

_BLOCKED_COMMANDS = frozenset({
    "rm", "del", "erase", "format", "fdisk",
    "shutdown", "reboot", "poweroff", "halt",
    "mkfs", "mkswap",
})

_ALLOWED_COMMANDS_HINT = (
    "常用命令如：ls, dir, cat, type, echo, mkdir, cp, copy, mv, move, "
    "git, python, pip, node, npm, pytest, black, ruff, mypy, "
    "pip list, pip install, python -c, 等"
)


def _check_command_safety(command: str) -> str | None:
    """Pre-check command for dangerous patterns. Returns error message or None."""
    cmd = command.strip()
    if len(cmd) > _MAX_COMMAND_LEN:
        return f"命令过长（{len(cmd)} > {_MAX_COMMAND_LEN} 字符）"

    cmd_lower = cmd.lower()

    for pattern, label in _BLOCKED_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return f"禁止的命令模式: {label}"

    first_word = cmd.split()[0].lower() if cmd.split() else ""
    if first_word in _BLOCKED_COMMANDS:
        if not _is_safe_usage(first_word, cmd):
            return f"禁止的命令: {first_word}（该命令可能导致数据丢失或系统危险）"

    return None


def _is_safe_usage(cmd: str, full: str) -> bool:
    """Check if a blocked command is being used in a safe way."""
    if cmd in ("rm", "del", "erase"):
        return False
    return False


def _format_output(stdout: str, stderr: str, returncode: int) -> str:
    parts: list[str] = []
    if stdout.strip():
        parts.append(stdout.rstrip())
    if stderr.strip():
        parts.append(f"[stderr]\n{stderr.strip()}")
    if returncode != 0:
        parts.append(f"[exit code: {returncode}]")

    output = "\n".join(parts) if parts else "(无输出)"

    if len(output) > _MAX_OUTPUT:
        output = output[:_MAX_OUTPUT] + "\n\n（输出过长，已截断）"
    return output


@register
@tool
def shell_executor(command: str, timeout: int = 30) -> str:
    """在项目目录内安全执行 Shell 命令。

    用于运行开发命令、安装依赖、运行测试、查看文件列表等。
    所有命令在项目根目录下执行，有超时保护。

    Args:
        command: 要执行的命令，如 "ls -la"、"pip list"、"python -c 'print(1+1)'"、"git status"
        timeout: 超时时间（秒），默认 30 秒，最大 60 秒
    """
    safety_err = _check_command_safety(command)
    if safety_err:
        return f"[安全拦截] {safety_err}\n\n提示：{_ALLOWED_COMMANDS_HINT}"

    timeout = max(1, min(timeout, 60))

    try:
        root = os.path.realpath(settings.FILE_READER_ROOT)
        os.makedirs(root, exist_ok=True)

        cmd = command.strip()
        shell = sys.platform == "win32"

        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=root,
            env={
                **os.environ,
                "PYTHONHASHSEED": "0",
            },
        )

        return _format_output(result.stdout, result.stderr, result.returncode)

    except subprocess.TimeoutExpired:
        return f"[超时] 命令执行超过 {timeout} 秒：{command}"
    except OSError as exc:
        return f"[错误] 命令执行失败：{exc}"
    except Exception as exc:
        return f"[错误] {exc}"


@register
@tool
def shell_executor_multi(commands: str, timeout: int = 60) -> str:
    """依次执行多条 Shell 命令，返回所有结果。

    命令之间用 && 或换行分隔。适合需要连续执行的场景。

    Args:
        commands: 要执行的多条命令，用 && 或换行分隔
        timeout: 总超时时间（秒），默认 60 秒
    """
    parts: list[str] = []
    cmd_list = [c.strip() for c in re.split(r'&&|\n', commands) if c.strip()]

    if not cmd_list:
        return "命令为空。"

    for cmd in cmd_list:
        safety_err = _check_command_safety(cmd)
        if safety_err:
            parts.append(f"[跳过] {cmd} — {safety_err}")
            continue

        try:
            root = os.path.realpath(settings.FILE_READER_ROOT)
            shell = sys.platform == "win32"
            result = subprocess.run(
                cmd,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=root,
                env={**os.environ, "PYTHONHASHSEED": "0"},
            )
            output = _format_output(result.stdout, result.stderr, result.returncode)
            parts.append(f"$ {cmd}\n{output}")
        except subprocess.TimeoutExpired:
            parts.append(f"$ {cmd}\n[超时]")
        except Exception as exc:
            parts.append(f"$ {cmd}\n[错误] {exc}")

    return "\n\n".join(parts)