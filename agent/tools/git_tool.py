"""Git 版本控制工具 — 查看仓库状态、日志、差异等。

纯 subprocess 实现，直接调用本地 git 命令。
在项目根目录内执行 git 操作。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

from langchain.tools import tool

from agent.config import settings
from agent.tools.registry import register

_MAX_OUTPUT = 10_000
_TIMEOUT = 30

_GIT_BIN = "git"


def _run_git(args: list[str], timeout: int = _TIMEOUT) -> tuple[int, str, str]:
    """运行 git 命令，返回 (returncode, stdout, stderr)。"""
    cmd = [_GIT_BIN] + args
    root = os.path.realpath(settings.FILE_READER_ROOT)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=root,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "git 命令超时"
    except FileNotFoundError:
        return -1, "", "未找到 git 命令，请确保已安装 Git"
    except OSError as exc:
        return -1, "", str(exc)


def _is_git_repo() -> bool:
    """检查当前目录是否为 git 仓库。"""
    returncode, stdout, _ = _run_git(["rev-parse", "--git-dir"])
    return returncode == 0


def _truncate(text: str, max_size: int = _MAX_OUTPUT) -> str:
    if len(text) <= max_size:
        return text
    return text[:max_size] + f"\n\n（输出过长，已截断 {len(text)} → {max_size} 字符）"


def _check_git_available() -> str | None:
    """检查 git 是否可用，返回错误信息或 None。"""
    returncode, _, stderr = _run_git(["--version"])
    if returncode != 0:
        return f"❌ Git 不可用：{stderr}"
    if not _is_git_repo():
        return "⚠️ 当前目录不是 Git 仓库。请在一个 git 仓库中使用此工具。"
    return None


@register
@tool
def git_status(porcelain: bool = False) -> str:
    """查看 Git 仓库的当前状态。

    显示工作区和暂存区的文件变更情况。

    Args:
        porcelain: 是否使用简洁的 porcelain 格式
    """
    git_err = _check_git_available()
    if git_err:
        return git_err

    args = ["status"]
    if porcelain:
        args.append("--porcelain")

    returncode, stdout, stderr = _run_git(args)
    if returncode != 0:
        return f"❌ git status 失败：{stderr}"

    return _truncate(stdout.strip() or "(工作区干净，无变更)")


@register
@tool
def git_log(
    max_count: int = 10,
    oneline: bool = False,
    author: str = "",
    since: str = "",
) -> str:
    """查看 Git 提交历史日志。

    Args:
        max_count: 显示的最大提交数，默认 10
        oneline: 是否使用单行精简格式
        author: 按作者筛选，如 "Zhang"
        since: 显示某时间之后的提交，如 "2024-01-01"、"7 days ago"
    """
    git_err = _check_git_available()
    if git_err:
        return git_err

    args = ["log", f"--max-count={max_count}"]
    if oneline:
        args.append("--oneline")
    if author:
        args.append(f"--author={author}")
    if since:
        args.append(f"--since={since}")

    returncode, stdout, stderr = _run_git(args)
    if returncode != 0:
        return f"❌ git log 失败：{stderr}"

    return _truncate(stdout.strip() or "(无提交记录)")


@register
@tool
def git_diff(
    target: str = "HEAD",
    staged: bool = False,
    context_lines: int = 3,
) -> str:
    """查看 Git 文件差异。

    Args:
        target: 对比的目标，如 "HEAD"、"main~1"、"abc123"
        staged: 是否显示暂存区的差异
        context_lines: 上下文行数，默认 3
    """
    git_err = _check_git_available()
    if git_err:
        return git_err

    args = ["diff", f"-U{context_lines}"]
    if staged:
        args.append("--cached")
    if target:
        args.append(target)

    returncode, stdout, stderr = _run_git(args)
    if returncode != 0:
        return f"❌ git diff 失败：{stderr}"

    return _truncate(stdout.strip() or "(无差异)")


@register
@tool
def git_branch(list_all: bool = False) -> str:
    """查看 Git 分支列表。

    Args:
        list_all: 是否显示所有分支（含远程）
    """
    git_err = _check_git_available()
    if git_err:
        return git_err

    args = ["branch"]
    if list_all:
        args.append("-a")

    returncode, stdout, stderr = _run_git(args)
    if returncode != 0:
        return f"❌ git branch 失败：{stderr}"

    return _truncate(stdout.strip() or "(无分支)")


@register
@tool
def git_remote() -> str:
    """查看 Git 远程仓库配置。"""
    git_err = _check_git_available()
    if git_err:
        return git_err

    returncode, stdout, stderr = _run_git(["remote", "-v"])
    if returncode != 0:
        return f"❌ git remote 失败：{stderr}"

    return _truncate(stdout.strip() or "(无远程仓库)")


@register
@tool
def git_show(commit: str, stat: bool = True) -> str:
    """查看指定提交的详细信息。

    Args:
        commit: 提交哈希或引用，如 "HEAD"、"abc123"、"v1.0"
        stat: 是否显示文件变更统计
    """
    git_err = _check_git_available()
    if git_err:
        return git_err

    args = ["show", commit]
    if stat:
        args.append("--stat")

    returncode, stdout, stderr = _run_git(args)
    if returncode != 0:
        return f"❌ git show 失败：{stderr}"

    return _truncate(stdout.strip())


@register
@tool
def git_add(files: str = ".") -> str:
    """将文件添加到 Git 暂存区。

    Args:
        files: 要添加的文件，多个用空格分隔，默认 "."（所有文件）
    """
    git_err = _check_git_available()
    if git_err:
        return git_err

    file_list = files.split()
    args = ["add"] + file_list

    returncode, stdout, stderr = _run_git(args)
    if returncode != 0:
        return f"❌ git add 失败：{stderr}"

    return f"✅ 已添加到暂存区：{files}"


@register
@tool
def git_commit(message: str) -> str:
    """创建 Git 提交。

    Args:
        message: 提交信息（使用 -m 格式）
    """
    git_err = _check_git_available()
    if git_err:
        return git_err

    if not message.strip():
        return "❌ 提交信息不能为空"

    args = ["commit", "-m", message]
    returncode, stdout, stderr = _run_git(args)
    if returncode != 0:
        return f"❌ git commit 失败：{stderr}\n\n提示：请先使用 git_add 添加文件到暂存区。"

    return _truncate(stdout.strip())


@register
@tool
def git_pull(remote: str = "origin", branch: str = "") -> str:
    """从远程仓库拉取更新。

    Args:
        remote: 远程仓库名称，默认 "origin"
        branch: 分支名，默认使用当前分支
    """
    git_err = _check_git_available()
    if git_err:
        return git_err

    args = ["pull", remote]
    if branch:
        args.append(branch)

    returncode, stdout, stderr = _run_git(args, timeout=60)
    if returncode != 0:
        return f"❌ git pull 失败：{stderr}"

    return _truncate(stdout.strip())


@register
@tool
def git_push(remote: str = "origin", branch: str = "") -> str:
    """推送本地提交到远程仓库。

    Args:
        remote: 远程仓库名称，默认 "origin"
        branch: 分支名，默认使用当前分支
    """
    git_err = _check_git_available()
    if git_err:
        return git_err

    args = ["push", remote]
    if branch:
        args.append(branch)

    returncode, stdout, stderr = _run_git(args, timeout=60)
    if returncode != 0:
        return f"❌ git push 失败：{stderr}"

    return _truncate(stdout.strip())


@register
@tool
def git_diff_staged() -> str:
    """查看暂存区的变更差异。"""
    return git_diff(target="HEAD", staged=True)


@register
@tool
def git_log_since(since: str = "7 days ago", max_count: int = 20) -> str:
    """查看最近一段时间的 Git 提交记录。

    Args:
        since: 时间范围，如 "7 days ago"、"2024-01-01"、"1 week ago"
        max_count: 最大提交数，默认 20
    """
    return git_log(max_count=max_count, oneline=True, since=since)