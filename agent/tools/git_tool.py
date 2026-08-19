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


@register(tags={"dev", "git"})
@tool
def git_status(porcelain: bool = False) -> str:
    """查看 Git 仓库状态（工作区/暂存区变更）。"""
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


@register(tags={"dev", "git"})
@tool
def git_log(
    max_count: int = 10,
    oneline: bool = False,
    author: str = "",
    since: str = "",
) -> str:
    """查看 Git 提交历史日志。"""
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


@register(tags={"dev", "git"})
@tool
def git_diff(
    target: str = "HEAD",
    staged: bool = False,
    context_lines: int = 3,
) -> str:
    """查看 Git 文件差异。"""
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


@register(tags={"dev", "git"})
@tool
def git_branch(list_all: bool = False) -> str:
    """查看 Git 分支列表。"""
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


@register(tags={"dev", "git"})
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


@register(tags={"dev", "git"})
@tool
def git_show(commit: str, stat: bool = True) -> str:
    """查看指定提交的详细信息。"""
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


@register(tags={"dev", "git"})
@tool
def git_add(files: str = ".") -> str:
    """将文件添加到 Git 暂存区。"""
    git_err = _check_git_available()
    if git_err:
        return git_err

    file_list = files.split()
    args = ["add"] + file_list

    returncode, stdout, stderr = _run_git(args)
    if returncode != 0:
        return f"❌ git add 失败：{stderr}"

    return f"✅ 已添加到暂存区：{files}"


@register(tags={"dev", "git"})
@tool
def git_commit(message: str) -> str:
    """创建 Git 提交。"""
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


@register(tags={"dev", "git"})
@tool
def git_pull(remote: str = "origin", branch: str = "") -> str:
    """从远程仓库拉取更新。"""
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


@register(tags={"dev", "git"})
@tool
def git_push(remote: str = "origin", branch: str = "") -> str:
    """推送本地提交到远程仓库。"""
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


@register(tags={"dev", "git"})
@tool
def git_diff_staged() -> str:
    """查看暂存区的变更差异。"""
    return git_diff(target="HEAD", staged=True)


@register(tags={"dev", "git"})
@tool
def git_log_since(since: str = "7 days ago", max_count: int = 20) -> str:
    """查看最近一段时间的 Git 提交记录。"""
    return git_log(max_count=max_count, oneline=True, since=since)