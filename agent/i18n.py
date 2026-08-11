"""Internationalization (i18n) strings — single source of truth for all UI text.

To add a new language:
  1. Add a new dict (e.g., _ZH, _EN)
  2. Set LANG env var or change I18N.locale

Usage:
    from agent.i18n import I18N
    msg = I18N.get("cancel.stopped")  # → "⏹ 已停止"
"""

from __future__ import annotations

import os

_ZH = {
    # Cancel / Stop
    "cancel.stopped": "⏹ 已停止",
    "cancel.llm_cancelled": "LLM 调用已被用户取消",
    "cancel.stopped_by_user": "⏹ *Stopped.*",

    # Agent status
    "agent.thinking": "Thinking...",
    "agent.ready": "Ready",
    "agent.running": "Running",
    "agent.powered_by": "Powered by ReAct Framework",

    # Orchestrator
    "orch.analyzing": "📋 **Analyzing goal and creating execution plan...**",
    "orch.plan_created": "📋 **Execution Plan Created**",
    "orch.time_budget_exceeded": "⏰ **Global time budget exceeded ({elapsed:.0f}s), stopping execution**",
    "orch.step_start": "▶️ **Step {step_id}/{total}**: {description}",
    "orch.step_completed": "✅ Step {step_id} completed: {status}",
    "orch.step_completed_partial": "✅ Step {step_id} completed: {status} (partial - reached iteration limit)",
    "orch.step_limit": "⏱️ Step {step_id} reached limit, extracting partial results...",
    "orch.reflecting": "🤔 **Reflecting on Step {step_id} results...**",
    "orch.reflection_result": "💭 Reflection: {thought}",
    "orch.goal_achieved": "🎯 **Goal achieved! Finalizing...**",
    "orch.replan_needed": "🔄 **Plan needs adjustment, re-planning remaining steps...**",
    "orch.step_failed": "⚠️ **Step {step_id} failed: {error}**",
    "orch.analyzing_failure": "🔍 **Analyzing failure and attempting recovery...**",
    "orch.retrying": "🔄 **Retrying with adjusted approach...**",
    "orch.skipping": "⏭️ **Skipping failed step, continuing with remaining plan...**",
    "orch.replanned": "📋 **Re-planned: {count} new steps remaining**",
    "orch.verifying": "🔍 **Verifying completion...**",
    "orch.verification_result": "✅ Verification: {summary}",

    # Task results
    "task.completed": "## 任务完成\n",
    "task.completed_partial": "## 任务完成（部分）\n",
    "task.goal": "**目标**: {goal}\n",
    "task.status_success": "**执行状态**: ✅ 成功 ({steps} 步)\n",
    "task.status_partial": "**执行状态**: ⚠️ 部分完成 ({completed}/{total} 步)\n",
    "task.main_result": "\n### 主要结果\n",
    "task.detail_result": "\n### 详细结果",
    "task.no_output": "\n*(已完成所有步骤，但无具体输出)*",
    "task.completed_steps": "\n### 已完成步骤",
    "task.failed_steps": "\n### 未完成步骤",
    "task.result_summary": "\n### 结果摘要",
    "task.fallback": "已处理请求，但未产生结果。目标：{goal}",

    # Memory
    "memory.no_short_term": "（无近期对话）",
    "memory.cleared_count": "已清除 {mem_count} 条记忆和 {sum_count} 条对话摘要",
    "memory.clear_failed": "清除失败: {error}",
    "memory.no_long_term": "No long-term memory (simplified mode)",

    # Code executor
    "code.too_large": "Code too large ({size} > {max_size} chars), rejected",
    "code.forbidden_pattern": "Forbidden pattern detected: {label}",
    "code.syntax_error": "Syntax error in code: {exc}",
    "code.forbidden_node": "Forbidden AST node type: {node_type} (line {line})",
    "code.forbidden_import": "Forbidden import: {module} (line {line})",
    "code.forbidden_from_import": "Forbidden import: from {module} import ... (line {line})",
    "code.forbidden_call": "Forbidden function call: {func}() (line {line})",
    "code.forbidden_method": "Forbidden method call: .{method}() (line {line})",
    "code.forbidden_getattr": "Forbidden: getattr() can bypass import restrictions (line {line})",
    "code.forbidden_attr": "Forbidden attribute access: .{attr} (line {line})",
    "code.timeout": "[Timeout] Code execution exceeded 30 seconds.",
    "code.no_output": "(no output)",

    # Browser
    "browser.launch_failed": "Browser launch failed.",
    "browser.edge_error": "- Edge: {error}",
    "browser.chromium_error": "- Chromium: {error}",
    "browser.install_hint": "Ensure Edge is installed or run: python -m playwright install chromium",
    "browser.closed": "✅ Browser closed",
    "browser.started": "[Browser] Browser started ({method}, headless={headless})",
    "browser.nav_failed": "❌ Navigation failed: {exc}",
    "browser.click_failed": "❌ Click failed: {exc}",
    "browser.type_failed": "❌ Type failed: {exc}",
    "browser.press_key_failed": "❌ Press key failed: {exc}",
    "browser.snapshot_failed": "❌ Snapshot failed: {exc}",
    "browser.screenshot_failed": "❌ Screenshot failed: {exc}",
    "browser.no_page": "(no page)",
    "browser.no_content": "(unable to extract page content)",
    "browser.no_interactive": "  (no interactive elements detected)",
    "browser.no_inputs": "  (no input elements detected)",
    "browser.empty_content": "⚠️ Page content is empty or could not be extracted (JS-rendered page?).",

    # Planner
    "planner.creating_plan": "[Planner] Creating plan for: {input}",
    "planner.plan_created": "[Planner] Plan created: {steps} steps for goal: {goal}",
    "planner.empty_plan": "[Planner] Attempt {attempt} produced empty plan, retrying...",
    "planner.attempt_failed": "[Planner] Attempt {attempt} failed: {error}",
    "planner.all_failed": "[Planner] All attempts failed, using fallback plan",
    "planner.fallback_plan": "[Planner] Fallback plan: {steps} steps for goal: {goal}",

    # LLM retry
    "llm.timeout": "LLM 调用超时 ({timeout:.0f}s)，线程级超时",
    "llm.retry_timeout": "LLM 调用超时 (第 {attempt}/{max} 次): {exc}，{delay:.1f}s 后重试",
    "llm.retry_failed": "LLM 调用失败 (第 {attempt}/{max} 次): {exc}，{delay:.1f}s 后重试",
    "llm.all_failed": "LLM 调用彻底失败: {exc}",
    "llm.call_timeout": "调用超时 (第 {attempt}/{max} 次)，{delay:.1f}s 后重试...",
    "llm.call_failed": "调用失败 (第 {attempt}/{max} 次): {exc}，{delay:.1f}s 后重试...",
    "llm.call_timeout_final": "调用超时 (已达最大重试次数 {max})",
    "llm.call_failed_final": "调用失败 (已达最大重试次数 {max}): {exc}",
    "llm.non_retryable": "调用发生不可重试的异常: {exc}",

    # Runner
    "run.completed": "[Run] Completed in {elapsed:.1f}s, {iterations} iterations",
    "run.hit_limit": "[Run] Hit limit: {msg}",
    "run.value_error": "[Run] ValueError: {exc}",
    "run.cancelled": "[Run] Cancelled by user after {elapsed:.1f}s",
    "run.failed": "[Run] Failed: {exc}",
    "run.connection_error": "[Run] {type}: {exc}",

    # Executor
    "exec.starting": "[Exec] Starting tool-calling loop: {input}",
    "exec.error": "[Error] {error}",

    # Agent
    "agent.fast_path": "[Agent] Fast path (simple query): {input}",
    "agent.plan_path": "[Agent] Plan-and-Execute path: {input}",
    "agent.cancel_requested": "Cancel requested",

    # Orchestrator log
    "orch.starting": "[Orch] Starting Plan-and-Execute for: {input}",
    "orch.cancelled_before_step": "[Orch] Cancelled before step {id}",
    "orch.time_budget_log": "[Orch] Global time budget exceeded ({elapsed:.1f}s > {budget:.1f}s)",
    "orch.step_failed_log": "[Orch] Step {id} failed, attempting recovery",
    "orch.hit_limit_log": "[Orch] Step {id} hit limit, extracting partial results",
    "orch.extraction_failed_log": "[Orch] Partial result extraction failed: {exc}",
    "orch.reflection_failed_log": "[Orch] Reflection failed: {exc}",
    "orch.recovery_failed_log": "[Orch] Recovery failed: {exc}",
    "orch.replan_failed_log": "[Orch] Re-plan failed: {exc}",
    "orch.verification_failed_log": "[Orch] Verification failed: {exc}",
    "orch.completed_log": "[Orch] Completed | status={status} | steps={steps} | elapsed={elapsed:.1f}s",
}


class _I18N:
    """Lazy-init i18n store. Loads locale dict based on LANG env var."""

    _LOCALES = {
        "zh": _ZH,
        "zh-CN": _ZH,
        "zh_CN": _ZH,
    }

    def __init__(self) -> None:
        self._locale = os.environ.get("LANG", "zh")

    @property
    def locale(self) -> str:
        return self._locale

    def set_locale(self, locale: str) -> None:
        """Switch to a different locale (e.g., 'en', 'zh')."""
        self._locale = locale

    def get(self, key: str, **kwargs: str | int | float) -> str:
        """Get a localized string by key, with optional format args.

        Args:
            key: Dot-separated key (e.g. "cancel.stopped")
            **kwargs: Format arguments for the template string

        Returns:
            Localized string, or the key itself if not found
        """
        locale_dict = self._LOCALES.get(self._locale, _ZH)
        template = locale_dict.get(key)
        if template is None:
            # Fallback to zh
            template = _ZH.get(key, key)
        if kwargs:
            return template.format(**kwargs)
        return template


I18N = _I18N()
