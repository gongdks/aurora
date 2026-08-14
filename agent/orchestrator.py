"""Orchestrator — AutoGen-based Plan-and-Execute with multi-agent collaboration.

Replaces the hand-written Plan-Execute-Reflect-Verify loop with AutoGen's
GroupChat, where four specialized agents (Planner, Executor, Verifier,
UserProxy) collaborate under a deterministic state-machine speaker selection.

The state machine enforces: planner → executor → (user_proxy ↔ executor) → verifier
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

try:
    from autogen import (
        AssistantAgent,
        GroupChat,
        GroupChatManager,
        UserProxyAgent,
    )
    _HAS_AUTOGEN = True
except ImportError:
    _HAS_AUTOGEN = False
    AssistantAgent = GroupChat = GroupChatManager = UserProxyAgent = None  # type: ignore

from agent.config import settings
from agent.progress import make_log, make_tool
from agent.tools.registry import list_tools
from agent.utils.message_extractor import extract_answer, extract_plan

logger = logging.getLogger(__name__)

# ============================================================================
# Agent system messages
# ============================================================================

_PLANNER_SYSTEM_MSG = """\
You are a task planning specialist.

When you receive a user goal, create a clear, step-by-step execution plan.
Break the goal into 2-5 concrete, actionable steps. Number each step.

After outputting the plan, your job is DONE. Do NOT try to execute steps yourself.
Do NOT comment on execution results unless explicitly asked by the verifier to re-plan.

The executor has access to these tool categories:
- File operations: file_reader, file_writer, file_editor, file_opener
- Web search: web_search
- Code execution: code_executor
- Browser control: browser_navigate, browser_click, browser_type, etc.
- Code search: grep, glob, code_outline
- Notes: note_save, note_read

Output format:
## 执行计划
1. [Step description]
2. [Step description]
...

Then hand off to the executor."""

_EXECUTOR_SYSTEM_MSG = """\
You are a task execution specialist. Your job is to CARRY OUT steps, not describe them.

CRITICAL — YOU MUST USE FUNCTION CALLING:
When you need to perform an action, IMMEDIATELY call the relevant function.
- To list a directory → call file_reader(path="D:\\", list_dir=True)
- To read a file → call file_reader(path="D:\\file.txt")
- To search the web → call web_search(query="...")
- To open a file → call file_opener(path="D:\\file.txt")
- To execute code → call code_executor(code="...")

NEVER say "I'll use tool X to do Y". Instead, just CALL the function.
NEVER say you can't access local files — you have file tools, USE THEM.

After receiving a function result, briefly summarize what you found.
Then call the next function if needed, or report the step is complete.

Keep responses concise — one function call at a time, then report results."""

_VERIFIER_SYSTEM_MSG = """\
You are a final verification specialist.

Review ALL completed steps and their results against the user's original goal.

If the goal is FULLY achieved:
- Provide a clear, well-formatted summary of what was accomplished
- End your message with the word TERMINATE

If the goal is NOT fully achieved:
- Clearly state what is missing
- Ask the planner to create a new plan for the remaining work

Your final message MUST end with TERMINATE when the goal is achieved."""


# ============================================================================
# AutoGenOrchestrator
# ============================================================================

class AutoGenOrchestrator:
    """Plan-and-Execute orchestrator powered by AutoGen GroupChat.

    Four specialized agents collaborate in a deterministic flow:
        planner → executor → user_proxy (tool exec) → executor → verifier

    Speaker selection uses a state machine (not LLM auto-select) to
    guarantee correct routing, especially for tool call execution.
    """

    def __init__(self) -> None:
        if not _HAS_AUTOGEN:
            raise ImportError(
                "AutoGen 未安装。请运行: pip install pyautogen"
            )
        self._llm_config = _build_llm_config()
        self._cancel_event: threading.Event | None = None
        self._progress_cb: Callable[[dict[str, Any]], None] | None = None

    # ---- Public API ----

    def run(
        self,
        user_input: str,
        chat_history_text: str,
        chat_history_messages: list,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        """Run Plan-and-Execute via AutoGen GroupChat.

        Returns just the answer string (backward compatible).
        """
        result = self._run_groupchat(
            user_input, chat_history_text, progress_callback, cancel_event,
        )
        return result["answer"]

    def run_complex(
        self,
        user_input: str,
        chat_history_text: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
        plan_rounds: int = 0,
        previous_plan: str = "",
        previous_result: str = "",
    ) -> dict[str, str | int]:
        """Run GroupChat with re-plan support for LangGraph integration.

        Returns dict with keys: answer, plan, plan_rounds.
        """
        result = self._run_groupchat(
            user_input, chat_history_text, progress_callback, cancel_event,
            plan_rounds=plan_rounds,
            previous_plan=previous_plan,
            previous_result=previous_result,
        )
        return result

    def _run_groupchat(
        self,
        user_input: str,
        chat_history_text: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
        plan_rounds: int = 0,
        previous_plan: str = "",
        previous_result: str = "",
    ) -> dict[str, str | int]:
        """Internal: run GroupChat and return answer + plan + metadata."""
        logger.info("[AutoGenOrch] Starting GroupChat for: %s", user_input[:60])

        self._cancel_event = cancel_event
        self._progress_cb = progress_callback

        if cancel_event and cancel_event.is_set():
            return {"answer": "⏹ 已停止", "plan": "", "plan_rounds": plan_rounds}

        if progress_callback and plan_rounds == 0:
            progress_callback(make_log("📋 **Starting Plan-and-Execute via AutoGen GroupChat...**"))
        elif progress_callback and plan_rounds > 0:
            progress_callback(make_log(f"🔄 **Re-plan round {plan_rounds + 1}...**"))

        user_proxy, planner, executor, verifier = _create_agents(self._llm_config)

        group_chat = GroupChat(
            agents=[user_proxy, planner, executor, verifier],
            messages=[],
            max_round=30,
            speaker_selection_method=_make_speaker_hook(
                self._cancel_event, self._progress_cb,
            ),
            role_for_select_speaker_messages="user",
        )
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=self._llm_config,
        )

        initial_msg = (
            f"## 用户目标\n{user_input}\n\n"
            f"对话历史摘要:\n{chat_history_text}\n\n"
        )

        if plan_rounds > 0 and previous_plan:
            initial_msg += (
                f"## 上一轮计划\n{previous_plan}\n\n"
                f"## 之前的结果\n{previous_result}\n\n"
                "请 verifier 反馈不完整的原因，planner 制定补充计划。"
            )
        else:
            initial_msg += "请 planner 先分析目标并制定执行计划。"

        try:
            user_proxy.initiate_chat(manager, message=initial_msg)
        except Exception as exc:
            logger.error("[AutoGenOrch] GroupChat error: %s", exc, exc_info=True)
            if progress_callback:
                progress_callback(make_log(f"❌ AutoGen error: {exc}"))
            return {"answer": f"Error: {exc}", "plan": "", "plan_rounds": plan_rounds}

        if cancel_event and cancel_event.is_set():
            return {"answer": "⏹ 已停止", "plan": "", "plan_rounds": plan_rounds}

        messages = group_chat.messages
        answer = extract_answer(messages)
        plan_text = extract_plan(messages)

        return {
            "answer": answer,
            "plan": plan_text,
            "plan_rounds": plan_rounds + 1,
        }


# ============================================================================
# Internal helpers
# ============================================================================

def _build_llm_config() -> dict:
    """Convert project Settings to AG2 config_list format."""
    cfg = settings.get_llm_config()
    provider = cfg["provider"]
    if provider == "ollama":
        return {
            "config_list": [{
                "model": cfg["model"],
                "base_url": f"{settings.OLLAMA_BASE_URL}/v1",
                "api_key": "ollama",
            }],
            "temperature": cfg["temperature"],
        }
    else:
        return {
            "config_list": [{
                "model": cfg["model"],
                "base_url": settings.OPENAI_BASE_URL,
                "api_key": settings.OPENAI_API_KEY,
            }],
            "temperature": cfg["temperature"],
        }


def _create_agents(
    llm_config: dict,
) -> tuple[UserProxyAgent, AssistantAgent, AssistantAgent, AssistantAgent]:
    """Create the four specialized agents for the GroupChat.

    Returns:
        (user_proxy, planner, executor, verifier)
    """
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: (
            x.get("content", "").rstrip().endswith("TERMINATE")
        ),
        code_execution_config=False,
        description="Tool executor — runs functions called by other agents",
    )

    planner = AssistantAgent(
        name="planner",
        system_message=_PLANNER_SYSTEM_MSG,
        llm_config=llm_config,
        description="Task planner — creates step-by-step plans from user goals",
    )
    executor = AssistantAgent(
        name="executor",
        system_message=_EXECUTOR_SYSTEM_MSG,
        llm_config=llm_config,
        description="Task executor — calls tools to carry out steps",
    )
    verifier = AssistantAgent(
        name="verifier",
        system_message=_VERIFIER_SYSTEM_MSG,
        llm_config=llm_config,
        description="Verifier — confirms goal achievement and produces final summary",
    )

    # Register all LangChain tools with executor (declaration) and user_proxy (execution)
    for t in list_tools():
        executor.register_for_llm(
            name=t.name,
            description=t.description,
        )(t.func)
        user_proxy.register_for_execution(name=t.name)(t.func)

    return user_proxy, planner, executor, verifier


class _SpeakerState:
    """Per-invocation mutable state for speaker selection."""

    def __init__(self) -> None:
        self.executor_rounds: int = 0


def _make_speaker_hook(
    cancel_event: threading.Event | None,
    progress_cb: Callable[[dict[str, Any]], None] | None,
) -> Callable:
    """Create a deterministic state-machine speaker selection function.

    Flow: planner → executor → (user_proxy ↔ executor)* → verifier

    AutoGen custom speaker selection functions must return:
      - An Agent object → that agent speaks next
      - None → terminate the conversation
      - "auto" → delegate to LLM-based selection

    Also bridges AutoGen events to the project's progress_callback protocol.
    """
    _ICONS = {
        "planner": "📋",
        "executor": "🔧",
        "verifier": "🔍",
        "user_proxy": "⚙️",
    }

    _state = _SpeakerState()

    def _agent_by_name(name: str, groupchat: Any) -> Any | None:
        """Look up an agent by name from the groupchat's agents list."""
        for a in groupchat.agents:
            if a.name == name:
                return a
        return None

    def _hook(last_speaker: Any, groupchat: Any) -> Any | None:
        # 1) Cancel check
        if cancel_event and cancel_event.is_set():
            return None

        # 2) Progress events from the last message
        messages = groupchat.messages
        if messages and progress_cb:
            last_msg = messages[-1]
            name = last_msg.get("name", "")
            content = last_msg.get("content", "")

            if name and content:
                icon = _ICONS.get(name, "💬")
                preview = content[:300].replace("\n", " ")
                progress_cb(make_log(f"{icon} **{name}**: {preview}"))

                # Emit tool events for function calls
                tool_calls = last_msg.get("tool_calls", [])
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        t_name = tc.get("function", {}).get("name", "unknown")
                        t_args = tc.get("function", {}).get("arguments", "")
                        progress_cb(make_tool(t_name, t_args))

        # 3) Deterministic state machine for speaker selection
        if not messages:
            return _agent_by_name("planner", groupchat)

        last_msg = messages[-1]
        name = last_msg.get("name", "")
        content = last_msg.get("content", "")
        tool_calls = last_msg.get("tool_calls", [])

        # First speaker: always planner
        if name in ("chat_manager", "", "user_proxy") and len(messages) <= 2:
            return _agent_by_name("planner", groupchat)

        # Planner → Executor
        if name == "planner":
            _state.executor_rounds = 0
            return _agent_by_name("executor", groupchat)

        # Executor with tool_call → UserProxy to execute it
        if name == "executor" and tool_calls:
            return _agent_by_name("user_proxy", groupchat)

        # UserProxy just executed a tool → Executor to process result
        if name == "user_proxy":
            _state.executor_rounds += 1
            # Safety: if executor keeps calling tools, eventually go to verifier
            if _state.executor_rounds > 10:
                _state.executor_rounds = 0
                return _agent_by_name("verifier", groupchat)
            return _agent_by_name("executor", groupchat)

        # Executor without tool_call (reporting) → Verifier to check completion
        if name == "executor" and not tool_calls:
            return _agent_by_name("verifier", groupchat)

        # Verifier → check for TERMINATE or re-plan
        if name == "verifier":
            if "TERMINATE" in content:
                # Return None to let user_proxy see TERMINATE and end conversation
                return None
            # Verifier says not done → back to planner for re-plan
            return _agent_by_name("planner", groupchat)

        # Fallback: let LLM decide (should rarely be reached)
        return "auto"

    return _hook