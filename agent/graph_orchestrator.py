"""GraphOrchestrator — pure LangGraph Plan-and-Execute orchestration.

Architecture:

  ┌──────────┐    ┌──────────────┐    ┌──────────┐
  │ classify  │───→│ react_fast   │───→│   END    │
  └────┬─────┘    └──────────────┘    └──────────┘
       │ complex
       ▼
  ┌──────────┐    ┌──────────────┐    ┌──────────┐
  │   plan    │───→│ execute_step │───→│  verify  │
  └──────────┘    └──────┬───────┘    └────┬─────┘
                          │ more steps       │ not complete
                          ▼                 ▼
                     check_steps       re-plan → plan

Pure LangGraph provides: declarative routing, checkpointing,
debug visualization, and typed state management.

Each plan step is executed via the LangChain ReAct tool-calling
executor, giving full tool access (file, web, code, browser, etc.).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.config import settings
from agent.llm.factory import create_llm
from agent.progress import make_log, make_streaming_token
from agent.runner import (
    StreamingCallbackHandler,
    _BaseToolEventTracker,
    build_react_executor,
    run_react_step,
)
from agent.tools.registry import list_tools

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """Shared state across all LangGraph nodes."""

    user_input: str
    short_term_text: str
    chat_history_messages: list
    classification: str
    plan: list[str]
    current_step: int
    results: list[str]
    result: str
    is_done: bool
    plan_rounds: int


class GraphOrchestrator:
    """Pure LangGraph Plan-and-Execute orchestrator.

    Combines LangGraph's deterministic routing and state management
    with LangChain ReAct tool-calling for execution.

    The graph:
      1. classify → simple → react_fast → END
      2. classify → complex → plan → execute_step → verify → END/re-plan
    """

    def __init__(self) -> None:
        self._cancel_event = threading.Event()
        self._progress_cb: Callable[[dict[str, Any]], None] | None = None
        self._llm_provider = create_llm()
        self._llm = self._llm_provider.get_model()
        self._executor = build_react_executor(
            self._llm,
            max_iterations=settings.MAX_ITERATIONS,
            max_execution_time=settings.MAX_EXECUTION_TIME_SEC,
            verbose=True,
        )
        self._last_token_count = 0
        try:
            self._graph = self._build_graph()
        except Exception as exc:
            logger.error("[GraphOrch] Graph build failed: %s", exc, exc_info=True)
            self._graph = None
        logger.info(
            "[GraphOrch] LangGraph ready | LLM: %s",
            self._llm_provider.model_name,
        )

    def request_stop(self) -> None:
        """Signal cancellation to all LangGraph nodes."""
        self._cancel_event.set()

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)

        graph.add_node("classify", self._node_classify)
        graph.add_node("react_fast", self._node_react_fast)
        graph.add_node("plan", self._node_plan)
        graph.add_node("execute_step", self._node_execute_step)
        graph.add_node("check_steps", self._node_check_steps)
        graph.add_node("verify", self._node_verify)
        graph.add_node("re_plan", self._node_re_plan)

        graph.add_edge(START, "classify")

        graph.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {
                "simple": "react_fast",
                "complex": "plan",
            },
        )

        graph.add_edge("react_fast", END)

        graph.add_edge("plan", "execute_step")

        graph.add_edge("execute_step", "check_steps")

        graph.add_conditional_edges(
            "check_steps",
            self._route_after_check,
            {
                "continue": "execute_step",
                "verify": "verify",
            },
        )

        graph.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {
                "complete": END,
                "incomplete": "re_plan",
            },
        )

        graph.add_edge("re_plan", "execute_step")

        return graph.compile()

    def run(
        self,
        user_input: str,
        short_term_text: str,
        chat_history_messages: list,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        self._cancel_event = cancel_event or threading.Event()
        self._progress_cb = progress_callback

        if self._graph is None:
            return "[Error] LangGraph not available. Please check the logs for build errors."

        initial_state: AgentState = {
            "user_input": user_input,
            "short_term_text": short_term_text,
            "chat_history_messages": chat_history_messages,
            "classification": "",
            "plan": [],
            "current_step": 0,
            "results": [],
            "result": "",
            "is_done": False,
            "plan_rounds": 0,
        }

        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as exc:
            logger.error("[GraphOrch] Graph error: %s", exc, exc_info=True)
            if progress_callback:
                progress_callback(make_log(f"❌ Graph error: {exc}"))
            return f"Error: {exc}"

        return final_state.get("result", "已完成，但未产生文本输出。")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_chat_history(messages: list) -> str:
        """Convert LangChain message list to readable transcript."""
        if not messages:
            return ""
        parts: list[str] = []
        for msg in messages[-10:]:
            role = getattr(msg, "type", "") or ""
            content = getattr(msg, "content", "") or ""
            if role == "human":
                parts.append(f"User: {content}")
            elif role == "ai":
                parts.append(f"Assistant: {content}")
            elif role == "system":
                parts.append(f"System: {content[:200]}")
        return "\n".join(parts)

    @staticmethod
    def _build_tools_description() -> str:
        """Dynamically build tool description from the registry."""
        tools = list_tools()
        if not tools:
            return "(no tools available)"
        lines = ["Available tools:"]
        for t in tools:
            name = getattr(t, "name", "unknown")
            desc = (getattr(t, "description", "") or "")[:120]
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def _execute_step_with_tools(
        self,
        input_text: str,
        chat_history_messages: list,
    ) -> str:
        """Run a single ReAct tool-calling step with cancellation support."""
        tracker = _BaseToolEventTracker(self._progress_cb)
        streaming_handler = StreamingCallbackHandler(self._progress_cb)

        result = run_react_step(
            self._executor,
            input_text,
            chat_history=chat_history_messages,
            progress_callback=self._progress_cb,
            tracker=tracker,
            cancel_event=self._cancel_event,
            extra_callbacks=[streaming_handler],
        )

        if self._cancel_event.is_set():
            return "⏹ 已停止"
        if result["status"] == "completed":
            return result["result"]
        if result["status"] == "cancelled":
            return "⏹ 已停止"
        return f"[Error] {result.get('error', 'Execution failed')}"

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _node_classify(self, state: AgentState) -> dict[str, Any]:
        """Classify query complexity — simple (fast path) or complex (plan+execute)."""
        from agent.utils.classifier import classify_query

        if self._cancel_event and self._cancel_event.is_set():
            return {"result": "⏹ 已停止", "is_done": True}

        if self._progress_cb:
            self._progress_cb(make_log("🤔 **Analyzing query complexity...**"))

        classification = classify_query(self._llm, state["user_input"])

        if self._progress_cb:
            icon = "⚡" if classification == "simple" else "🧩"
            self._progress_cb(
                make_log(
                    f"{icon} **Route: {classification}** — "
                    f"{'fast path' if classification == 'simple' else 'multi-step Plan-and-Execute'}"
                )
            )
            self._progress_cb(make_streaming_token(f" → {classification}"))

        return {"classification": classification}

    def _node_react_fast(self, state: AgentState) -> dict[str, Any]:
        """Execute a simple query via the LangChain ReAct fast path."""
        if self._cancel_event and self._cancel_event.is_set():
            return {"result": "⏹ 已停止", "is_done": True}

        if self._progress_cb:
            self._progress_cb(make_log("⚡ **Running fast ReAct path...**"))

        result = self._execute_step_with_tools(
            state["user_input"],
            state.get("chat_history_messages", []),
        )

        return {"result": result, "is_done": True}

    def _node_plan(self, state: AgentState) -> dict[str, Any]:
        """Create a step-by-step execution plan via LLM."""
        if self._cancel_event and self._cancel_event.is_set():
            return {"result": "⏹ 已停止", "is_done": True}

        plan_rounds = state.get("plan_rounds", 0)

        if self._progress_cb:
            if plan_rounds == 0:
                self._progress_cb(make_log("📋 **Creating execution plan...**"))
            else:
                self._progress_cb(make_log(f"🔄 **Re-planning (round {plan_rounds + 1})...**"))

        user_input = state.get("user_input", "")
        previous_plan = state.get("plan", [])
        previous_results = state.get("results", [])
        chat_history_text = self._format_chat_history(state.get("chat_history_messages", []))
        tools_desc = self._build_tools_description()

        context_parts = [f"User goal: {user_input}"]

        if chat_history_text:
            context_parts.append(f"Conversation history:\n{chat_history_text}")

        if previous_plan:
            context_parts.append("Previous plan:")
            for i, step in enumerate(previous_plan):
                status = previous_results[i] if i < len(previous_results) else "(not executed)"
                context_parts.append(f"  Step {i + 1}: {step} → {status[:200]}")

        context = "\n".join(context_parts)

        prompt = f"""{context}

{tools_desc}

Create a clear, step-by-step execution plan to achieve the user's goal.
Break it into 2-5 concrete, actionable steps. Each step should be a single tool call or a simple action.

Respond with exactly one step per line, numbered:
1. [Step description — be specific about what to do]
2. [Step description]
...

Only output the numbered steps, nothing else."""

        try:
            plan_text = self._stream_llm_response(prompt)
        except Exception as exc:
            logger.error("[GraphOrch] Plan generation failed: %s", exc)
            if self._progress_cb:
                self._progress_cb(make_log("⚠️ Plan generation failed, using fallback plan"))
            plan_text = f"1. Analyze the request: {user_input}\n2. Execute the required action\n3. Summarize results"

        plan = self._parse_plan(plan_text)

        if not plan:
            plan = [
                f"Analyze request: {user_input}",
                "Execute the required action",
                "Summarize the results",
            ]

        if self._progress_cb:
            self._progress_cb(make_log(f"📋 **Plan created: {len(plan)} steps**"))
            for i, step in enumerate(plan):
                self._progress_cb(make_log(f"  {i + 1}. {step[:120]}"))

        return {
            "plan": plan,
            "current_step": 0,
            "results": [],
        }

    def _node_execute_step(self, state: AgentState) -> dict[str, Any]:
        """Execute the current plan step using the ReAct executor."""
        if self._cancel_event and self._cancel_event.is_set():
            return {"result": "⏹ 已停止", "is_done": True}

        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])

        if current_step >= len(plan):
            return {}

        step_description = plan[current_step]

        if self._progress_cb:
            self._progress_cb(
                make_log(
                    f"▶️ **Step {current_step + 1}/{len(plan)}**: {step_description[:120]}"
                )
            )

        user_input = state.get("user_input", "")
        short_term_text = state.get("short_term_text", "")
        chat_history = state.get("chat_history_messages", [])

        context_sections = []
        if short_term_text:
            context_sections.append(f"Previous context:\n{short_term_text[:800]}")

        step_prompt = f"""Execute this step to achieve the user's goal.

User's original goal: {user_input}
Current step: {step_description}
{chr(10).join(context_sections)}

Perform the required action using available tools. If you need to read files,
search the web, or run code, use the appropriate tools now.

After completing this step, provide a brief summary of what you did and what you found."""

        result = self._execute_step_with_tools(step_prompt, chat_history)

        results = list(state.get("results", []))
        while len(results) <= current_step:
            results.append("")
        results[current_step] = result

        if self._progress_cb:
            self._progress_cb(
                make_log(f"✅ Step {current_step + 1} completed: {result[:200]}")
            )

        return {
            "results": results,
            "current_step": current_step + 1,
        }

    def _node_check_steps(self, state: AgentState) -> dict[str, Any]:
        """Check if all plan steps are completed."""
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)

        if current_step >= len(plan):
            if self._progress_cb:
                self._progress_cb(make_log("📋 **All plan steps completed**"))
            return {}

        return {}

    def _node_verify(self, state: AgentState) -> dict[str, Any]:
        """Verify whether the goal is achieved and produce final result."""
        user_input = state.get("user_input", "")
        results = state.get("results", [])
        plan = state.get("plan", [])
        plan_rounds = state.get("plan_rounds", 0)
        chat_history_text = self._format_chat_history(state.get("chat_history_messages", []))

        if self._cancel_event and self._cancel_event.is_set():
            return {"result": "⏹ 已停止", "is_done": True}

        if not results:
            return {"is_done": True}

        if plan_rounds >= 3:
            logger.info("[GraphOrch] Max re-plan rounds (%d) reached, ending", plan_rounds)
            return {"is_done": True}

        if self._progress_cb:
            self._progress_cb(make_log("🔍 **Verifying goal achievement...**"))

        summary_parts = ["## Execution Results\n"]
        for i, (step, result) in enumerate(zip(plan, results)):
            summary_parts.append(f"**Step {i + 1}**: {step}")
            summary_parts.append(f"Result: {result[:300]}")
            summary_parts.append("")

        full_summary = "\n".join(summary_parts)

        context_block = ""
        if chat_history_text:
            context_block = f"\n\nConversation context:\n{chat_history_text[:600]}"

        prompt = f"""{full_summary}{context_block}

Original user goal: {user_input}

Based on the execution results above, is the user's goal fully achieved?

Answer exactly 'yes' or 'no', then provide a brief explanation.
If yes, also provide a concise final answer to the user.
If no, explain what is still missing."""

        try:
            verification = self._stream_llm_response(prompt)

            if self._progress_cb:
                self._progress_cb(make_log(f"🔍 Verification: {verification[:200]}"))

            is_complete = verification.strip().lower().startswith("yes")

            if is_complete:
                final_answer = self._extract_final_answer(verification, user_input, results)
                if self._progress_cb:
                    self._progress_cb(make_log("🎯 **Goal achieved!**"))
                return {
                    "result": final_answer,
                    "is_done": True,
                }

            if plan_rounds >= 2:
                if self._progress_cb:
                    self._progress_cb(
                        make_log("⚠️ Max re-plans reached, accepting current results")
                    )
                final_answer = self._extract_final_answer(verification, user_input, results)
                return {
                    "result": final_answer,
                    "is_done": True,
                }

            if self._progress_cb:
                self._progress_cb(make_log("🔄 **Goal not fully achieved, re-planning...**"))

            return {
                "result": verification,
                "is_done": False,
            }

        except Exception as exc:
            logger.warning("[GraphOrch] Verification failed: %s", exc)
            if self._progress_cb:
                self._progress_cb(make_log("⚠️ Verification failed, proceeding with results"))
            final_answer = self._extract_final_answer(
                "Verification failed", user_input, results
            )
            return {
                "result": final_answer,
                "is_done": True,
            }

    def _node_re_plan(self, state: AgentState) -> dict[str, Any]:
        """Re-plan remaining steps after verification found gaps."""
        if self._cancel_event and self._cancel_event.is_set():
            return {"result": "⏹ 已停止", "is_done": True}

        plan_rounds = state.get("plan_rounds", 0)
        new_plan_rounds = plan_rounds + 1

        if self._progress_cb:
            self._progress_cb(
                make_log(f"📋 **Re-planning (round {new_plan_rounds})...**")
            )

        user_input = state.get("user_input", "")
        results = state.get("results", [])
        plan = state.get("plan", [])
        verification_feedback = state.get("result", "")
        chat_history_text = self._format_chat_history(state.get("chat_history_messages", []))

        prompt = f"""Original goal: {user_input}

Previous plan:
{chr(10).join(f'{i + 1}. {s}' for i, s in enumerate(plan))}

Previous results:
{chr(10).join(f'Step {i + 1}: {r[:200]}' for i, r in enumerate(results))}

Verification feedback: {verification_feedback[:500]}
{chr(10).join('Conversation context:' if chat_history_text else '')}
{chat_history_text[:400] if chat_history_text else ''}

Create a NEW execution plan to address the remaining work.
Only list steps that still need to be done.
Respond with numbered steps only."""

        try:
            plan_text = self._stream_llm_response(prompt)
            new_plan = self._parse_plan(plan_text)
        except Exception as exc:
            logger.error("[GraphOrch] Re-plan failed: %s", exc)
            new_plan = [f"Complete remaining work for: {user_input}"]

        if not new_plan:
            new_plan = [f"Complete remaining work for: {user_input}"]

        if self._progress_cb:
            self._progress_cb(
                make_log(f"📋 **New plan: {len(new_plan)} steps**")
            )
            for i, step in enumerate(new_plan):
                self._progress_cb(make_log(f"  {i + 1}. {step[:120]}"))

        return {
            "plan": new_plan,
            "current_step": 0,
            "results": [],
            "plan_rounds": new_plan_rounds,
        }

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _stream_llm_response(self, prompt: str) -> str:
        """Stream LLM response, emitting tokens via progress_callback."""
        collected: list[str] = []
        token_count = 0
        try:
            for chunk in self._llm.stream(prompt):
                if self._cancel_event and self._cancel_event.is_set():
                    break
                token = chunk.content
                if token:
                    collected.append(token)
                    token_count += 1
                    if self._progress_cb:
                        self._progress_cb(make_streaming_token(token))
        except Exception:
            pass
        self._last_token_count = token_count
        return "".join(collected)

    @staticmethod
    def _parse_plan(plan_text: str) -> list[str]:
        """Parse a numbered plan from LLM output.

        Expected format: lines starting with '1.', '2.', etc.
        """
        steps: list[str] = []
        for line in plan_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit():
                for sep in (".", "、", ")", ":", "："):
                    if sep in line:
                        idx = line.index(sep)
                        step_text = line[idx + 1:].strip()
                        if step_text:
                            steps.append(step_text)
                        break
                else:
                    steps.append(line)
        return steps

    @staticmethod
    def _extract_final_answer(
        verification: str, user_input: str, results: list[str]
    ) -> str:
        """Extract a clean final answer from verification output."""
        lines = verification.strip().split("\n")

        answer_lines: list[str] = []
        found_yes = False

        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("yes") and not found_yes:
                found_yes = True
                remainder = stripped[3:].strip(".,;:：,。； ")
                if remainder and len(remainder) > 10:
                    answer_lines.append(remainder)
                continue
            if found_yes and stripped:
                answer_lines.append(stripped)

        if answer_lines:
            return "\n".join(answer_lines)

        if results:
            return "\n\n".join(
                f"**Result {i + 1}**: {r[:500]}" for i, r in enumerate(results)
            )

        return f"Task completed for: {user_input}"

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    @staticmethod
    def _route_after_classify(state: AgentState) -> str:
        return state.get("classification", "complex")

    @staticmethod
    def _route_after_check(state: AgentState) -> str:
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        if current_step < len(plan):
            return "continue"
        return "verify"

    @staticmethod
    def _route_after_verify(state: AgentState) -> str:
        return "complete" if state.get("is_done") else "incomplete"