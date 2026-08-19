"""GraphOrchestrator — pure LangGraph Plan-and-Execute orchestration.

Architecture:

  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
  │ classify  │───→│ react_fast   │───→│ adaptive_chk │───→│   END    │
  └────┬─────┘    └──────────────┘    └──────┬───────┘    └──────────┘
       │ complex                              │ budget exceeded
       ▼                                      ▼
  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
  │   plan    │───→│ execute_step │───→│  verify      │───→│   END    │
  └──────────┘    └──────┬───────┘    └──────┬───────┘    └──────────┘
                          │ more steps        │ not complete
                          ▼                 ▼
                     check_steps         re-plan → plan

Adaptive features:
  - Tier 1: Learned classifications (feedback from past runs)
  - Tier 2: Keyword + heuristic scoring
  - Tier 3: LLM classification (fallback)
  - Runtime: Budget monitoring auto-upgrades/downgrades paths
    * Simple path exceeding iterations/tool_calls/time → auto-upgrade to complex
    * Complex path finishing in <=1 step and <10s → learned as simple

Pure LangGraph provides: declarative routing, checkpointing,
debug visualization, and typed state management.

Each plan step is executed via the LangChain ReAct tool-calling
executor, giving full tool access (file, web, code, browser, etc.).
"""

from __future__ import annotations

import logging
import threading
import time
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
from agent.tools.registry import list_tools, ToolRouter

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
    graph_start_time: float
    fast_iterations: int
    fast_tool_calls: int
    fast_elapsed: float
    fast_hit_limit: bool
    complex_steps_executed: int
    complex_elapsed: float


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
        self._scene_executors: dict[str, AgentExecutor] = {}
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

    def _get_or_build_executor(self, scene: str | None = None) -> AgentExecutor:
        """Get or create a scene-specific tool executor.

        Main executor (all tools) is kept as fallback.
        Scene executors are cached for reuse.
        """
        if not scene:
            return self._executor
        if scene not in self._scene_executors:
            self._scene_executors[scene] = build_react_executor(
                self._llm,
                max_iterations=settings.MAX_ITERATIONS,
                max_execution_time=settings.MAX_EXECUTION_TIME_SEC,
                verbose=True,
                scene=scene,
            )
        return self._scene_executors[scene]

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)

        graph.add_node("classify", self._node_classify)
        graph.add_node("react_fast", self._node_react_fast)
        graph.add_node("adaptive_check", self._node_adaptive_check)
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

        graph.add_edge("react_fast", "adaptive_check")

        graph.add_conditional_edges(
            "adaptive_check",
            self._route_after_adaptive,
            {
                "end": END,
                "upgrade": "plan",
            },
        )

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
            "graph_start_time": time.time(),
            "fast_iterations": 0,
            "fast_tool_calls": 0,
            "fast_elapsed": 0.0,
            "fast_hit_limit": False,
            "complex_steps_executed": 0,
            "complex_elapsed": 0.0,
        }

        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as exc:
            logger.error("[GraphOrch] Graph error: %s", exc, exc_info=True)
            if progress_callback:
                progress_callback(make_log(f"❌ Graph error: {exc}"))
            return f"Error: {exc}"

        self._record_complex_downgrade_feedback(final_state)

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
        scene: str | None = None,
    ) -> dict[str, Any]:
        """Run a single ReAct tool-calling step with cancellation support.

        Args:
            input_text: Step instruction or user query
            chat_history_messages: Conversation history
            scene: Optional scene hint for tool routing (auto-detected if not set)

        Returns dict with: result, status, iterations, time, hit_limit,
        tool_calls, llm_calls.
        """
        from agent.tools.registry import ToolRouter

        if not scene and input_text:
            scene, _ = ToolRouter().smart_route(input_text)

        executor = self._get_or_build_executor(scene)

        tracker = _BaseToolEventTracker(self._progress_cb)
        streaming_handler = StreamingCallbackHandler(self._progress_cb)

        result = run_react_step(
            executor,
            input_text,
            chat_history=chat_history_messages,
            progress_callback=self._progress_cb,
            tracker=tracker,
            cancel_event=self._cancel_event,
            extra_callbacks=[streaming_handler],
        )

        if self._cancel_event.is_set():
            return {
                "result": "⏹ 已停止",
                "status": "cancelled",
                "iterations": 0,
                "time": 0.0,
                "hit_limit": False,
                "tool_calls": 0,
                "llm_calls": 0,
            }

        enriched = dict(result)
        enriched["tool_calls"] = tracker._call_counter
        enriched["llm_calls"] = tracker._llm_call_count
        return enriched

    def _record_complex_downgrade_feedback(self, final_state: AgentState) -> None:
        """After graph completion: learn if complex queries could have been simple.

        If a query was classified as "complex" but finished in <=1 plan step
        and under 10 seconds, it would have been better handled by the simple
        path. Record this feedback so the classifier learns for next time.
        """
        from agent.utils.classifier import record_feedback

        classification = final_state.get("classification", "")
        if classification != "complex":
            return

        graph_start = final_state.get("graph_start_time", time.time())
        total_elapsed = time.time() - graph_start
        steps_executed = final_state.get("complex_steps_executed", 0)

        user_input = final_state.get("user_input", "")
        record_feedback(
            user_input=user_input,
            original_label="complex",
            steps_executed=steps_executed,
            elapsed=total_elapsed,
        )

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _node_classify(self, state: AgentState) -> dict[str, Any]:
        """Classify query complexity — simple (fast path) or complex (plan+execute)."""
        from agent.utils.classifier import classify_query

        if self._cancel_event and self._cancel_event.is_set():
            return {"result": "⏹ 已停止", "is_done": True}

        # if self._progress_cb:
        #     self._progress_cb(make_log("🤔 **Analyzing query complexity...**"))

        classification = classify_query(self._llm, state["user_input"])

        # if self._progress_cb:
        #     self._progress_cb(make_log(f"⚡ **Route: {classification}**"))

        return {"classification": classification}

    def _node_react_fast(self, state: AgentState) -> dict[str, Any]:
        """Execute a simple query via the LangChain ReAct fast path with budget tracking."""
        if self._cancel_event and self._cancel_event.is_set():
            return {"result": "⏹ 已停止", "is_done": True}

        # if self._progress_cb:
        #     self._progress_cb(make_log("⚡ **Running fast ReAct path...**"))

        exec_result = self._execute_step_with_tools(
            state["user_input"],
            state.get("chat_history_messages", []),
        )

        result_text = exec_result.get("result", "")
        if exec_result.get("status") == "cancelled":
            result_text = "⏹ 已停止"

        return {
            "result": result_text,
            "is_done": True,
            "fast_iterations": exec_result.get("iterations", 0),
            "fast_tool_calls": exec_result.get("tool_calls", 0),
            "fast_elapsed": exec_result.get("time", 0.0),
            "fast_hit_limit": exec_result.get("hit_limit", False),
        }

    def _node_adaptive_check(self, state: AgentState) -> dict[str, Any]:
        """After fast path: check if budget was exceeded and upgrade to complex if needed.

        This is the core of the adaptive routing. If a query was classified
        as "simple" but actually used too many iterations, tool calls, or
        time, we reclassify it as "complex" and route to the plan+execute
        path. The feedback is recorded so future identical queries will
        be correctly classified from the start.
        """
        from agent.utils.classifier import record_feedback, get_budget

        if self._cancel_event and self._cancel_event.is_set():
            return {"result": "⏹ 已停止", "is_done": True}

        classification = state.get("classification", "simple")
        iterations = state.get("fast_iterations", 0)
        tool_calls = state.get("fast_tool_calls", 0)
        elapsed = state.get("fast_elapsed", 0.0)
        hit_limit = state.get("fast_hit_limit", False)

        if classification != "simple":
            return {}

        budget = get_budget()
        exceeded = budget.exceeded_simple_budget(iterations, tool_calls, elapsed)

        if hit_limit or exceeded:
            if self._progress_cb:
                reason_parts = []
                if hit_limit:
                    reason_parts.append("hit iteration/time limit")
                if iterations >= budget.MAX_SIMPLE_ITERATIONS:
                    reason_parts.append(f"iter={iterations}>={budget.MAX_SIMPLE_ITERATIONS}")
                if tool_calls >= budget.MAX_SIMPLE_TOOL_CALLS:
                    reason_parts.append(f"tools={tool_calls}>={budget.MAX_SIMPLE_TOOL_CALLS}")
                if elapsed >= budget.MAX_SIMPLE_TIME_SEC:
                    reason_parts.append(f"time={elapsed:.1f}s>={budget.MAX_SIMPLE_TIME_SEC}s")
                reason = ", ".join(reason_parts) if reason_parts else "budget exceeded"
                self._progress_cb(make_log(
                    f"⚠️ **Fast path budget exceeded** ({reason}) → upgrading to complex path"
                ))

            user_input = state.get("user_input", "")
            record_feedback(
                user_input=user_input,
                original_label="simple",
                iterations=iterations,
                tool_calls=tool_calls,
                elapsed=elapsed,
            )

            return {
                "classification": "complex",
                "result": state.get("result", ""),
                "is_done": False,
            }

        logger.info(
            "[Adaptive] Fast path OK: iter=%d tools=%d time=%.1fss (budget: max_iter=%d max_tools=%d max_time=%.1fs)",
            iterations, tool_calls, elapsed,
            budget.MAX_SIMPLE_ITERATIONS, budget.MAX_SIMPLE_TOOL_CALLS, budget.MAX_SIMPLE_TIME_SEC,
        )
        return {}

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

IMPORTANT: You MUST respond in Chinese (中文). All step descriptions must be in Chinese.

Respond with exactly one step per line, numbered:
1. [步骤描述 — 具体说明要做什么]
2. [步骤描述]
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

IMPORTANT: You MUST respond in Chinese (中文). Think and respond in Chinese.

After completing this step, provide a brief summary of what you did and what you found."""

        result = self._execute_step_with_tools(step_prompt, chat_history)

        result_text = result.get("result", "") if isinstance(result, dict) else str(result)

        results = list(state.get("results", []))
        while len(results) <= current_step:
            results.append("")
        results[current_step] = result_text

        complex_steps = state.get("complex_steps_executed", 0) + 1

        if self._progress_cb:
            self._progress_cb(
                make_log(f"✅ Step {current_step + 1} completed: {result_text[:200]}")
            )

        return {
            "results": results,
            "current_step": current_step + 1,
            "complex_steps_executed": complex_steps,
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

IMPORTANT: You MUST respond in Chinese (中文).

Answer exactly 'yes' or 'no', then provide a brief explanation in Chinese.
If yes, also provide a concise final answer to the user in Chinese.
If no, explain what is still missing in Chinese."""

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

IMPORTANT: You MUST respond in Chinese (中文). All step descriptions must be in Chinese.

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
    def _route_after_adaptive(state: AgentState) -> str:
        """After adaptive check: either end (fast path OK) or upgrade to complex."""
        if state.get("classification") == "complex":
            return "upgrade"
        return "end"

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