"""Autonomous graph — LangGraph-native autonomous execution engine.

Replaces the old manual threading/event-loop (autonomous_loop.py)
with a proper LangGraph StateGraph that manages goal lifecycle,
state transitions, and re-planning.

Architecture (LangGraph StateGraph):

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ perceive │───→│  select  │───→│   act    │───→│ reflect  │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
       ▲                                               │
       │  continue                                     ▼
       │  ┌──────────┐    complete ┌──────┐            │
       └──│   check   │────────────→│ END  │◄───────────┘
          └──────────┘              └──────┘
               │ ▲
               │ │ replan
               └─┘

Key improvements over the old implementation:
  1. StateGraph with TypedDict + Annotated reducers for clean state flow
  2. Goals and events flow through graph state (not side-effect mutation)
  3. Command(goto=...) for clean loop control
  4. Checkpointer for state persistence across cycles
  5. Config-based runtime context (no threading.local())
  6. Proper reducer-based goal updates
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

logger = logging.getLogger(__name__)

_DEFAULT_GOALS_DIR = "./agent_goals"


# ---------------------------------------------------------------------------
# Goal data model
# ---------------------------------------------------------------------------


@dataclass
class Goal:
    """A persistent goal that the agent pursues autonomously."""

    goal_id: str = ""
    description: str = ""
    priority: int = 0
    progress: float = 0.0
    status: str = "pending"
    created_at: float = 0.0
    updated_at: float = 0.0
    deadline: float | None = None
    context: dict[str, Any] = field(default_factory=dict)
    steps_completed: int = 0
    steps_total: int = 0
    last_error: str = ""

    def __post_init__(self) -> None:
        if not self.goal_id:
            self.goal_id = hashlib.md5(
                self.description.encode("utf-8")
            ).hexdigest()[:12]
        now = time.time()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    @property
    def is_complete(self) -> bool:
        return self.status == "completed"

    @property
    def is_blocked(self) -> bool:
        return self.status == "blocked"

    @property
    def is_expired(self) -> bool:
        if self.deadline is None:
            return False
        return time.time() > self.deadline

    def update_progress(self, completed: int, total: int) -> None:
        self.steps_completed = completed
        self.steps_total = total
        self.progress = completed / max(total, 1)
        self.updated_at = time.time()

    def mark_complete(self) -> None:
        self.status = "completed"
        self.progress = 1.0
        self.updated_at = time.time()

    def mark_blocked(self, error: str = "") -> None:
        self.status = "blocked"
        self.last_error = error
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "priority": self.priority,
            "progress": self.progress,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deadline": self.deadline,
            "context": self.context,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Goal:
        return cls(
            goal_id=data.get("goal_id", ""),
            description=data.get("description", ""),
            priority=data.get("priority", 0),
            progress=data.get("progress", 0.0),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            deadline=data.get("deadline"),
            context=data.get("context", {}),
            steps_completed=data.get("steps_completed", 0),
            steps_total=data.get("steps_total", 0),
            last_error=data.get("last_error", ""),
        )


# ---------------------------------------------------------------------------
# Reducers (LangGraph Annotated)
# ---------------------------------------------------------------------------


def _merge_goals(existing: list, updates: list) -> list:
    """Reducer: merge goal updates into the goals list by goal_id."""
    existing_ids = {g["goal_id"] for g in existing if isinstance(g, dict)}
    result = list(existing)
    for update in updates:
        if not isinstance(update, dict):
            continue
        gid = update.get("goal_id", "")
        if gid in existing_ids:
            for i, g in enumerate(result):
                if isinstance(g, dict) and g.get("goal_id") == gid:
                    result[i] = {**g, **update}
                    break
        else:
            result.append(update)
            if gid:
                existing_ids.add(gid)
    return result


def _append_event(existing: list, updates: list) -> list:
    """Reducer: append new events to the event log (bounded to 200)."""
    result = list(existing) + list(updates)
    if len(result) > 200:
        result = result[-200:]
    return result


# ---------------------------------------------------------------------------
# Autonomous state (LangGraph TypedDict with Annotated reducers)
# ---------------------------------------------------------------------------


class AutonomousState(TypedDict, total=False):
    goals: Annotated[list, _merge_goals]
    active_goal_id: str | None
    perception_data: dict[str, Any]
    cycle_count: int
    should_continue: bool
    events: Annotated[list, _append_event]
    max_cycles: int
    action_result: str
    reflection_insights: list[str]
    stop_requested: bool


def _make_initial_state(goals: list[Goal] | None = None) -> AutonomousState:
    """Create initial state for the autonomous graph."""
    return {
        "goals": [g.to_dict() for g in goals] if goals else [],
        "active_goal_id": None,
        "perception_data": {},
        "cycle_count": 0,
        "should_continue": True,
        "events": [],
        "max_cycles": 1000,
        "action_result": "",
        "reflection_insights": [],
        "stop_requested": False,
    }


# ---------------------------------------------------------------------------
# Graph node implementations
# ---------------------------------------------------------------------------


class AutonomousGraph:
    """LangGraph-based autonomous execution engine.

    Uses TypedDict state with Annotated reducers for goals and events.
    State flows through the graph — no side-effect mutation.

    Usage:
        engine = AutonomousGraph(action_callback=my_action)
        engine.add_goal(Goal(description="监控项目进度"))
        engine.run_cycle()   # runs one cycle through the LangGraph
    """

    def __init__(
        self,
        llm: Any = None,
        action_callback: Callable[[Goal], str] | None = None,
        reflection_engine: Any = None,
        cycle_interval: float = 5.0,
        max_cycles: int = 1000,
        checkpointer: Any | None = None,
    ) -> None:
        self._llm = llm
        self._action_callback = action_callback
        self._reflection_engine = reflection_engine
        self._cycle_interval = cycle_interval
        self._max_cycles = max_cycles

        self._goals: dict[str, Goal] = {}
        self._active_goal_id: str | None = None
        self._max_event_log = 200
        self._cycle_count = 0
        self._running = False

        if checkpointer is None:
            checkpointer = MemorySaver()
        self._checkpointer = checkpointer

        self.graph = self._build_graph()
        logger.info("[AutoGraph] LangGraph-based engine ready (TypedState + reducers)")

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(AutonomousState)

        graph.add_node("perceive", self._node_perceive)
        graph.add_node("select_goal", self._node_select_goal)
        graph.add_node("act", self._node_act)
        graph.add_node("reflect", self._node_reflect)
        graph.add_node("check", self._node_check)

        graph.add_edge(START, "perceive")
        graph.add_edge("perceive", "select_goal")
        graph.add_edge("select_goal", "act")
        graph.add_edge("act", "reflect")
        graph.add_edge("reflect", "check")

        return graph.compile(checkpointer=self._checkpointer)

    # ------------------------------------------------------------------
    # Node implementations (all operate on state, not self._goals)
    # ------------------------------------------------------------------

    @staticmethod
    def _node_perceive(state: AutonomousState) -> dict[str, Any]:
        """Perceive environment state, update goal statuses via reducer."""
        goals = state.get("goals", [])
        cycle_count = state.get("cycle_count", 0) + 1

        updated_goals: list[dict[str, Any]] = []
        new_events: list[dict[str, Any]] = []

        for g_data in goals:
            if not isinstance(g_data, dict):
                continue
            if g_data.get("status") not in ("pending", "in_progress", "blocked"):
                continue

            deadline = g_data.get("deadline")
            if deadline is not None and time.time() > deadline:
                updated = {**g_data, "status": "blocked", "last_error": "Deadline exceeded", "updated_at": time.time()}
                updated_goals.append(updated)
                new_events.append({
                    "type": "goal_expired",
                    "goal_id": g_data.get("goal_id", ""),
                    "timestamp": time.time(),
                    "cycle": cycle_count,
                })
            else:
                updated_goals.append(g_data)

        active_count = len([
            g for g in goals
            if isinstance(g, dict) and g.get("status") in ("pending", "in_progress", "blocked")
        ])

        return {
            "goals": updated_goals,
            "cycle_count": cycle_count,
            "perception_data": {
                "active_goals": active_count,
                "expired": len(new_events),
                "timestamp": time.time(),
            },
            "events": new_events,
        }

    @staticmethod
    def _node_select_goal(state: AutonomousState) -> dict[str, Any]:
        """Select the highest-priority active goal. Returns active_goal_id via state."""
        goals = state.get("goals", [])

        active_goals = sorted(
            [g for g in goals if isinstance(g, dict) and g.get("status") in ("pending", "in_progress", "blocked")],
            key=lambda g: -g.get("priority", 0),
        )

        if not active_goals:
            return {"active_goal_id": None, "should_continue": True}

        for g in active_goals:
            deadline = g.get("deadline")
            if deadline is not None and time.time() > deadline:
                continue
            return {
                "active_goal_id": g.get("goal_id", ""),
                "should_continue": True,
            }

        return {"active_goal_id": None, "should_continue": True}

    def _node_act(self, state: AutonomousState) -> dict[str, Any]:
        """Execute action for the selected goal. Updates goal via state reducer."""
        goal_id = state.get("active_goal_id")
        if not goal_id:
            return {"action_result": ""}

        goals = state.get("goals", [])
        goal_data = None
        for g in goals:
            if isinstance(g, dict) and g.get("goal_id") == goal_id:
                goal_data = g
                break

        if goal_data is None:
            return {"action_result": ""}

        status = goal_data.get("status", "")
        progress = goal_data.get("progress", 0.0)
        steps_completed = goal_data.get("steps_completed", 0)
        steps_total = goal_data.get("steps_total", 0)
        last_error = goal_data.get("last_error", "")
        new_events: list[dict[str, Any]] = []

        if status == "blocked":
            if last_error and self._reflection_engine and self._llm:
                goal_data = {
                    **goal_data,
                    "status": "in_progress",
                    "last_error": "",
                    "progress": max(0.1, progress * 0.8),
                    "updated_at": time.time(),
                }
                new_events.append({
                    "type": "goal_replan",
                    "goal_id": goal_id,
                    "timestamp": time.time(),
                    "cycle": state.get("cycle_count", 0),
                })
                return {"goals": [goal_data], "action_result": "replan", "events": new_events}
            return {"action_result": "wait"}

        if progress >= 1.0:
            goal_data = {
                **goal_data,
                "status": "completed",
                "progress": 1.0,
                "updated_at": time.time(),
            }
            new_events.append({
                "type": "goal_completed",
                "goal_id": goal_id,
                "timestamp": time.time(),
                "cycle": state.get("cycle_count", 0),
            })
            return {"goals": [goal_data], "action_result": "complete", "events": new_events}

        if self._action_callback is None:
            goal_data = {
                **goal_data,
                "steps_total": steps_total + 1,
                "updated_at": time.time(),
            }
            return {"goals": [goal_data], "action_result": "wait"}

        goal_obj = Goal.from_dict(goal_data)
        try:
            result = self._action_callback(goal_obj)
            if result:
                completed = steps_completed + 1
                new_progress = completed / max(steps_total, completed)
                goal_data = {
                    **goal_data,
                    "steps_completed": completed,
                    "steps_total": max(steps_total, completed),
                    "progress": new_progress,
                    "updated_at": time.time(),
                }
                new_events.append({
                    "type": "goal_step_complete",
                    "goal_id": goal_id,
                    "progress": new_progress,
                    "timestamp": time.time(),
                    "cycle": state.get("cycle_count", 0),
                })
                if new_progress >= 1.0:
                    goal_data["status"] = "completed"
                    goal_data["progress"] = 1.0
                    new_events.append({
                        "type": "goal_completed",
                        "goal_id": goal_id,
                        "timestamp": time.time(),
                        "cycle": state.get("cycle_count", 0),
                    })
            else:
                goal_data = {
                    **goal_data,
                    "status": "blocked",
                    "last_error": "Action returned empty result",
                    "updated_at": time.time(),
                }
                new_events.append({
                    "type": "goal_blocked",
                    "goal_id": goal_id,
                    "error": "empty_result",
                    "timestamp": time.time(),
                    "cycle": state.get("cycle_count", 0),
                })
        except Exception as exc:
            logger.warning("[AutoGraph] Action failed for %s: %s", goal_id, exc)
            goal_data = {
                **goal_data,
                "status": "blocked",
                "last_error": str(exc),
                "updated_at": time.time(),
            }
            new_events.append({
                "type": "goal_blocked",
                "goal_id": goal_id,
                "error": str(exc),
                "timestamp": time.time(),
                "cycle": state.get("cycle_count", 0),
            })

        return {"goals": [goal_data], "action_result": result or "", "events": new_events}

    def _node_reflect(self, state: AutonomousState) -> dict[str, Any]:
        """Reflect on action results. Returns insights via state."""
        action_result = state.get("action_result", "")
        goal_id = state.get("active_goal_id")

        if not goal_id or not action_result:
            return {"reflection_insights": []}

        goals = state.get("goals", [])
        goal_data = None
        for g in goals:
            if isinstance(g, dict) and g.get("goal_id") == goal_id:
                goal_data = g
                break

        if goal_data is None:
            return {"reflection_insights": []}

        insights: list[str] = []

        if goal_data.get("status") == "blocked":
            insights.append(f"Goal blocked: {goal_data.get('last_error', '')}")
            if self._reflection_engine and self._llm:
                try:
                    reflection = self._reflection_engine.reflect(
                        user_input=goal_data.get("description", ""),
                        plan=[goal_data.get("description", "")],
                        results=[goal_data.get("last_error", "") or "blocked"],
                        execution_metrics={"steps": 1, "tool_calls": 0, "time": 0},
                    )
                    if reflection.went_wrong:
                        insights.extend(reflection.went_wrong[:3])
                except Exception:
                    pass

        return {"reflection_insights": insights}

    @staticmethod
    def _node_check(state: AutonomousState) -> Command:
        """Check if the loop should continue or end."""
        goals = state.get("goals", [])
        active_count = len([
            g for g in goals
            if isinstance(g, dict) and g.get("status") in ("pending", "in_progress", "blocked")
        ])

        max_cycles = state.get("max_cycles", 1000)
        cycle_count = state.get("cycle_count", 0)
        stop_requested = state.get("stop_requested", False)

        should_continue = (
            not stop_requested
            and active_count > 0
            and cycle_count < max_cycles
        )

        if cycle_count % 50 == 0:
            logger.info(
                "[AutoGraph] Cycle %d | active=%d | should_continue=%s",
                cycle_count, active_count, should_continue,
            )

        if should_continue:
            return Command(goto="perceive", update={"should_continue": True})
        return Command(goto=END, update={"should_continue": False})

    # ------------------------------------------------------------------
    # Goal management (updates in-memory cache, syncs to state on run)
    # ------------------------------------------------------------------

    def add_goal(
        self,
        description: str,
        priority: int = 0,
        deadline: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> Goal:
        """Add a new goal to pursue autonomously."""
        goal = Goal(
            description=description,
            priority=priority,
            deadline=deadline,
            context=context or {},
        )
        self._goals[goal.goal_id] = goal
        logger.info("[AutoGraph] Goal added: %s", description)
        return goal

    def remove_goal(self, goal_id: str) -> bool:
        """Remove a goal by ID."""
        existed = self._goals.pop(goal_id, None) is not None
        return existed

    def get_active_goals(self) -> list[Goal]:
        """Get all non-completed goals."""
        return [
            g for g in self._goals.values()
            if g.status in ("pending", "in_progress", "blocked")
        ]

    def get_goal(self, goal_id: str) -> Goal | None:
        """Get a goal by ID."""
        return self._goals.get(goal_id)

    def get_event_log(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent events from the last run's state."""
        return self._last_events[-limit:] if hasattr(self, '_last_events') else []

    # ------------------------------------------------------------------
    # Execution control
    # ------------------------------------------------------------------

    def run_cycle(self) -> dict[str, Any]:
        """Run a single autonomous cycle through the LangGraph.

        Reads goals from cache, invokes the graph, then syncs
        updated state back to the cache.
        """
        initial = _make_initial_state(list(self._goals.values()))
        initial["cycle_count"] = self._cycle_count

        config = {"configurable": {"thread_id": f"autonomous_{id(self)}"}}

        try:
            final = self.graph.invoke(initial, config=config)
            self._cycle_count = final.get("cycle_count", self._cycle_count)
            self._active_goal_id = final.get("active_goal_id")
            self._last_events = final.get("events", [])

            # Sync updated goals back to cache
            updated_goals = final.get("goals", [])
            for g_data in updated_goals:
                if isinstance(g_data, dict):
                    gid = g_data.get("goal_id", "")
                    if gid and gid in self._goals:
                        self._goals[gid] = Goal.from_dict(g_data)
                    elif gid:
                        self._goals[gid] = Goal.from_dict(g_data)

            # Remove goals that are completed and no longer in state
            updated_ids = {
                g.get("goal_id", "")
                for g in updated_goals
                if isinstance(g, dict) and g.get("goal_id")
            }
            for gid in list(self._goals.keys()):
                if gid not in updated_ids and self._goals[gid].is_complete:
                    del self._goals[gid]

            return final
        except Exception as exc:
            logger.error("[AutoGraph] Cycle error: %s", exc, exc_info=True)
            return {"error": str(exc), "cycle_count": self._cycle_count}

    def run(self, max_cycles: int | None = None) -> None:
        """Run the autonomous loop (blocking) for up to max_cycles."""
        self._running = True
        cycles = 0
        limit = max_cycles or self._max_cycles

        while self._running and cycles < limit:
            result = self.run_cycle()
            cycles += 1

            if not result.get("should_continue", False):
                logger.info("[AutoGraph] No active goals, stopping")
                break

            if cycles % 20 == 0:
                logger.info("[AutoGraph] Completed %d cycles", cycles)

        self._running = False

    def start(self) -> None:
        """Start the autonomous loop in a background thread."""
        import threading
        if self._running:
            return
        self._running = True
        thread = threading.Thread(target=self._run_background, daemon=True)
        thread.start()
        logger.info("[AutoGraph] Started in background")

    def stop(self) -> None:
        """Stop the autonomous loop gracefully by setting stop_requested in state."""
        if self._goals:
            initial = _make_initial_state(list(self._goals.values()))
            initial["stop_requested"] = True
            config = {"configurable": {"thread_id": f"autonomous_stop_{id(self)}"}}
            try:
                self.graph.invoke(initial, config=config)
            except Exception:
                pass
        self._running = False
        logger.info("[AutoGraph] Stop requested via state")

    def _run_background(self) -> None:
        """Background thread target."""
        while self._running:
            self.run_cycle()
            if self._cycle_count % 20 == 0:
                logger.info("[AutoGraph] Background cycle %d", self._cycle_count)

    def set_environment_signal(self, signal: str, value: Any = True) -> None:
        """Set an environment signal."""
        logger.info("[AutoGraph] Signal set: %s=%s", signal, value)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def stats(self) -> dict[str, Any]:
        active = self.get_active_goals()
        return {
            "running": self._running,
            "cycles": self._cycle_count,
            "total_goals": len(self._goals),
            "active_goals": len(active),
            "blocked_goals": sum(1 for g in active if g.is_blocked),
            "completed_goals": sum(1 for g in self._goals.values() if g.is_complete),
        }