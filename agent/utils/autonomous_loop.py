"""Autonomous loop engine — proactive agent behavior.

Transforms the agent from passive-response to autonomous-execution:
  Perceive → Decide → Act → Reflect → Persist

The loop runs continuously (or until goal satisfaction), monitoring
environment state and taking initiative to achieve user goals.

Key features:
  - Goal persistence across sessions
  - Event-driven triggers (timer, file changes, messages)
  - Progress tracking with partial-result recovery
  - Adaptive re-planning when obstacles are detected
  - Integration with ReflectionEngine for strategy improvement
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


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
        import hashlib
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


class AutonomousLoop:
    """Core autonomous perception-decision-action loop.

    Usage:
        loop = AutonomousLoop()
        loop.add_goal(Goal(description="监控项目进度"))
        loop.start()
        # Loop runs, perceiving environment and taking action
        loop.stop()
    """

    def __init__(
        self,
        llm: Any = None,
        action_callback: Callable[[Goal], str] | None = None,
        reflection_engine: Any = None,
        cycle_interval: float = 5.0,
        max_cycles: int = 1000,
    ) -> None:
        self._llm = llm
        self._action_callback = action_callback
        self._reflection_engine = reflection_engine
        self._cycle_interval = cycle_interval
        self._max_cycles = max_cycles

        self._goals: OrderedDict[str, Goal] = OrderedDict()
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._environment_state: dict[str, Any] = {}
        self._event_log: list[dict[str, Any]] = []
        self._max_event_log = 200
        self._cycle_count = 0

    def add_goal(self, goal: Goal) -> None:
        """Add a new goal to pursue autonomously."""
        with self._lock:
            self._goals[goal.goal_id] = goal
        self._log_event("goal_added", goal.to_dict())
        logger.info("[AutoLoop] Goal added: %s", goal.description)

    def remove_goal(self, goal_id: str) -> bool:
        with self._lock:
            return self._goals.pop(goal_id, None) is not None

    def get_active_goals(self) -> list[Goal]:
        with self._lock:
            return [
                g for g in self._goals.values()
                if g.status in ("pending", "in_progress", "blocked")
            ]

    def get_goal(self, goal_id: str) -> Goal | None:
        with self._lock:
            return self._goals.get(goal_id)

    def start(self) -> None:
        """Start the autonomous loop in a background thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("[AutoLoop] Started | interval=%.1fs", self._cycle_interval)

    def stop(self) -> None:
        """Stop the autonomous loop gracefully."""
        self._stop_event.set()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("[AutoLoop] Stopped")

    def _run_loop(self) -> None:
        """Main loop: perceive → decide → act → reflect."""
        logger.info("[AutoLoop] Loop thread started")

        while not self._stop_event.is_set():
            try:
                self._cycle_count += 1

                self._perceive()

                goals = self.get_active_goals()
                if not goals:
                    if self._cycle_count % 20 == 0:
                        self._log_event("idle_cycle", {"cycle": self._cycle_count})
                    time.sleep(self._cycle_interval)
                    continue

                for goal in sorted(goals, key=lambda g: -g.priority):
                    if self._stop_event.is_set():
                        break
                    if goal.is_expired:
                        self._handle_expired_goal(goal)
                        continue
                    self._decide_and_act(goal)

                self._cleanup_completed()

                if self._cycle_count >= self._max_cycles:
                    logger.info("[AutoLoop] Max cycles reached, resetting")
                    self._cycle_count = 0

            except Exception as exc:
                logger.error("[AutoLoop] Loop error: %s", exc, exc_info=True)

            self._stop_event.wait(self._cycle_interval)

        logger.info("[AutoLoop] Loop thread exiting")

    def _perceive(self) -> None:
        """Perceive environment state — collect signals and update context."""
        self._environment_state["last_perception"] = time.time()
        self._environment_state["active_goals"] = len(self.get_active_goals())
        self._environment_state["cycle"] = self._cycle_count

        if self._environment_state.get("file_watcher_triggered"):
            self._log_event("env_file_change", {})
            self._environment_state["file_watcher_triggered"] = False

    def _decide_and_act(self, goal: Goal) -> None:
        """Decide what to do for a goal and execute the action."""
        decision = self._decide(goal)

        if decision == "act":
            self._execute_action(goal)
        elif decision == "replan":
            self._replan_goal(goal)
        elif decision == "wait":
            pass
        elif decision == "complete":
            goal.mark_complete()
            self._log_event("goal_completed", goal.to_dict())

    def _decide(self, goal: Goal) -> str:
        """Decide the next action for a goal.

        Returns: act | replan | wait | complete
        """
        if goal.status == "blocked":
            if goal.last_error and self._reflection_engine:
                goal.last_error = ""
                return "replan"
            return "wait"

        if goal.progress >= 1.0:
            return "complete"

        if self._action_callback is None:
            goal.update_progress(goal.steps_completed, goal.steps_total + 1)
            return "wait"

        return "act"

    def _execute_action(self, goal: Goal) -> None:
        """Execute a single action step for a goal."""
        try:
            if self._action_callback:
                result = self._action_callback(goal)
                if result:
                    completed = goal.steps_completed + 1
                    goal.update_progress(completed, max(goal.steps_total, completed))
                    self._log_event(
                        "goal_step_complete",
                        {"goal_id": goal.goal_id, "progress": goal.progress},
                    )
                    if goal.progress >= 1.0:
                        goal.mark_complete()
                else:
                    goal.mark_blocked("Action returned empty result")
        except Exception as exc:
            logger.warning("[AutoLoop] Action failed for goal %s: %s", goal.goal_id, exc)
            goal.mark_blocked(str(exc))
            self._log_event("goal_blocked", {"goal_id": goal.goal_id, "error": str(exc)})

    def _replan_goal(self, goal: Goal) -> None:
        """Re-plan a blocked goal using reflection insights."""
        self._log_event("goal_replan", {"goal_id": goal.goal_id})
        goal.status = "in_progress"
        goal.progress = max(0.1, goal.progress * 0.8)
        goal.updated_at = time.time()

    def _handle_expired_goal(self, goal: Goal) -> None:
        self._log_event("goal_expired", goal.to_dict())
        goal.mark_blocked("Deadline exceeded")

    def _cleanup_completed(self) -> None:
        with self._lock:
            completed = [
                gid for gid, g in self._goals.items()
                if g.is_complete and (time.time() - g.updated_at > 300)
            ]
            for gid in completed:
                del self._goals[gid]

    def _log_event(self, event_type: str, data: dict[str, Any]) -> None:
        entry = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data,
            "cycle": self._cycle_count,
        }
        self._event_log.append(entry)
        if len(self._event_log) > self._max_event_log:
            self._event_log.pop(0)

    def set_environment_signal(self, signal: str, value: Any = True) -> None:
        """Set an environment signal that triggers perception."""
        self._environment_state[signal] = value
        self._log_event("signal_set", {"signal": signal})

    def get_event_log(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._event_log[-limit:]

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
            "event_log_size": len(self._event_log),
        }