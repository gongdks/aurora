"""Tests for agent.graph_orchestrator — Plan-and-Execute LangGraph orchestrator."""

import threading
from unittest.mock import patch, MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent.config import settings
from agent.graph_orchestrator import (
    AgentState,
    GraphOrchestrator,
    _merge_results,
    _pending_parallel_reduce,
    _sum_complex_steps,
)


@pytest.fixture(autouse=True)
def _memory_saver(monkeypatch):
    """Force all GraphOrchestrator instances to use MemorySaver (no file locks)."""
    monkeypatch.setattr(
        GraphOrchestrator, "_init_checkpointer", lambda self: MemorySaver()
    )


@pytest.fixture
def orch():
    """Provide a fresh GraphOrchestrator with no real LLM calls."""
    with patch("agent.graph_orchestrator.create_llm") as mock_create:
        mock_llm_provider = MagicMock()
        mock_llm_provider.get_model.return_value = MagicMock()
        mock_create.return_value = mock_llm_provider

        with patch.object(GraphOrchestrator, "_build_tools_description", return_value=""):
            orchestrator = GraphOrchestrator()
    return orchestrator


class TestReducers:
    def test_merge_results_empty_updates(self):
        result = _merge_results(["a", "b"], [])
        assert result == []

    def test_merge_results_tuple_updates(self):
        result = _merge_results([""] * 3, [(0, "first"), (2, "third")])
        assert result[0] == "first"
        assert result[2] == "third"
        assert len(result) == 3

    def test_merge_results_dict_updates(self):
        result = _merge_results([""] * 3, [{"step_index": 1, "result": "second"}])
        assert result[1] == "second"

    def test_merge_results_auto_extend(self):
        result = _merge_results([], [(3, "step4")])
        assert len(result) == 4
        assert result[3] == "step4"

    def test_pending_parallel_reduce_set_and_decrement(self):
        assert _pending_parallel_reduce(0, 3) == 3
        assert _pending_parallel_reduce(3, -1) == 2
        assert _pending_parallel_reduce(2, -1) == 1
        assert _pending_parallel_reduce(1, -1) == 0

    def test_pending_parallel_reduce_zero_floor(self):
        assert _pending_parallel_reduce(0, -1) == 0
        assert _pending_parallel_reduce(0, -5) == 0

    def test_sum_complex_steps(self):
        assert _sum_complex_steps(5, 3) == 8
        assert _sum_complex_steps(0, 0) == 0


class TestAgentState:
    def test_state_has_required_keys(self):
        state = AgentState(
            user_input="test",
            classification="simple",
            plan=[],
            current_step=0,
            results=[],
        )
        assert state["user_input"] == "test"
        assert state["classification"] == "simple"

    def test_state_optional_keys(self):
        state = AgentState(user_input="test")
        assert "graph_context" not in state

    def test_state_reducer_fields_exist(self):
        annotations = AgentState.__annotations__
        assert "results" in annotations
        assert "pending_parallel" in annotations
        assert "complex_steps_executed" in annotations


class TestGraphOrchestratorInit:
    def test_init_without_errors(self, orch):
        assert orch._graph is not None
        assert orch._llm is not None

    def test_graph_structure_info(self, orch):
        info = orch.graph_structure_info()
        assert info["type"] == "StateGraph (Plan-and-Execute)"
        assert "classify" in info["nodes"]
        assert "react_fast" in info["nodes"]
        assert "plan" in info["nodes"]
        assert "multi_agent_execute" in info["nodes"]
        assert any("MultiAgentGraph" in s.get("name", "") for s in info["subgraphs"])

    def test_graph_has_correct_edges(self, orch):
        info = orch.graph_structure_info()
        edge_str = str(info["edges"])
        assert "classify" in edge_str
        assert "multi_agent" in edge_str
        assert "END" in edge_str


class TestRoutingLogic:
    def test_route_after_classify_simple(self):
        route = GraphOrchestrator._route_after_classify({"classification": "simple"})
        assert route == "simple"

    def test_route_after_classify_complex(self):
        route = GraphOrchestrator._route_after_classify({"classification": "complex"})
        assert route == "complex"

    def test_route_after_classify_multi_agent(self):
        route = GraphOrchestrator._route_after_classify({"classification": "multi_agent"})
        assert route == "multi_agent"

    def test_route_after_classify_default(self):
        route = GraphOrchestrator._route_after_classify({})
        assert route == "complex"

    def test_route_after_adaptive_simple(self):
        route = GraphOrchestrator._route_after_adaptive({"classification": "simple"})
        assert route == "end"

    def test_route_after_adaptive_complex(self):
        route = GraphOrchestrator._route_after_adaptive({"classification": "complex"})
        assert route == "upgrade"

    def test_route_after_reflect_done(self):
        route = GraphOrchestrator._route_after_reflect({"is_done": True})
        assert route == "complete"

    def test_route_after_reflect_no_replan(self):
        route = GraphOrchestrator._route_after_reflect({
            "is_done": False,
            "reflection_should_replan": False,
        })
        assert route == "complete"

    def test_route_after_reflect_should_replan(self):
        route = GraphOrchestrator._route_after_reflect({
            "is_done": False,
            "reflection_should_replan": True,
            "plan_rounds": 1,
            "max_plan_rounds": 3,
        })
        assert route == "replan"

    def test_route_after_reflect_max_rounds(self):
        route = GraphOrchestrator._route_after_reflect({
            "is_done": False,
            "reflection_should_replan": True,
            "plan_rounds": 3,
            "max_plan_rounds": 3,
        })
        assert route == "complete"


class TestMultiAgentIntegration:
    def test_enable_multi_agent_mode(self, orch):
        assert orch._use_multi_agent is False
        orch.enable_multi_agent_mode(True)
        assert orch._use_multi_agent is True
        orch.enable_multi_agent_mode(False)
        assert orch._use_multi_agent is False

    def test_classify_routes_to_multi_agent(self, orch):
        orch.enable_multi_agent_mode(True)
        with patch("agent.graph_orchestrator.classify_query", return_value="complex"):
            result = orch._node_classify(
                {"user_input": "请团队协作完成一个综合分析任务", "plan_rounds": 0},
                {"configurable": {}},
            )
        assert result["classification"] == "multi_agent"

    def test_classify_does_not_route_to_multi_agent_when_disabled(self, orch):
        with patch("agent.graph_orchestrator.classify_query", return_value="complex"):
            result = orch._node_classify(
                {"user_input": "请团队协作完成一个综合分析任务", "plan_rounds": 0},
                {"configurable": {}},
            )
        assert result["classification"] != "multi_agent"

    def test_classify_routes_simple_query_to_simple(self, orch):
        orch.enable_multi_agent_mode(True)
        with patch("agent.graph_orchestrator.classify_query", return_value="simple"):
            result = orch._node_classify(
                {"user_input": "你好", "plan_rounds": 0},
                {"configurable": {}},
            )
        assert result["classification"] == "simple"

    def test_node_multi_agent_execute_handles_cancel(self, orch):
        cancel_evt = threading.Event()
        cancel_evt.set()
        result = orch._node_multi_agent_execute(
            {"user_input": "test", "plan_rounds": 0},
            {"configurable": {"cancel_event": cancel_evt}},
        )
        assert result.get("is_done") is True
        assert "停止" in result.get("result", "")


class TestPlanAndExecuteFlow:
    def test_node_plan_creates_plan(self, orch):
        with patch.object(orch, "_llm_invoke", return_value="1. 分析请求\n2. 执行操作\n3. 总结结果"):
            result = orch._node_plan(
                {
                    "user_input": "分析代码结构",
                    "graph_context": "",
                    "results": [],
                    "plan": [],
                    "plan_rounds": 0,
                },
                {"configurable": {}},
            )
        assert len(result.get("plan", [])) > 0
        assert result.get("plan_rounds") == 1

    def test_node_react_fast_returns_result(self, orch):
        mock_exec = MagicMock(return_value={
            "status": "completed",
            "result": "你好！有什么可以帮助你的吗？",
            "iterations": 1,
            "time": 0.5,
            "hit_limit": False,
            "tool_calls": 0,
            "llm_calls": 1,
        })
        with patch.object(orch._tool_executor, "execute", mock_exec):
            result = orch._node_react_fast(
                {
                    "user_input": "你好",
                    "chat_history_messages": [],
                },
                {"configurable": {}},
            )
        assert result.get("is_done") is True
        assert "result" in result
        assert result.get("fast_iterations") == 1

    def test_node_adaptive_check_passes_when_ok(self, orch):
        result = orch._node_adaptive_check(
            {
                "classification": "simple",
                "fast_iterations": 1,
                "fast_tool_calls": 1,
                "fast_elapsed": 1.0,
                "fast_hit_limit": False,
            },
            {"configurable": {}},
        )
        assert result.get("classification") != "complex"

    def test_node_adaptive_check_upgrades_when_exceeded(self, orch):
        result = orch._node_adaptive_check(
            {
                "classification": "simple",
                "fast_iterations": 1000,
                "fast_tool_calls": 1000,
                "fast_elapsed": 999.0,
                "fast_hit_limit": True,
                "user_input": "test",
            },
            {"configurable": {}},
        )
        assert result.get("classification") == "complex"
        assert result.get("is_done") is False

    def test_node_dispatch_all_steps_done(self, orch):
        result = orch._node_dispatch(
            {
                "plan": ["step1"],
                "current_step": 1,
                "parallel_groups": [],
                "parallel_execution": True,
            },
            {"configurable": {}},
        )
        assert isinstance(result, Command)
        assert result.goto == "verify"

    def test_node_dispatch_serial_step(self, orch):
        result = orch._node_dispatch(
            {
                "plan": ["step1", "step2"],
                "current_step": 0,
                "parallel_groups": [],
                "parallel_execution": True,
            },
            {"configurable": {}},
        )
        assert isinstance(result, Command)
        assert result.goto == "execute_step"

    def test_node_dispatch_parallel_sends(self, orch):
        result = orch._node_dispatch(
            {
                "plan": ["s0", "s1", "s2"],
                "current_step": 0,
                "parallel_groups": [[0, 1]],
                "parallel_execution": True,
            },
            {"configurable": {}},
        )
        assert isinstance(result, Command)
        assert isinstance(result.goto, list)
        assert len(result.goto) == 2

    def test_node_check_steps_waits_for_parallel(self, orch):
        result = orch._node_check_steps(
            {
                "plan": ["step1", "step2", "step3"],
                "current_step": 0,
                "pending_parallel": 2,
            },
            {"configurable": {}},
        )
        assert isinstance(result, Command)

    def test_node_check_steps_advances(self, orch):
        result = orch._node_check_steps(
            {
                "plan": ["step1", "step2"],
                "current_step": 0,
                "pending_parallel": 0,
                "parallel_groups": [],
                "parallel_execution": True,
            },
            {"configurable": {}},
        )
        assert isinstance(result, Command)
        assert result.goto == "dispatch"

    def test_node_check_steps_all_done(self, orch):
        result = orch._node_check_steps(
            {
                "plan": ["step1"],
                "current_step": 1,
                "pending_parallel": 0,
            },
            {"configurable": {}},
        )
        assert isinstance(result, Command)
        assert result.goto == "verify"

    def test_node_execute_step_cancelled(self, orch):
        cancel_evt = threading.Event()
        cancel_evt.set()
        result = orch._node_execute_step(
            {
                "current_step": 0,
                "plan": ["step1"],
                "user_input": "test",
                "graph_context": "",
                "chat_history_messages": [],
                "parallel_groups": [],
            },
            {"configurable": {"cancel_event": cancel_evt}},
        )
        assert "停止" in result.get("result", "")

    def test_node_execute_step_runs_tool(self, orch):
        mock_exec = MagicMock(return_value={
            "status": "completed",
            "result": "执行结果",
            "iterations": 1,
            "time": 0.5,
            "hit_limit": False,
            "tool_calls": 1,
            "llm_calls": 1,
        })
        with patch.object(orch._tool_executor, "execute", mock_exec):
            result = orch._node_execute_step(
                {
                    "current_step": 0,
                    "plan": ["执行测试步骤"],
                    "user_input": "测试",
                    "graph_context": "",
                    "chat_history_messages": [],
                    "parallel_groups": [],
                },
                {"configurable": {}},
            )
        assert result.get("results") is not None
        assert result.get("complex_steps_executed") == 1

    def test_node_verify_empty_results(self, orch):
        result = orch._node_verify(
            {
                "user_input": "test",
                "results": [],
                "plan": [],
                "plan_rounds": 0,
                "graph_context": "",
            },
            {"configurable": {}},
        )
        assert result.get("is_done") is True

    def test_node_verify_max_rounds_reached(self, orch):
        result = orch._node_verify(
            {
                "user_input": "test",
                "results": ["result1", "result2"],
                "plan": ["step1", "step2"],
                "plan_rounds": settings.MAX_PLAN_ROUNDS + 1,
                "graph_context": "",
            },
            {"configurable": {}},
        )
        assert result.get("is_done") is True

    def test_node_verify_cancelled(self, orch):
        cancel_evt = threading.Event()
        cancel_evt.set()
        result = orch._node_verify(
            {
                "user_input": "test",
                "results": ["result"],
                "plan": ["step"],
                "plan_rounds": 0,
                "graph_context": "",
            },
            {"configurable": {"cancel_event": cancel_evt}},
        )
        assert result.get("is_done") is True
        assert "停止" in result.get("result", "")

    def test_node_reflect_cancelled(self, orch):
        cancel_evt = threading.Event()
        cancel_evt.set()
        result = orch._node_reflect(
            {
                "user_input": "test",
                "plan": ["step"],
                "results": ["result"],
                "result": "",
                "is_done": False,
                "complex_steps_executed": 1,
                "complex_elapsed": 1.0,
                "plan_rounds": 0,
            },
            {"configurable": {"cancel_event": cancel_evt}},
        )
        assert result.get("is_done") is True


class TestGraphCompilation:
    def test_graph_compiles_with_checkpointer(self, orch):
        assert orch._graph is not None

    def test_graph_has_expected_node_count(self, orch):
        info = orch.graph_structure_info()
        assert len(info["nodes"]) == 11

    def test_parallel_state_transitions(self):
        assert _pending_parallel_reduce(0, 3) == 3
        assert _pending_parallel_reduce(3, -1) == 2
        assert _pending_parallel_reduce(2, -1) == 1
        assert _pending_parallel_reduce(1, -1) == 0
        assert _pending_parallel_reduce(0, -1) == 0

    def test_results_merging_with_indices(self):
        existing = [""] * 5
        updates = [(1, "result1"), (3, "result3")]
        merged = _merge_results(existing, updates)
        assert merged[1] == "result1"
        assert merged[3] == "result3"
        assert merged[0] == ""
        assert merged[2] == ""
        assert merged[4] == ""

    def test_skill_learner_does_not_need_memory_manager(self):
        from agent.utils.skill_learner_graph import SkillLearnerGraph
        learner = SkillLearnerGraph()
        assert learner is not None
        result = learner.learn_from_execution(
            user_input="测试技能",
            plan=["步骤1", "步骤2"],
            results=["结果1", "结果2"],
            tool_calls=[{"name": "test_tool", "args": {}}],
            success=True,
            reflection_score=0.8,
        )
        assert result is not None
        assert result.confidence == 0.8