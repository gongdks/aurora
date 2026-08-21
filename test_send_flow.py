"""Minimal test to verify Send + edge behavior in LangGraph 1.2.x."""
from typing import Annotated, TypedDict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Command


class TestState(TypedDict, total=False):
    count: int
    results: Annotated[list, lambda a, b: list(a) + list(b)]
    plan: list[str]
    current_step: int


def dispatch(state: dict[str, Any]) -> Command:
    count = state.get("count", 0)
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    print(f"[dispatch] count={count}, plan={plan}, current_step={current_step}")

    if current_step >= len(plan):
        print("[dispatch] ALL DONE -> verify")
        return Command(goto="verify")

    exec_state = {
        "current_step": current_step,
        "plan": plan,
        "count": count,
    }
    print(f"[dispatch] Send(execute_step, {exec_state})")
    return Command(goto=Send("execute_step", exec_state))


def execute_step(state: dict[str, Any]) -> dict[str, Any]:
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])
    count = state.get("count", 0)
    print(f"[execute_step] current_step={current_step}, plan_len={len(plan)}, count={count}")

    if current_step >= len(plan):
        print("[execute_step] current_step >= plan length, returning empty")
        return {}

    step_desc = plan[current_step]
    result = f"step_{current_step}_({step_desc})_done"
    print(f"[execute_step] produced result: {result}")
    return {"results": [result]}


def check_steps(state: dict[str, Any]) -> Command:
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    results = state.get("results", [])
    print(f"[check_steps] current_step={current_step}, plan_len={len(plan)}, results={results}")

    if current_step + 1 >= len(plan):
        print("[check_steps] All steps done -> verify")
        return Command(goto="verify")

    next_step = current_step + 1
    print(f"[check_steps] Advancing to step {next_step}")
    return Command(goto="dispatch", update={"current_step": next_step})


def verify(state: dict[str, Any]) -> dict[str, Any]:
    results = state.get("results", [])
    print(f"[verify] results={results}")
    if not results:
        print("[verify] *** NO RESULTS ***")
    return {"final_result": "ALL DONE" if results else "NO OUTPUT"}


graph = StateGraph(TestState)
graph.add_node("dispatch", dispatch)
graph.add_node("execute_step", execute_step)
graph.add_node("check_steps", check_steps)
graph.add_node("verify", verify)

graph.add_edge(START, "dispatch")
graph.add_edge("execute_step", "check_steps")

app = graph.compile()

print("=== Running test ===")
try:
    result = app.invoke({
        "count": 0,
        "results": [],
        "plan": ["step_a", "step_b", "step_c"],
        "current_step": 0,
    })
    print(f"\n=== Final result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    results = result.get("results", [])
    if len(results) == 3:
        print("\n*** TEST PASSED: All 3 results collected correctly! ***")
    else:
        print(f"\n*** TEST FAILED: Expected 3 results, got {len(results)} ***")
except Exception as e:
    import traceback
    traceback.print_exc()