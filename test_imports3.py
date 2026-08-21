import sys
import traceback

with open("test_results3.txt", "w") as out:
    def test(name, fn):
        try:
            fn()
            out.write(f"OK: {name}\n")
        except Exception as e:
            out.write(f"FAIL: {name} -> {type(e).__name__}: {e}\n")
            traceback.print_exc(file=out)

    out.write("=== Testing LangGraph imports ===\n\n")

    test("langgraph.graph.StateGraph",
         lambda: __import__("langgraph.graph", fromlist=["StateGraph"]))

    test("langgraph.checkpoint.memory.MemorySaver",
         lambda: __import__("langgraph.checkpoint.memory", fromlist=["MemorySaver"]))

    test("langgraph.prebuilt.ToolNode",
         lambda: __import__("langgraph.prebuilt", fromlist=["ToolNode"]))

    test("langgraph.checkpoint.sqlite.SqliteSaver",
         lambda: __import__("langgraph.checkpoint.sqlite", fromlist=["SqliteSaver"]))

    test("langgraph.types.Send",
         lambda: __import__("langgraph.types", fromlist=["Send"]))

    out.write("\n=== Testing ToolExecutorGraph ===\n\n")
    try:
        from agent.utils.tool_executor_graph import ToolExecutorGraph, create_tool_executor
        out.write("OK: ToolExecutorGraph imported\n")
    except Exception as e:
        out.write(f"FAIL: ToolExecutorGraph -> {type(e).__name__}: {e}\n")
        traceback.print_exc(file=out)

    out.write("\n=== Testing GraphOrchestrator ===\n\n")
    try:
        from agent.graph_orchestrator import GraphOrchestrator
        out.write("OK: GraphOrchestrator imported\n")
    except Exception as e:
        out.write(f"FAIL: GraphOrchestrator -> {type(e).__name__}: {e}\n")
        traceback.print_exc(file=out)

    out.write("\nDONE\n")