import sys
import traceback

with open("verify_result.txt", "w") as out:
    try:
        from langgraph.graph import StateGraph, END, START
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.prebuilt import ToolNode
        from langgraph.types import Send
        out.write("OK: All LangGraph core imports work\n")
    except Exception as e:
        out.write(f"FAIL: {e}\n")
        traceback.print_exc(file=out)
        sys.exit(1)

    try:
        from agent.utils.tool_executor_graph import ToolExecutorGraph, create_tool_executor
        out.write("OK: ToolExecutorGraph imported\n")
    except Exception as e:
        out.write(f"FAIL ToolExecutorGraph: {e}\n")
        traceback.print_exc(file=out)

    try:
        from agent.graph_orchestrator import GraphOrchestrator
        out.write("OK: GraphOrchestrator imported\n")
    except Exception as e:
        out.write(f"FAIL GraphOrchestrator: {e}\n")
        traceback.print_exc(file=out)

    out.write("DONE\n")