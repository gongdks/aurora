import sys
import traceback

results = []

def test(name, fn):
    try:
        fn()
        results.append(f"OK: {name}")
    except Exception as e:
        results.append(f"FAIL: {name} -> {e}")
        traceback.print_exc()

test("langgraph.graph.StateGraph", lambda: __import__("langgraph.graph", fromlist=["StateGraph"]))
test("langgraph.checkpoint.memory.MemorySaver", lambda: __import__("langgraph.checkpoint.memory", fromlist=["MemorySaver"]))
test("langgraph.prebuilt.ToolNode", lambda: __import__("langgraph.prebuilt", fromlist=["ToolNode"]))
test("langgraph.checkpoint.sqlite.SqliteSaver", lambda: __import__("langgraph.checkpoint.sqlite", fromlist=["SqliteSaver"]))
test("langgraph.types.Send", lambda: __import__("langgraph.types", fromlist=["Send"]))

print("\n".join(results))