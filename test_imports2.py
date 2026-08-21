import sys
import traceback

with open("test_results2.txt", "w") as f:
    def test(name, fn):
        try:
            fn()
            f.write(f"OK: {name}\n")
        except Exception as e:
            f.write(f"FAIL: {name} -> {e}\n")
            traceback.print_exc(file=f)

    test("langgraph.graph.StateGraph", lambda: __import__("langgraph.graph", fromlist=["StateGraph"]))
    test("langgraph.checkpoint.memory.MemorySaver", lambda: __import__("langgraph.checkpoint.memory", fromlist=["MemorySaver"]))
    test("langgraph.prebuilt.ToolNode", lambda: __import__("langgraph.prebuilt", fromlist=["ToolNode"]))
    test("langgraph.checkpoint.sqlite.SqliteSaver", lambda: __import__("langgraph.checkpoint.sqlite", fromlist=["SqliteSaver"]))
    test("langgraph.types.Send", lambda: __import__("langgraph.types", fromlist=["Send"]))
    test("langgraph.prebuilt.ToolNode direct import", lambda: None)

    from langgraph.prebuilt import ToolNode
    f.write(f"ToolNode: {ToolNode}\n")

    f.write("DONE\n")