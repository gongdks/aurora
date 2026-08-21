with open("final_verify.txt", "w") as f:
    try:
        from agent.graph_orchestrator import GraphOrchestrator
        f.write("OK: GraphOrchestrator imported successfully\n")
    except Exception as e:
        import traceback
        f.write(f"FAIL GraphOrchestrator: {e}\n")
        traceback.print_exc(file=f)