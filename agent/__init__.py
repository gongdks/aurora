"""AI Agent package — LangGraph + AutoGen hybrid multi-agent agent.

Layers:
    llm/                — Chat model (remote API / local Ollama, switch via .env)
    memory/             — Short-term conversation sliding window
    tools/              — Tool ecosystem (calculator, search, file R/W, code exec, etc.)
    agent.py            — AgentSession core (mode switcher: "graph" / "autogen")
    graph_orchestrator.py — LangGraph + AutoGen hybrid orchestrator (recommended)
    orchestrator.py     — AutoGen GroupChat orchestrator (legacy, still available)
    executor.py         — Single-step ReAct executor (for simple queries)
    prompts.py          — Tool-calling agent prompt template
    config.py           — Unified config (.env → Settings)
"""

__version__ = "5.0.0"