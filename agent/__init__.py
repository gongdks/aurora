"""AI Agent package — pure LangGraph Plan-and-Execute agent.

Layers:
    llm/                — Chat model (remote API / local Ollama, switch via .env)
    memory/             — Short-term sliding window + long-term SQLite storage
    tools/              — Tool ecosystem (calculator, search, file R/W, code exec, etc.)
    ui/                 — PyQt6 desktop GUI components
    agent.py            — AgentSession core (pure LangGraph orchestration)
    graph_orchestrator.py — LangGraph Plan-and-Execute orchestrator
    runner.py           — Shared ReAct executor (tool-calling, streaming, tracking)
    prompts.py          — Tool-calling agent prompt template
    config.py           — Unified config (.env → Settings)
"""

__version__ = "6.0.0"