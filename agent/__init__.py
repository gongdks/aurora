"""AI Agent package — pure LangGraph Plan-and-Execute agent.

Layers:
    llm/                — Chat model (API-based, switch via .env)
    memory/             — Short-term sliding window + long-term SQLite storage
    tools/              — Tool ecosystem (calculator, search, file R/W, code exec, etc.)
    ui/                 — PyQt6 desktop GUI components
    utils/              — LangGraph subgraphs (tool executor, reflection, multi-agent, etc.)
    agent.py            — AgentSession core (pure LangGraph orchestration)
    graph_orchestrator.py — LangGraph Plan-and-Execute orchestrator
    config.py           — Unified config (.env → Settings)
"""

__version__ = "6.0.0"