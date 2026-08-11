"""AI Agent package — AutoGen-powered multi-agent agent.

Layers:
    llm/            — Chat model (remote API / local Ollama, switch via .env)
    memory/         — Short-term conversation sliding window
    tools/          — Tool ecosystem (calculator, search, file R/W, code exec, etc.)
    agent.py        — AgentSession core
    orchestrator.py — AutoGen GroupChat orchestrator (Planner + Executor + Verifier)
    executor.py     — Single-step ReAct executor (for simple queries)
    prompts.py      — Tool-calling agent prompt template
    config.py       — Unified config (.env → Settings)
"""

__version__ = "5.0.0"