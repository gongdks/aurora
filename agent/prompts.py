"""Prompt templates for the AI Agent."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI Agent that can use tools to assist the user.\n\nYou have access to the following tools:\n\n{tool_names}\n\nUse tools when needed to answer the user's question. If you can answer\ndirectly without tools, do so. Always be helpful and concise.\n\nIMPORTANT: You MUST always respond in Chinese (中文). All your thoughts, tool calls, and final answers must be in Chinese."),
    ("human", "Conversation history:\n{chat_history}\n\nUser's question:\n{input}\n\nThink step by step and use the appropriate tools to get the best answer. Respond in Chinese."),
    ("placeholder", "{agent_scratchpad}"),
])