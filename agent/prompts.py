"""Prompt templates for the AI Agent — unified Chinese prompts."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个智能 AI 助手，可以使用工具来帮助用户完成任务。\n\n"
        "你可以使用以下工具：\n\n{tool_names}\n\n"
        "在需要时使用工具来回答用户的问题。如果你可以直接回答而不需要工具，请直接回答。"
        "始终保持友好和简洁。\n\n"
        "重要：你必须始终使用中文回复。你的所有思考、工具调用和最终答案都必须使用中文。",
    ),
    (
        "human",
        "对话历史：\n{chat_history}\n\n"
        "用户问题：\n{input}\n\n"
        "请逐步思考并使用适当的工具来获得最佳答案。用中文回复。",
    ),
    (
        "placeholder",
        "{agent_scratchpad}",
    ),
])