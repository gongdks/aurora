"""LLM 工厂函数 —— 根据配置返回对应提供者。

延迟导入：只加载实际使用的依赖，不用的不会触发 importError。
"""

from agent.config import settings
from agent.llm.base import BaseLLMProvider


def create_llm() -> BaseLLMProvider:
    """根据 settings.LLM_PROVIDER 创建 LLM 提供者。

    Returns:
        对应的 BaseLLMProvider 实例。

    Raises:
        ValueError: 未知的 LLM_PROVIDER 值。
    """
    provider = settings.LLM_PROVIDER.strip().lower()

    if provider == "openai":
        from agent.llm.openai_compatible import OpenAICompatibleLLM
        return OpenAICompatibleLLM()

    if provider == "ollama":
        from agent.llm.ollama_llm import OllamaLLM
        return OllamaLLM()

    raise ValueError(
        f"未知的 LLM_PROVIDER '{provider}'，"
        f"可选值：openai, ollama"
    )
