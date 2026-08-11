"""OpenAI 兼容 API 提供者。

覆盖 DeepSeek、GPT、Claude 及任何兼容 OpenAI 协议的远端服务。
"""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from agent.config import settings
from agent.llm.base import BaseLLMProvider


class OpenAICompatibleLLM(BaseLLMProvider):
    """通过 ChatOpenAI 连接任何 OpenAI 兼容 API。

    支持：DeepSeek、OpenAI GPT、Claude（通过代理）等。
    """

    def __init__(self) -> None:
        self._model = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            temperature=settings.OPENAI_TEMPERATURE,
            streaming=True,
        )

    def get_model(self) -> BaseChatModel:
        return self._model

    @property
    def model_name(self) -> str:
        return f"openai-compatible:{settings.OPENAI_MODEL}"
