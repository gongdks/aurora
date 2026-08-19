"""本地 Ollama 提供者。

完全离线运行，不依赖外部 API。
"""

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from agent.config import settings
from agent.llm.base import BaseLLMProvider


class OllamaLLM(BaseLLMProvider):
    """通过 ChatOllama 连接本地 Ollama 服务。"""

    def __init__(self) -> None:
        self._model = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.OLLAMA_TEMPERATURE,
            streaming=True,
            timeout=settings.LLM_TIMEOUT_SEC,
        )

    def get_model(self) -> BaseChatModel:
        return self._model

    @property
    def model_name(self) -> str:
        return f"ollama:{settings.OLLAMA_MODEL}"