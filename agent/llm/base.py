"""LLM 抽象接口。"""

from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel


class BaseLLMProvider(ABC):
    """抽象 LLM 提供者。

    每个实现只需返回一个 LangChain 兼容的 chat model 实例，
    agent 层不关心底层是远端 API 还是本地模型。
    """

    @abstractmethod
    def get_model(self) -> BaseChatModel:
        """返回已配置好的 LangChain chat model 实例。"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """人类可读的模型标识，用于日志和显示。"""
        ...
