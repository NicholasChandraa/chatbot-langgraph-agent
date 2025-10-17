"""
Ollama Provider Implementation
Supports: Local LLMs (Llama 3, Mistral, Qwen, etc.)
"""
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel

from app.llm.base_provider import BaseLLMProvider
from app.utils.logger import logger


class OllamaProvider(BaseLLMProvider):
    """Ollama Local LLM Provider using LangChain"""

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Ollama Provider

        Args:
            model_name: Ollama model name (e.g., 'llama3.1:8b', 'mistral')
            base_url: Ollama server URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens (mapped to num_predict)
            **kwargs: Additional parameters
        """
        self.base_url = base_url
        super().__init__(
            model_name=model_name,
            api_key=None,  # Ollama doesn't need API key
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    def validate_config(self) -> bool:
        """
        Validate Ollama configuration

        Note:
            Ollama doesn't require API key, just checks base_url
        """
        logger.info(
            f"✅ Ollama Provider initialized | "
            f"model={self.model_name} | base_url={self.base_url}"
        )
        return True

    def get_client(self) -> BaseChatModel:
        """
        Get or create ChatOllama client

        Returns:
            ChatOllama: LangChain Ollama chat model
        """
        if self._client is None:
            self._client = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens,  # Ollama uses num_predict instead of max_tokens
                **self.kwargs
            )
            logger.debug(
                f"ChatOllama client created | "
                f"model={self.model_name} | base_url={self.base_url}"
            )

        return self._client