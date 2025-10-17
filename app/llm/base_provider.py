"""
Base LLM Provider
Abstract base class untuk semua LLM providers
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.utils.logger import logger


class BaseLLMProvider(ABC):
    """
    Abstract base class untuk LLM providers
    Semua provider harus implement interface ini
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize LLM Provider

        Args:
            model_name: Model identifier (e.g., 'gpt-4', 'claude-3-opus')
            api_key: API key untuk provider (optional untuk Ollama)
            temperature: Sampling temperature (0.0 - 1.0)
            max_tokens: Maximum tokens untuk response
            **kwargs: Additional provider-specific parameters
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self._client: Optional[BaseChatModel] = None

        # Validate configuration saat initialization
        self.validate_config()

    @abstractmethod
    def get_client(self) -> BaseChatModel:
        """
        Get atau create LangChain chat model client

        Returns:
            BaseChatModel: LangChain chat model instance

        Note:
            Implementation harus support lazy initialization (create once, reuse)
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate provider configuration (API keys, model availability, etc)

        Returns:
            bool: True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def invoke(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
        """
        Synchronously invoke LLM with messages

        Args:
            messages: List of LangChain messages
            **kwargs: Additional parameters untuk invoke

        Returns:
            BaseMessage: LLM response
        """
        client = self.get_client()
        return client.invoke(messages, **kwargs)

    async def ainvoke(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
        """
        Asynchronously invoke LLM with messages

        Args:
            messages: List of LangChain messages
            **kwargs: Additional parameters untuk invoke

        Returns:
            BaseMessage: LLM response
        """
        client = self.get_client()
        return await client.ainvoke(messages, **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get provider and model information

        Returns:
            Dict with provider metadata
        """
        return {
            "provider": self.__class__.__name__.replace("Provider", "").lower(),
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
    
    def extract_token_usage(self, response: BaseMessage) -> dict:
        """
        Extract token usage from LLM response.
        Works for OpenAI, Anthropic, Gemini, and Ollama.
        
        Returns standardized format:
        {
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int
        }
        """
        usage = {}

        # Langchain stores usage in response_metadata
        if hasattr(response, 'response_metadata'):
            metadata = response.response_metadata

            # Different providers use different keys
            # OpenAI, Anthropic format
            if 'token_usage' in metadata:
                usage = metadata['token_usage']
            
            # Gemini, Ollama  format
            elif 'usage_metadata' in metadata:
                usage = metadata['usage_metadata']
            
            # Ollama format kadang attributenya cuma 'usage'
            elif 'usage' in metadata:
                usage = metadata['usage']

            # Fallback: Mencoba extract dari top-level metadata
            else:
                # Some providers put tokens directly in metadata
                if 'prompt_tokens' in metadata or 'input_tokens' in metadata:
                    usage = metadata

        # Normalize to standard format
        prompt_tokens = (
            usage.get('prompt_tokens', 0) or
            usage.get('input_tokens', 0) or
            usage.get('prompt_token_count', 0) or
            0
        )

        completion_tokens = (
            usage.get('completion_tokens', 0) or
            usage.get('output_tokens', 0) or
            usage.get('completion_token_count', 0) or
            usage.get('candidates_token_count', 0) or
            0
        )

        total_tokens = (
            usage.get('total_tokens', 0) or
            usage.get('total_token_count', 0) or
            (prompt_tokens + completion_tokens)
        )

        # Normalize to standard format
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }

    def __repr__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(model={self.model_name}, temperature={self.temperature})"