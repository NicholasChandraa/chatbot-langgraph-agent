"""
LLM Provider Factory
Factory pattern untuk create LLM providers berdasarkan configuration
"""
from typing import Optional, Dict, Type
from functools import lru_cache

from app.llm.base_provider import BaseLLMProvider
from app.llm.providers import (
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
)
from app.config.settings.settings import get_settings
from app.utils.logger import logger


class LLMProviderFactory:
    """
    Factory untuk create LLM providers
    Automatically selects provider berdasarkan configuration
    """

    # Registry: mapping provider name ke provider class
    _providers: Dict[str, Type[BaseLLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "ollama": OllamaProvider,
    }

    @classmethod
    def create(
        cls,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        Create LLM provider instance and return LangChain chat model

        Args:
            provider_name: Provider name (openai, anthropic, gemini, ollama)
                          If None, uses DEFAULT_LLM_PROVIDER from settings
            model_name: Model name to use
                       If None, uses default model for the provider from settings
            temperature: Sampling temperature
                        If None, uses default from settings or provider default
            **kwargs: Additional provider-specific parameters

        Returns:
            BaseChatModel: LangChain chat model instance (ready for use with agents/toolkits)

        Raises:
            ValueError: If provider is unknown or configuration is invalid
        """
        settings = get_settings()

        # Use default provider if not specified
        if provider_name is None:
            provider_name = settings.DEFAULT_LLM_PROVIDER.lower()

        # Validate provider exists
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown LLM provider: '{provider_name}'. "
                f"Available providers: {available}"
            )

        # Get provider configuration
        provider_config = cls._get_provider_config(
            provider_name, model_name, temperature, **kwargs
        )

        # Create provider instance
        provider_class = cls._providers[provider_name]
        provider = provider_class(**provider_config)

        logger.info(
            f"🤖 LLM Provider created | "
            f"provider={provider_name} | "
            f"model={provider_config.get('model_name', 'default')}"
        )

        # Return the actual LangChain chat model client
        return provider.get_client()

    @classmethod
    def _get_provider_config(
        cls,
        provider_name: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """
        Get configuration untuk specific provider dari settings

        Args:
            provider_name: Provider name
            model_name: Optional model override
            temperature: Optional temperature override
            **kwargs: Additional parameters

        Returns:
            Dict: Provider configuration
        """
        settings = get_settings()

        # Base configuration
        config = {}

        # Provider-specific configuration
        if provider_name == "openai":
            config = {
                "model_name": model_name or settings.OPENAI_MODEL,
                "api_key": settings.OPENAI_API_KEY,
                "temperature": temperature or 0.4,
            }

        elif provider_name == "anthropic":
            config = {
                "model_name": model_name or settings.ANTHROPIC_MODEL,
                "api_key": settings.ANTHROPIC_API_KEY,
                "temperature": temperature or 0.4,
            }

        elif provider_name == "gemini":
            config = {
                "model_name": model_name or settings.GEMINI_MODEL,
                "api_key": settings.GEMINI_API_KEY,
                "temperature": temperature or 0.4,
            }

        elif provider_name == "ollama":
            config = {
                "model_name": model_name or settings.OLLAMA_MODEL,
                "base_url": settings.OLLAMA_BASE_URL,
                "temperature": temperature or 0.4,
            }

        # Merge with additional kwargs
        config.update(kwargs)

        return config

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """
        Get list of available provider names

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLMProvider]):
        """
        Register custom provider (untuk extensibility)

        Args:
            name: Provider name
            provider_class: Provider class (must inherit from BaseLLMProvider)
        """
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError(
                f"Provider class must inherit from BaseLLMProvider, "
                f"got {provider_class.__name__}"
            )

        cls._providers[name.lower()] = provider_class
        logger.info(f"✅ Custom provider registered | name={name}")


@lru_cache()
def get_llm(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Get cached LLM provider instance (singleton pattern)

    Args:
        provider_name: Provider name
        model_name: Model name
        **kwargs: Additional parameters

    Returns:
        BaseLLMProvider: Cached provider instance

    Note:
        Uses @lru_cache untuk reuse provider instances.
        Useful untuk avoid repeated initialization.
    """
    return LLMProviderFactory.create(provider_name, model_name, **kwargs)