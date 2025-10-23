"""
LLM Provider
Supports multiple providers: OpenAI, Anthropic, Google Gemini
"""
import os
from typing import Optional, Dict, Any
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.utils.logger import logger


class LLMProviderFactory:
    """
    Factory untuk create LLM instances dari berbagai providers
    menggunakan LangChain's unified init_chat_model
    """

    # Supported providers
    SUPPORTED_PROVIDERS = ["openai", "anthropic", "ollama", "google-genai"]

    @staticmethod
    def create(
        provider_name: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> BaseChatModel:
        """
        Create LLM instance using init_chat_model.

        Args:
            provider_name: Provider name (openai, anthropic, ollama, google-genai)
            model_name: Model name (gpt-4, claude-3-5-sonnet-latest, etc.)
            temperature: Temperature for generation (0.0 - 1.0)
            max_tokens: Maximum tokens for response
            **kwargs: Additional provider-specific parameters

        Returns:
            BaseChatModel instance

        Raises:
            ValueError: If provider not supported or API key missing
            Exception: If model initialization fails

        Examples:
            - # OpenAI
            - llm = LLMProviderFactory.create("openai", "gpt-4", temperature=0.5)

            - # Anthropic
            - llm = LLMProviderFactory.create("anthropic", "claude-3-5-sonnet-latest")

            - # Ollama (local)
            - llm = LLMProviderFactory.create("ollama", "llama3.2", base_url="http://localhost:11434")
        """

        provider_name = provider_name.lower().strip()

        # Validate provider
        if provider_name not in LLMProviderFactory.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Provider '{provider_name}' not supported. "
                f"Saat ini hanya support untuk {LLMProviderFactory.SUPPORTED_PROVIDERS}"
            )
        
        # Cek API keys (Kecuali untuk ollama)
        if provider_name != "ollama":
            LLMProviderFactory._validate_api_key(provider_name)
        
        logger.info(
            f"🤖 Membuat LLM | provider = {provider_name} | model = {model_name} | "
            f"temperature = {temperature} | max_tokens = {max_tokens}"
        )

        try:
            # Build parameters
            params = {
                "model": model_name,
                "model_provider": provider_name,
                "temperature": temperature
            }

            if max_tokens:
                params["max_tokens"] = max_tokens
            
            # Gabungkan kwargs tambahan kalau ada
            params.update(kwargs)

            # Buat model pake unified init_chat_model
            llm = init_chat_model(**params)

            logger.info(f"✅ LLM created successfully: {provider_name}/{model_name}")
            return llm
        
        except Exception as e:
            logger.error(f"❌ Gagal membuat LLM: {e}")
            raise Exception(f"Gagal untuk inisialisasi {provider_name}/{model_name}: {str(e)}")

    @staticmethod
    def _validate_api_key(provider_name: str) -> None:
        """
        Validate API Key yang dibutuhkan ada di environment.
        """
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google-genai": "GOOGLE_API_KEY"
        }

        env_var = key_mapping.get(provider_name)
        if not env_var:
            return # Tidak ada validasi yang diperlukan
        
        if not os.getenv(env_var):
            raise ValueError(
                f"Ga ada API KEY untuk provider {provider_name}."
                f"Mohon untuk set API KEY nya di environment variable: {env_var}"
            )
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> BaseChatModel:
        """
        Buat LLM dari agent config dictionary

        Args:
            config: Config dict with keys: llm_provider, model_name, temperature, max_tokens, config_metadata

        Returns:
            BaseChatModel instance

        Example:
            - config = await get_agent_config("supervisor", db)
            - llm = LLMProviderFactory.create_from_config(config)
        """
        # Ekstrak base params
        provider = config["llm_provider"]
        model = config["model_name"]
        temperature = config.get("temperature", 0.4)
        max_tokens = config.get("max_tokens")

        # Get extra params dari metadata
        extra_params = config.get("config_metadata", {})

        return LLMProviderFactory.create(
            provider_name=provider,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_params
        )
