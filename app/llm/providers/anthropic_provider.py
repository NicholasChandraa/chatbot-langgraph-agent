"""
Anthropic Provider Implementation
Supports: Claude 3 (Opus, Sonnet, Haiku), Claude 3.5, etc.
"""
from typing import Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from app.llm.base_provider import BaseLLMProvider
from app.utils.logger import logger


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM Provider using LangChain"""

    def validate_config(self) -> bool:
        """
        Validate Anthropic configuration

        Raises:
            ValueError: If API key is missing
        """
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. "
                "Set ANTHROPIC_API_KEY in environment variables."
            )

        logger.info(f"✅ Anthropic Provider initialized | model={self.model_name}")
        return True

    def get_client(self) -> BaseChatModel:
        """
        Get or create ChatAnthropic client

        Returns:
            ChatAnthropic: LangChain Anthropic chat model
        """
        if self._client is None:
            # Claude requires max_tokens to be set
            max_tokens = self.max_tokens or 4096

            self._client = ChatAnthropic(
                model=self.model_name,
                anthropic_api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=max_tokens,
                **self.kwargs
            )
            logger.debug(f"ChatAnthropic client created | model={self.model_name}")

        return self._client