"""
OpenAI Provider Implementation
Supports: GPT-4, GPT-4-turbo, GPT-3.5-turbo, etc.
"""
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from app.llm.base_provider import BaseLLMProvider
from app.utils.logger import logger


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM Provider using LangChain"""

    def validate_config(self) -> bool:
        """
        Validate OpenAI configuration

        Raises:
            ValueError: If API key is missing
        """
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY in environment variables."
            )

        logger.info(f"✅ OpenAI Provider initialized | model={self.model_name}")
        return True

    def get_client(self) -> BaseChatModel:
        """
        Get or create ChatOpenAI client

        Returns:
            ChatOpenAI: LangChain OpenAI chat model
        """
        if self._client is None:
            self._client = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self.kwargs
            )
            logger.debug(f"ChatOpenAI client created | model={self.model_name}")

        return self._client