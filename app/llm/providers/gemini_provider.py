"""
Google Gemini Provider Implementation
Supports: Gemini Pro, Gemini Ultra, Gemini Flash, etc.
"""
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from app.llm.base_provider import BaseLLMProvider
from app.utils.logger import logger


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM Provider using LangChain"""

    def validate_config(self) -> bool:
        """
        Validate Gemini configuration

        Raises:
            ValueError: If API key is missing
        """
        if not self.api_key:
            raise ValueError(
                "Google Gemini API key is required. "
                "Set GEMINI_API_KEY in environment variables."
            )

        logger.info(f"✅ Gemini Provider initialized | model={self.model_name}")
        return True

    def get_client(self) -> BaseChatModel:
        """
        Get or create ChatGoogleGenerativeAI client

        Returns:
            ChatGoogleGenerativeAI: LangChain Gemini chat model
        """
        if self._client is None:
            self._client = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                **self.kwargs
            )
            logger.debug(f"ChatGoogleGenerativeAI client created | model={self.model_name}")

        return self._client