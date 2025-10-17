"""
Test Script - LLM Provider System
Test connectivity ke berbagai LLM providers
"""
import asyncio
from langchain_core.messages import HumanMessage

from app.llm.provider_factory import get_llm, LLMProviderFactory
from app.config.settings import get_settings
from app.utils.logger import logger


async def test_provider(provider_name: str):
    """
    Test specific LLM provider

    Args:
        provider_name: Provider name (openai, anthropic, gemini, ollama)
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {provider_name.upper()} Provider")
        logger.info(f"{'='*60}")

        # Create provider
        provider = LLMProviderFactory.create(provider_name=provider_name)

        # Get model info
        info = provider.get_model_info()
        logger.info(f"Provider Info: {info}")

        # Test message
        test_message = "Halo! Sebutkan 3 rasa donut yang populer."
        messages = [HumanMessage(content=test_message)]

        logger.info(f"Sending test message: {test_message}")

        # Invoke LLM (async)
        response = await provider.ainvoke(messages)

        logger.info(f"✅ Response received:")
        logger.info(f"{response.content}")

        return True

    except Exception as e:
        logger.error(f"❌ {provider_name.upper()} Provider failed: {e}")
        return False


async def test_all_providers():
    """Test all configured providers"""
    settings = get_settings()

    logger.info(f"\n{'#'*60}")
    logger.info(f"LLM Provider System Test")
    logger.info(f"Default Provider: {settings.DEFAULT_LLM_PROVIDER}")
    logger.info(f"{'#'*60}\n")

    # Get available providers
    available = LLMProviderFactory.get_available_providers()
    logger.info(f"Available Providers: {', '.join(available)}\n")

    results = {}

    # Test default provider (Gemini)
    if settings.GEMINI_API_KEY:
        results['gemini'] = await test_provider('gemini')
    else:
        logger.warning("⚠️  Gemini API key not set, skipping...")

    # Test OpenAI (if configured)
    if settings.OPENAI_API_KEY:
        results['openai'] = await test_provider('openai')
    else:
        logger.warning("⚠️  OpenAI API key not set, skipping...")

    # Test Anthropic (if configured)
    if settings.ANTHROPIC_API_KEY:
        results['anthropic'] = await test_provider('anthropic')
    else:
        logger.warning("⚠️  Anthropic API key not set, skipping...")

    # Test Ollama (if running locally)
    try:
        results['ollama'] = await test_provider('ollama')
    except Exception as e:
        logger.warning(f"⚠️  Ollama not available: {e}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    for provider, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{provider.upper()}: {status}")


async def test_quick():
    """Quick test - hanya test default provider"""
    settings = get_settings()
    provider_name = settings.DEFAULT_LLM_PROVIDER

    logger.info(f"🚀 Quick Test: {provider_name.upper()} Provider\n")

    success = await test_provider(provider_name)

    if success:
        logger.info(f"\n✅ {provider_name.upper()} Provider is working!")
    else:
        logger.error(f"\n❌ {provider_name.upper()} Provider failed!")


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            # Test all providers
            asyncio.run(test_all_providers())
        else:
            # Test specific provider
            provider = sys.argv[1].lower()
            asyncio.run(test_provider(provider))
    else:
        # Default: quick test
        asyncio.run(test_quick())