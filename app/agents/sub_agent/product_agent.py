from langchain.agents import create_agent
from langchain.agents.middleware import after_model
from langchain_core.tools import tool
from deepagents import CompiledSubAgent

from app.repositories.product_repository import ProductRepository
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger
from app.prompt.product_agent_prompt import get_product_agent_prompt
from app.services.token_tracking_service import track_subagent_direct_tokens

async def create_product_agent(repo: ProductRepository) -> CompiledSubAgent:
    """
    Product Agent - Handles product-related queries.

    Scope:
    - Product information
    - Product categories
    - Product availability

    Tables: product only

    Args:
        repo: ProductRepository instance (injected via DI)

    Returns:
        CompiledSubAgent ready to be used by supervisor
    """
    logger.info("🤖 Creating Product Agent...")

    # Load config from database
    config = await repo.get_config()

    # Create LLM for ReAct agent
    llm = LLMProviderFactory.create(
        provider_name=config["llm_provider"],
        model_name=config["model_name"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )

    # Define tool using @tool decorator
    @tool("product_dynamic_query")
    async def product_query(question: str) -> str:
        """
        Query product database using natural language.

        This tool automatically:
        1. Converts your natural language question to SQL
        2. Executes the query safely
        3. Returns formatted results

        Use this tool to get product information like:
        - Product names, PLU codes
        - Product search by keyword
        - Product availability
    
        Args:
            question (str): Natural language question about products

        Returns:
            str: Query results as formatted string
        
        Examples:
            "Tampilkan semua produk coklat"
            "Cari produk dengan PLU 000000906"
        """
        try:
            logger.info(f"[product_agent] Tool called: {question[:50]}...")

            # Execute query via repository (repository handles caching, metrics, etc.)
            result = await repo.execute_query(question)

            logger.info(f"[product_agent] Tool completed successfully")
            return result
        
        except Exception as e:
            error_msg = f"Error querying products: {str(e)}"
            logger.error(f"[product_agent] {error_msg}", exc_info=True)
            return error_msg

    # Middleware to track token usage
    @after_model
    async def track_tokens(request, response):
        """Track LLM token usage from product_agent"""
        try:
            usage_metadata = getattr(response, 'usage_metadata', None)
            if usage_metadata:
                input_tokens = usage_metadata.get('input_tokens', 0)
                output_tokens = usage_metadata.get('output_tokens', 0)
                total_tokens = usage_metadata.get('total_tokens', input_tokens + output_tokens)

                cache_read = usage_metadata.get('input_token_details', {}).get('cache_read', 0)
                reasoning = usage_metadata.get('output_token_details', {}).get('reasoning', 0)

                logger.info(f"📊 [product_agent] LLM tokens: {total_tokens}")

                track_subagent_direct_tokens(
                    agent_name="product_agent",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cache_read_tokens=cache_read,
                    reasoning_tokens=reasoning
                )
        except Exception as e:
            logger.error(f"[product_agent] Token tracking error: {e}")

        return response

    # Create React Agent with single tool
    agent_graph = create_agent(
        llm,
        tools=[product_query],
        system_prompt=get_product_agent_prompt(),
        name="product_agent",
        middleware=[track_tokens]
    )

    # Wrap in CompiledSubAgent for deepagents
    compiled_subagent = CompiledSubAgent(
        name="product_agent",
        description=(
            "Product specialist for handling product-related queries. "
            "Use for: product information, pricing, PLU codes, product categories, availability."
        ),
        runnable=agent_graph
    )

    logger.info("✅ Product Agent created as CompiledSubAgent with dynamic_query tool")
    return compiled_subagent
