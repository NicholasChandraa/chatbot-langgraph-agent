from langchain.agents import create_agent
from langchain.agents.middleware import after_model
from langchain_core.tools import tool
from deepagents import CompiledSubAgent

from app.repositories.sales_repository import SalesRepository
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger
from app.prompt.sales_agent_prompt import get_sales_agent_prompt
from app.services.token_tracking_service import track_subagent_direct_tokens

async def create_sales_agent(repo: SalesRepository) -> CompiledSubAgent:
    """
    Sales Agent - Handles sales analytics queries.

    Scope:
    - Sales revenue and trends
    - Top selling products
    - Store performance
    - Sales by date/period

    Tables: store_daily_single_item, product, store_master, branch

    Args:
        repo: SalesRepository instance (injected via DI)

    Returns:
        CompiledSubAgent ready for supervisor
    """
    logger.info("🤖 Creating Sales Agent...")

    config = await repo.get_config()

    llm = LLMProviderFactory.create(
        provider_name=config["llm_provider"],
        model_name=config["model_name"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )

    # Define tool with @tool decorator
    @tool("sales_dynamic_query")
    async def sales_query(question: str) -> str:
        """
        Query sales analytics database using natural language

        This tool automatically:
        1. Converts your natural langauge question to SQL
        2. Handles complex joins across sales, product, store, branch tables
        3. Executes the query safely
        4. Returns formatted results

        Use this tool to get sales analytics like:
        - Total revenue by date/period
        - Top selling products
        - Store performance comparison
        - Sales trends and patterns

        Args:
            question (str): Natural language question about sales

        Returns:
            str: Query results as formatted string
        
        Examples:
            "Berapa total penjualan kemarin?"
            "Tampilkan 5 produk terlaris minggu ini"
            "Bandingkan performa toko FS Palembang vs FS Bandung"
        """
        try:
            logger.info(f"[sales_agent] Tool called: {question[:50]}...")

            # Execute query via repository
            result = await repo.execute_query(question)

            logger.info(f"[sales_agent] Tool completed successfully")
            return result
        
        except Exception as e:
            error_msg = f"Error querying sales data: {str(e)}"
            logger.error(f"[sales_agent] {error_msg}", exc_info=True)
            return error_msg

    # Middleware to track token usage
    @after_model
    async def track_tokens(request, response):
        """Track LLM token usage from sales_agent"""
        try:
            usage_metadata = getattr(response, 'usage_metadata', None)
            if usage_metadata:
                input_tokens = usage_metadata.get('input_tokens', 0)
                output_tokens = usage_metadata.get('output_tokens', 0)
                total_tokens = usage_metadata.get('total_tokens', input_tokens + output_tokens)

                cache_read = usage_metadata.get('input_token_details', {}).get('cache_read', 0)
                reasoning = usage_metadata.get('output_token_details', {}).get('reasoning', 0)

                logger.info(f"📊 [sales_agent] LLM tokens: {total_tokens}")

                track_subagent_direct_tokens(
                    agent_name="sales_agent",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cache_read_tokens=cache_read,
                    reasoning_tokens=reasoning
                )
        except Exception as e:
            logger.error(f"[sales_agent] Token tracking error: {e}")

        return response

    agent_graph = create_agent(
        llm,
        tools=[sales_query],
        system_prompt=get_sales_agent_prompt(),
        name="sales_agent",
        middleware=[track_tokens]
    )

    # Wrap in CompiledSubAgent for deepagents
    compiled_subagent = CompiledSubAgent(
        name="sales_agent",
        description=(
            "Sales analytics specialist for handling sales-related queries. "
            "Use for: sales revenue, trends, top products, performance analysis, sales reports."
        ),
        runnable=agent_graph
    )

    logger.info("✅ Sales Agent created as CompiledSubAgent with dynamic_query tool")
    return compiled_subagent