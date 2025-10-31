from langchain.agents import create_agent
from langchain.agents.middleware import after_model
from langchain_core.tools import tool
from deepagents import CompiledSubAgent

from app.repositories.store_repository import StoreRepository
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger
from app.prompt.store_agent_prompt import get_store_agent_prompt
from app.services.token_tracking_service import track_subagent_direct_tokens

async def create_store_agent(repo: StoreRepository) -> CompiledSubAgent:
    """
    Store Agent - Handles store and branch queries.

    Scope:
    - Store information (name, code, location)
    - Branch management
    - Store-branch relationships

    Tables: store_master, branch

    Args:
        repo: StoreRepository instance (injected via DI)

    Returns:
        CompiledSubAgent ready for supervisor
    """
    logger.info("🤖 Creating Store Agent...")

    config = await repo.get_config()

    llm = LLMProviderFactory.create(
        provider_name=config["llm_provider"],
        model_name=config["model_name"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )

    # Define tool with @tool decorator
    @tool("store_dynamic_query")
    async def store_query(question: str) -> str:
        """
        Query store database using natural language

        This tool automatically:
        1. Converts your natural language question to SQL
        2. Handles joins across store and branch tables
        3. Executes the query safely
        4. Returns formatted results

        Use this tool to get store information like:
        - Store locations and addresses
        - Branch information
        - Store status and availability

        Args:
            question (str): Natural language question about stores

        Returns:
            str: Query results as formatted string
        
        Examples:
            "Berapa jumlah toko yang ada"
            "Cari toko dengan kode TPLG"
        """
        try:
            logger.info(f"[store_agent] Tool called: {question[:50]}")

            # Execute query via repository
            result = await repo.execute_query(question)

            logger.info(f"[store_agent] Tool completed successfully")
            return result
        
        except Exception as e:
            error_msg = f"Error querying store data: {str(e)}"
            logger.error(f"[store_agent] {error_msg}", exc_info=True)
            return error_msg

    # Middleware to track token usage
    @after_model
    async def track_tokens(request, response):
        """Track LLM token usage from store_agent"""
        try:
            usage_metadata = getattr(response, 'usage_metadata', None)
            if usage_metadata:
                input_tokens = usage_metadata.get('input_tokens', 0)
                output_tokens = usage_metadata.get('output_tokens', 0)
                total_tokens = usage_metadata.get('total_tokens', input_tokens + output_tokens)

                cache_read = usage_metadata.get('input_token_details', {}).get('cache_read', 0)
                reasoning = usage_metadata.get('output_token_details', {}).get('reasoning', 0)

                logger.info(f"📊 [store_agent] LLM tokens: {total_tokens}")

                track_subagent_direct_tokens(
                    agent_name="store_agent",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cache_read_tokens=cache_read,
                    reasoning_tokens=reasoning
                )
        except Exception as e:
            logger.error(f"[store_agent] Token tracking error: {e}")

        return response

    agent_graph = create_agent(
        llm,
        tools=[store_query],
        system_prompt=get_store_agent_prompt(),
        name="store_agent",
        middleware=[track_tokens]
    )

    # Wrap in CompiledSubAgent for deepagents
    compiled_subagent = CompiledSubAgent(
        name="store_agent",
        description=(
            "Store information specialist for handling store and branch queries. "
            "Use for: store details, branch information, locations, store counts."
        ),
        runnable=agent_graph
    )

    logger.info("✅ Store Agent created as CompiledSubAgent with dynamic_query tool")
    return compiled_subagent
