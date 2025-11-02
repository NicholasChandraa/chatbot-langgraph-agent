from langchain.agents import create_agent
from langchain_core.tools import tool
from deepagents import CompiledSubAgent

from app.repositories.store_repository import StoreRepository
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger
from app.prompt.store_agent_prompt import get_store_agent_prompt
from app.services.token_tracking_service import create_token_tracking_middleware

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

    # Create token tracking middleware using factory
    track_tokens = create_token_tracking_middleware("store_agent")

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
