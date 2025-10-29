import os

from deepagents import create_deep_agent

from app.agents.sub_agent.product_agent import create_product_agent
from app.agents.sub_agent.sales_agent import create_sales_agent
from app.agents.sub_agent.store_agent import create_store_agent

from app.repositories.repository_container import RepositoryContainer
from app.config.agent_config.agent_config_manager import get_agent_config
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger
from app.prompt.supervisor_prompt import get_supervisor_base_prompt, inject_user_context

async def create_supervisor_agent(
    repos: RepositoryContainer, 
    checkpointer=None, 
    store=None, 
    user_context: str = ""
):
    """
    Create supervisor that manages specialized business intelligence agents using DeepAgents.

    The supervisor delegates tasks to appropriate subagents based on content:
    - Product questions -> product_agent
    - Sales questions -> sales_agent
    - Store questions -> store_agent

    Args:
        db: Database session for loading agent configs
        checkpointer: Optional checkpointer for persistent memory
        store: Optional store for long-term memory
        user_context: User context string from long-term memory

    Returns:
        Compiled deep agent ready to process queries
    """

    logger.info("☑️ Creating Supervisor Agent with DeepAgents...")

    # Load supervisor config
    # Note: We use product repo temporarily for supervisor config
    # TODO: Consider creating dedicated SupervisorRepository if supervisor needs its own data access
    supervisor_config = await repos.supervisor.get_config()

    # Create supervisor LLM
    supervisor_llm = LLMProviderFactory.create_from_config(supervisor_config)

    logger.info(f"✅ Supervisor LLM created: {supervisor_config['model_name']}")
    
    # panggil sub agent
    product_agent = await create_product_agent(repos.product)
    sales_agent = await create_sales_agent(repos.sales)
    store_agent = await create_store_agent(repos.store)

    subagents = [product_agent, sales_agent, store_agent]

    # Define subagents as dictionaries (DeepAgents pattern)
    logger.info("📦 Defining subagents...")

    # Build base system prompt
    base_prompt = get_supervisor_base_prompt()
    system_prompt = inject_user_context(base_prompt, user_context)

    supervisor_agent = create_deep_agent(
        model=supervisor_llm,
        subagents=subagents,
        checkpointer=checkpointer,
        store=store,
        system_prompt=system_prompt,
    )

    memory_status = "with checkpointer" if checkpointer else "without checkpointer"
    logger.info(f"✅ Supervisor Agent Created ({memory_status})")

    return supervisor_agent
