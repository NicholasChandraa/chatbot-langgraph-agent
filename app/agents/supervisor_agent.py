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
from app.agents.tools.memory_tools import (
    save_user_info,
    save_preference,
    remember_fact,
    recall_facts,
    recall_preferences,
    Context
)

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

    Also provides long-term memory tools for user personalization.
    
    Args:
        repos: Repository container for database access
        checkpointer: Optional checkpointer for conversation memory (short-term)
        store: Optional store for long-term memory (cross-session user data)
        user_context: Pre-loaded user context string to inject into prompt
        
    Returns:
        Compiled deep agent ready to process queries
    """

    logger.info("☑️ Creating Supervisor Agent...")

    # Load supervisor config
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

    # Build system prompt
    base_prompt = get_supervisor_base_prompt()
    system_prompt = inject_user_context(base_prompt, user_context)
    
    # Memory tools (only if store available)
    memory_tools = []
    if store:
        logger.info("✅ Store available - enabling long-term memory tools")
        memory_tools = [
            save_user_info,
            save_preference,
            remember_fact,
            recall_facts,
            recall_preferences
        ]
    else:
        logger.warning("⚠️ Store not available - long-term memory disabled")

    # Create supervisor agent
    supervisor_agent = create_deep_agent(
        model=supervisor_llm,
        subagents=subagents,
        tools=memory_tools,
        checkpointer=checkpointer,
        store=store,
        context_schema=Context,  # Required for tools to access user_id
        system_prompt=system_prompt,
    )

    memory_status = "with checkpointer" if checkpointer else "without checkpointer"
    store_status = "with persistent store" if store else "without store"
    tools_status = f"with {len(memory_tools)} memory tools" if memory_tools else "without memory tools"

    logger.info(f"✅ Supervisor Agent Created ({memory_status}, {store_status}, {tools_status})")

    return supervisor_agent
