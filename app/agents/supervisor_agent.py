from langgraph_supervisor import create_supervisor
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.sub_agent.product_agent import create_product_agent
from app.agents.sub_agent.sales_agent import create_sales_agent
from app.agents.sub_agent.store_agent import create_store_agent

from app.config.agent_config.agent_config_manager import get_agent_config
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger

async def create_supervisor_agent(db: AsyncSession):
    """
    Create supervisor that manages specialized business intelligence agents.

    The supervisor routes questions to appropriate agents based on content:
    - Product questions -> Product Agent
    - Sales questions -> Sales Agent
    - Store questions -> Store Agent

    Args:
        db: Database session for loading agent configs
    
    Returns:
        Compiled supervisor graph ready to process queries
    """

    logger.info("☑️ Createing Supervisor Agent...")

    # Create worker agents
    product_agent = await create_product_agent(db)
    sales_agent = await create_sales_agent(db)
    store_agent = await create_store_agent(db)

    # Load supervisor config
    config = await get_agent_config("supervisor", db)

    # Create supervisor LLM
    supervisor_llm = LLMProviderFactory.create(
        provider_name=config["llm_provider"],
        model_name=config["model_name"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )

    # Create supervisor using built-in function
    supervisor = create_supervisor(
        model=supervisor_llm,
        agents=[product_agent, sales_agent, store_agent],
        prompt=(
            "You are an Intelligence Supervisor managing specialized agents for a Mister Donut Store.\n\n"

            "AVAILABLE AGENTS:\n"
            "1. product_agent\n"
            "- Handles: Product information, pricing, PLU codes, categories\n"
            "- Keywords: product, produk, harga, price, PLU, donut, beverage, cakes, snack\n"
            "- Examples: 'Berapa harga donut glazed?', 'Apakah ada donut coklat?'\n\n"

            "2. sales_agent\n"
            "   - Handles: Sales analytics, revenue, trends, top products, store performance\n"
            "   - Keywords: sales, penjualan, revenue, pendapatan, terlaris, best seller, trend, analisis\n"
            "   - Examples: 'Berapa total penjualan hari ini?', 'Produk apa yang paling laris?'\n\n"
            
            "3. store_agent\n"
            "   - Handles: Store information, branch details, locations\n"
            "   - Keywords: store, toko, branch, cabang, lokasi, location, outlet\n"
            "   - Examples: 'Di mana toko Flagship BSD?', 'Berapa jumlah toko aktif?'\n\n"
        
            "ROUTING RULES:\n"
            "- Analyze the user's question carefully\n"
            "- Choose the MOST RELEVANT agent based on the primary intent\n"
            "- For questions needing multiple domains (e.g., 'Which store sells the most glazed donuts?'):\n"
            "  * Start with the agent that handles the PRIMARY data (sales_agent for sales data)\n"
            "  * That agent can access related tables (product, store) via JOINs\n"
            "- Only route to ONE agent at a time\n"
            "- If the agent's answer is incomplete, you can route to another agent\n\n"
            
            "IMPORTANT:\n"
            "- DO NOT attempt to answer questions yourself\n"
            "- ALWAYS delegate to an agent\n"
            "- Let agents use their SQL tools - they have the data\n"
            "- When answer is complete, finish the conversation\n\n"
            
            "LANGUAGE:\n"
            "- User may ask in Indonesian or English\n"
            "- Agents will respond in the same language as the question"
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
        supervisor_name="supervisor_agent"
    ).compile()

    logger.info("✅ Supervisor Agent Created")

    return supervisor