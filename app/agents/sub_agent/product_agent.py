from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection.sql_database import get_sql_database
from app.config.agent_config.agent_config_manager import get_agent_config
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger

async def create_product_agent(db: AsyncSession):
    """
    Product Agent - Handles product-related queries.

    Scope:
    - Product information
    - Product categories
    - Product availability
    
    Tables: product only
    """
    logger.info("🤖 Creating Product Agent...")

    # Load config from database
    config = await get_agent_config("product_agent", db)

    # Create LLM
    llm = LLMProviderFactory.create(
        provider_name=config["llm_provider"],
        model_name=config["model_name"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )

    # Create SQL Database with specific tables
    sql_db = get_sql_database(
        include_tables=["product"]
    )

    # Create toolkit
    toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)

    # Create React Agent
    agent = create_react_agent(
        llm,
        tools=toolkit.get_tools(),
        prompt=(
            "You are a Product Information Specialist for a Mister Donut Store.\n\n"
            
            "YOUR EXPERTISE:\n"
            "- Product names, prices, PLU codes\n"
            "- Product categories and variants\n"
            "- Product availability\n\n"

            "DATABASE CONTEXT:\n"
            "- Table: product\n"
            "- Product names are in UPPERCASE (e.g., 'GLAZED DONUT', 'ICED CHOCOLATE (REG)')\n"
            "- PLU codes are strings with leading zeros (e.g., '01040109', '00000220')\n"
            "- Prices are in Indonesian Rupiah (IDR) without decimals\n\n"
            
            "SEARCH GUIDELINES:\n"
            "- Use ILIKE with wildcards for case-insensitive search: plu_name ILIKE '%glazed%'\n"
            "- For multiple matches, show all relevant products\n"
            "- Always include PLU code in your response\n\n"
            
            "RESPONSE FORMAT:\n"
            "- Format prices as: Rp 15.000 (use thousand separator)\n"
            "- Be concise but informative\n"
            "- If product not found, suggest similar products"
        ),
        name="product_agent"
    )

    logger.info("✅ Product Agent created")
    return agent
