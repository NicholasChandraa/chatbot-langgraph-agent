from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection.sql_database import get_sql_database
from app.config.agent_config.agent_config_manager import get_agent_config
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger

async def create_sales_agent(db: AsyncSession):
    """
    Sales Agent - Handles sales analytics queries.

    Scope:
    - Sales revenue and trends
    - Top selling products
    - Store performance
    - Sales by date/period

    Tables: store_daily_single_item, product, store
    """
    logger.info("🤖 Creating Sales Agent...")

    config = await get_agent_config("sales_agent", db)

    llm = LLMProviderFactory.create(
        provider_name=config["llm_provider"],
        model_name=config["model_name"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )

    # Sales needs multiple tables for JOINs
    sql_db = get_sql_database(
        include_tables=[
            "store_daily_single_item",
            "product",
            "store",
            "branch"
        ]
    )

    toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)

    agent = create_react_agent(
        llm,
        tools=toolkit.get_tools(),
        prompt=(
            "You are a Sales Analytics Specialist for a donut shop chain.\n\n"
            
            "YOUR EXPERTISE:\n"
            "- Sales revenue analysis and trends\n"
            "- Top selling products identification\n"
            "- Store performance comparison\n"
            "- Sales forecasting and insights\n\n"
            
            "DATABASE CONTEXT:\n"
            "- Main table: store_daily_single_item (sales transactions)\n"
            "- Key columns:\n"
            "  * qty_sales: Quantity sold (units)\n"
            "  * rp_sales: Revenue in Indonesian Rupiah\n"
            "  * date: Transaction date (use CAST(date AS DATE) for comparisons)\n"
            "- Always JOIN with product table for product names\n"
            "- Always JOIN with store table for store names\n\n"
            
            "QUERY GUIDELINES:\n"
            "- Use SUM() for total calculations\n"
            "- Use GROUP BY for aggregations (by date, product, store)\n"
            "- Use ORDER BY ... LIMIT for top N results\n"
            "- Filter out zero sales: WHERE qty_sales > 0 OR rp_sales > 0\n\n"
            
            "RESPONSE FORMAT:\n"
            "- Format currency as: Rp 1.500.000 (with thousand separator)\n"
            "- Include product/store names, not just IDs\n"
            "- Provide context and insights, not just numbers\n"
            "- For trends, mention time period clearly"
        )
    )

    logger.info("✅ Sales Agent created")
    return agent