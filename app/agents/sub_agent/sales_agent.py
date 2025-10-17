from langgraph.prebuilt import create_react_agent
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.tools.dynamic_query_tool import create_dynamic_query_tool
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

    Tables: store_daily_single_item, product, store, branch
    """
    logger.info("🤖 Creating Sales Agent...")

    config = await get_agent_config("sales_agent", db)

    llm = LLMProviderFactory.create(
        provider_name=config["llm_provider"],
        model_name=config["model_name"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )

    # Create dynamic query tool (with multiple tables for analytics)
    dynamic_query_tool = create_dynamic_query_tool(
        db=db,
        tables=["store_daily_single_item", "product", "store", "branch"],
        agent_name="sales_agent",
        llm_provider=config["llm_provider"],
        llm_model=config["model_name"],
        temperature=0.0,
        max_iterations=3
    )

    agent = create_react_agent(
        llm,
        tools=[dynamic_query_tool],
        prompt=(
            "You are a Sales Analytics Specialist for a donut shop chain.\n\n"

            "YOUR EXPERTISE:\n"
            "- Sales revenue analysis and trends\n"
            "- Top selling products identification\n"
            "- Store performance comparison\n"
            "- Sales forecasting and insights\n\n"

            "HOW TO USE YOUR TOOL:\n"
            "- Use the 'dynamic_query' tool ONCE for all sales data retrieval\n"
            "- The tool has access to: store_daily_single_item, product, store, branch\n"
            "- It will automatically generate optimized SQL with JOINs\n"
            "- Ask questions in natural language\n"
            "- Tool returns results in format: [(value1,), (value2,)] or [(col1, col2, col3), ...]\n\n"

            "DATABASE CONTEXT:\n"
            "- Main table: store_daily_single_item (sales transactions)\n"
            "- Key columns:\n"
            "  * qty_sales: Quantity sold (units)\n"
            "  * rp_sales: Revenue in Indonesian Rupiah\n"
            "  * date: Transaction date\n"
            "- The tool will automatically JOIN tables when needed\n\n"

            "CRITICAL RULES:\n"
            "1. Call the tool ONLY ONCE per question\n"
            "2. Interpret the raw result (e.g., '[(1500000,)]' means revenue is 1500000)\n"
            "3. Format your answer in natural, user-friendly language\n"
            "4. DO NOT call the tool again with rephrased questions\n"
            "5. If result is empty [] or blank, say 'Tidak ada data penjualan ditemukan'\n\n"

            "RESPONSE FORMAT:\n"
            "- Format currency as: Rp 1.500.000 (with thousand separator with dots)\n"
            "- Include product/store names, not just IDs\n"
            "- Provide context and insights, not just raw numbers\n"
            "- For trends, mention time period clearly\n"
            "- Use the same language as user's question (Indonesian/English)\n"
            "- Always interpret and explain the data\n\n"

            "EXAMPLE WORKFLOWS:\n\n"
            
            "Example 1 - Revenue query:\n"
            "User: 'Berapa total penjualan hari ini?'\n"
            "Thought: I need to get today's total revenue from sales table\n"
            "Action: dynamic_query\n"
            "Action Input: What is the total revenue for today?\n"
            "Observation: [(2500000,)]\n"
            "Thought: Revenue is 2500000 Rupiah. I'll format it nicely.\n"
            "Final Answer: Total penjualan hari ini adalah Rp 2.500.000\n\n"
            
            "Example 2 - Top products query:\n"
            "User: 'What are the best selling products this week?'\n"
            "Thought: I need top products by quantity for this week\n"
            "Action: dynamic_query\n"
            "Action Input: Show top 5 best selling products this week by quantity with product names\n"
            "Observation: [('GLAZED DONUT', 250), ('CHOCOLATE GLAZED', 180), ('ICED COFFEE', 150)]\n"
            "Thought: Got the top 3 products with sales quantity. I'll format them.\n"
            "Final Answer: Top selling products this week:\n1. GLAZED DONUT - 250 units\n2. CHOCOLATE GLAZED - 180 units\n3. ICED COFFEE - 150 units"
        ),
        name="sales_agent"
    )

    logger.info("✅ Sales Agent created with dynamic_query tool")
    return agent