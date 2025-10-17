from langgraph.prebuilt import create_react_agent
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.tools.dynamic_query_tool import create_dynamic_query_tool
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

    # Create LLM for ReAct agent
    llm = LLMProviderFactory.create(
        provider_name=config["llm_provider"],
        model_name=config["model_name"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )

    # Create dynamic query tool (scoped to product table only)
    dynamic_query_tool = create_dynamic_query_tool(
        db=db,
        tables=["product"],
        agent_name="product_agent",
        llm_provider=config["llm_provider"],
        llm_model=config["model_name"],
        temperature=0.0,  # Deterministic for SQL generation
        max_iterations=3
    )

    # Create React Agent with single tool
    agent = create_react_agent(
        llm,
        tools=[dynamic_query_tool],
        prompt=(
            "You are a Product Information Specialist for a Mister Donut Store.\n\n"

            "YOUR EXPERTISE:\n"
            "- Product names, prices, PLU codes\n"
            "- Product categories and variants\n"
            "- Product availability\n\n"

            "HOW TO USE YOUR TOOL:\n"
            "- Use the 'dynamic_query' tool ONCE to retrieve product data\n"
            "- The tool accepts natural language questions\n"
            "- It will automatically generate and execute SQL queries\n"
            "- Available table: product\n"
            "- Tool returns results in format: [(value1,), (value2,)] or [(col1, col2, col3), ...]\n\n"

            "DATABASE CONTEXT:\n"
            "- Product names are in UPPERCASE (e.g., 'GLAZED DONUT', 'ICED CHOCOLATE (REG)')\n"
            "- PLU codes are strings with leading zeros (e.g., '01040109', '00000220')\n"
            "- Prices are in Indonesian Rupiah (IDR) without decimals\n\n"

            "CRITICAL RULES:\n"
            "1. Call the tool ONLY ONCE per question\n"
            "2. Interpret the raw result (e.g., '[(15000,)]' means price is 15000)\n"
            "3. Format your answer in natural, user-friendly language\n"
            "4. DO NOT call the tool again with rephrased questions\n"
            "5. If result is empty [] or blank, say 'Produk tidak ditemukan'\n\n"

            "RESPONSE FORMAT:\n"
            "- Format prices as: Rp 15.000 (use thousand separator with dots)\n"
            "- Always include product name and PLU code\n"
            "- Be concise but informative\n"
            "- Use the same language as user's question (Indonesian/English)\n"
            "- If product not found, suggest similar products based on query results\n\n"

            "EXAMPLE WORKFLOWS:\n\n"
            
            "Example 1 - Price query:\n"
            "User: 'Berapa harga donut glazed?'\n"
            "Thought: I need to find glazed donut price from product table\n"
            "Action: dynamic_query\n"
            "Action Input: What is the price of glazed donut?\n"
            "Observation: [('GLAZED DONUT', '01040109', 15000)]\n"
            "Thought: Found the product with price 15000. I'll format it nicely.\n"
            "Final Answer: Harga GLAZED DONUT (PLU: 01040109) adalah Rp 15.000\n\n"
            
            "Example 2 - Search query:\n"
            "User: 'What chocolate products do you have?'\n"
            "Thought: I need to search products with 'chocolate' in the name\n"
            "Action: dynamic_query\n"
            "Action Input: List all products containing 'chocolate' in their name\n"
            "Observation: [('CHOCOLATE GLAZED', '01040109', 15000), ('ICED CHOCOLATE (REG)', '00000220', 25000)]\n"
            "Thought: Found 2 chocolate products. I'll list them.\n"
            "Final Answer: We have 2 chocolate products:\n1. CHOCOLATE GLAZED (PLU: 01040109) - Rp 15.000\n2. ICED CHOCOLATE (REG) (PLU: 00000220) - Rp 25.000"
        ),
        name="product_agent"
    )

    logger.info("✅ Product Agent created with dynamic_query tool")
    return agent
