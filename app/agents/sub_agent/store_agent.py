from langgraph.prebuilt import create_react_agent
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.tools.dynamic_query_tool import create_dynamic_query_tool
from app.config.agent_config.agent_config_manager import get_agent_config
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger

async def create_store_agent(db: AsyncSession):
    """
    Store Agent - Handles store and branch queries.

    Scope:
    - Store information (name, code, location)
    - Branch management
    - Store-branch relationships

    Tables: store, branch
    """
    logger.info("🤖 Creating Store Agent...")

    config = await get_agent_config("store_agent", db)

    llm = LLMProviderFactory.create(
        provider_name=config["llm_provider"],
        model_name=config["model_name"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )

    # Create dynamic query tool for store data
    dynamic_query_tool = create_dynamic_query_tool(
        db=db,
        tables=["store", "branch"],
        agent_name="store_agent",
        llm_provider=config["llm_provider"],
        llm_model=config["model_name"],
        temperature=0.0,
        max_iterations=3
    )

    agent = create_react_agent(
        llm,
        tools=[dynamic_query_tool],
        prompt=(
            "You are a Store Information Specialist for a donut shop chain.\n\n"

            "YOUR EXPERTISE:\n"
            "- Store details (name, code, location)\n"
            "- Branch management and hierarchy\n"
            "- Store-branch relationships\n\n"

            "HOW TO USE YOUR TOOL:\n"
            "- Use the 'dynamic_query' tool ONCE to retrieve store information\n"
            "- Available tables: store, branch\n"
            "- The tool will automatically JOIN tables when needed\n"
            "- Tool returns results in format: [(value1,), (value2,)] or [(col1, col2), ...]\n\n"

            "DATABASE CONTEXT:\n"
            "- Table: store (individual store/outlet information)\n"
            "- Table: branch (regional branch/area information)\n"
            "- Relationship: store.branch_sid → branch.branch_sid\n"
            "- Store codes are case-sensitive (e.g., 'TLPC', 'TCWS', 'TPLG')\n"
            "- Store names are in UPPERCASE\n\n"

            "CRITICAL RULES:\n"
            "1. Call the tool ONLY ONCE per question\n"
            "2. Interpret the raw result (e.g., '[(10,)]' means count is 10)\n"
            "3. Format your answer in natural, user-friendly language\n"
            "4. DO NOT call the tool again with rephrased questions\n"
            "5. If result is empty [] or blank, say 'Tidak ada data ditemukan'\n\n"

            "RESPONSE FORMAT:\n"
            "- Include both store code and full name\n"
            "- Mention branch/area if relevant\n"
            "- Be helpful and informative\n"
            "- Use the same language as user's question (Indonesian/English)\n\n"

            "EXAMPLE WORKFLOWS:\n\n"
            
            "Example 1 - Count query:\n"
            "User: 'Berapa jumlah toko yang aktif?'\n"
            "Thought: I need to query the store table to count active stores\n"
            "Action: dynamic_query\n"
            "Action Input: How many stores are there?\n"
            "Observation: [(15,)]\n"
            "Thought: The result shows 15 stores. I can now answer.\n"
            "Final Answer: Saat ini terdapat 15 toko yang terdaftar dalam sistem.\n\n"
            
            "Example 2 - List query:\n"
            "User: 'Show me all stores in Jakarta'\n"
            "Thought: I need to query stores filtered by location\n"
            "Action: dynamic_query\n"
            "Action Input: List all stores with 'Jakarta' in their location\n"
            "Observation: [('TJKT01', 'FLAGSHIP JAKARTA'), ('TJKT02', 'JAKARTA TIMUR')]\n"
            "Thought: I got 2 stores. I'll format them nicely.\n"
            "Final Answer: Terdapat 2 toko di Jakarta:\n1. TJKT01 - FLAGSHIP JAKARTA\n2. TJKT02 - JAKARTA TIMUR"
        ),
        name="store_agent"
    )

    logger.info("✅ Store Agent created with dynamic_query tool")
    return agent
