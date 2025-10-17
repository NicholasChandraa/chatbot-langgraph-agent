from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection.sql_database import get_sql_database
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
    
    sql_db = get_sql_database(
        include_tables=["store", "branch"]
    )
    
    toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)
    
    agent = create_react_agent(
        llm,
        tools=toolkit.get_tools(),
        state_modifier=(
            "You are a Store Information Specialist for a donut shop chain.\n\n"
            
            "YOUR EXPERTISE:\n"
            "- Store details (name, code, location)\n"
            "- Branch management and hierarchy\n"
            "- Store-branch relationships\n\n"
            
            "DATABASE CONTEXT:\n"
            "- Table: store (individual store/outlet information)\n"
            "- Table: branch (regional branch/area information)\n"
            "- Relationship: store.branch_sid → branch.branch_sid\n"
            "- Store codes are case-sensitive (e.g., 'TLPC', 'TCWS', 'TPLG')\n"
            "- Store names are in UPPERCASE\n\n"
            
            "SEARCH GUIDELINES:\n"
            "- Use ILIKE for case-insensitive store name search\n"
            "- Use exact match (=) for store codes (case-sensitive)\n"
            "- Always JOIN with branch table for complete info\n\n"
            
            "RESPONSE FORMAT:\n"
            "- Include both store code and full name\n"
            "- Mention branch/area if relevant\n"
            "- Be helpful and informative"
        )
    )
    
    logger.info("✅ Store Agent created")
    return agent
