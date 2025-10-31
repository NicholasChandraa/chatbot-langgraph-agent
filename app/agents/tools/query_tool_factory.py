"""
Query Tool Factory
Factory for creating LangChain tools that wrap SQL Agent workflow
"""
from typing import List
from langchain.tools import tool
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection.async_sql_database import AsyncSQLDatabase
from app.agents.workflows.sql_agent_workflow import SQLAgentWorkflow
from app.utils.logger import logger


def create_dynamic_query_tool(
    db: AsyncSession,
    tables: List[str],
    agent_name: str,
    llm_provider: str,
    llm_model: str,
    temperature: float = 0.0,
    max_iterations: int = 3
):
    """
    Factory function to create table-scoped dynamic query tool.

    This creates a LangChain tool that wraps the full SQL Agent workflow,
    allowing ReAct agents to query databases using natural language.

    Args:
        db: AsyncSession for database access
        tables: List of allowed tables for this agent
        agent_name: Name for logging
        llm_provider: LLM provider (gemini, openai, etc.)
        llm_model: LLM model name
        temperature: LLM temperature (default: 0.0 for deterministic SQL)
        max_iterations: Max retry for query generation

    Returns:
        LangChain tool ready to be used by ReAct agent

    Usage:
        # In agent creation:
        tool = create_dynamic_query_tool(
            db=db_session,
            tables=["product"],
            agent_name="product_agent",
            llm_provider="gemini",
            llm_model="gemini-2.0-flash-exp"
        )

        agent = create_react_agent(llm, tools=[tool], ...)
    """

    # Create AsyncSQLDatabase wrapper
    sql_db = AsyncSQLDatabase(
        db=db,
        include_tables=tables,
        sample_rows_in_table_info=3,
        max_string_length=300
    )

    # Create SQL Agent Workflow
    workflow = SQLAgentWorkflow(
        sql_db=sql_db,
        tables=tables,
        agent_name=agent_name,
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=temperature,
        max_iterations=max_iterations
    )

    logger.info(
        f"✅ Dynamic query tool created | "
        f"agent={agent_name} | "
        f"tables={tables}"
    )

    # Define the async tool
    @tool
    async def dynamic_query(question: str) -> str:
        """
        Query database using natural language.

        This tool automatically:
        1. Analyzes your question
        2. Generates optimized SQL query
        3. Validates for safety and correctness
        4. Executes and returns results

        Use this tool whenever you need to retrieve data from the database.

        Args:
            question: Natural language question about the data

        Returns:
            Query results as formatted string

        Examples:
            - "What products contain chocolate?"
            - "Show me top 10 selling products this month"
            - "How many stores are active?"
        """
        try:
            logger.info(f"[{agent_name}] Tool called with question: {question}")

            # Execute workflow
            result = await workflow.execute(question)

            # Log token usage
            if "tokens" in result:
                tokens = result["tokens"]
                logger.info(
                    f"[{agent_name}] Tool execution complete | "
                    f"tokens={tokens.get('total_tokens', 0)} | "
                    f"status={result.get('status')}"
                )

            # Return answer
            answer = result.get("answer", "No results")

            return answer

        except Exception as e:
            error_msg = f"Error executing dynamic query: {str(e)}"
            logger.error(f"[{agent_name}] {error_msg}", exc_info=True)
            return error_msg

    # Customize tool metadata
    dynamic_query.name = "dynamic_query"
    dynamic_query.description = f"""
Query {', '.join(tables)} table(s) using natural language.

This tool automatically generates and executes SQL queries based on your question.
It includes built-in validation and safety checks.

Available tables: {', '.join(tables)}

Use this tool whenever you need to retrieve data to answer the user's question.
""".strip()

    return dynamic_query
