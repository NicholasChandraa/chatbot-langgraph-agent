"""
LangGraph-based SQL Agent Workflow
Handles natural language to SQL query generation with validation and retry logic
"""
import re
import sqlparse
from typing import List, Optional, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain.messages import HumanMessage

from app.database.connection.async_sql_database import AsyncSQLDatabase
from app.llm.provider_factory import LLMProviderFactory
from app.utils.logger import logger


def clean_llm_output(text: str) -> str:
    """
    Clean LLM output by removing thinking tags and markdown code blocks.

    Args:
        text: Raw LLM output text

    Returns:
        Cleaned text
    """
    if not text:
        return text

    # Remove <think> tags from models like Qwen
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Remove markdown code blocks if present
    if text.startswith('```'):
        text = text.split('```')[1]
        if text.startswith('sql'):
            text = text[3:]
        text = text.strip()

    return text


def extract_token_usage(response: Any) -> Dict[str, int]:
    """
    Extract token usage from LangChain model response.
    Supports: Gemini, OpenAI, Anthropic, Ollama

    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens
    """
    token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

    try:
        # Priority 1: Check for usage_metadata as direct attribute (Gemini 2.5+)
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            # Handle both dict and object formats
            if isinstance(usage, dict):
                token_usage["prompt_tokens"] = usage.get('input_tokens', 0) or usage.get('prompt_token_count', 0)
                token_usage["completion_tokens"] = usage.get('output_tokens', 0) or usage.get('candidates_token_count', 0)
                token_usage["total_tokens"] = usage.get('total_tokens', 0) or usage.get('total_token_count', 0)
            else:
                # Object format
                token_usage["prompt_tokens"] = getattr(usage, 'input_tokens', 0) or getattr(usage, 'prompt_token_count', 0)
                token_usage["completion_tokens"] = getattr(usage, 'output_tokens', 0) or getattr(usage, 'candidates_token_count', 0)
                token_usage["total_tokens"] = getattr(usage, 'total_tokens', 0) or getattr(usage, 'total_token_count', 0)

            return token_usage

        # Priority 2: Check for response_metadata
        if hasattr(response, 'response_metadata') and response.response_metadata:
            metadata = response.response_metadata

            # Gemini format (older)
            if 'usage_metadata' in metadata:
                usage = metadata['usage_metadata']
                token_usage["prompt_tokens"] = usage.get('prompt_token_count', 0)
                token_usage["completion_tokens"] = usage.get('candidates_token_count', 0)
                token_usage["total_tokens"] = usage.get('total_token_count', 0)

            # OpenAI format
            elif 'token_usage' in metadata:
                usage = metadata['token_usage']
                token_usage["prompt_tokens"] = usage.get('prompt_tokens', 0)
                token_usage["completion_tokens"] = usage.get('completion_tokens', 0)
                token_usage["total_tokens"] = usage.get('total_tokens', 0)

            # Ollama format
            elif 'eval_count' in metadata and 'prompt_eval_count' in metadata:
                token_usage["prompt_tokens"] = metadata.get('prompt_eval_count', 0)
                token_usage["completion_tokens"] = metadata.get('eval_count', 0)
                token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]

    except Exception as e:
        logger.warning(f"Failed to extract token usage: {e}")

    return token_usage


class SQLAgentState(TypedDict):
    """State for SQL Agent workflow"""
    # Input
    question: str

    # Database context
    available_tables: List[str]
    relevant_tables: List[str]
    table_info: str

    # Query generation
    generated_query: Optional[str]
    query_valid: bool
    validation_message: Optional[str]

    # Execution
    query_result: Optional[str]
    execution_error: Optional[str]

    # Token tracking
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int

    # Control flow
    iteration: int
    max_iterations: int


class SQLAgentWorkflow:
    """
    LangGraph-based SQL query workflow.
    Generates, validates, and executes SQL queries from natural language.
    """

    FORBIDDEN_KEYWORDS = {
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'TRUNCATE',
        'ALTER', 'CREATE', 'GRANT', 'REVOKE'
    }

    def __init__(
        self,
        sql_db: AsyncSQLDatabase,
        tables: List[str],
        agent_name: str,
        llm_provider: str,
        llm_model: str,
        temperature: float = 0.0,
        max_iterations: int = 3
    ):
        self.sql_db = sql_db
        self.tables = tables
        self.agent_name = agent_name
        self.max_iterations = max_iterations

        # Create LLM
        self.llm = LLMProviderFactory.create(
            provider_name=llm_provider,
            model_name=llm_model,
            temperature=temperature
        )

        # Build workflow graph
        self.graph = self._build_graph()

        # logger.info(
        #     f"✅ SQL Agent Workflow initialized | "
        #     f"agent={agent_name} | "
        #     f"tables={len(tables)} | "
        #     f"llm={llm_provider}/{llm_model}"
        # )

    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(SQLAgentState)

        # Add nodes
        workflow.add_node("list_tables", self._list_tables_node)
        workflow.add_node("get_schema", self._get_schema_node)
        workflow.add_node("generate_query", self._generate_query_node)
        workflow.add_node("validate_query", self._validate_query_node)
        workflow.add_node("execute_query", self._execute_query_node)

        # Define edges
        workflow.set_entry_point("list_tables")
        workflow.add_edge("list_tables", "get_schema")
        workflow.add_edge("get_schema", "generate_query")
        workflow.add_edge("generate_query", "validate_query")

        # Conditional: validate → execute or retry
        workflow.add_conditional_edges(
            "validate_query",
            self._should_execute_or_retry,
            {
                "execute": "execute_query",
                "retry": "generate_query",
                "end": END
            }
        )

        workflow.add_edge("execute_query", END)

        return workflow.compile()

    # ==================== WORKFLOW NODES ====================

    async def _list_tables_node(self, state: SQLAgentState) -> SQLAgentState:
        """Step 1: List available tables"""
        # logger.info(f"📝 Step 1: Listing tables")

        state["available_tables"] = self.tables
        state["relevant_tables"] = self.tables  # Use all scoped tables

        # logger.info(f"Using {len(self.tables)} tables: {self.tables}")
        return state

    async def _get_schema_node(self, state: SQLAgentState) -> SQLAgentState:
        """Step 2: Get table schemas"""
        # logger.info(f"📖 Step 2: Getting schemas")

        try:
            table_info = await self.sql_db.get_table_info(
                table_names=state["relevant_tables"],
                include_comments=True
            )
            state["table_info"] = table_info

            # logger.info(f"Schema info retrieved: \n {table_info})")

        except Exception as e:
            logger.error(f"[{self.agent_name}] Failed to get schema: {e}")
            state["table_info"] = ""

        return state

    async def _generate_query_node(self, state: SQLAgentState) -> SQLAgentState:
        """Step 3: Generate SQL query"""
        state["iteration"] = state.get("iteration", 0) + 1

        # logger.info(f"✍️ Step 3: Generating SQL (iteration {state['iteration']})")

        # Build feedback from previous validation
        feedback = ""
        if state.get("validation_message") and state["iteration"] > 1:
            feedback = f"""
PREVIOUS ATTEMPT FAILED VALIDATION:
{state['validation_message']}

FIX THESE ISSUES in your new query!
"""

        prompt = f"""
You are a PostgreSQL expert for a donut shop business intelligence system.

Question: {state['question']}

Database Schema:
{state['table_info']}

CRITICAL RULES:
1. ALWAYS add LIMIT clause (default 100, or 10-20 for rankings) unless query has aggregation without GROUP BY
2. NEVER use SELECT * - always specify columns explicitly
3. Add SQL comments (--) to explain your logic
4. For sales queries with qty_sales/rp_sales: MUST filter (qty_sales > 0 OR rp_sales > 0)

BEST PRACTICES:
- Use ILIKE for case-insensitive text search
- Use CAST(date AS DATE) for date comparisons
- Add date filters only when question asks about specific time period
- For transaction counts: COUNT(DISTINCT NULLIF(sales_code, ''))

Date handling:
- Today: CURRENT_DATE
- Yesterday: CURRENT_DATE - INTERVAL '1 day'
- Last N days: CURRENT_DATE - INTERVAL 'N days'
- This month: DATE_TRUNC('month', CURRENT_DATE)

{feedback}

Generate ONLY the SQL query (no explanation, no markdown).
Add -- comments in the SQL to explain each step.
""".strip()
        
        # logger.info(f"[PROMPT] Generate Query Node: {prompt}")

        try:
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            query = clean_llm_output(response.content.strip())

            # logger.info(f"[RESPONSE] Generate Query Node: {query}")

            # Extract token usage
            token_usage = extract_token_usage(response)
            state["total_prompt_tokens"] += token_usage["prompt_tokens"]
            state["total_completion_tokens"] += token_usage["completion_tokens"]
            state["total_tokens"] += token_usage["total_tokens"]

            state["generated_query"] = query

            # logger.info(f"Token usage: {token_usage}")

        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            state["generated_query"] = None

        return state

    async def _validate_query_node(self, state: SQLAgentState) -> SQLAgentState:
        """Step 4: Validate generated query (Rule-based + LLM semantic)"""
        # logger.info(f"⚠️ Step 4: Validating query")

        query = state.get("generated_query")

        if not query:
            state["query_valid"] = False
            state["validation_message"] = "No query generated"
            logger.error(f"[{self.agent_name}] No query to validate")
            return state

        issues = []

        try:
            # 1. SQL parsing
            parsed = sqlparse.parse(query)
            if not parsed:
                issues.append("Cannot parse SQL syntax")

            # 2. Check statement type
            query_cleaned = sqlparse.format(query, strip_comments=True).strip()
            query_upper = query_cleaned.upper()

            if not (query_upper.startswith('SELECT') or query_upper.startswith('WITH')):
                issues.append("Query must start with SELECT or WITH")

            # 3. Check for forbidden operations
            for keyword in self.FORBIDDEN_KEYWORDS:
                if f' {keyword} ' in f' {query_upper} ':
                    issues.append(f"Forbidden operation: {keyword}")

            # 4. Safety checks

            # 4a. Valid sales filter (if using store_daily_single_item)
            if 'STORE_DAILY_SINGLE_ITEM' in query_upper:
                if ('QTY_SALES' in query_upper or 'RP_SALES' in query_upper):
                    pattern = r'\(\s*[\w\.]*qty_sales\s*>\s*0\s+OR\s+[\w\.]*rp_sales\s*>\s*0\s*\)'
                    has_filter = bool(re.search(pattern, query, re.IGNORECASE))

                    if not has_filter:
                        issues.append("Missing valid sales filter: (qty_sales > 0 OR rp_sales > 0)")

            # 4b. LIMIT check
            has_aggregate = any(agg in query_upper for agg in ['SUM(', 'COUNT(', 'AVG(', 'MIN(', 'MAX('])
            has_group_by = 'GROUP BY' in query_upper

            if 'LIMIT' not in query_upper:
                if not has_aggregate or has_group_by:
                    issues.append("Missing LIMIT clause")

            # 4c. SELECT * check
            if 'SELECT *' in query_upper:
                issues.append("Avoid SELECT * - specify columns")

            # 5. LLM semantic validation - ONLY if rule-based passed
            if not issues:
                # logger.info(f"Rule-based validation passed, checking semantic issues with LLM")
                
                # Truncate schema context to save tokens
                # schema_context = state.get("table_info", "")[:1000]
                
                validation_prompt = f"""
Review this PostgreSQL query for CRITICAL issues only:

QUERY:
{query}

CHECK FOR:
1. Cartesian products (missing JOIN conditions between multiple tables)
2. Invalid column references (columns not in schema)
3. Type mismatches in comparisons
4. Missing JOIN conditions between tables

RULES:
- If multiple tables are referenced, ensure they are properly JOINed with ON conditions
- Verify all column names exist in the schema
- DO NOT flag missing date filters, optional WHERE clauses, or performance issues
- DO NOT flag queries that use only ONE table

RESPOND:
- If no critical issues, response with just "OK"
- If critical issues found: List each issue on a new line
""".strip()     
                # logger.info(f"[PROMPT] Validate Query Node: {validation_prompt}")

                try:
                    response = await self.llm.ainvoke([HumanMessage(content=validation_prompt)])
                    llm_result = clean_llm_output(response.content.strip())

                    # logger.info(f"[RESPONSE] Validate Query Node: {llm_result}")

                    # Extract and track token usage
                    token_usage = extract_token_usage(response)
                    state["total_prompt_tokens"] += token_usage["prompt_tokens"]
                    state["total_completion_tokens"] += token_usage["completion_tokens"]
                    state["total_tokens"] += token_usage["total_tokens"]

                    # logger.info(f"[{self.agent_name}] LLM validation result: {llm_result}")
                    # logger.info(f"[{self.agent_name}] Token usage (semantic validation): {token_usage}")

                    if llm_result.upper() != "OK":
                        issues.append(f"Semantic issue: {llm_result}")

                except Exception as e:
                    logger.warning(f"[{self.agent_name}] LLM validation failed (continuing): {e}")
                    # Don't fail validation if LLM validation errors out

            # Set result
            state["query_valid"] = len(issues) == 0
            state["validation_message"] = "\n".join(issues) if issues else "Query is valid"

            if state["query_valid"]:
                # logger.info(f"[{self.agent_name}] ✅ Validation PASSED")
                pass
            else:
                logger.warning(f"[{self.agent_name}] ❌ Validation FAILED:\n{state['validation_message']}")

        except Exception as e:
            logger.error(f"[{self.agent_name}] Validation error: {e}")
            state["query_valid"] = False
            state["validation_message"] = f"Validation error: {str(e)}"

        return state

    def _should_execute_or_retry(self, state: SQLAgentState) -> str:
        """Decision: execute query or retry generation"""
        if state["query_valid"]:
            return "execute"

        max_iter = state.get("max_iterations", 3)
        current_iter = state.get("iteration", 0)

        if current_iter >= max_iter:
            logger.warning(f"Max iterations ({max_iter}) reached")

            # Best effort: try to execute anyway if query exists
            if state.get("generated_query"):
                logger.warning(f"Attempting best-effort execution")
                return "execute"
            else:
                return "end"

        # logger.info(f"Retrying generation ({current_iter}/{max_iter})")
        return "retry"

    async def _execute_query_node(self, state: SQLAgentState) -> SQLAgentState:
        """⚙️ Step 5: Execute validated query"""
        # logger.info(f"Step 5: Executing query")

        query = state.get("generated_query")
        if not query:
            state["execution_error"] = "No query to execute"
            logger.error(f"No query")
            return state

        try:
            result = await self.sql_db.run(query)

            # logger.info(f"[RESPONSE] result from db: {result}")

            state["query_result"] = result
            state["execution_error"] = None

            # logger.info(f"✅ Execution successful ({len(str(result))} chars)")
            logger.debug(f"Query result preview:\n{str(result)[:500]}...")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Execution failed: {error_msg}")

            state["query_result"] = None
            state["execution_error"] = error_msg

        return state

    # ==================== PUBLIC API ====================

    async def execute(self, question: str) -> Dict[str, Any]:
        """
        Execute workflow with natural language question.

        Args:
            question: Natural language question

        Returns:
            Dict with answer, query, tokens, status
        """
        try:
            # logger.info(f"Processing query: {question}")

            initial_state = {
                "question": question,
                "available_tables": [],
                "relevant_tables": [],
                "table_info": "",
                "generated_query": None,
                "query_valid": False,
                "validation_message": None,
                "query_result": None,
                "execution_error": None,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "iteration": 0,
                "max_iterations": self.max_iterations
            }

            # Run workflow
            final_state = await self.graph.ainvoke(initial_state)

            # logger.info(f"✅ Workflow completed")

            # Prepare result
            query_result = final_state.get("query_result")
            execution_error = final_state.get("execution_error")
            generated_query = final_state.get("generated_query")

            if execution_error:
                answer = f"Error executing query:\n{execution_error}"
            elif query_result:
                answer = str(query_result)
            else:
                answer = "No results from query."

            return {
                "status": "success" if not execution_error else "error",
                "answer": answer,
                "query": generated_query,
                "tokens": {
                    "prompt_tokens": final_state.get("total_prompt_tokens", 0),
                    "completion_tokens": final_state.get("total_completion_tokens", 0),
                    "total_tokens": final_state.get("total_tokens", 0)
                }
            }

        except Exception as e:
            error_msg = f"Workflow failed: {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            return {
                "status": "error",
                "answer": f"Sorry, an error occurred: {str(e)}",
                "error": error_msg
            }
