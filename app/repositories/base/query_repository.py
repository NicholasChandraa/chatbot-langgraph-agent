"""
Query Repository - Core SQl Execution Logic
Centralized query execution with caching, metrics, and validation
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection.async_sql_database import AsyncSQLDatabase
from app.agents.tools.sql_agent_workflow import SQLAgentWorkflow
from app.config.agent_config.agent_config_manager import get_agent_config
from app.utils.logger import logger

class QueryRepository:
    """
    Centralized repository for executing natural language queries.

    Features:
    - Natural language > SQL conversion via LLM
    - Automatic query validation and safety checks
    - Built-in caching (in-memory, upgradeable to Redis)
    - Metrics collection (query duration, success/failure)
    - Error handling and rollback

    This class is used by all domain repositories to execute queries,
    ensuring consistent behavior across all agents.
    """

    def __init__(
        self,
        db: AsyncSession,
        agent_name: str,
        tables: List[str],
        cache_ttl_seconds: int = 300, # 5 minutes default
    ):
        """
        Initialize query repository.

        Args:
            db: SQLAlchemy async session
            agent_name: Name of agent using this repo (for loggin/metrics)
            tables: List of table names this repo can access
            cache_ttl_seconds: Cache time-to-live (default: 5 minutes)
        """
        self.db = db
        self.agent_name = agent_name
        self.tables = tables
        self._cache_ttl = cache_ttl_seconds

        # Internal components (hidden from agents)
        self._sql_db = AsyncSQLDatabase(
            db=db,
            include_tables=tables,
            sample_rows_in_table_info=3,
            # max_string_length=
        )

        # Lazy-initialized workflow (only create when first query executed)
        self._workflow: Optional[SQLAgentWorkflow] = None
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Simple in-memory cache

        logger.info(
            f"✅ QueryRepository initialized | agent = {agent_name} | tables = {tables} | cache_ttl = {cache_ttl_seconds}s"
        )

    def _get_cache_key(self, question: str) -> str:
        """
        Generate cache key from question.

        Uses hash of lowercase question for case-insensitive caching.

        Args:
            question: The query question
        Returns:
            Cache key string
        """
        # Case-insensitive caching
        normalized = question.lower().strip()
        return f"{self.agent_name}:{hash(normalized)}"
    
    def _get_from_cache(self, question: str) -> Optional[str]:
        """
        Retrieve cached result if available and not expired.

        Args:
            question: The query question
        Returns:
            Cached answer if valid, None otherwise
        """
        cache_key = self._get_cache_key(question)

        if cache_key not in self._cache:
            return None
        
        cached_entry = self._cache[cache_key]
        cached_time = cached_entry["timestamp"]

        # Check if cache expired
        age_seconds = (datetime.now() - cached_time).total_seconds()
        if age_seconds > self._cache_ttl:
            # Expired, remove from cache
            del self._cache[cache_key]
            return None
        
        return cached_entry["result"]

    async def get_config(self) -> Dict[str, Any]:
        """
        Get agent configuration from database.
        Uses the centralized agent config manager which has its own caching.

        Returns:
            Agent configuration dictionary
        Raises:
            ValueError: If agent config not found
        """
        return await get_agent_config(self.agent_name, self.db)

    async def _initialize_workflow(self):
        """
        Lazy-initialize SQL Agent Workflow.

        This is called only on first query to avoid unnecessary
        initialization overhead if repository is never used.
        """
        logger.info(f"[{self.agent_name}] Initializing SQL workflow...")

        config = await self.get_config()

        self._workflow = SQLAgentWorkflow(
            sql_db=self._sql_db,
            tables=self.tables,
            agent_name=self.agent_name,
            llm_provider=config["llm_provider"],
            llm_model=config["model_name"],
            temperature=config["temperature"],
            max_iterations=3
        )

        logger.info(f"[{self.agent_name}] ✅ Workflow initialized | llm={config['llm_provider']}/{config['model_name']}")
        
    def _add_to_cache(self, question: str, result: str):
        """
        Add query result

        Args:
            question (str): The query question
            result (str): The query result to cache
        """
        cache_key = self._get_cache_key(question)

        self._cache[cache_key] = {
            "result": result,
            "timestamp": datetime.now()
        }

        # Simple cache size management (keep only 100 most recent)
        if len(self._cache) > 100:
            # Remove oldest entry
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k]["timestamp"]
            )
            del self._cache[oldest_key]
    
    def _record_metrics(
        self,
        question: str,
        duration: float,
        success: bool,
        tokens: Optional[Dict] = None,
        error: Optional[str] = None
    ):
        """
        Record query metrics for monitoring

        Currently logs to console. Can be extended to send to:
        - Prometheus
        - DataDog
        - CloudWatch
        - Custom metrics database

        Args:
            question (str): The query question
            duration (float): Execution duration in seconds
            success (bool): Wheter query succeeded
            tokens (Optional[Dict], optional): Token usage info (optional). Defaults to None.
            error (Optional[str], optional): Error message if failed. Defaults to None.
        """
        metric_data = {
            "agent": self.agent_name,
            "duration_seconds": round(duration, 3),
            "success": success,
            "question_length": len(question),
            "timestamp": datetime.now().isoformat()
        }

        if tokens:
            metric_data["tokens"] = tokens.get("total_tokens", 0)

        if error:
            metric_data["error"] = error[:200]   # Truncate long errors
        
        # Log metrics (can be extended to push to metrics service)
        logger.info(f"[METRICS] {metric_data}")

        # TODO: Send to metrics service
        # await metrics_client.record(metric_data)

    async def execute_nl_query(self, question: str) -> str:
        """
        Execute natural language query with full workflow

        Workflow:
        1. Check cache for previous identical query
        2. If not cached, initialize SQL workflow (lazy)
        3. Execute query via workflow (NL > SQL > Execute)
        4. Cache successful results
        5. Record metrics
        6. Return formatted answer

        Args:
            question: Natural language question
        
        Returns:
            Formatted query resutls as string
        
        Examples:
            > result = await repo.execute_nl_query("Show top 5 products with highest selling")
            > print(result)
            "[('GLAZED DONUT', 'MOCHIDO CHOCOLATE', ...)]"
        """
        start_time = datetime.now()

        # Step 1: Check cache
        cached_result = self._get_from_cache(question)
        if cached_result is not None:
            logger.info(f"[{self.agent_name}] Cache HIT | question: {question}...")
            return cached_result
        
        logger.info(f"[{self.agent_name}] Cache MISS | question={question}...")
        
        try:
            # Step 2: Lazy-initialize workflow
            if self._workflow is None:
                await self._initialize_workflow()
            
            # Step 3: Execute query
            logger.info(f"[{self.agent_name}] 🚀 Executing query...")
            result = await self._workflow.execute(question)

            # Extract answer from workflow result
            answer = result.get("answer", "")

            # Step 4: Cache successful result
            self._add_to_cache(question, answer)

            # Step 5: Record metrics
            duration = (datetime.now() - start_time).total_seconds()
            self._record_metrics(
                question=question,
                duration=duration,
                success=True,
                tokens=result.get("tokens", {})
            )

            logger.info(
                f"[{self.agent_name}] ✅ Query executed | duration = {duration:.2f}s | answer_length={len(answer)}"
            )

            return answer
        except Exception as e:
            # Record failuer metrics
            duration = (datetime.now() - start_time).total_seconds()
            self._record_metrics(
                question=question,
                duration=duration,
                success=False,
                error=str(e)
            )

            logger.error(
                f"[{self.agent_name}] ❌ Query failed | duration = {duration:.2f}s | error = {str(e)}", exc_info=True
            )

            # Return user-friendly error
            return f"Error executing query: {str(e)}"

    def clear_cache(self):
        """Clear all cached query results."""
        cache_size = len(self._cache)
        self._cache.clear()
        logger.info(f"[{self.agent_name}] 🪣 Cache cleared | entries = {cache_size}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring.

        Returns:
            Dict[str, Any]: Dict with caches stats
        """
        return {
            "agent_name": self.agent_name,
            "cache_size": len(self._cache),
            "cache_ttl_seconds": self._cache_ttl,
            "tables": self.tables
        }