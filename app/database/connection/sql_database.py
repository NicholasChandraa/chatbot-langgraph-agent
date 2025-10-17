from typing import Dict
from langchain_community.utilities import SQLDatabase

from app.config.settings.settings import get_settings
from app.utils.logger import logger

class SQLDatabaseManager:
    """
    Manager for LangChain SQLDatabase instances.
    Creates separate instances per table scope to support different agents.
    """

    def __init__(self):
        self.settings = get_settings()
        self._sql_databases: Dict[str, SQLDatabase] = {}  # Cache per table combination

    def get_sql_database(self, include_tables: list[str] = None) -> SQLDatabase:
        """
        Get LangChain SQLDatabase instance for SQL Agent.
        Creates separate instances per unique table combination.

        Args:
            include_tables: List of tables to include (default: all 4 tables)

        Returns:
            SQLDatabase instance configured for specified tables

        Example:
            # ProductAgent - only product table
            db = get_sql_database(["product"])

            # SalesAgent - multiple tables
            db = get_sql_database(["store_daily_single_item", "product", "store"])
        """
        # Default: All 4 tables used by AI
        if include_tables is None:
            include_tables = [
                "branch",
                "store",
                "product",
                "store_daily_single_item"
            ]

        # Create cache key from sorted table list
        cache_key = "|".join(sorted(include_tables))

        # Return cached instance if exists
        if cache_key in self._sql_databases:
            logger.debug(f"📦 Using cached SQLDatabase for tables: {include_tables}")
            return self._sql_databases[cache_key]

        # Create new SQLDatabase instance
        logger.info(f"🔨 Creating SQLDatabase with tables: {include_tables}")

        # Create SQLDatabase (uses sync connection)
        # Note: LangChain's SQLDatabase requires psycopg2 (sync), not asyncpg
        sql_database = SQLDatabase.from_uri(
            database_uri=self._get_sync_database_url(),
            include_tables=include_tables,
            sample_rows_in_table_info=3,  # Include 3 sample rows for context
            view_support=True,             # Support database views
            max_string_length=1000,        # Limit string column preview
        )

        # Cache for reuse
        self._sql_databases[cache_key] = sql_database

        logger.info(f"✅ SQLDatabase initialized | tables={len(include_tables)} | cached_instances={len(self._sql_databases)}")

        return sql_database
    
    def _get_sync_database_url(self) -> str:
        """
        Get sync database URL (not async).
        SQLDatabase requires psycopg2, not asyncpg.
        """
        return self.settings.SQLAGENT_DATABASE_URL

    def clear_cache(self, include_tables: list[str] = None):
        """
        Clear cached SQLDatabase instances.

        Args:
            include_tables: Specific table combination to clear, or None to clear all

        Example:
            # Clear specific cache
            sql_db_manager.clear_cache(["product"])

            # Clear all cache
            sql_db_manager.clear_cache()
        """
        if include_tables:
            cache_key = "|".join(sorted(include_tables))
            if cache_key in self._sql_databases:
                del self._sql_databases[cache_key]
                logger.info(f"🗑️ Cleared SQLDatabase cache for tables: {include_tables}")
        else:
            self._sql_databases.clear()
            logger.info("🗑️ Cleared all SQLDatabase cache")

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache info for monitoring
        """
        return {
            "cached_instances": len(self._sql_databases),
            "table_combinations": list(self._sql_databases.keys())
        }


# Singleton instance
sql_db_manager = SQLDatabaseManager()

def get_sql_database(include_tables: list[str] = None) -> SQLDatabase:
    """Get SQLDatabase instance for SQL Agent"""
    return sql_db_manager.get_sql_database(include_tables)