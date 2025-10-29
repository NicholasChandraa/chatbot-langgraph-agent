"""
Async SQL Database Utility
Drop-in async replacement for LangChain's SQLDatabase
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, inspect, MetaData, Table
from sqlalchemy.schema import CreateTable

from app.utils.logger import logger
from app.database.connection.connection import get_db


class AsyncSQLDatabase:
    """
    Async SQL Database wrapper for LangGraph SQL Agent.
    Provides same interface as LangChain's SQLDatabase but fully async.
    """

    def __init__(
        self,
        db: AsyncSession,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 3,
        max_string_length: int = 300
    ):
        """
        Initialize async SQL database.

        Args:
            db: AsyncSession from SQLAlchemy
            include_tables: List of tables to include (default: all)
            sample_rows_in_table_info: Number of sample rows in schema
            max_string_length: Max length for string values
        """
        self.db = db
        self.include_tables = include_tables
        self.sample_rows_in_table_info = sample_rows_in_table_info
        self.max_string_length = max_string_length
        self._metadata: Optional[MetaData] = None
        self._table_cache: Dict[str, str] = {}

    async def get_usable_table_names(self) -> List[str]:
        """
        Get list of available table names.

        Returns:
            List of table names
        """
        try:
            if self.include_tables:
                return self.include_tables

            # Query pg_tables for all user tables
            result = await self.db.execute(text("""
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename
            """))

            tables = [row[0] for row in result.fetchall()]
            logger.debug(f"Found {len(tables)} tables: {tables}")

            return tables

        except Exception as e:
            logger.error(f"Failed to get table names: {e}")
            return []

    async def get_table_info(
        self,
        table_names: Optional[List[str]] = None,
        include_comments: bool = True
    ) -> str:
        """
        Get table schema information with CREATE TABLE statements and sample rows.

        Args:
            table_names: Specific tables to get info for (default: all usable tables)
            include_comments: Include database comments (default: True)

        Returns:
            Formatted string with table schemas and sample data
        """
        try:
            if table_names is None:
                table_names = await self.get_usable_table_names()

            table_infos = []

            for table_name in table_names:
                # Check cache first
                cache_key = f"{table_name}:{include_comments}"
                if cache_key in self._table_cache:
                    table_infos.append(self._table_cache[cache_key])
                    continue

                # Get CREATE TABLE statement
                create_table = await self._get_create_table_statement(table_name)

                # Get table comment if requested
                table_comment = ""
                if include_comments:
                    table_comment = await self._get_table_comment(table_name)

                # Get column comments if requested
                column_comments = ""
                if include_comments:
                    column_comments = await self._get_column_comments(table_name)

                # Get sample rows
                sample_rows = await self._get_sample_rows(table_name)

                # Build table info
                info_parts = []

                # CREATE TABLE
                info_parts.append(create_table)

                # Table comment
                if table_comment:
                    info_parts.append(f"\n/* Table: {table_name}")
                    info_parts.append(f"Description: {table_comment}")

                # Column comments
                if column_comments:
                    info_parts.append("\nColumns:")
                    info_parts.append(column_comments)

                if table_comment or column_comments:
                    info_parts.append("*/")

                # Sample rows
                if sample_rows:
                    info_parts.append(f"\n/*\n{sample_rows}\n*/")

                table_info = "\n".join(info_parts)

                # Cache it
                self._table_cache[cache_key] = table_info
                table_infos.append(table_info)

            result = "\n\n".join(table_infos)
            logger.debug(f"Generated table info for {len(table_names)} tables ({len(result)} chars)")

            return result

        except Exception as e:
            logger.error(f"Failed to get table info: {e}", exc_info=True)
            return f"Error getting table info: {str(e)}"

    async def _get_create_table_statement(self, table_name: str) -> str:
        """Get CREATE TABLE statement for a table"""
        try:
            # Get columns and types
            result = await self.db.execute(text(f"""
                SELECT
                    column_name,
                    data_type,
                    character_maximum_length,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = :table_name
                ORDER BY ordinal_position
            """), {"table_name": table_name})

            columns = []
            for row in result.fetchall():
                col_name, data_type, max_len, nullable, default = row

                # Format type
                type_str = data_type.upper()
                if max_len and data_type in ('character varying', 'character'):
                    type_str = f"VARCHAR({max_len})"

                # Format column
                col_str = f"  {col_name} {type_str}"

                if nullable == 'NO':
                    col_str += " NOT NULL"

                if default:
                    col_str += f" DEFAULT {default}"

                columns.append(col_str)

            # Get primary key (using subquery to avoid ::regclass parameter binding issue)
            pk_result = await self.db.execute(text("""
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                JOIN pg_class c ON c.oid = i.indrelid
                WHERE c.relname = :table_name
                AND c.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
                AND i.indisprimary
            """), {"table_name": table_name})

            pk_cols = [row[0] for row in pk_result.fetchall()]
            if pk_cols:
                columns.append(f"  PRIMARY KEY ({', '.join(pk_cols)})")

            create_statement = f"CREATE TABLE {table_name} (\n" + ",\n".join(columns) + "\n)"
            return create_statement

        except Exception as e:
            logger.warning(f"Failed to get CREATE TABLE for {table_name}: {e}")
            # Rollback to prevent transaction abort
            await self.db.rollback()
            return f"-- Table: {table_name} (schema unavailable)"

    async def _get_table_comment(self, table_name: str) -> str:
        """Get table comment from pg_description"""
        try:
            result = await self.db.execute(text("""
                SELECT obj_description(c.oid)
                FROM pg_class c
                WHERE c.relname = :table_name
                AND c.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
            """), {"table_name": table_name})

            row = result.fetchone()
            return row[0] if row and row[0] else ""

        except Exception as e:
            logger.debug(f"No comment for table {table_name}: {e}")
            # Rollback to prevent transaction abort
            await self.db.rollback()
            return ""

    async def _get_column_comments(self, table_name: str) -> str:
        """Get column comments from pg_description"""
        try:
            result = await self.db.execute(text("""
                SELECT
                    a.attname as column_name,
                    col_description(a.attrelid, a.attnum) as comment,
                    format_type(a.atttypid, a.atttypmod) as data_type
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                WHERE c.relname = :table_name
                AND a.attnum > 0
                AND NOT a.attisdropped
                ORDER BY a.attnum
            """), {"table_name": table_name})

            comments = []
            for row in result.fetchall():
                col_name, comment, data_type = row
                if comment:
                    comments.append(f"  - {col_name} ({data_type}): {comment}")
                else:
                    comments.append(f"  - {col_name} ({data_type})")

            return "\n".join(comments) if comments else ""

        except Exception as e:
            logger.debug(f"No column comments for {table_name}: {e}")
            # Rollback to prevent transaction abort
            await self.db.rollback()
            return ""

    async def _get_sample_rows(self, table_name: str) -> str:
        """Get sample rows from table"""
        try:
            limit = self.sample_rows_in_table_info

            result = await self.db.execute(
                text(f'SELECT * FROM "{table_name}" LIMIT {limit}'),
            )

            rows = result.fetchall()
            if not rows:
                return f"{limit} rows from {table_name} table:\n(No data)"

            # Get column names
            columns = result.keys()
            columns_str = "\t".join(columns)

            # Format rows (truncate long strings)
            rows_str = []
            for row in rows:
                row_values = []
                for val in row:
                    val_str = str(val)[:self.max_string_length]
                    row_values.append(val_str)
                rows_str.append("\t".join(row_values))

            return (
                f"{limit} rows from {table_name} table:\n"
                f"{columns_str}\n"
                f"\n".join(rows_str)
            )

        except Exception as e:
            logger.warning(f"Failed to get sample rows for {table_name}: {e}")
            # Rollback to prevent transaction abort
            await self.db.rollback()
            return f"{self.sample_rows_in_table_info} rows from {table_name} table:\n(Error: {e})"

    async def run(self, query: str) -> str:
        """
        Execute SQL query and return formatted results.

        Args:
            query: SQL query to execute

        Returns:
            String representation of query results
        """
        try:
            logger.info(f"Executing query: {query}...")

            result = await self.db.execute(text(query))

            # Check if query returns rows
            if result.returns_rows:
                rows = result.fetchall()

                if not rows:
                    return ""

                # Format as list of tuples (matching LangChain SQLDatabase format)
                formatted_rows = []
                for row in rows:
                    # Truncate long strings
                    truncated_values = []
                    for val in row:
                        if isinstance(val, str) and len(val) > self.max_string_length:
                            truncated_values.append(val[:self.max_string_length] + "...")
                        else:
                            truncated_values.append(val)
                    formatted_rows.append(tuple(truncated_values))

                return str(formatted_rows)
            else:
                # Query doesn't return rows (INSERT, UPDATE, etc.)
                return ""

        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            logger.error(error_msg)
            # Rollback to prevent transaction abort
            await self.db.rollback()
            return error_msg

    def clear_cache(self):
        """Clear table info cache"""
        self._table_cache.clear()
        logger.debug("Table info cache cleared")


async def get_async_sql_database(
    include_tables: Optional[List[str]] = None,
    sample_rows: int = 3
) -> AsyncSQLDatabase:
    """
    Factory function to create AsyncSQLDatabase instance.

    Args:
        include_tables: List of tables to include
        sample_rows: Number of sample rows in schema

    Returns:
        AsyncSQLDatabase instance

    Usage:
        async for db_session in get_db():
            sql_db = await get_async_sql_database(
                include_tables=["product", "store_master"]
            )
            table_info = await sql_db.get_table_info()
            result = await sql_db.run("SELECT * FROM product LIMIT 10")
            break
    """
    # Get async session
    async for db_session in get_db():
        return AsyncSQLDatabase(
            db=db_session,
            include_tables=include_tables,
            sample_rows_in_table_info=sample_rows
        )

    raise RuntimeError("Failed to get database session")
