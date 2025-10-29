"""
Sales Repository
Handles sales analytics data access
"""
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.base.base_repository import BaseRepository
from app.repositories.base.query_repository import QueryRepository
from app.utils.logger import logger

class SalesRepository(BaseRepository):
    """
    Repository for sales analytics domain.

    Scope:
    - Sales revenue and trends
    - Top selling products
    - Store performance comparison

    Tables: store_daily_single_item, product, store_master, branch
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize sales repository.

        Args:
            db (AsyncSession): SQLAlchemy async session
        """
        self._query_repo = QueryRepository(
            db=db,
            agent_name="sales_agent",
            tables=["store_daily_single_item", "product", "store_master", "branch"],
            cache_ttl_seconds=180  # 3 menit karena data sales sering berubah
        )

        logger.debug("SalesRepository Initialized")

    async def get_config(self) -> Dict[str, Any]:
        """
        Get sales agent configuration.

        Returns:
            Dict[str, Any]: Agent config dict
        """
        return await self._query_repo.get_config()

    async def execute_query(self, question) -> str:
        """
        Execute sales analytics query.

        Args:
            question (_type_): Natural language question about sales

        Returns:
            str: Formatted query results

        Examples:
            > result = await repo.execute_query("total penjualan kemarin?")
            > print(result)
            "[(3020000,)]"
        """
    
    # Domain-specific convenience methods
    
    async def get_total_revenue(self, date: str = "today") -> str:
        """
        Get total revenue for specific date.

        Args:
            date (str, optional): Date string (e.g., "today", "2025-10-28"). Defaults to "today".

        Returns:
            str: Total revenue
        """
        question = f"Berapa total revenue untuk {date}?"
        return await self.execute_query(question)
    
    async def get_top_products(self, limit: int = 10, period: str = "today") -> str:
        """
        Get top selling products.

        Args:
            limit (int, optional): Number of products to return. Defaults to 10.
            period (str, optional): Time period. Defaults to "today".

        Returns:
            str: Top products by sales
        """
        question = (
            f"Tampilkan {limit} produk terlaris untuk {period} "
            f"beserta nama produk dan jumlah terjual"
        )
        return await self.execute_query(question)
    
    async def get_store_performance(self, period: str = "this month") -> str:
        """
        Get store performance comparison

        Args:
            period (str, optional): Time Period. Defaults to "this month".

        Returns:
            str: Store performance data
        """
        question = f"Bandingkan performa penjualan semua toko untuk {period}"
        return await self.execute_query(question)
    
    async def get_sales_trend(self, days: int = 7) -> str:
        """
        Get sales trend for last N days.

        Args:
            days (int, optional): Number of days to analyze. Defaults to 7.

        Returns:
            str: Sales trend data
        """
        question = f"Tampilkan trend penjualan untuk {days} hari terakhir"
        return await self.execute_query(question)
    
    def clear_cache(self):
        """Clear cached sales queries."""
        self._query_repo.clear_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get repository statistics."""
        return self._query_repo.get_cache_stats()