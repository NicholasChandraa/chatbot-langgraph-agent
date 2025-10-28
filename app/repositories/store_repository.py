"""
Store Repository
Handles store-related data access
"""
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.base.base_repository import BaseRepository
from app.repositories.base.query_repository import QueryRepository
from app.utils.logger import logger


class StoreRepository(BaseRepository):
    """
    Repository for store domain.

    Scope:
    - Store information and locations
    - Branch details
    - Store status and availability

    Tables: store, branch
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize store repository.

        Args:
            db (AsyncSession): db: SQLAlchemy async session
        """
        self._query_repo = QueryRepository(
            db=db,
            agent_name="store_agent",
            tables=["store", "branch"],
            cache_ttl_seconds=600  # 10 minutes (data toko jarang berubah)
        )

        logger.debug("StoreRepository initialized")
    
    async def get_config(self) -> Dict[str, Any]:
        """
        Get store agent configuration.

        Returns:
            Dict[str, Any]: Agent config dict
        """
        return await self._query_repo.get_config()
    
    async def execute_query(self, question: str) -> str:
        """
        Execute store-related query.

        Args:
            question (str): Natural language question about stores

        Returns:
            str: Formatted query results

        Examples:
            > result = await repo.execute("toko di cabang CENTRAL KITCHEN ANCOL")
            > print(result)
            "[('FX SUDIRMAN', 'TKLPS', 'Jakarta')]"
        """
        return await self._query_repo.execute_nl_query(question)
    
    # Domain-specific convenience methods

    async def search_stores(self, keyword: str) -> str:
        """
        Search stores by keyword (name or location).

        Args:
            keyword (str): Search keyword

        Returns:
            str: Matching stores
        """
        question = f"Cari toko yang namanya atau lokasinya mengandung '{keyword}'"
        return await self.execute_query(question)
    
    async def get_stores_by_code(self, store_code: str) -> str:
        """
        Get store by store code.

        Args:
            store_code (str): Store code

        Returns:
            str: Store information
        """
        question = f"Ambil informasi toko dengan kode '{store_code}'"
        return await self.execute_query(question)
    
    async def get_all_stores(self) -> str:
        """
        Get all active stores.

        Returns:
            str: All stores
        """
        question = "Tampilkan semua toko yang aktif beserta lokasinya"
        return await self.execute_query(question)
    
    async def get_stores_by_location(self, location: str) -> str:
        """
        Get stores in specific location

        Args:
            location (str): location/city name

        Returns:
            str: Stores in location
        """
        question = f"Tampilkan semua toko yang ada di {location}"
        return await self.execute_query(question)
    
    def clear_cache(self):
        """Clear cached store queries."""
        self._query_repo.clear_cache()

    def get_stats(self) -> Dict[str, Any]:
        """Get repository statistics"""
        return self._query_repo.get_cache_stats()