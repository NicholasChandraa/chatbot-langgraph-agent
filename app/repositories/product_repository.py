"""
Product Repository
Handles product-related data access
"""
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.base.base_repository import BaseRepository
from app.repositories.base.query_repository import QueryRepository
from app.utils.logger import logger

class ProductRepository(BaseRepository):
    """
    Repository for product domain

    Scope:
    - Product information (name, PLU, price)
    - Product categories and variants
    - Product search and filtering

    Tables: product
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize product repository

        Args:
            db (AsyncSession): SQLAlchemy async session
        """
        self._query_repo = QueryRepository(
            db=db,
            agent_name="product_agent",
            tables=["product"],
            cache_ttl_seconds=300  # 5 minutes
        )

        logger.debug("ProductRepository initialized")
    
    async def get_config(self) -> Dict[str, Any]:
        """
        Get product agent configuration.

        Returns:
            Dict[str, Any]: Agent config dict
        """
        return await self._query_repo.get_config()
    
    async def execute_query(self, question):
        """
        Execute product-related query.

        Args:
            question (_type_): Natural language question about products
        
        Returns:
            Formatted query results
        
        Examples:
            > result = await repo.execute_query("Kamu punya produk apa aja?")
            > print(result)
            "[('GLAZED DONUT', 'MOCHIDO CHOCOLATE', ...)]"
        """
        return await self._query_repo.execute_nl_query(question)

    # Domain-specific convenience methods
    async def search_products(self, keyword: str) -> str:
        """
        Search products by keyword.

        Args:
            keyword (str): Product name keyword

        Returns:
            str: Matchin products
        """
        question = f"Cari semua produk yang namanya mengandung '{keyword}'"
        return await self.execute_query(question)
    
    async def get_product_by_plu(self, plu: str) -> str:
        """
        Get product by PLU code.

        Args:
            plu: Product PLU code
        
        Returns:
            Product information
        """
        question = f"Ambil informasi produk dengan PLU '{plu}'"
        return await self.execute_query(question)
    
    async def get_products_by_price_range(
        self,
        min_price: int,
        max_price: int
    ) -> str:
        """
        Get products within price range.

        Args:
            min_price (int): Minimum price (IDR)
            max_price (int): Maximum price (IDR)

        Returns:
            str: Products in price range
        """
        question = (
            f"Tampilkan produk dengan harga antara "
            f"{min_price} dan {max_price} rupiah"
        )
        return await self.execute_query(question)
    
    def clear_cache(self):
        """Clear cached product queries."""
        self._query_repo.clear_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get repository statistics."""
        return self._query_repo.get_cache_stats()
