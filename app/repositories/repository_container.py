"""
Repository Container
Dependency Injection container for all repositories
"""
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.supervisor_repository import SupervisorRepository
from app.repositories.product_repository import ProductRepository
from app.repositories.sales_repository import SalesRepository
from app.repositories.store_repository import StoreRepository
from app.repositories.token_usage_repository import TokenUsageRepository
from app.utils.logger import logger

@dataclass
class RepositoryContainer:
    """
    Container for all domain repositories.

    This provides a single injection point for all repositories,
    making it easy to pass repository dependencies through the application.

    Usage in dependency injection:
        @router.post("/chat")
        async def chat(repos: RepositoryContainer = Depends(get_repositories)):
            # Use repos.product, repos.sales, repos.store
            ...

    Benefits:
    - Single injection point (not multiple parameters)
    - Type-sage access to repositories
    - Easy to extend with new repositories
    - Clear documentation of available repositories
    """

    supervisor: SupervisorRepository
    product: ProductRepository
    sales: SalesRepository
    store: StoreRepository
    token_usage: TokenUsageRepository

    # Add more repositories bisa disini
    # contoh: inventory: InventoryRepository

    @classmethod
    def create(cls, db: AsyncSession) -> "RepositoryContainer":
        """
        Factory method to create repostiroy container

        This is the main entry point for creating all repositories
        from a single database session.

        Args:
            db (AsyncSession): SQLAlchemy async session

        Returns:
            RepositoryContainer: RepositoryContainer with all repositories initialized
        
        Example:
              async def some_function(db: AsyncSession):
                  repos = RepositoryContainer.create(db)

                  # Access repositories
                  products = await repos.product.search_products("donut")
                  sales = await repos.sales.get_total_revenue("today")
        """
        logger.debug("Creating RepositoryContainer...")

        container = cls(
            supervisor=SupervisorRepository(db),
            product=ProductRepository(db),
            sales=SalesRepository(db),
            store=StoreRepository(db),
            token_usage=TokenUsageRepository(db),
        )

        logger.info("✅ RepositoryContainer created with all repositories")

        return container
    
    def clear_all_caches(self):
        """
        Clear caches for all repositories.

        Useful for:
        - Testing
        - Manual cache invalidation
        - Freeing memory
        """
        # Clear caches for repositories that have them
        self.product.clear_cache()
        self.sales.clear_cache()
        self.store.clear_cache()
        
        logger.info(f"🗑️ Clearing all repository caches...")

    def get_all_stats(self) -> dict:
        """
        Get statistics for all repositories

        Returns:
            dict: Dict with stats for each repository
        """
        return {
            "product": self.product.get_stats(),
            "sales": self.sales.get_stats(),
            "store": self.store.get_stats(),
        }