"""
LangGraph Store Manager
Provides long-term memory storage using PostgreSQL
"""
from typing import Optional
from langgraph.store.postgres.aio import AsyncPostgresStore
from app.config.settings.settings import get_settings
from app.utils.logger import logger

class StoreManager:
    """
    Manages LangGraph store for long-term memory.
    
    Uses AsyncPostgresStore to store:
    - User preferences across sessions
    - User facts and profile
    - Purchase history
    - Business context
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._store: Optional[AsyncPostgresStore] = None
        self._store_cm = None # Store context manager for cleanup
    
    async def init(self):
        """
        Initialize PostgreSQL store.
        Creates store tables if not exits.
        """
        if self._store is not None:
            logger.warning("Store already initialized, skipping...")
            return

        try:
            logger.info("🧠 Initializing long-term memory (store)...")

            # Create async store with connection string
            conn_string = self.settings.POSTGRES_URL

            # Enter the async context manager
            self._store_cm = AsyncPostgresStore.from_conn_string(conn_string)
            self._store = await self._store_cm.__aenter__()

            # Setup store tables (creates if not exist)
            await self._store.setup()

            logger.info("✅ Long-term memory initialized with PostgreSQL")
        except Exception as e:
            logger.error(f"❌ Failed to initialize store: {e}", exc_info=True)
            raise e

    async def close(self):
        """Close store connections"""
        if self._store is None:
            logger.warning("⚠️ Store not initialized, skipping close...")
            return
        
        try:
            logger.info("Closing long-term memory...")
            
            # Exit the context manager properly
            if self._store_cm is not None:
                await self._store_cm.__aexit__(None, None, None)
                
                self._store = None
                self._store_cm = None
                
                logger.info("✅ Long-term memory closed")
        except Exception as e:
            logger.error(f"❌ Error closing store: {e}", exc_info=True)
    
    def get_store(self) -> AsyncPostgresStore:
        """
        Get store instance
        
        Returns:
            AsyncPostgresStore instance

        Raises:
            RuntimeError: If store not initialized
        """
        if self._store is None:
            raise RuntimeError(
                "Store not initialized. Call store_manager.init() first."
            )
        return self._store

# Singleton intance
store_manager = StoreManager()