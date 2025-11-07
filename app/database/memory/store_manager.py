"""
LangGraph Store Manager
Provides long-term memory storage using PostgreSQL
"""
from typing import Optional
from langgraph.store.postgres.aio import AsyncPostgresStore
from langchain.embeddings import init_embeddings
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
        Initialize PostgreSQL store with semantic search enabled.
        Creates store tables if not exits.
        """
        if self._store is not None:
            logger.warning("Store already initialized, skipping...")
            return

        try:
            logger.info("🧠 Initializing long-term memory (store) with semantic search...")

            # Create async store with connection string
            conn_string = self.settings.POSTGRES_URL

            # Log masked DB URL for debugging
            masked_url = conn_string.replace(self.settings.DB_PASSWORD, "***")
            logger.info(f"📊 Database URL: {masked_url}")

            # Enter the async context manager with semantic search configuration
            try:
                self._store_cm = AsyncPostgresStore.from_conn_string(
                    conn_string,
                    index={
                        "embed": init_embeddings("qwen3-embedding:4b", provider="ollama", base_url="http://localhost:11434"),
                        "dims": 2000,
                        "fields": ["$"]  # "$" means embed all fields
                    }
                )
                self._store = await self._store_cm.__aenter__()
                logger.info("✅ Store context manager initialized")

            except Exception as e:
                logger.error(f"❌ Failed to create store context manager: {e}", exc_info=True)
                raise

            # Setup store tables (creates if not exist)
            try:
                logger.info("📋 Setting up store tables (store, store_vectors)...")
                await self._store.setup()
                logger.info("✅ Store tables created/verified")

            except Exception as e:
                logger.error(f"❌ Failed to setup store tables: {e}", exc_info=True)
                logger.error("💡 Troubleshooting:")
                logger.error("   1. Make sure PostgreSQL user has CREATE TABLE permissions")
                logger.error("   2. Verify pgvector extension is installed: CREATE EXTENSION vector;")
                logger.error("   3. Check database connection is working")
                raise

            logger.info("✅ Long-term memory initialized with PostgreSQL + Semantic Search")

        except Exception as e:
            logger.error(f"❌ Failed to initialize store: {e}", exc_info=True)
            # Clean up on failure
            if self._store_cm is not None:
                try:
                    await self._store_cm.__aexit__(None, None, None)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup store on error: {cleanup_error}")
                self._store_cm = None
                self._store = None
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