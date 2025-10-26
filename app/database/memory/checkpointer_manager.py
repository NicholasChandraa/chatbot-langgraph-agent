"""
LangGraph Checkpointer Manager
Provides persistent conversation memory using PostgreSQL
"""
from typing import Optional
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.config.settings.settings import get_settings
from app.utils.logger import logger

class CheckpointerManager:
    """
    Manages LangGraph checkpointer for conversation persistence.

    Uses AsyncPostgresSaver to store:
    - Conversation state per thread (session_id)
    - Message history
    - Agent state snapshots
    """

    def __init__(self):
        self.settings = get_settings()
        self._checkpointer: Optional[AsyncPostgresSaver] = None
        self._checkpointer_cm = None  # Store context manager for cleanup
    
    async def init(self):
        """
        Initialize PostgreSQL checkpointer.
        Creates checkpoint tables if not exist.
        """
        if self._checkpointer is not None:
            logger.warning("Checkpointer already initialized, skipping...")
            return

        try:
            logger.info("🧠 Initializing conversation memory (checkpointer)...")

            # Create async checkpointer with connection string
            # from_conn_string returns context manager, we need to enter it
            conn_string = self.settings.POSTGRES_URL
            
            # Enter the async context manager
            self._checkpointer_cm = AsyncPostgresSaver.from_conn_string(conn_string)
            self._checkpointer = await self._checkpointer_cm.__aenter__()

            # Setup checkpoint tables (creates if not exist)
            await self._checkpointer.setup()

            logger.info("✅ Conversation memory initialized with PostgreSQL")
        except Exception as e:
            logger.error(f"❌ Failed to initialize checkpointer: {e}", exc_info=True)
            raise
    
    async def close(self):
        """Close checkpointer connections"""
        if self._checkpointer is None:
            logger.warning("Checkpointer not initialized, skipping close...")
            return

        try:
            logger.info("Closing conversation memory...")

            # Exit the context manager properly
            if self._checkpointer_cm is not None:
                await self._checkpointer_cm.__aexit__(None, None, None)
            
            self._checkpointer = None
            self._checkpointer_cm = None

            logger.info("✅ Conversation memory closed")

        except Exception as e:
            logger.error(f"❌ Error closing checkpointer: {e}", exc_info=True)
    
    def get_checkpointer(self) -> AsyncPostgresSaver:
        """
        Get checkpointer instance.

        Returns:
            AsyncPostgresSaver instance
        
        Raises:
            RuntimeError: If checkpointer not initialized
        """
        if self._checkpointer is None:
            raise RuntimeError(
                "Checkpointer not initialized. Call checkpointer_manager.init() first."
            )
        return self._checkpointer

# Singleton instance
checkpointer_manager = CheckpointerManager()
