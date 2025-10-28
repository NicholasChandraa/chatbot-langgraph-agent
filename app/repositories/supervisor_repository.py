"""
Supervisor Repository
Handles data access for supervisor configuration
"""
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.agent_config.agent_config_manager import get_agent_config
from app.repositories.base.base_repository import BaseRepository
from app.utils.logger import logger

class SupervisorRepository(BaseRepository):
    """
    Repository for Supervisor Domain

    Scope:
    - Supervisor for handling sub-agents    
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize product repository

        Args:
            db (AsyncSession): SQLAlchemy async session
        """
        self.db = db
    
    async def get_config(self) -> Dict[str, Any]:
        """
        Get Supervisor Agent configuration

        Returns:
            Dict[str, Any]: Agent config dict
        """
        return await get_agent_config("supervisor", self.db)
    
    async def execute_query(self, question: str) -> str:
        """
        Not implemented for supervisor.

        Supervisor doesn't execute queries directly - it delegates
        to specialized sub-agents (product_agent, sales_agent, store_agent).

        Args:
            question: Natural language question

        Returns:
            Error message explaining supervisor doesn't query directly

        Raises:
            NotImplementedError: Always, as supervisor delegates to sub-agents
        """
        error_msg = (
            "Supervisor doesn't execute queries directly. "
            "It delegates to specialized sub-agents (product, sales, store)."
        )
        logger.warning(f"[supervisor] execute_query called but not implemented: {error_msg}")
        raise NotImplementedError(error_msg)