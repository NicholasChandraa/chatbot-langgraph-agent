"""
Base Repository Abstract Class
Menentukan kontrak interface utnuk semua domain repositories
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseRepository(ABC):
    """
    Abstract base class for all repositories.

    All domain repositories should inherit from this and implement
    the required methods to ensure consistent interface.

    This enables:
    - Polymorphism for testing (easy mocking)
    - Contract enforcement via type hints
    - Clear documentation of repository capabilities
    """

    @abstractmethod
    async def get_config(self) -> Dict[str, Any]:
        """
        Get agent configuration for this domain.

        Returns:
            Dict containing:
                - agent_name: str
                - llm_provider: str (gemini, openai, etc.)
                - model_name: str
                - temperature: float
                - max_tokens: int
                - config_metadata: Dict (optional extras)
        Raises:
            ValueError: If config not found in database
        """
        pass

    @abstractmethod
    async def execute_query(self, question: str) -> str:
        """
        Execute natural language query agains domain tables.

        This is the primary method for data retrieval. It handles:
        1. Natual language -> SQL conversion
        2. Query validation and safety checks
        3. Execution with error handling
        4. Result formatting

        Args:
            question: Natural language question about the domain
        
        Returns:
            Formatted string with query results
        
        Examples:
            > await repo.execute_query("Show me all stores name")
            [('FS PALEMBANG', 'FS BANDUNG', 'FX SUDIRMAN')]
        """
        pass