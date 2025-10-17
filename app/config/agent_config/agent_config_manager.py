from typing import Optional, Dict
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.model.agent_config import AgentConfig
from app.utils.logger import logger

class AgentConfigManager:
    """
    Simple database-only agent config manager with caching.
    Cache TTL: 5 minutes (configurable)
    """

    def __init__(self, cache_ttl_seconds: int = 300):
        """
        Initialize config manager.

        Args:
            cache_ttl_seconds: Cache time-to-live in seconds (default: 5 minutes)
        """
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = cache_ttl_seconds
        self._last_refresh: Dict[str, datetime] = {}

    async def get_config(
        self,
        agent_name: str,
        db: AsyncSession
    ) -> Dict[str, any]:
        """
        Get agent configuration from database (with cache).

        Args:
            agent_name: Name of agent (e.g., 'sql_agent')
            db: Database session

        Returns:
            Dict with: provider, model_name, temperature, max_tokens, etc.

        Raises:
            ValueError: If agent config not found in database
        """
        # Check cache first
        if self._is_cache_valid(agent_name):
            logger.debug(f"✅ Using cached config for '{agent_name}'")
            return self._cache[agent_name]
        
        # Load fom database
        logger.debug(f"Loading config for '{agent_name}' from database...")
        config = await self._load_from_db(agent_name, db)

        if config is None:
            raise ValueError(
                f"Agent config not found for '{agent_name}'."
                f"Please ensure agent_configs table has entry for this agent."
            )
        
        # Update cache
        self._update_cache(agent_name, config)
        logger.info(
            f"✅ Config loaded for '{agent_name}': "
            f"{config['llm_provider']}/{config['model_name']} "
            f"(temp={config['temperature']})"
        )

        return config
    
    async def _load_from_db(
            self,
            agent_name: str,
            db: AsyncSession
    ) -> Optional[Dict[str, any]]:
        """Load config from database"""
        stmt = select(AgentConfig).where(
            AgentConfig.agent_name == agent_name,
            AgentConfig.is_active == True
        )

        result = await db.execute(stmt)
        config_row = result.scalar_one_or_none()

        if config_row is None:
            return None
        
        return {
            "agent_name": config_row.agent_name,
            "llm_provider": config_row.llm_provider,
            "model_name": config_row.model_name,
            "temperature": config_row.temperature,
            "max_tokens": config_row.max_tokens,
            "config_metadata": config_row.config_metadata or {}
        }
    
    def _is_cache_valid(self, agent_name: str) -> bool:
        """Check if cache entry is still valid"""
        if agent_name not in self._cache:
            return False
        
        last_refresh = self._last_refresh.get(agent_name)
        if last_refresh is None:
            return False
        
        age_seconds = (datetime.now() - last_refresh).total_seconds()

        return age_seconds < self._cache_ttl
    
    def _update_cache(self, agent_name: str, config: Dict):
        """Update cache with new config"""
        self._cache[agent_name] = config
        self._last_refresh[agent_name] = datetime.now()

    def invalidate_cache(self, agent_name: Optional[str] = None):
        """
        Invalidate cache for specific agent or all agents.
        Call this after updating config in database.

        Args:
            agent_name: Specific agent to invalidate, or None for all
        """
        if agent_name:
            self._cache.pop(agent_name, None)
            self._last_refresh.pop(agent_name, None)
            logger.info(f"🗑️ Cache invalidated for '{agent_name}'")
        else:
            self._cache.clear()
            self._last_refresh.clear()
            logger.info(f"🗑️ All config cache invalidated")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics (for monitoring)"""
        return {
            "cached_agents": list(self._cache.keys()),
            "cache_size": len(self._cache),
            "cache_ttl_seconds": self._cache_ttl,
        }

# Singleton instance
agent_config_manager = AgentConfigManager(cache_ttl_seconds=300)

async def get_agent_config(
    agent_name: str,
    db: AsyncSession
) -> Dict[str, any]:
    """
    Get agent configuration from database.
    
    Usage:
        config = await get_agent_config("sql_agent", db)
        llm = LLMProviderFactory.create(
            provider_name=config["provider"],
            model_name=config["model_name"],
            temperature=config["temperature"]
        )
    
    Args:
        agent_name: Name of agent
        db: Database session
    
    Returns:
        Configuration dictionary
    
    Raises:
        ValueError: If config not found
    """
    return await agent_config_manager.get_config(agent_name, db)