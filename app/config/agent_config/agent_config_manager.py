import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.database.model.agent_config import AgentConfig
from app.utils.logger import logger
from app.database.cache.redis_cache_manager import get_redis_cache

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

    # ---------------------- Redis Helpers ----------------------
    @staticmethod
    def _redis_key(agent_name: str) -> str:
        return f"agent_config:{agent_name}"
    
    async def _redis_get(self, agent_name: str) -> Optional[Dict[str, Any]]:
        cache = await get_redis_cache()
        if not cache.is_enabled():
            return None
        
        try:
            # Gunakan get_json bila tersedia
            data = await cache.get_json(self._redis_key(agent_name))

            if isinstance(data, dict):
                return data
            
            # fallback kalau backend hanya simpan string
            if isinstance(data, str):
                return json.loads(data)
        except Exception as e:
            logger.warning(f"[AgentConfig] Redis GET error for '{agent_name}': {e}")
        
        return None

    async def _redis_set(self, agent_name: str, config: Dict[str, Any]) -> None:
        cache = await get_redis_cache()
        if not cache.is_enabled():
            return
        
        try:
            await cache.set_json(self._redis_key(agent_name), config, ttl=self._cache_ttl)
        except Exception as e:
            logger.warning(f"[AgentConfig] Redis SET error for '{agent_name}': {e}")


    async def get_config(
        self,
        agent_name: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Ambil konfigurasi Agent.

        Urutan lookup:
        1) Redis (jika aktif)
        2) Local hot cache (in-memory, TTL), fallback jika redis tidak aktif
        3) Database (source of truth), lalu tuliskan kembali ke Redis + local
        """
        # 1) Redis first
        redis_cfg = await self._redis_get(agent_name)
        if redis_cfg:
            logger.debug(f"✅ Using Redis cached config for '{agent_name}'")

            # segarkan hot cache lokal untuk akses cepat
            self._update_cache(agent_name, redis_cfg)
            return redis_cfg
        
        # 2) Local hot cache (fallback)
        if self._is_cache_valid(agent_name):
            logger.debug(f"✅ Using local cached config for '{agent_name}'")
            return self._cache[agent_name]
        
        # 3) DB -> set ke Redis + local
        config = await self._load_config_from_db(agent_name, db)
        await self._redis_set(agent_name, config)
        self._update_cache(agent_name, config)
        
        return config


    async def _load_config_from_db(
            self,
            agent_name: str,
            db: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """Load config from database"""
        stmt = select(AgentConfig).where(
            AgentConfig.agent_name == agent_name,
            AgentConfig.is_active.is_(True)
        )

        try:
            result = await db.execute(stmt)
            config_row = result.scalar_one_or_none()
        except SQLAlchemyError as e:
            # Timeout, koneksi putus, dsb.
            logger.exception("DB error saat load AgentConfig untuk %s", agent_name)
            # Naikkan exception yang lebih netral ke service layer
            raise RuntimeError("Gagal mengakses database") from e
        
        if config_row is None:
                logger.warning(f"[DANGEROUS] agent config tidak ditemukan dengan nama {agent_name}")
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
        Invalidate cache lokal (in-memory) - sinkron & cepat
        """
        if agent_name:
            self._cache.pop(agent_name, None)
            self._last_refresh.pop(agent_name, None)
            logger.info(f"🗑️ Cache invalidated for '{agent_name}'")
        else:
            self._cache.clear()
            self._last_refresh.clear()
            logger.info(f"🗑️ All config cache invalidated")

    async def invalidate_cache_redis(self, agent_name: Optional[str] = None) -> int:
        """
        Invalidate cache di Redis. Return jumlah key yang terhapus.

        Dibuat async supaya route bisa 'await' tanpa blocking.

        Args:
            agent_name (Optional[str], optional): nama agent yang ingin di invalidate cachenya. Defaults to None.
        """
        cache = await get_redis_cache()
        if not cache.is_enabled():
            return 0
        
        pattern = self._redis_key(agent_name) if agent_name else "agent_config:*"

        try:
            deleted = await cache.clear_pattern(pattern)

            logger.info(f"🗑️ Redis cache invalidated | pattern={pattern} | deleted = {deleted}")
            
            return deleted
        except Exception as e:
            logger.warning(f"[AgentConfig] Redis invalidate error (pattern={pattern}): {e}")
            return 0
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics (for monitoring)"""
        return {
            "local_cache_size": len(self._cache),
            "cache_ttl_seconds": self._cache_ttl,
            "redis_namespace": "agent_config:*"
        }

# Singleton instance
agent_config_manager = AgentConfigManager(cache_ttl_seconds=300)

async def get_agent_config(
    agent_name: str,
    db: AsyncSession
) -> Dict[str, Any]:
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