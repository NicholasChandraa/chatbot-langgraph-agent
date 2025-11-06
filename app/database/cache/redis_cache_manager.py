"""
Redis Cache Manager
Centralized caching service using Redis for distributed, persistent caching
"""
from typing import Optional, Any, Dict, List
import json
from datetime import timedelta
from redis.asyncio import Redis, from_url
from redis.exceptions import RedisError

from app.config.settings import get_settings
from app.utils.logger import logger


class RedisCacheManager:
    """
    Centralized Redis cache manager for the application.

    Features:
    - Async Redis operations
    - JSON serialization/deserialization
    - TTL support
    - Pattern-based key operations
    - Health checks and stats
    - Graceful fallback on errors

    Usage:
        cache = RedisCacheManager()
        await cache.init()

        # Simple cache
        await cache.set("key", "value", ttl=300)
        value = await cache.get("key")

        # JSON cache
        await cache.set_json("config:agent", {"model": "gpt-5"}, ttl=600)
        config = await cache.get_json("config:agent")

        # Pattern operations
        await cache.clear_pattern("query:product_agent:*")
    """

    def __init__(self):
        """Initialize Redis cache manager"""
        self.settings = get_settings()
        self._redis: Optional[Redis] = None
        self._enabled = True  # bis di disabled kalo redis ga tersedia
    
    def _build_redis_url(self) -> str:
        """Build Redis connection URL from settings"""
        password_part = f":{self.settings.REDIS_PASSWORD}@" if self.settings.REDIS_PASSWORD else ""
        return (
            f"redis://{password_part}"
            f"{self.settings.REDIS_HOST}:{self.settings.REDIS_PORT}/"
            f"{self.settings.REDIS_DB}"
        )
    
    def _check_enabled(self) -> bool:
        """Check if Redis is enabled and connected"""
        if not self._enabled or self._redis is None:
            logger.debug("Redis not available, skipping cache operation")
            return False
        
        return True

    async def init(self) -> None:
        """
        Initialize Redis connection.

        Creates async Redis client and tests connection.
        If connetion failes, disables caching gracefully.
        """
        if self._redis is not None:
            logger.warning("Redis already initialized, skipping...")
            return
        
        try:
            logger.info("🔂 Connecting to Redis...")

            # Build Redis URL
            redis_url = self._build_redis_url()

            # Create async Redis client
            self._redis = from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                socket_timeout=5,
                socket_connect_timeout=5
            )

            # Test connection
            await self._redis.ping()

            logger.info(f"✅ Redis connected: {self.settings.REDIS_HOST}:{self.settings.REDIS_PORT}")
            self._enabled = True
        
        except RedisError as e:
            logger.error(f"❌ Redis connection failed: {e}")
            logger.warning(f"⚠️ Continuing without Redis cache (degraded mode)")
            self._redis = None
            self._enabled = False

        except Exception as e:
            logger.error(f"❌ Unexpected error initializing Redis: {e}", exc_info=True)
            self._redis = None
            self._enabled = False

    async def close(self) -> None:
        """Close Redis connection gracefully"""
        if self._redis is None:
            return
        
        try:
            logger.info("Closing Redis connection...")

            await self._redis.close()
            self._redis = None

            logger.info("✅ Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis: {e}")

    # ==================== BASIC OPERATIONS ====================

    async def get(self, key: str) -> Optional[str]:
        """
        Get string value from cache.

        Args:
            key (str): Cache key

        Returns:
            Optional[str]: Cached value or None if not found/error
        
        Example:
            value = await cache.get("user:12:name")
        """
        if not self._check_enabled():
            return None
        
        try:
            value = await self._redis.get(key)

            if value:
                logger.debug(f"Cache HIT: {key}")
            else:
                logger.debug(f"Cache MISS: {key}")
            
            return value
            
        except Exception as e:
            logger.warning(f"Redis GET error for key '{key}': {e}")
            return None
        
    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set string value in cache

        Args:
            key (str): Cache key
            value (str): String value to cache
            ttl (Optional[int], optional): Time-to-live in seconds. Defaults to None.

        Returns:
            bool: True if successful, False otherwise
        
        Example:
            await cache.set("session:abc", "user_data", ttl=3600)
        """
        if not self._check_enabled():
            return False
        
        try:
            if ttl:
                await self._redis.setex(key, ttl, value)
            else:
                await self._redis.set(key, value)
            
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True
        
        except RedisError as e:
            logger.warning(f"Redis SET error for key '{key}': {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache

        Args:
            key (str): Cache key to delete

        Returns:
            bool: True if deleted, False otherwise
        """
        if not self._check_enabled():
            return False
        
        try:
            deleted = await self._redis.delete(key)
            logger.debug(f"Cache DELETE: {key} (deleted: {deleted})")

            return deleted > 0

        except RedisError as e:
            logger.warning(f"Redis DELETE error for key '{key}': {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key (str): Cache key

        Returns:
            bool: True if exists, False otherwise
        """
        if not self._check_enabled():
            return False
        
        try:
            exists = await self._redis.exists(key)
            return exists > 0
        
        except RedisError as e:
            logger.warning(f"Redis EXISTS error for key '{key}': {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for existing key.

        Args:
            key (str): Cache key
            ttl (int): Time-to-live in seconds

        Returns:
            bool: True if successful, False otherwise
        """
        if not self._check_enabled():
            return False
        
        try:
            result = await self._redis.expire(key, ttl)
            return result
        except RedisError as e:
            logger.warning(f"Redis EXPIRE error for key '{key}': {e}")
            return False
    
    # ======================== JSON OPERATIONS ========================

    async def get_json(self, key: str) -> Optional[Any]:
        """
        Get JSON value from cache and deserialize.

        Args:
            key (str): Cache key

        Returns:
            Optional[Any]: Deserialized Python object or None

        Example:
            config = await cache.get_json("config:agent")
            # Returns: {"model": "gpt-4", "temperature": 0.7}
        """
        value = await self.get(key)

        if value is None:
            return None
        
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for key '{key}': {e}")
            return None
    
    async def set_json(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Serialize Python object to JSON and cache.

        Args:
            key (str): Cache key
            value (Any): Pyt
            ttl (Optional[int], optional): _description_. Defaults to None.

        Returns:
            bool: _description_
        """
        try:
            json_str = json.dumps(value, ensure_ascii=False)
            return await self.set(key, json_str, ttl)
        except (TypeError, ValueError) as e:
            logger.error(f"JSON encode error for key '{key}': {e}")
            return False
        
    
    # ==================== PATTERN OPERATIONS ====================
    async def get_keys(self, pattern: str) -> List[str]:
        """
        Get all keys matching pattern.

        Args:
            pattern (str): Redis key pattern (e.g., "query:product_agent:*)

        Returns:
            List[str]: List of matching keys
        """
        if not self._check_enabled():
            return []
        
        try:
            keys = await self._redis.keys(pattern)
            return keys

        except RedisError as e:
            logger.warning(f"Redis KEYS error for pattern '{pattern}': {e}")
            return []
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.

        Args:
            pattern (str): Redis key pattern

        Returns:
            int: Number of keys deleted
        
        Example:
            deleted = await cache.clear_pattern("query:product_agent:*")
        """
        if not self._check_enabled():
            return 0
        
        try:
            keys = await self.get_keys(pattern)

            if not keys:
                logger.debug(f"No keys found for pattern: {pattern}")
            
            deleted = await self._redis.delete(*keys)
            logger.info(f"🗑️ Deleted {deleted} keys matching '{pattern}'")
            return deleted
        
        except RedisError as e:
            logger.warning(f"Redis clear pattern error for '{pattern}': {e}")
    
    async def flush_all(self) -> bool:
        """
        Clear ALL keys in current database.
        USE WITH CAUTION - deletes everything!

        Returns:
            bool: True if successful, False otherwise
        """
        if not self._check_enabled():
            return False
        
        try:
            await self._redis.flushdb()
            logger.warning(f"🗑️ Redis database flushed (all keys deleted)")
            return True
        
        except RedisError as e:
            logger.error(f"Redis FLUSHDB error: {e}")
            return False
    

    # ==================== MONITORING & HEALTH ====================
    async def health_check(self) -> Dict[str, Any]:
        """
        Check Redis health and return status.

        Returns:
            Dict[str, Any]: Dict with health status, latency, and info
        """
        if not self._enabled or self._redis is None:
            return {
                "healthy": False,
                "enabled": False,
                "error": "Redis not initialized or disabled"
            }
        
        try:
            # Measure ping latency
            import time
            start = time.time()
            await self._redis.ping()
            latency_ms = (time.time() - start) * 1000

            # Get Redis info
            info = await self._redis.info("stats")

            return {
                "healthy": True,
                "enabled": True,
                "latency_ms": round(latency_ms, 2),
                "host": self.settings.REDIS_HOST,
                "port": self.settings.REDIS_PORT,
                "db": self.settings.REDIS_DB,
                "total_commands_processed": info.get("total_commands_processed", 0),
                "connected_clients": info.get("connected_clients", 0)
            }

        except RedisError as e:
            return {
                "healthy": False,
                "enabled": True,
                "error": str(e)
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Any]: Dict with cache stats (memory, keys, hits/misses, etc.)
        """
        if not self._check_enabled():
            return {
                "enabled": False,
                "message": "Redis not available"
            }
        
        try:
            info = await self._redis.info()

            # Extract key metrics
            stats = {
                "enabled": True,
                "redis_version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "used_memory_peak_human": info.get("used_memory_peak_human"),
                "total_keys": await self._redis.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "evicted_keys": info.get("evicted_keys", 0),
                "expired_keys": info.get("expired_keys", 0),
                "connected_clients": info.get("connected_clients", 0)
            }

            # Calculate hit rate
            total_requests = stats["hits"] + stats["misses"]
            if total_requests > 0:
                stats["hit_rate"] = round(stats["hits"] / total_requests * 100, 2)
            else:
                stats["hit_rate"] = 0.0

            return stats
        
        except RedisError as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {
                "enabled": True,
                "error": str(e)
            }
    
    def is_enabled(self) -> bool:
        """Check if Redis cache is enabled and available"""
        return self._enabled and self._redis is not None

# Singleton instance
redis_cache_manager = RedisCacheManager()

async def get_redis_cache() -> RedisCacheManager:
    """
    Get Redis cache manager singleton.

    Returns:
        RedisCacheManager instance
    
    Usage:
        cache = await get_redis_cache()
        await cache.set("key", "value")
    """
    return redis_cache_manager