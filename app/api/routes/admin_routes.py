from fastapi import APIRouter, Depends
from app.config.agent_config.agent_config_manager import agent_config_manager
from app.database.cache.redis_cache_manager import get_redis_cache

router = APIRouter()

@router.post("/cache/invalidate")
async def invalidate_cache(agent_name: str = None):
    """
    Invalidate config cache.
    Useful after updating agent_configs in database.

    Args:
        agent_name: Specific agent to invalidate, or None for all
    """
    # Invalidate local cache (sync)
    agent_config_manager.invalidate_cache(agent_name)

    # Invalidate Redis cache (async)
    redis_deleted = await agent_config_manager.invalidate_cache_redis(agent_name)

    return {
        "message": f"Cache invalidated for {agent_name or 'all agents'}",
        "local_cache_cleared": True,
        "redis_keys_deleted": redis_deleted,
        "cache_stats": agent_config_manager.get_cache_stats()
    }

@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics
    """
    local_stats = agent_config_manager.get_cache_stats()

    cache = await get_redis_cache()
    redis_stats = await cache.get_stats()

    return {
        "local_cache": local_stats,
        "redis_cache": redis_stats
    }


@router.get("/cache/health")
async def cache_health_check():
    """
    Check Redis cache health
    """
    cache = await get_redis_cache()
    health = await cache.health_check()

    return {
        "redis": health,
        "local_cache": {
            "enabled": True,
            "size": agent_config_manager.get_cache_stats()["local_cache_size"]
        }
    }
