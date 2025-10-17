from fastapi import APIRouter, Depends
from app.config.agent_config.agent_config_manager import agent_config_manager

router = APIRouter()

@router.post("/cache/invalidate")
async def invalidate_cache(agent_name: str = None):
    """
    Invalidate config cache.
    Useful after updating agent_configs in database.

    Args:
        agent_name: Specific agent to invalidate, or None for all
    """
    agent_config_manager.invalidate_cache(agent_name)

    return {
        "message": f"Cache invalidated for {agent_name or 'all agents'}",
        "cache_stats": agent_config_manager.get_cache_stats()
    }

@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics
    """
    return agent_config_manager.get_cache_stats()
