"""
Memory Context Loader
Helper to pre-load user context from LangGraph Store
"""
from typing import Dict, Any, Optional
from app.database.memory.store_manager import store_manager
from app.utils.logger import logger

async def load_user_context(user_id: str) -> Dict[str, Any]:
    """
    Pre-load user context from LangGraph Store for system prompt injection
    
    Loads:
    - Profile (name, phone, email, job)
    - Preferences (favorite products, dietary restrictions)

    Args:
        user_id (str): User identifier

    Returns:
        Dict[str, Any]: Dictionary with user context:
                        {
                            "has_data": bool,
                            "name": str | None,
                            "phone": str | None,
                            "email": str | None,
                            "job": str | None,
                            "preferences": list[dict]
                        }
    """
    try:
        store = store_manager.get_store()
        
        context = {
            "has_data": False,
            "name": None,
            "phone": None,
            "email": None,
            "job": None,
            "preferences": []
        }
        
        # Load profile (use async method)
        profile = await store.aget(("users", user_id), "profile")
        if profile:
            context["has_data"] = True
            context["name"] = profile.value.get("name")
            context["phone"] = profile.value.get("phone")
            context["email"] = profile.value.get("email")
            context["job"] = profile.value.get("job")

        # Load preference (get all without search, use async method)
        preferences = await store.asearch(("users", user_id, "preferences"), limit=10)
        if preferences:
            context["has_data"] = True
            context["preferences"] = [
                {
                    "type": item.value.get("type"),
                    "value": item.value.get("value")
                }
                for item in preferences
            ]
        
        if context["has_data"]:
            logger.info(f"✅ Loaded user context for {user_id}")
        else:
            logger.debug(f"No stored context found for {user_id}")
        
        return context
    
    except RuntimeError as e:
        # Store not initialized
        logger.warning(f"⚠️ Store not initialized: {e}")
        return {"has_data": False}

    except Exception as e:
        logger.error(f"❌ Failed to load user context: {e}", exc_info=True)
        return {"has_data": False}
    
def format_user_context_for_prompt(context: Dict[str, Any]) -> str:
    """
    Format user context for system prompt injection

    Args:
        context (Dict[str, Any]): User context from load_user_context()

    Returns:
        str: Formatted string for prompt injection (empty if no data)
    
    Example output:
        INFORMASI USER:
        - Nama: Nicholas Chandra
        - Telepon: 08123456789
        - Preferensi: favorite_products = Glazed Donut
    """
    if not context.get("has_data"):
        return ""
    
    lines = []
    
    # Profile info
    if context.get("name"):
        lines.append(f"- Nama: {context['name']}")
    if context.get("phone"):
        lines.append(f"- Phone: {context['phone']}")
    if context.get("email"):
        lines.append(f"- Email: {context['email']}")
    if context.get("job"):
        lines.append(f"- Job: {context['job']}")
    
    # Preferences
    preferences = context.get("preferences", [])
    if preferences:
        lines.append("\nPreferensi:")
        for pref in preferences:
            pref_type = pref.get("type", "unknown")
            pref_value = pref.get("value", "")
            lines.append(f" • {pref_type}: {pref_value}")
    
    if lines:
        return "INFORMASI USER:\n" + "\n".join(lines)
    
    return ""