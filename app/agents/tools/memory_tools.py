"""
Memory Tools
Tools untuk agent berinteraksi dengan long-term memory
"""
from langchain_core.tools import tool
from app.services.memory.memory_service import memory_service

@tool
async def save_user_info(user_id: str, key: str, value: str) -> str:
    """
    Save user information to long-term memory.

    Args:
        user_id: User identifier
        key: Information key (e.g., 'name', 'favorite_product')
        value: Information value

    Returns:
        Confirmation message
    """
    await memory_service.save_user_preference(user_id, key, value)
    return f"Saved {key} for user {user_id}"


@tool
async def get_user_info(user_id: str, key: str) -> str:
    """
    Get user information from long-term memory.

    Args:
        user_id: User identifier
        key: Information key

    Returns:
        Information value or "not found"
    """
    value = await memory_service.get_user_preference(user_id, key)
    if value:
        return f"User {user_id}'s {key}: {value}"
    return f"No {key} found for user {user_id}"


@tool
async def remember_fact(user_id: str, fact: str) -> str:
    """
    Remember a fact about the user.

    Args:
        user_id: User identifier
        fact: Fact to remember

    Returns:
        Confirmation message
    """
    await memory_service.add_user_memory(user_id, fact, "fact")
    return f"I'll remember that: {fact}"


@tool
async def recall_facts(user_id: str, query: str) -> str:
    """
    Recall facts about user using semantic search.

    Args:
        user_id: User identifier
        query: Search query

    Returns:
        Found facts
    """
    memories = await memory_service.search_user_memories(user_id, query, limit=3)

    if not memories:
        return "I don't recall anything about that."

    facts = [f"- {m['text']}" for m in memories]
    return "I remember:\n" + "\n".join(facts)