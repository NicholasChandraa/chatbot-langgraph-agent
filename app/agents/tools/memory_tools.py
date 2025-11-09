"""
Long-term Memory Tools for LangGraph
Uses LangGraph Store for persistent user memory across conversations
"""
from typing import Any
from dataclasses import dataclass
from uuid import uuid4
from datetime import datetime

from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from app.utils.logger import logger


@dataclass
class Context:
    """Context schema for passing user_id to tools"""
    user_id: str


@tool
async def save_user_info(key: str, value: str, runtime: ToolRuntime[Context]) -> str:
    """
    Save user profile information (name, phone, email, etc).

    Use this to save core user profile data like:
    - name: User's name
    - phone: Phone number
    - email: Email address
    - job: User's job/occupation

    Args:
        key: Profile field name (e.g., 'name', 'phone', 'email', 'job')
        value: Value to save

    Returns:
        Confirmation message

    Example:
        save_user_info(key="name", value="Nicholas Chandra")
        save_user_info(key="phone", value="081234567890")
    """
    try:
        store = runtime.store
        user_id = runtime.context.user_id

        # Get existing profile or create new (use async method)
        existing = await store.aget(("users", user_id), "profile")
        profile = existing.value if existing else {}

        # Update field
        profile[key] = value
        profile["updated_at"] = datetime.now().isoformat()

        if "created_at" not in profile:
            profile["created_at"] = datetime.now().isoformat()

        # Save back to store (no embedding needed for profile, use async method)
        await store.aput(("users", user_id), "profile", profile, index=False)

        logger.info(f"💾 Saved user profile: {user_id}/{key}={value}")
        return f"✅ Berhasil menyimpan {key}: {value}"

    except Exception as e:
        logger.error(f"❌ Failed to save user info: {e}", exc_info=True)
        # Production: Fail soft
        # return f"⚠️ Gagal menyimpan {key}. Silakan coba lagi nanti."
        # Development: Fail hard
        raise e


@tool
async def save_preference(preference_type: str, value: str, runtime: ToolRuntime[Context]) -> str:
    """
    Save user preferences (favorite products, dietary restrictions, etc).

    Use this for user preferences like:
    - favorite_products: Products user likes
    - dietary_restrictions: Food restrictions (vegetarian, halal, allergies)
    - communication_preference: Preferred contact method

    Args:
        preference_type: Type of preference (e.g., 'favorite_products', 'dietary_restrictions')
        value: Preference value

    Returns:
        Confirmation message

    Example:
        save_preference(preference_type="favorite_products", value="Glazed Donut")
        save_preference(preference_type="dietary_restrictions", value="vegetarian")
    """
    try:
        store = runtime.store
        user_id = runtime.context.user_id

        # Store preference with semantic indexing for search
        preference_data = {
            "type": preference_type,
            "value": value,
            "created_at": datetime.now().isoformat()
        }

        # Use preference_type as key (allows updates, use async method)
        await store.aput(
            ("users", user_id, "preferences"),
            preference_type,
            preference_data,
            index=["value"]  # Only embed the value field
        )

        logger.info(f"💾 Saved preference: {user_id}/{preference_type}={value}")
        return f"✅ Berhasil menyimpan preferensi: {preference_type} = {value}"

    except Exception as e:
        logger.error(f"❌ Failed to save preference: {e}", exc_info=True)
        # Production: Fail soft
        # return f"⚠️ Gagal menyimpan preferensi. Silakan coba lagi nanti."
        # Development: Fail hard
        raise e


@tool
async def remember_fact(fact: str, context: str = "", runtime: ToolRuntime[Context] = None) -> str:
    """
    Remember a fact about user with semantic indexing.

    Use this to save important information user mentions that doesn't fit
    in profile or preferences. Facts are searchable semantically.

    Args:
        fact: The fact to remember (e.g., "User celebrates birthday in December")
        context: Optional context about when this was mentioned

    Returns:
        Confirmation message

    Example:
        remember_fact(fact="User merayakan ulang tahun di bulan Desember")
        remember_fact(fact="User suka beli donut untuk acara kantor", context="Diskusi tentang bulk order")
    """
    try:
        store = runtime.store
        user_id = runtime.context.user_id

        # Create unique ID for this fact
        fact_id = str(uuid4())

        fact_data = {
            "text": fact,
            "context": context,
            "created_at": datetime.now().isoformat()
        }

        # Store with full semantic indexing (use async method)
        await store.aput(
            ("users", user_id, "facts"),
            fact_id,
            fact_data,
            index=["text", "context"]  # Both fields searchable
        )

        logger.info(f"🧠 Remembered fact for {user_id}: {fact}")
        return f"✅ Saya akan mengingat: {fact}"

    except Exception as e:
        logger.error(f"❌ Failed to remember fact: {e}", exc_info=True)
        # Production: Fail soft
        # return "⚠️ Gagal menyimpan informasi. Silakan coba lagi nanti."
        # Development: Fail hard
        raise e


@tool
async def recall_facts(query: str, limit: int = 3, runtime: ToolRuntime[Context] = None) -> str:
    """
    Recall facts about user using semantic search.

    Use this to search for information user previously mentioned.
    Searches semantically (by meaning, not exact keywords).

    Args:
        query: Search query (e.g., "food allergies", "birthday", "favorite donut")
        limit: Maximum number of facts to return (default 3)

    Returns:
        Found facts or message if none found

    Example:
        recall_facts(query="makanan yang tidak boleh dimakan")
        recall_facts(query="kapan ulang tahun user")
    """
    try:
        store = runtime.store
        user_id = runtime.context.user_id

        # Semantic search in facts (use async method)
        results = await store.asearch(
            ("users", user_id, "facts"),
            query=query,
            limit=limit
        )

        if not results:
            logger.debug(f"No facts found for query: {query} (user: {user_id})")
            return "Saya tidak menemukan informasi terkait dengan itu."

        # Format results
        facts = []
        for item in results:
            fact_text = item.value.get("text", "")
            context = item.value.get("context", "")
            if context:
                facts.append(f"- {fact_text} (konteks: {context})")
            else:
                facts.append(f"- {fact_text}")

        result = "Saya ingat:\n" + "\n".join(facts)
        logger.info(f"📚 Recalled {len(results)} facts for user {user_id}")
        return result

    except Exception as e:
        logger.error(f"❌ Failed to recall facts: {e}", exc_info=True)
        # Production: Fail soft
        # return "⚠️ Gagal mengakses informasi tersimpan."
        # Development: Fail hard
        raise e


@tool
async def recall_preferences(query: str, limit: int = 3, runtime: ToolRuntime[Context] = None) -> str:
    """
    Recall user preferences using semantic search.

    Search for user preferences like favorite products, dietary restrictions, etc.

    Args:
        query: Search query (e.g., "favorite food", "allergies")
        limit: Maximum number of preferences to return (default 3)

    Returns:
        Found preferences or message if none found

    Example:
        recall_preferences(query="produk favorit")
        recall_preferences(query="pantangan makanan")
    """
    try:
        store = runtime.store
        user_id = runtime.context.user_id

        # Semantic search in preferences (use async method)
        results = await store.asearch(
            ("users", user_id, "preferences"),
            query=query,
            limit=limit
        )

        if not results:
            logger.debug(f"No preferences found for query: {query} (user: {user_id})")
            return "Saya tidak menemukan preferensi terkait dengan itu."

        # Format results
        prefs = []
        for item in results:
            pref_type = item.value.get("type", "")
            pref_value = item.value.get("value", "")
            prefs.append(f"- {pref_type}: {pref_value}")

        result = "Preferensi Anda:\n" + "\n".join(prefs)
        logger.info(f"📋 Recalled {len(results)} preferences for user {user_id}")
        return result

    except Exception as e:
        logger.error(f"❌ Failed to recall preferences: {e}", exc_info=True)
        # Production: Fail soft
        # return "⚠️ Gagal mengakses preferensi tersimpan."
        # Development: Fail hard
        raise e
