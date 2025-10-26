"""
Memory Service
Helper function for long-term memory operations
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.database.memory.store_manager import store_manager
from app.utils.logger import logger

class MemoryService:
    """Service for managing long-term memory operations"""
    
    @staticmethod
    async def save_user_preference(
        user_id: str,
        key: str,
        value: Any
    ) -> None:
        """
        Save user preference to long-term memory
        
        Args:
            user_id: User identifier
            key: Preference key (e.g., "favorite_product", "dietary_restrictions")
            value: Preference value
        """
        try:
            store = store_manager.get_store()
            
            await store.aput(
                namespace=("users", user_id, "preferences"),
                key=key,
                value={
                    "data": value,
                    "updated_at": datetime.now().isoformat()
                }
            )
            
            logger.info(f"💾 Saved user preference: {user_id}/{key}")
        except Exception as e:
            logger.error(f"❌ Failed to save user preference: {e}")
            
    @staticmethod
    async def get_user_preference(
        user_id: str,
        key: str
    ) -> Optional[Any]:
        """
        Get user preference from long-term memory

        Args:
            user_id (str): User identifier
            key (str): Preference key

        Returns:
            Optional[Any]: Preference value or None if not found
        """
        try:
            store = store_manager.get_store()
            
            result = await store.aget(
                namespace=("users", user_id, "preferences"),
                key=key
            )
            
            if result:
                return result.value.get("data")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get user preference: {e}")
            return None
    
    @staticmethod
    async def save_user_profile(
        user_id: str,
        profile_data: Dict[str, Any]
    ) -> None:
        """
        Save user profile information

        Args:
            user_id (str): User identifier
            profile_data (Dict[str, Any]): Profile data (name, phone, etc.)
        """
        try:
            store = store_manager.get_store()
            
            await store.aput(
                namespace=("users", user_id, "profile"),
                key="info",
                value={
                    **profile_data,
                    "updated_at": datetime.now().isoformat()
                }
            )
            
            logger.info(f"💾 Saved user profile: {user_id}")
        except Exception as e:
            logger.error(f"❌ Failed to save user profile: {e}")
    
    @staticmethod
    async def save_user_profile(
        user_id: str,
        profile_data: Dict[str, Any]
    ) -> None:
        """
        Save user profile information

        Args:
            user_id (str): User identifier
            profile_data (Dict[str, Any]): Profile data (name, phone, etc.)
        """
        try:
            store = store_manager.get_store()
            
            await store.aput(
                namespace=("users", user_id, "profile"),
                key="info",
                value={
                    **profile_data,
                    "updated_at": datetime.now().isoformat()
                }
            )
            
            logger.info(f"💾 Saved user profile: {user_id}")
        except Exception as e:
            logger.error(f"❌ Failed to save user profile: {e}")
            
    @staticmethod
    async def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user profile information

        Args:
            user_id (str): User identifier

        Returns:
            Optional[Dict[str, Any]]: Profile data or None if not found
        """
        try:
            store = store_manager.get_store()
            
            result = await store.aget(
                namespace=("users", user_id, "profile"),
                key="info"
            )
            
            if result:
                return result.value
            
            return None
        
        except Exception as e:
            logger.error(f"❌ Failed to get user profile: {e}")
            return None
    
    @staticmethod
    async def add_user_memory(
        user_id: str,
        memory_text: str,
        memory_type: str = "general"
    ) -> None:
        """
        Add a memory/fact about user

        Args:
            user_id (str): User identifier
            memory_text (str): Memory text
            memory_type (str, optional): Type of memory (e.g., "preference", "fact", "general"). Defaults to "general".
        """
        try:
            store = store_manager.get_store()
            
            # Use timestamp as unique key
            key = f"{memory_type}_{int(datetime.now().timestamp() * 1000)}"
            
            await store.aput(
                namespace=("users", user_id, "memories"),
                key=key,
                value={
                    "text": memory_text,
                    "type": memory_type,
                    "created_at": datetime.now().isoformat()
                }
            )
            
            logger.info(f"💾 Saved user memory: {user_id}/{memory_type}")
        except Exception as e:
            logger.error(f"❌ Failed to save user memory: {e}")
            
    @staticmethod
    async def search_user_memories(
        user_id: str,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search user memories with semantic search
        
        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum results to return
        """
        try:
            store = store_manager.get_store()
            
            results = await store.asearch(
                namespace_prefix=("users", user_id, "memories"),
                query=query,
                limit=limit
            )
            
            memories = []
            for item in results:
                memories.append({
                    "text": item.value.get("text"),
                    "type": item.values.get("type"),
                    "created_at": item.value.get("created_at"),
                    "score": item.score  # Similarity score
                })
                
            logger.info(f"Found {len(memories)} memories for user {user_id}")
            return memories
        except Exception as e:
            logger.error(f"❌ Failed to search user memories: {e}")
            return []
    
    @staticmethod
    async def get_all_user_preferences(user_id: str) -> Dict[str, Any]:
        """
        Get all preferences for a user

        Args:
            user_id (str): User identifier

        Returns:
            Dict[str, Any]: Dictionary pf all preferences
        """
        try:
            store = store_manager.get_store()
            
            results = await store.asearch(
                namespace_prefix=("users", user_id, "preferences"),
                limit=100
            )
            
            preferences = {}
            for item in results:
                preferences[item.key] = item.value.get("data")
            
            return preferences
        except Exception as e:
            logger.error(f"❌ Failed to get user preferences: {e}")
            return {}
        
# Singleton instance
memory_service = MemoryService() 

    