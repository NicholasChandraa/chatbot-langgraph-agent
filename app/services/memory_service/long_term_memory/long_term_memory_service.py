"""
Composite Backend Factory for DeepAgents
Menyediakan filesystem-based memory dengan persistent dan transient storage
"""
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from app.utils.logger import logger

def create_memory_backend(runtime):
    """
    Create composite backend for DeepAgents long-term memory.

    Storage Architecture:
    - Regular files (/notes.txt, /draft.md) -> StateBackend (transient, per-thread)
    - Memory files (/memories/*) -> StoreBackend (persistent, cross-thread)

    Args:
        runtime (_type_): DeepAgents runtime context (automatically injected)

    Returns:
        CompositeBackend instance with routing configuration

    Example paths:
        - /notes.txt -> Transient (deleted after thread ends)
        - /memories/user_preference.txt -> Persistent (survives across threads)
        - /memories/instructions.txt -> Persistent (agent self-improvement)
    """
    try:
        backend = CompositeBackend(
            default=StateBackend(runtime),  # Default: transient storage
            routes={
                "/memories/": StoreBackend(runtime)   # /memories/* -> persistent
            }
        )

        logger.info("✅ Composite backend created (transient + persistent storage)")
        return backend
    
    except Exception as e:
        logger.error(f"❌ Failed to create composite backend: {e}")
        return StateBackend(runtime)