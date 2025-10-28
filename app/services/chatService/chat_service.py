"""
Chat Service - Business Logic Layer
Handles chat message processing and agent orchestration
"""
import time
from typing import Dict, Any
from langchain_core.messages import HumanMessage

from app.schemas.chat_schema import ChatResponse
from app.utils.logger import logger
from app.agents.supervisor_agent import create_supervisor_agent
from app.repositories.repository_container import RepositoryContainer
from app.services.chatService.base_chat_service import BaseChatService
from langgraph.graph.state import CompiledStateGraph

class ChatService(BaseChatService):
    """
    Service for handling regular chat operations.

    Inherits shared logic from BaseChatService:
    - Memory management (_get_checkpointer, _get_store)
    - User context loading (_load_user_context, _format_user_context_string)
    - Message processing (_extract_current_turn_response, _extract_text_from_content)
    - Debug utilities (_debug_log_messages)
    """
    
    @staticmethod
    async def process_message(
        message: str,
        user_id: str,
        session_id: str,
        repos: RepositoryContainer
    ) -> ChatResponse:
        """
        Process a chat message through the AI agent system
        
        Args:
            message: User's message text
            user_id: User identifier
            session_id: Conversation session identifier
            repos: Repository Container
            
        Returns:
            ChatResponse with AI response and metadata
        """
        start_time = time.time()
        
        logger.info(f"Chat Request | user={user_id} | session={session_id}")
        logger.info(f"Question: {message}")

        # Get checkpointer for conversation memory
        checkpointer = ChatService._get_checkpointer(session_id)
        store = ChatService._get_store(session_id)

        # Load user context from long-term memory
        user_context = await ChatService._load_user_context(user_id)
        user_context_string = ChatService._format_user_context_string(user_context)

        # Create supervisor agent with checkpointer, store, and user context
        app = await create_supervisor_agent(
            repos,
            checkpointer=checkpointer,
            store=store,
            user_context=user_context_string
        )
        
        # Invoke agent with message
        result = await ChatService._invoke_agent(
            app=app,
            message=message,
            session_id=session_id,
            user_id=user_id
        )
        
        logger.warning(f"RESULT COMPILE-AN SUPERVISOR: {result}")
        
        # Extract messages from result
        all_messages = result.get("messages", [])
        
        # Debug logging for message flow
        ChatService._debug_log_messages(all_messages)
        
        # Extract AI responses from current turn only
        response_text = ChatService._extract_current_turn_response(all_messages)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"✅ Completed | "
            f"time={processing_time:.2f}ms | "
            f"messages={len(all_messages)}"
        )
        
        # Build response
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            metadata={
                "supervisor": "supervisor_agent",
                "processing_time_ms": round(processing_time, 2),
                "message_count": len(all_messages),
                "memory_enabled": checkpointer is not None,
                "store_enabled": store is not None,
                "user_context_loaded": user_context.get("has_data", False),
                "thread_id": session_id
            }
        )

    @staticmethod
    async def _invoke_agent(
        app: CompiledStateGraph,
        message: str,
        session_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Invoke agent with user message
        
        Args:
            app: Compiled graph application
            message: User message
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            Agent execution result
        """
        config = {
            "configurable": {
                "thread_id": session_id,
                "user_id": user_id,
            }
        }
        
        logger.info("🚀 Invoking supervisor...")
        result = await app.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )

        # async for event in app.astream(
        #     {"messages": [HumanMessage(content=message)]},
        #     config=config
        # ):
        #     logger.warning(f"EVENT ASTREAM: {event}")

        return result


# Singleton instance
chat_service = ChatService()
