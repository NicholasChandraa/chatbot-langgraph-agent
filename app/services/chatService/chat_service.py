"""
Chat Service - Business Logic Layer
Handles chat message processing and agent orchestration
"""
import time
from typing import Dict, List, Any, AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from app.schemas.chat_schema import ChatResponse
from app.utils.logger import logger
from app.agents.supervisor_agent import create_supervisor_agent
from app.database.memory.checkpointer import checkpointer_manager
from langgraph.graph.state import CompiledStateGraph

class ChatService:
    """Service for handling chat operations"""
    
    @staticmethod
    async def process_message(
        message: str,
        user_id: str,
        session_id: str,
        db: AsyncSession
    ) -> ChatResponse:
        """
        Process a chat message through the AI agent system
        
        Args:
            message: User's message text
            user_id: User identifier
            session_id: Conversation session identifier
            db: Database session
            
        Returns:
            ChatResponse with AI response and metadata
        """
        start_time = time.time()
        
        logger.info(f"Chat Request | user={user_id} | session={session_id}")
        logger.info(f"Question: {message}")
        
        # Get checkpointer for conversation memory
        checkpointer = ChatService._get_checkpointer(session_id)
        
        # Create supervisor agent with checkpointer (already compiled by DeepAgents)
        app = await create_supervisor_agent(db, checkpointer=checkpointer)
        
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
                "thread_id": session_id
            }
        )
    
    @staticmethod
    def _get_checkpointer(session_id: str):
        """
        Get checkpointer for conversation memory
        
        Args:
            session_id: Session identifier for logging
            
        Returns:
            Checkpointer instance or None if unavailable
        """
        try:
            checkpointer = checkpointer_manager.get_checkpointer()
            logger.info(f"🧠 Using persistent memory | session={session_id}")
            return checkpointer
        except Exception as e:
            logger.warning(f"⚠️ Checkpointer unavailable, using stateless mode: {e}")
            return None
    
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
    
    @staticmethod
    def _debug_log_messages(messages: List) -> None:
        """
        Log all messages for debugging purposes
        
        Args:
            messages: List of message objects
        """
        logger.debug(f"📨 Total messages: {len(messages)}")
        
        for i, msg in enumerate(messages):
            # Determine message type
            if isinstance(msg, HumanMessage):
                msg_type = "human"
            elif isinstance(msg, AIMessage):
                msg_type = "ai"
            elif isinstance(msg, ToolMessage):
                msg_type = "tool"
            elif isinstance(msg, SystemMessage):
                msg_type = "system"
            else:
                msg_type = "unknown"
            
            msg_name = getattr(msg, 'name', 'N/A')
            has_tool_calls = hasattr(msg, 'tool_calls') and bool(msg.tool_calls)
            
            # Extract content preview (handle both string and list formats)
            raw_content = getattr(msg, 'content', '')
            if isinstance(raw_content, list):
                # Extract text from content blocks for preview
                content_preview = ChatService._extract_text_from_content(raw_content)[:200]
            else:
                content_preview = str(raw_content)[:200]

            # Log tool call details if present
            tool_info = ""
            if has_tool_calls:
                tool_names = [tc.get('name', 'unknown') for tc in msg.tool_calls]
                tool_info = f" | ToolsCalled={tool_names}"

            # Special detection for forwarding tool usage
            if isinstance(msg, ToolMessage) and 'forward' in str(msg_name).lower():
                logger.debug(f"  ⏩ [FORWARDING DETECTED] Message [{i}] uses forwarding mechanism")

            logger.debug(
                f"  [{i}] Type={msg_type} | Name={msg_name} | "
                f"HasTools={has_tool_calls}{tool_info} | Content={content_preview}"
            )
    
    @staticmethod
    def _extract_current_turn_response(messages: List) -> str:
        """
        Extract AI responses from the current conversation turn only
        
        This avoids returning responses from conversation history
        by finding the last HumanMessage and extracting only AI responses after it.
        
        Args:
            messages: List of message objects
            
        Returns:
            Combined AI response text
        """
        # Find the LAST HumanMessage (current user input)
        last_human_idx = -1
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                last_human_idx = i
        
        # Collect AI responses ONLY after the last HumanMessage
        ai_responses = []
        if last_human_idx >= 0:
            current_turn_messages = messages[last_human_idx + 1:]
            logger.debug(
                f"Processing {len(current_turn_messages)} messages from current turn "
                f"(after index {last_human_idx})"
            )
            
            for msg in current_turn_messages:
                if isinstance(msg, AIMessage):
                    # Extract text content (handle both string and list formats)
                    text_content = ChatService._extract_text_from_content(msg.content)
                    
                    if text_content and text_content.strip():
                        ai_responses.append(text_content)
                        logger.debug(
                            f"Collected AI response from {getattr(msg, 'name', 'unknown')}: "
                            f"{text_content[:100]}..."
                        )
        
        # Combine all responses
        if ai_responses:
            response_text = "\n\n".join(ai_responses)
            logger.info(f"✅ Combined {len(ai_responses)} AI responses into final answer")
            return response_text
        else:
            logger.warning("⚠️ No valid AI responses found")
            return "Maaf, saya tidak dapat menjawab pertanyaan ini."
    
    @staticmethod
    def _extract_text_from_content(content) -> str:
        """
        Extract text from message content (handles both string and list formats)
        
        LangChain content can be:
        1. String: "Some text"
        2. List of content blocks: [{'type': 'text', 'text': 'Some text', 'extras': {...}}]
        
        Args:
            content: Message content (string or list)
            
        Returns:
            Extracted text string
        """
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
            return " ".join(text_parts)
        
        return ""

# Singleton instance
chat_service = ChatService()
