"""
Chat Stream Service - Business Logic Layer
Handles streaming chat message processing
"""
import time
import json
from typing import Dict, Any, AsyncIterator
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.utils.logger import logger
from app.agents.supervisor_agent import create_supervisor_agent
from app.repositories.repository_container import RepositoryContainer
from app.services.chatService.base_chat_service import BaseChatService

class ChatStreamService(BaseChatService):
    """
    Service for handling streaming chat operations.

    Inherits shared logic from BaseChatService.
    Only implements streaming-specific logic.
    """
    
    @staticmethod
    async def process_message_stream(
        message: str,
        user_id: str,
        session_id: str,
        repos: RepositoryContainer
    ) -> AsyncIterator[str]:
        """
        Process a chat message with streaming (astream)
        
        Yields events as Server-Sent Events (SSE) format:
        - start: Stream started
        - agent_start: Agent/subagent started
        - tool_start: Tool execution started
        - tool_end: Tool execution completed
        - message: AI message chunk
        - end: Stream completed with metadata
        - error: Error occurred
        
        Args:
            message: User's message text
            user_id: User identifier
            session_id: Conversation session identifier
            db: Database session
            
        Yields:
            SSE formatted strings with event data
        """
        start_time = time.time()
        
        try:
            logger.info(f"Chat Stream Request | user={user_id} | session={session_id}")
            logger.info(f"Question: {message}")
            
            # Send start event
            yield ChatStreamService._format_sse("start", {
                "session_id": session_id,
                "message": message
            })
            
            # Get checkpointer for short-term memory
            checkpointer = ChatStreamService._get_checkpointer(session_id)

            # Get store for long-term memory
            store = ChatStreamService._get_store(session_id)

            # Load user context from long-term memory
            user_context = await ChatStreamService._load_user_context(user_id)
            user_context_string = ChatStreamService._format_user_context_string(user_context)

            # Create supervisor agent with full context
            app = await create_supervisor_agent(
                repos,
                checkpointer=checkpointer,
                store=store,
                user_context=user_context_string
            )
            
            # Config for agent
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "user_id": user_id,
                }
            }
            
            # Stream events from agent
            collected_messages = []
            
            logger.info("🚀 Starting agent stream...")
            
            async for chunk in app.astream(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                stream_mode="values"  # Get full state values
            ):
                # In "values" mode, chunk is the full state
                logger.debug(f"Stream chunk: {chunk}")
                
                # Extract messages from state
                if "messages" in chunk:
                    messages = chunk["messages"]
                    
                    # Get new messages (compare with collected)
                    new_messages = messages[len(collected_messages):]
                    
                    for msg in new_messages:
                        collected_messages.append(msg)
                        
                        # Stream AI messages
                        if isinstance(msg, AIMessage):
                            text_content = ChatStreamService._extract_text_from_content(msg.content)
                            if text_content and text_content.strip():
                                # Try to determine agent from message metadata
                                agent_name = ChatStreamService._determine_agent_name(msg)
                                
                                yield ChatStreamService._format_sse("message", {
                                    "content": text_content,
                                    "name": agent_name,
                                    "type": "ai"
                                })
                        
                        # Stream tool messages
                        elif isinstance(msg, ToolMessage):
                            tool_name = getattr(msg, 'name', 'unknown')
                            yield ChatStreamService._format_sse("tool_result", {
                                "name": tool_name,
                                "type": "tool"
                            })
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Extract final response
            response_text = ChatStreamService._extract_current_turn_response(collected_messages)
            
            logger.info(
                f"✅ Stream Completed | "
                f"time={processing_time:.2f}ms | "
                f"messages={len(collected_messages)}"
            )
            
            # Send completion event
            yield ChatStreamService._format_sse("end", {
                "response": response_text,
                "metadata": {
                    "processing_time_ms": round(processing_time, 2),
                    "message_count": len(collected_messages),
                    "memory_enabled": checkpointer is not None,
                    "store_enabled": store is not None,
                    "user_context_loaded": user_context.get("has_data", False)
                }
            })
            
        except Exception as e:
            logger.error(f"Error in stream processing: {str(e)}", exc_info=True)
            yield ChatStreamService._format_sse("error", {
                "error": str(e)
            })

    # ============================================================
    # STREAMING-SPECIFIC METHODS
    # ============================================================

    @staticmethod
    def _determine_agent_name(msg: AIMessage) -> str:
        """
        Determine agent name from AIMessage
        
        Args:
            msg: AIMessage instance
            
        Returns:
            Agent name string
        """
        # Try to get name from message attribute
        if hasattr(msg, 'name') and msg.name:
            return msg.name
        
        # Try to infer from message ID or metadata
        msg_id = getattr(msg, 'id', '')
        if msg_id:
            # LangGraph includes agent info in message ID sometimes
            if 'product' in msg_id.lower():
                return 'product_agent'
            elif 'sales' in msg_id.lower():
                return 'sales_agent'
            elif 'store' in msg_id.lower():
                return 'store_agent'
        
        # Check response_metadata for hints
        response_metadata = getattr(msg, 'response_metadata', {})
        model_name = response_metadata.get('model_name', '')
        
        # Check if message has tool calls - likely from supervisor
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get('name', '')
                if tool_name == 'task':
                    # This is supervisor delegating to subagent
                    return 'supervisor'
        
        # Check content for hints (last resort)
        content = ChatStreamService._extract_text_from_content(msg.content)
        if content:
            # Subagents tend to give specific data responses
            if any(word in content.lower() for word in ['toko', 'store', 'jumlah toko']):
                return 'store_agent'
            elif any(word in content.lower() for word in ['produk', 'product', 'harga', 'plu']):
                return 'product_agent'
            elif any(word in content.lower() for word in ['penjualan', 'sales', 'revenue']):
                return 'sales_agent'
        
        # Default to supervisor
        return 'supervisor'
    
    @staticmethod
    def _format_sse(event: str, data: Dict[str, Any]) -> str:
        """
        Format data as Server-Sent Event
        
        Args:
            event: Event type (e.g., "message", "error", "end")
            data: Event data to send
            
        Returns:
            SSE formatted string
        """
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# Singleton instance
chat_stream_service = ChatStreamService()
