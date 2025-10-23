"""
Chat Service - Business Logic Layer
Handles chat message processing and agent orchestration
"""
import time
import json
from typing import Dict, List, Any, Optional, AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from app.schemas.chat_schema import ChatResponse
from app.utils.logger import logger
from app.agents.supervisor_agent import create_supervisor_agent
from app.database.memory.checkpointer import checkpointer_manager
from langgraph.graph.state import CompiledStateGraph

class ChatStreamEventService:
    """Service for handling chat operations"""
    
    @staticmethod
    async def process_message_stream_events(
        message: str,
        user_id: str,
        session_id: str,
        db: AsyncSession
    ) -> AsyncIterator[str]:
        """
        Process a chat message with token-by-token streaming (astream_events)
        
        Yields events as Server-Sent Events (SSE) format:
        - start: Stream started
        - token: Individual token from AI model (real-time typing effect)
        - tool_start: Tool execution started
        - tool_end: Tool execution completed
        - agent_start: Agent/subagent started
        - agent_end: Agent/subagent completed
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
            logger.info(f"Chat Stream Events Request | user={user_id} | session={session_id}")
            logger.info(f"Question: {message}")
            
            # Send start event
            yield ChatStreamEventService._format_sse("start", {
                "session_id": session_id,
                "message": message
            })
            
            # Get checkpointer
            checkpointer = ChatStreamEventService._get_checkpointer(session_id)
            
            # Create supervisor agent
            app: CompiledStateGraph = await create_supervisor_agent(db, checkpointer=checkpointer)
            
            # Config for agent
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "user_id": user_id,
                }
            }
            
            # Track state
            current_agent = "supervisor"
            current_tool = None
            token_buffer = []
            collected_messages = []
            
            logger.info("🚀 Starting event stream...")
            
            # Stream events with version v2
            async for event in app.astream_events(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                version="v2"
            ):
                event_type = event.get("event")
                event_name = event.get("name", "")
                event_data = event.get("data", {})
                
                logger.debug(f"Event: {event_type} | Name: {event_name}")
                
                # Handle different event types
                if event_type == "on_chat_model_start":
                    # Model started generating
                    current_agent = ChatStreamEventService._extract_agent_from_event_name(event_name)
                    yield ChatStreamEventService._format_sse("agent_start", {
                        "agent": current_agent,
                        "model": event_name
                    })
                
                elif event_type == "on_chat_model_stream":
                    # Token streaming from model
                    chunk = event_data.get("chunk", {})
                    
                    # Extract token content
                    if hasattr(chunk, 'content'):
                        content = chunk.content
                        if isinstance(content, str) and content:
                            token_buffer.append(content)
                            yield ChatStreamEventService._format_sse("token", {
                                "token": content,
                                "agent": current_agent
                            })
                        elif isinstance(content, list):
                            # Handle list of content blocks
                            for block in content:
                                if isinstance(block, dict) and block.get('type') == 'text':
                                    token = block.get('text', '')
                                    if token:
                                        token_buffer.append(token)
                                        yield ChatStreamEventService._format_sse("token", {
                                            "token": token,
                                            "agent": current_agent
                                        })
                
                elif event_type == "on_chat_model_end":
                    # Model finished generating
                    output = event_data.get("output", {})
                    
                    # Collect full message
                    if hasattr(output, 'content'):
                        full_content = ChatStreamEventService._extract_text_from_content(output.content)
                        if full_content:
                            collected_messages.append(output)
                    
                    # Clear token buffer and send complete message event
                    if token_buffer:
                        complete_text = "".join(token_buffer)
                        yield ChatStreamEventService._format_sse("message_complete", {
                            "content": complete_text,
                            "agent": current_agent
                        })
                        token_buffer = []
                    
                    yield ChatStreamEventService._format_sse("agent_end", {
                        "agent": current_agent
                    })
                
                elif event_type == "on_tool_start":
                    # Tool execution started
                    tool_name = event_name.split(".")[-1] if "." in event_name else event_name
                    current_tool = tool_name
                    
                    yield ChatStreamEventService._format_sse("tool_start", {
                        "tool": tool_name,
                        "input": str(event_data.get("input", ""))[:200]  # Truncate long inputs
                    })
                
                elif event_type == "on_tool_end":
                    # Tool execution completed
                    tool_name = current_tool or event_name.split(".")[-1]
                    output = event_data.get("output", "")
                    
                    yield ChatStreamEventService._format_sse("tool_end", {
                        "tool": tool_name,
                        "output": str(output)[:200]  # Truncate long outputs
                    })
                    current_tool = None
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Extract final response
            final_response = "".join(token_buffer) if token_buffer else ""
            
            logger.info(
                f"✅ Event Stream Completed | "
                f"time={processing_time:.2f}ms | "
                f"messages={len(collected_messages)}"
            )
            
            # Send completion event
            yield ChatStreamEventService._format_sse("end", {
                "response": final_response,
                "metadata": {
                    "processing_time_ms": round(processing_time, 2),
                    "message_count": len(collected_messages),
                    "memory_enabled": checkpointer is not None
                }
            })
            
        except Exception as e:
            logger.error(f"Error in event stream processing: {str(e)}", exc_info=True)
            yield ChatStreamEventService._format_sse("error", {
                "error": str(e)
            })

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
    def _extract_agent_from_event_name(event_name: str) -> str:
        """
        Extract agent name from event name
        
        Args:
            event_name: Event name string (e.g., "ChatGoogleGenerativeAI")
            
        Returns:
            Agent name string
        """
        event_lower = event_name.lower()
        
        if 'product' in event_lower:
            return 'product_agent'
        elif 'sales' in event_lower:
            return 'sales_agent'
        elif 'store' in event_lower:
            return 'store_agent'
        else:
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
chat_stream_event_service = ChatStreamEventService()
