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
from app.database.memory.checkpointer_manager import checkpointer_manager
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
    
    @staticmethod
    async def process_message_stream(
        message: str,
        user_id: str,
        session_id: str,
        db: AsyncSession
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
            yield ChatService._format_sse("start", {
                "session_id": session_id,
                "message": message
            })
            
            # Get checkpointer
            checkpointer = ChatService._get_checkpointer(session_id)
            
            # Create supervisor agent
            app = await create_supervisor_agent(db, checkpointer=checkpointer)
            
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
                            text_content = ChatService._extract_text_from_content(msg.content)
                            if text_content and text_content.strip():
                                # Try to determine agent from message metadata
                                agent_name = ChatService._determine_agent_name(msg)
                                
                                yield ChatService._format_sse("message", {
                                    "content": text_content,
                                    "name": agent_name,
                                    "type": "ai"
                                })
                        
                        # Stream tool messages
                        elif isinstance(msg, ToolMessage):
                            tool_name = getattr(msg, 'name', 'unknown')
                            yield ChatService._format_sse("tool_result", {
                                "name": tool_name,
                                "type": "tool"
                            })
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Extract final response
            response_text = ChatService._extract_current_turn_response(collected_messages)
            
            logger.info(
                f"✅ Stream Completed | "
                f"time={processing_time:.2f}ms | "
                f"messages={len(collected_messages)}"
            )
            
            # Send completion event
            yield ChatService._format_sse("end", {
                "response": response_text,
                "metadata": {
                    "processing_time_ms": round(processing_time, 2),
                    "message_count": len(collected_messages),
                    "memory_enabled": checkpointer is not None
                }
            })
            
        except Exception as e:
            logger.error(f"Error in stream processing: {str(e)}", exc_info=True)
            yield ChatService._format_sse("error", {
                "error": str(e)
            })
    
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
        content = ChatService._extract_text_from_content(msg.content)
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
            yield ChatService._format_sse("start", {
                "session_id": session_id,
                "message": message
            })
            
            # Get checkpointer
            checkpointer = ChatService._get_checkpointer(session_id)
            
            # Create supervisor agent
            app = await create_supervisor_agent(db, checkpointer=checkpointer)
            
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
                    current_agent = ChatService._extract_agent_from_event_name(event_name)
                    yield ChatService._format_sse("agent_start", {
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
                            yield ChatService._format_sse("token", {
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
                                        yield ChatService._format_sse("token", {
                                            "token": token,
                                            "agent": current_agent
                                        })
                
                elif event_type == "on_chat_model_end":
                    # Model finished generating
                    output = event_data.get("output", {})
                    
                    # Collect full message
                    if hasattr(output, 'content'):
                        full_content = ChatService._extract_text_from_content(output.content)
                        if full_content:
                            collected_messages.append(output)
                    
                    # Clear token buffer and send complete message event
                    if token_buffer:
                        complete_text = "".join(token_buffer)
                        yield ChatService._format_sse("message_complete", {
                            "content": complete_text,
                            "agent": current_agent
                        })
                        token_buffer = []
                    
                    yield ChatService._format_sse("agent_end", {
                        "agent": current_agent
                    })
                
                elif event_type == "on_tool_start":
                    # Tool execution started
                    tool_name = event_name.split(".")[-1] if "." in event_name else event_name
                    current_tool = tool_name
                    
                    yield ChatService._format_sse("tool_start", {
                        "tool": tool_name,
                        "input": str(event_data.get("input", ""))[:200]  # Truncate long inputs
                    })
                
                elif event_type == "on_tool_end":
                    # Tool execution completed
                    tool_name = current_tool or event_name.split(".")[-1]
                    output = event_data.get("output", "")
                    
                    yield ChatService._format_sse("tool_end", {
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
            yield ChatService._format_sse("end", {
                "response": final_response,
                "metadata": {
                    "processing_time_ms": round(processing_time, 2),
                    "message_count": len(collected_messages),
                    "memory_enabled": checkpointer is not None
                }
            })
            
        except Exception as e:
            logger.error(f"Error in event stream processing: {str(e)}", exc_info=True)
            yield ChatService._format_sse("error", {
                "error": str(e)
            })
    
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
chat_service = ChatService()
