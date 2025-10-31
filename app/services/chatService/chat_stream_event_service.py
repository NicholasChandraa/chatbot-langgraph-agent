"""
Chat Stream Event Service - Business Logic Layer
Handles token-by-token streaming chat message processing
"""
import time
import json
from typing import Dict, Any, AsyncIterator
from langchain_core.messages import HumanMessage

from app.utils.logger import logger
from app.agents.supervisor_agent import create_supervisor_agent
from app.repositories.repository_container import RepositoryContainer
from app.services.chatService.base_chat_service import BaseChatService
from app.services.token_tracking_service import (
    init_token_tracking,
    track_supervisor_tokens,
    get_token_summary,
    clear_token_tracking
)


class ChatStreamEventService(BaseChatService):
    """
    Service for handling dual-mode streaming chat operations.

    Uses stream_mode=["messages", "updates"] for:
    - Token-by-token streaming (typing effect)
    - State updates after each agent completes

    Inherits shared logic from BaseChatService.
    """

    @staticmethod
    async def process_message_stream_events(
        message: str,
        user_id: str,
        session_id: str,
        repos: RepositoryContainer
    ) -> AsyncIterator[str]:
        """
        Process a chat message with dual-mode streaming.

        Uses LangGraph's stream_mode=["messages", "updates"] for:
        - Real-time token streaming (typing effect)
        - State updates after each agent completes

        Yields SSE events:
        - agent_start: Node started generating
        - chunk: Individual token from LLM (typing effect)
        - state_update: State changed (with agent & state keys)
        - agent_end: Node finished generating
        - end: Stream completed with metadata
        - error: Error occurred

        Args:
            message: User's message text
            user_id: User identifier
            session_id: Conversation session identifier
            repos: Repository container

        Yields:
            SSE formatted strings
        """
        start_time = time.time()

        try:
            # Initialize token tracking for this request
            init_token_tracking()

            logger.info(f"Chat Stream Events Request | user={user_id} | session={session_id}")
            logger.info(f"Question: {message}")

            # Setup memory
            checkpointer = ChatStreamEventService._get_checkpointer(session_id)
            store = ChatStreamEventService._get_store(session_id)
            user_context = await ChatStreamEventService._load_user_context(user_id)
            user_context_string = ChatStreamEventService._format_user_context_string(user_context)

            # Create supervisor agent
            app = await create_supervisor_agent(
                repos,
                checkpointer=checkpointer,
                store=store,
                user_context=user_context_string
            )

            # Config (only thread_id needed for memory)
            config = {"configurable": {"thread_id": session_id}}

            logger.info("🚀 Starting stream with mode=['messages', 'updates']...")

            # Track current agent for start/end events
            current_agent = None
            # Track subagent execution
            current_subagent = None

            # Stream with dual mode: tokens + state updates
            async for event in app.astream(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                stream_mode=["messages", "updates"]  # Dual streaming
            ):
                logger.warning(f"EVENT: {event}")
                # Event structure: (event_type, (data, metadata))
                event_type = event[0] if isinstance(event, tuple) else None

                # Handle "messages" events (token streaming)
                if event_type == "messages":
                    chunk, metadata = event[1]

                    logger.debug(f"TOKEN: {chunk}")
                    agent_name = metadata.get('langgraph_node', 'unknown')

                    # Track supervisor token usage if available
                    usage_metadata = getattr(chunk, 'usage_metadata', None)
                    if usage_metadata:
                        track_supervisor_tokens(usage_metadata)

                    # Check if this is a tool call with subagent_type
                    tool_calls = getattr(chunk, 'tool_calls', [])
                    if tool_calls:
                        for tool_call in tool_calls:
                            if tool_call.get('name') == 'task':
                                subagent_type = tool_call.get('args', {}).get('subagent_type')
                                if subagent_type:
                                    current_subagent = subagent_type
                                    logger.info(f"🎯 Detected subagent execution: {subagent_type}")

                    # Map to semantic agent name (returns None if should skip)
                    display_agent = ChatStreamEventService._map_agent_name(agent_name, current_subagent)
                    if display_agent is None:
                        logger.debug(f"Skipping internal node: {agent_name}")
                        continue

                    # Send agent_start event when agent changes
                    if display_agent != current_agent:
                        if current_agent:
                            yield ChatStreamEventService._format_sse("agent_end", {
                                "agent": current_agent
                            })
                        current_agent = display_agent
                        yield ChatStreamEventService._format_sse("agent_start", {
                            "agent": display_agent
                        })

                    # Extract and stream token content
                    content = getattr(chunk, 'content', None)

                    # Skip special/empty chunks
                    if not content or getattr(chunk, 'id', '') == '__remove_all__':
                        continue

                    # Extract text from content (handles various formats)
                    text_chunks = ChatStreamEventService._extract_text_from_content(content)
                    for text in text_chunks:
                        yield ChatStreamEventService._format_sse("chunk", {
                            "chunk": text,
                            "agent": display_agent
                        })

                # Handle "updates" events (state changes)
                elif event_type == "updates":
                    update_dict = event[1]  # Dict: {agent_name: update_data}

                    for agent_name, update_data in update_dict.items():
                        logger.debug(f"STATE UPDATE from {agent_name}")

                        # Map to semantic agent name (returns None if should skip)
                        display_agent = ChatStreamEventService._map_agent_name(agent_name, current_subagent)
                        if display_agent is None:
                            logger.debug(f"Skipping internal node update: {agent_name}")
                            continue

                        # Send state update event
                        yield ChatStreamEventService._format_sse("state_update", {
                            "agent": display_agent,
                            "has_data": update_data is not None
                        })

                        # Reset subagent tracking after tools complete
                        if agent_name == 'tools' and current_subagent:
                            logger.info(f"✅ Subagent execution completed: {current_subagent}")
                            current_subagent = None

            # End last agent
            if current_agent:
                yield ChatStreamEventService._format_sse("agent_end", {
                    "agent": current_agent
                })

            # Stream completed
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"✅ chunk stream completed | time={processing_time:.2f}ms")

            # Get token usage summary
            token_summary = get_token_summary()

            yield ChatStreamEventService._format_sse("end", {
                "metadata": {
                    "processing_time_ms": round(processing_time, 2),
                    "memory_enabled": checkpointer is not None,
                    "user_context_loaded": user_context.get("has_data", False),
                    "token_usage": token_summary
                }
            })

            # Clear token tracking for this request
            clear_token_tracking()

        except Exception as e:
            logger.error(f"❌ chunk stream error: {str(e)}", exc_info=True)
            yield ChatStreamEventService._format_sse("error", {"error": str(e)})

    @staticmethod
    def _format_sse(event: str, data: Dict[str, Any]) -> str:
        """Format data as Server-Sent Event"""
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    @staticmethod
    def _map_agent_name(agent_name: str, current_subagent: str = None) -> str | None:
        """
        Map LangGraph node name to semantic agent name for display.

        Mapping rules:
        - Internal nodes (containing '__') → None (skip)
        - Middleware nodes → 'supervisor_agent'
        - 'model' with subagent context → subagent name
        - 'model' without subagent → 'supervisor_agent'
        - 'tools' with subagent context → subagent name
        - 'tools' without subagent → 'supervisor_agent'
        - Other nodes → original name

        Args:
            agent_name: LangGraph node name
            current_subagent: Current executing subagent (if any)

        Returns:
            Semantic agent name for display, or None to skip
        """
        # Skip internal system nodes
        if '__' in agent_name:
            return None

        # Middleware coordination
        if 'Middleware' in agent_name:
            return 'supervisor_agent'

        # Model node
        if agent_name == 'model':
            return current_subagent if current_subagent else 'supervisor_agent'

        # Tools node
        if agent_name == 'tools':
            return current_subagent if current_subagent else 'supervisor_agent'

        # Other nodes keep original name
        return agent_name

    @staticmethod
    def _extract_text_from_content(content) -> list[str]:
        """
        Extract text content from various LangChain message formats.

        Args:
            content: Message content (can be string, list, or dict)

        Returns:
            List of text strings to stream
        """
        texts = []

        # Handle list of content blocks
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    # Text block: {'type': 'text', 'text': '...'}
                    if block.get('type') == 'text':
                        text = block.get('text', '')
                        if text:
                            texts.append(text)
                    # Tool call chunk (skip - not displayable text)
                    elif block.get('type') in ['tool_call', 'tool_call_chunk']:
                        continue
                elif isinstance(block, str):
                    texts.append(block)

        # Handle string content
        elif isinstance(content, str) and content:
            texts.append(content)

        return texts


# Singleton instance
chat_stream_event_service = ChatStreamEventService()
