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

                    # Send agent_start event when agent changes
                    if agent_name != current_agent:
                        if current_agent:
                            yield ChatStreamEventService._format_sse("agent_end", {
                                "agent": current_agent
                            })
                        current_agent = agent_name
                        yield ChatStreamEventService._format_sse("agent_start", {
                            "agent": agent_name
                        })

                    # Extract and stream token content
                    content = getattr(chunk, 'content', None)

                    # Skip special/empty chunks
                    if not content or getattr(chunk, 'id', '') == '__remove_all__':
                        continue

                    # Handle list of content blocks
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text = block.get('text', '')
                                if text:
                                    yield ChatStreamEventService._format_sse("chunk", {
                                        "chunk": text,
                                        "agent": agent_name
                                    })
                            elif isinstance(block, str):
                                yield ChatStreamEventService._format_sse("chunk", {
                                    "chunk": block,
                                    "agent": agent_name
                                })

                    # Handle string content
                    elif isinstance(content, str):
                        yield ChatStreamEventService._format_sse("chunk", {
                            "chunk": content,
                            "agent": agent_name
                        })

                # Handle "updates" events (state changes)
                elif event_type == "updates":
                    update_dict = event[1]  # Dict: {agent_name: update_data}

                    for agent_name, update_data in update_dict.items():
                        logger.debug(f"STATE UPDATE from {agent_name}")

                        # Send state update event
                        yield ChatStreamEventService._format_sse("state_update", {
                            "agent": agent_name,
                            "has_data": update_data is not None
                        })

            # End last agent
            if current_agent:
                yield ChatStreamEventService._format_sse("agent_end", {
                    "agent": current_agent
                })

            # Stream completed
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"✅ chunk stream completed | time={processing_time:.2f}ms")

            yield ChatStreamEventService._format_sse("end", {
                "metadata": {
                    "processing_time_ms": round(processing_time, 2),
                    "memory_enabled": checkpointer is not None,
                    "user_context_loaded": user_context.get("has_data", False)
                }
            })

        except Exception as e:
            logger.error(f"❌ chunk stream error: {str(e)}", exc_info=True)
            yield ChatStreamEventService._format_sse("error", {"error": str(e)})

    @staticmethod
    def _format_sse(event: str, data: Dict[str, Any]) -> str:
        """Format data as Server-Sent Event"""
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# Singleton instance
chat_stream_event_service = ChatStreamEventService()
