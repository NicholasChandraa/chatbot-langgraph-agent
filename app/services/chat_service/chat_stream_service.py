"""
Chat Stream Service - Business Logic Layer
Handles streaming chat message processing with step-by-step agent progress
"""
import time
import json
from typing import Dict, Any, AsyncIterator
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.utils.logger import logger
from app.agents.supervisor_agent import create_supervisor_agent
from app.repositories.repository_container import RepositoryContainer
from app.services.chat_service.base_chat_service import BaseChatService


class ChatStreamService(BaseChatService):
    """
    Service for handling streaming chat operations.

    Uses stream_mode="updates" to stream agent progress step-by-step.
    Each update contains the node name (agent name) and state changes.

    Inherits shared logic from BaseChatService.
    """

    @staticmethod
    async def process_message_stream(
        message: str,
        user_id: str,
        session_id: str,
        repos: RepositoryContainer
    ) -> AsyncIterator[str]:
        """
        Process a chat message with streaming agent progress.

        Uses LangGraph's stream_mode="updates" for step-by-step progress.
        Each chunk contains node name (agent name) and state updates.

        Yields SSE events:
        - agent_start: Node started processing
        - message: AI message content from node
        - tool_result: Tool execution completed
        - agent_end: Node finished processing
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
            logger.info(f"Chat Stream Request | user={user_id} | session={session_id}")
            logger.info(f"Question: {message}")

            # Setup memory
            checkpointer = ChatStreamService._get_checkpointer(session_id)
            store = ChatStreamService._get_store(session_id)
            user_context = await ChatStreamService._load_user_context(user_id)
            user_context_string = ChatStreamService._format_user_context_string(user_context)

            # Create supervisor agent
            app = await create_supervisor_agent(
                repos,
                checkpointer=checkpointer,
                store=store,
                user_context=user_context_string
            )

            # Config (only thread_id needed for memory)
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "user_id": user_id  # User id diperlukan untuk store akses
                }
            }

            logger.info("🚀 Starting stream with mode='updates'...")

            # Stream agent progress step-by-step
            async for chunk in app.astream(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                stream_mode="updates"
            ):
                # chunk format: {node_name: state_update}
                for node_name, state_update in chunk.items():

                    # Skip nodes with no state update (e.g., middleware)
                    if state_update is None:
                        continue
                    
                    logger.warning(f"📍 Node: {node_name}")
                    logger.warning(f"📍 Data: {state_update}")

                    # Agent started
                    yield ChatStreamService._format_sse("agent_start", {
                        "agent": node_name
                    })

                    # Extract messages from state update
                    messages = state_update.get('messages', [])
                    if messages:
                        last_msg = messages[-1]

                        # Stream AI message
                        if isinstance(last_msg, AIMessage):
                            text = ChatStreamService._extract_text_from_content(last_msg.content)
                            if text and text.strip():
                                yield ChatStreamService._format_sse("message", {
                                    "content": text,
                                    "agent": node_name,
                                    "type": "ai"
                                })

                        # Stream tool result
                        elif isinstance(last_msg, ToolMessage):
                            yield ChatStreamService._format_sse("tool_result", {
                                "tool": getattr(last_msg, 'name', 'unknown'),
                                "agent": node_name,
                                "type": "tool"
                            })

                    # Agent finished
                    yield ChatStreamService._format_sse("agent_end", {
                        "agent": node_name
                    })

            # Stream completed
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Stream completed | time={processing_time:.2f}ms")

            yield ChatStreamService._format_sse("end", {
                "metadata": {
                    "processing_time_ms": round(processing_time, 2),
                    "memory_enabled": checkpointer is not None,
                    "user_context_loaded": user_context.get("has_data", False)
                }
            })

        except Exception as e:
            logger.error(f"❌ Stream error: {str(e)}", exc_info=True)
            yield ChatStreamService._format_sse("error", {"error": str(e)})

    @staticmethod
    def _format_sse(event: str, data: Dict[str, Any]) -> str:
        """Format data as Server-Sent Event"""
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# Singleton instance
chat_stream_service = ChatStreamService()
