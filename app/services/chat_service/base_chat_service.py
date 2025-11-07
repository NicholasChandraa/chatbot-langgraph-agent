"""
Base Chat Service
Shared logic for all chat service variants
"""
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from app.database.memory.checkpointer_manager import checkpointer_manager
from app.database.memory.store_manager import store_manager
from app.utils.logger import logger


class BaseChatService:
    """
    Base class for all chat services.

    Contains shared methods used by:
    - ChatService (regular chat)
    - ChatStreamService (streaming)
    - ChatStreamEventService (token-by-token streaming)

    Benefits:
    - DRY: Code written once, reused by all services
    - Consistency: All services use same logic
    - Maintainability: Bug fix in one place, all services benefit
    - Testability: Test shared logic once
    """

    # ============================================================
    # MEMORY & CONTEXT MANAGEMENT
    # ============================================================

    @staticmethod
    def _get_checkpointer(session_id: str):
        """
        Get checkpointer for conversation memory (short-term).

        Checkpointer provides:
        - Conversation history within a session
        - Message-level memory
        - Thread-based context

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
    def _get_store(session_id: str):
        """
        Get store for long-term memory.

        Store provides:
        - User profiles across sessions
        - User preferences
        - Historical context

        Args:
            session_id: Session identifier for logging

        Returns:
            Store instance or None if unavailable
        """
        try:
            store = store_manager.get_store()
            logger.info(f"🧠 Using long-term memory | session={session_id}")
            return store
        except RuntimeError as e:
            # [UPDATE] Store not initialized (expected if tables don't exist)
            logger.warning(f"⚠️ Store not initialized: {e}")
            logger.warning("⚠️ Continuing without long-term memory. DeepAgents memory will be disabled.")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Store unavailable, continuing without long-term memory: {e}")
            return None

    @staticmethod
    async def _load_user_context(user_id: str) -> dict:
        """
        Load user context from long-term memory.

        Retrieves:
        - User profile (name, job, phone)
        - User preferences (favorite products, dietary restrictions)
        - Custom metadata

        Args:
            user_id: User identifier

        Returns:
            Dictionary containing user context data with structure:
            {
                "has_data": bool,
                "name": str | None,
                "job": str | None,
                "phone": str | None,
                "favorite_product": str | None,
                "dietary_restrictions": str | None,
                "other_preferences": dict
            }
        """
        try:
            from app.services.memory_service.memory_service import memory_service

            # Load profile and preferences from long-term store
            profile = await memory_service.get_user_profile(user_id)
            preferences = await memory_service.get_all_user_preferences(user_id)

            context = {
                "has_data": False,
                "name": None,
                "job": None,
                "phone": None,
                "favorite_product": None,
                "dietary_restrictions": None,
                "other_preferences": {}
            }

            # Extract profile data
            if profile:
                context["has_data"] = True
                context["name"] = profile.get("name")
                context["job"] = profile.get("job")
                context["phone"] = profile.get("phone")

            # Extract preferences
            if preferences:
                context["has_data"] = True
                context["favorite_product"] = preferences.get("favorite_product")
                context["dietary_restrictions"] = preferences.get("dietary_restrictions")

                # Store other preferences
                for key, value in preferences.items():
                    if key not in ["favorite_product", "dietary_restrictions"]:
                        context["other_preferences"][key] = value

            if context["has_data"]:
                logger.info(f"=� Loaded user context for {user_id}")
            else:
                logger.debug(f"No stored context found for {user_id}")

            return context

        except Exception as e:
            logger.warning(f"� Failed to load user context: {e}")
            return {"has_data": False}

    @staticmethod
    def _format_user_context_string(context: dict) -> str:
        """
        Format user context dictionary into readable string for LLM prompt.

        Converts structured context into natural language format
        that can be injected into system prompts.

        Args:
            context: User context dictionary from _load_user_context()

        Returns:
            Formatted context string (empty if no data)

        Example output:
            - Nama: John Doe
            - Pekerjaan: Software Engineer
            - Produk Favorit: Glazed Donut
        """
        if not context.get("has_data"):
            return ""

        lines = []

        if context.get("name"):
            lines.append(f"- Nama: {context['name']}")

        if context.get("job"):
            lines.append(f"- Pekerjaan: {context['job']}")

        if context.get("phone"):
            lines.append(f"- Telepon: {context['phone']}")

        if context.get("favorite_product"):
            lines.append(f"- Produk Favorit: {context['favorite_product']}")

        if context.get("dietary_restrictions"):
            lines.append(f"- Pantangan Makanan: {context['dietary_restrictions']}")

        if context.get("other_preferences"):
            for key, value in context["other_preferences"].items():
                lines.append(f"- {key.replace('_', ' ').title()}: {value}")

        if lines:
            return "\n".join(lines)
        return ""

    # ============================================================
    # MESSAGE PROCESSING UTILITIES
    # ============================================================

    @staticmethod
    def _extract_current_turn_response(messages: List) -> str:
        """
        Extract AI responses from the current conversation turn only.

        This avoids returning responses from conversation history
        by finding the last HumanMessage and extracting only AI responses after it.

        Important for multi-turn conversations where we only want
        the response to the current user query, not historical responses.

        Args:
            messages: List of message objects from LangGraph state

        Returns:
            Combined AI response text from current turn

        Logic:
        1. Find last HumanMessage (current user input)
        2. Collect all AIMessages after it
        3. Combine into single response
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
                    text_content = BaseChatService._extract_text_from_content(msg.content)

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
        Extract text from message content (handles both string and list formats).

        LangChain/LangGraph message content can be:
        1. String: "Some text"
        2. List of content blocks: [{'type': 'text', 'text': 'Some text'}, ...]

        This utility ensures we can extract text regardless of format.

        Args:
            content: Message content (string or list)

        Returns:
            Extracted text string

        Examples:
            > _extract_text_from_content("Hello")
            "Hello"

            > _extract_text_from_content([
                {'type': 'text', 'text': 'Hello'},
                {'type': 'text', 'text': 'World'}
            ])
            "Hello World"
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

    # ============================================================
    # DEBUG & LOGGING UTILITIES
    # ============================================================

    @staticmethod
    def _debug_log_messages(messages: List) -> None:
        """
        Log all messages for debugging purposes.

        Useful for understanding message flow through the agent system.
        Logs message types, names, tool calls, and content previews.

        Args:
            messages: List of message objects
        """
        logger.debug(f"=� Total messages: {len(messages)}")

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
                content_preview = BaseChatService._extract_text_from_content(raw_content)[:200]
            else:
                content_preview = str(raw_content)[:200]

            # Log tool call details if present
            tool_info = ""
            if has_tool_calls:
                tool_names = [tc.get('name', 'unknown') for tc in msg.tool_calls]
                tool_info = f" | ToolsCalled={tool_names}"

            # Special detection for forwarding tool usage
            if isinstance(msg, ToolMessage) and 'forward' in str(msg_name).lower():
                logger.debug(f"[FORWARDING DETECTED] Message [{i}] uses forwarding mechanism")

            logger.debug(
                f"  [{i}] Type={msg_type} | Name={msg_name} | "
                f"HasTools={has_tool_calls}{tool_info} | Content={content_preview}"
            )
