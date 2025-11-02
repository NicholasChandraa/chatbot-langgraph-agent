"""
Token Tracking Service
Centralized service for tracking and aggregating token usage across all agents
"""
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from contextvars import ContextVar
from langchain.agents.middleware import after_model
from app.utils.logger import logger

# Context variable for current conversation's token tracker
_current_tracker: ContextVar[Optional['ConversationTokenTracking']] = ContextVar('token_tracker', default=None)


@dataclass
class TokenUsage:
    """Token usage data structure"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, other: 'TokenUsage') -> None:
        """Add another TokenUsage to this one"""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.reasoning_tokens += other.reasoning_tokens

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ConversationTokenTracking:
    """
    Tracks token usage for entire conversation.

    Structure:
    - supervisor: Supervisor agent LLM tokens
    - subagents: {
        "agent_name": {
            "llm_tokens": Direct LLM calls from subagent (via middleware),
            "tool_tokens": Tool execution tokens (SQL workflows)
        }
      }
    """
    supervisor: TokenUsage = field(default_factory=TokenUsage)
    subagents: Dict[str, Dict[str, TokenUsage]] = field(default_factory=dict)

    def add_supervisor_tokens(self, usage_metadata: Dict[str, Any]) -> None:
        """
        Add supervisor LLM tokens from usage_metadata

        Args:
            usage_metadata: Dict with input_tokens, output_tokens, etc.
        """
        tokens = TokenUsage(
            input_tokens=usage_metadata.get('input_tokens', 0),
            output_tokens=usage_metadata.get('output_tokens', 0),
            total_tokens=usage_metadata.get('total_tokens', 0),
            cache_read_tokens=usage_metadata.get('input_token_details', {}).get('cache_read', 0),
            reasoning_tokens=usage_metadata.get('output_token_details', {}).get('reasoning', 0)
        )

        self.supervisor.add(tokens)

        logger.debug(
            f"📊 Supervisor tokens added | "
            f"input={tokens.input_tokens} | "
            f"output={tokens.output_tokens} | "
            f"total={tokens.total_tokens}"
        )

    def add_sql_workflow_tokens(
        self,
        agent_name: str,
        total_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        reasoning_tokens: int = 0
    ) -> None:
        """
        Add SQL workflow tokens (tool execution).

        Args:
            agent_name: Name of agent (e.g., 'store_agent')
            total_tokens: Total tokens used
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            cache_read_tokens: Cached input tokens
            reasoning_tokens: Reasoning output tokens
        """
        tokens = TokenUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=cache_read_tokens,
            reasoning_tokens=reasoning_tokens
        )

        # Initialize subagent entry if not exists
        if agent_name not in self.subagents:
            self.subagents[agent_name] = {
                "llm_tokens": TokenUsage(),
                "tool_tokens": TokenUsage()
            }

        # Add to tool_tokens (SQL workflow execution)
        self.subagents[agent_name]["tool_tokens"].add(tokens)

        logger.debug(
            f"📊 SQL workflow tokens added | "
            f"agent={agent_name} | "
            f"total={total_tokens}"
        )

    def get_total_tokens(self) -> int:
        """Calculate total tokens across all components"""
        total = self.supervisor.total_tokens

        # Add subagent tokens (both LLM and tool)
        for agent_data in self.subagents.values():
            llm_tokens = agent_data.get("llm_tokens", TokenUsage())
            tool_tokens = agent_data.get("tool_tokens", TokenUsage())

            total += llm_tokens.total_tokens
            total += tool_tokens.total_tokens

        return total

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive token usage summary

        Returns:
            Dict with total and breakdown by agent
        """
        # Calculate subagent totals
        subagents_summary = {}
        for agent_name, agent_data in self.subagents.items():
            llm_tokens = agent_data.get("llm_tokens", TokenUsage())
            tool_tokens = agent_data.get("tool_tokens", TokenUsage())

            agent_total = llm_tokens.total_tokens + tool_tokens.total_tokens

            subagents_summary[agent_name] = {
                "llm_tokens": llm_tokens.to_dict(),
                "tool_tokens": tool_tokens.to_dict(),
                "total": agent_total
            }

        total_subagent_tokens = sum(s["total"] for s in subagents_summary.values())

        summary = {
            "total_tokens": self.get_total_tokens(),
            "breakdown": {
                "supervisor": self.supervisor.to_dict(),
                "subagents": subagents_summary
            }
        }

        logger.info(
            f"📊 Token Summary | "
            f"Total: {summary['total_tokens']} | "
            f"Supervisor: {self.supervisor.total_tokens} | "
            f"Subagents: {total_subagent_tokens}"
        )

        return summary


# ==================== PUBLIC API ====================

def init_token_tracking() -> None:
    """
    Initialize token tracking for current request context.
    Call this at the start of each request/conversation turn.
    """
    tracker = ConversationTokenTracking()
    _current_tracker.set(tracker)
    logger.debug("📊 Token tracking initialized for request")


def get_current_tracker() -> Optional[ConversationTokenTracking]:
    """
    Get current token tracker from context.

    Returns:
        Current ConversationTokenTracking or None
    """
    return _current_tracker.get()


def track_supervisor_tokens(usage_metadata: Dict[str, Any]) -> None:
    """
    Track supervisor LLM tokens in current context.

    Args:
        usage_metadata: Dict with token usage from supervisor
    """
    tracker = get_current_tracker()
    if tracker:
        tracker.add_supervisor_tokens(usage_metadata)


def track_sql_workflow_tokens(
    agent_name: str,
    total_tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    reasoning_tokens: int = 0
) -> None:
    """
    Track SQL workflow tokens in current context.

    Args:
        agent_name: Name of agent
        total_tokens: Total tokens
        prompt_tokens: Input tokens
        completion_tokens: Output tokens
        cache_read_tokens: Cached input tokens
        reasoning_tokens: Reasoning output tokens
    """
    tracker = get_current_tracker()
    if tracker:
        tracker.add_sql_workflow_tokens(
            agent_name=agent_name,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            reasoning_tokens=reasoning_tokens
        )


def track_subagent_direct_tokens(
    agent_name: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cache_read_tokens: int = 0,
    reasoning_tokens: int = 0
) -> None:
    """
    Track sub-agent's direct LLM call tokens (not tool tokens).

    Args:
        agent_name: Name of agent
        input_tokens: Input tokens
        output_tokens: Output tokens
        total_tokens: Total tokens
        cache_read_tokens: Cached input tokens
        reasoning_tokens: Reasoning output tokens
    """
    tracker = get_current_tracker()
    if not tracker:
        return

    tokens = TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_read_tokens=cache_read_tokens,
        reasoning_tokens=reasoning_tokens
    )

    # Initialize subagent entry if not exists
    if agent_name not in tracker.subagents:
        tracker.subagents[agent_name] = {
            "llm_tokens": TokenUsage(),
            "tool_tokens": TokenUsage()
        }

    # Add to llm_tokens (Direct LLM calls from subagent via middleware)
    tracker.subagents[agent_name]["llm_tokens"].add(tokens)

    logger.debug(f"📊 {agent_name} LLM tokens | total={total_tokens}")


def get_token_summary() -> Optional[Dict[str, Any]]:
    """
    Get token usage summary for current request.

    Returns:
        Token summary dict or None if not initialized
    """
    tracker = get_current_tracker()
    if tracker:
        return tracker.get_summary()
    return None


def clear_token_tracking() -> None:
    """
    Clear token tracking for current context.
    Call this at the end of request.
    """
    _current_tracker.set(None)
    logger.debug("📊 Token tracking cleared")


def create_token_tracking_middleware(agent_name: str) -> Callable:
    """
    Factory function to create token tracking middleware for sub-agents.

    This eliminates code duplication across product_agent, sales_agent, store_agent.

    Args:
        agent_name: Name of the agent (e.g., 'product_agent', 'sales_agent')

    Returns:
        Async middleware function decorated with @after_model

    Example:
        ```python
        from app.services.token_tracking_service import create_token_tracking_middleware

        track_tokens = create_token_tracking_middleware("product_agent")

        agent_graph = create_agent(
            llm,
            tools=[product_query],
            middleware=[track_tokens]  # Already decorated
        )
        ```
    """
    @after_model
    async def track_tokens(request, response):
        """Track LLM token usage from sub-agent"""
        try:
            logger.warning(f"[REQUEST]: {request}")
            # Extract AIMessage from request (last message in the conversation)
            messages = request.get('messages', [])
            if not messages:
                return response
            
            # Get the last AIMessage (the LLM response)
            last_message = messages[-1]
            usage_metadata = getattr(last_message, 'usage_metadata', None)

            if usage_metadata:
                input_tokens = usage_metadata.get('input_tokens', 0)
                output_tokens = usage_metadata.get('output_tokens', 0)
                total_tokens = usage_metadata.get('total_tokens', input_tokens + output_tokens)

                cache_read = usage_metadata.get('input_token_details', {}).get('cache_read', 0)
                reasoning = usage_metadata.get('output_token_details', {}).get('reasoning', 0)

                logger.info(
                    f"📊 [{agent_name}] LLM tokens: {total_tokens} "
                    f"(input: {input_tokens}, output: {output_tokens}, "
                    f"cache: {cache_read}, reasoning: {reasoning})"
                )

                track_subagent_direct_tokens(
                    agent_name=agent_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cache_read_tokens=cache_read,
                    reasoning_tokens=reasoning
                )
        except Exception as e:
            logger.error(f"[{agent_name}] Token tracking error: {e}")

        return response

    return track_tokens


# PUBLIC API
async def save_token_usage_to_db(
    session_id: str,
    user_id: str,
    summary: Dict[str, Any],
    processing_time_ms: float,
    token_repo,  # TokenUsageRepository
    user_question: str = None
) -> Optional[int]:
    """
    Save token usage summary to database.

    Args:
        session_id (str): Conversation session ID
        user_id (str): User identifier
        summary (Dict[str, Any]): Token summary from get_token_summary()
        processing_time_ms (float): Request processing time
        token_repo (_type_): TokenUsageRepository instance
        user_question: User's question (optional)

    Returns:
        Optional[int]: usage_id if successful, None otherwise
    """
    if not summary:
        logger.warning("No token summary to save")
        return None
    
    try:
        usage_id = await token_repo.save_usage(
            session_id=session_id,
            user_id=user_id,
            summary=summary,
            processing_time_ms=processing_time_ms,
            user_question=user_question
        )
        
        return usage_id
    
    except Exception as e:
        logger.error(f"Error in save_token_usage_to_db: {e}")
        return None