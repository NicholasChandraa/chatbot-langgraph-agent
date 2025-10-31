"""
Token Tracking Service
Centralized service for tracking and aggregating token usage across all agents
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from contextvars import ContextVar
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
    """Tracks token usage for entire conversation"""
    supervisor: TokenUsage = field(default_factory=TokenUsage)
    subagents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sql_workflows: Dict[str, TokenUsage] = field(default_factory=dict)

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
        Add SQL workflow tokens

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

        if agent_name not in self.sql_workflows:
            self.sql_workflows[agent_name] = TokenUsage()

        self.sql_workflows[agent_name].add(tokens)

        # Track in subagents as well
        if agent_name not in self.subagents:
            self.subagents[agent_name] = {
                "tool_tokens": TokenUsage()
            }

        self.subagents[agent_name]["tool_tokens"].add(tokens)

        logger.debug(
            f"📊 SQL workflow tokens added | "
            f"agent={agent_name} | "
            f"total={total_tokens}"
        )

    def get_total_tokens(self) -> int:
        """Calculate total tokens across all components"""
        total = self.supervisor.total_tokens

        # Add subagent tokens
        for agent_data in self.subagents.values():
            agent_tokens = agent_data.get("agent_tokens", TokenUsage())
            if isinstance(agent_tokens, TokenUsage):
                total += agent_tokens.total_tokens

        # Add SQL workflow tokens
        for workflow_tokens in self.sql_workflows.values():
            total += workflow_tokens.total_tokens

        return total

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive token usage summary

        Returns:
            Dict with total and breakdown
        """
        # Calculate subagent totals
        subagents_summary = {}
        for agent_name, agent_data in self.subagents.items():
            agent_tokens = agent_data.get("agent_tokens", TokenUsage())
            tool_tokens = agent_data.get("tool_tokens", TokenUsage())

            agent_total = 0
            if isinstance(agent_tokens, TokenUsage):
                agent_total += agent_tokens.total_tokens
            if isinstance(tool_tokens, TokenUsage):
                agent_total += tool_tokens.total_tokens

            subagents_summary[agent_name] = {
                "agent_tokens": agent_tokens.to_dict() if isinstance(agent_tokens, TokenUsage) else agent_tokens,
                "tool_tokens": tool_tokens.to_dict() if isinstance(tool_tokens, TokenUsage) else tool_tokens,
                "total": agent_total
            }

        summary = {
            "total_tokens": self.get_total_tokens(),
            "breakdown": {
                "supervisor": self.supervisor.to_dict(),
                "subagents": subagents_summary,
                "sql_workflows": {
                    name: tokens.to_dict()
                    for name, tokens in self.sql_workflows.items()
                }
            }
        }

        logger.info(
            f"📊 Token Summary | "
            f"Total: {summary['total_tokens']} | "
            f"Supervisor: {self.supervisor.total_tokens} | "
            f"SQL Workflows: {sum(t.total_tokens for t in self.sql_workflows.values())}"
        )

        return summary

    def reset(self) -> None:
        """Reset all token counters"""
        self.supervisor = TokenUsage()
        self.subagents = {}
        self.sql_workflows = {}
        logger.debug("📊 Token tracking reset")


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
            "agent_tokens": TokenUsage(),
            "tool_tokens": TokenUsage()
        }

    # Add to agent_tokens (LLM calls from subagent itself)
    if "agent_tokens" not in tracker.subagents[agent_name]:
        tracker.subagents[agent_name]["agent_tokens"] = TokenUsage()

    tracker.subagents[agent_name]["agent_tokens"].add(tokens)

    logger.debug(f"📊 {agent_name} tokens | total={total_tokens}")


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
