"""
Token Usage Repository
Handles persistence of token usage data database
"""
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.utils.logger import logger

class TokenUsageRepository:
    """Repository for token usage tracking"""
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        
    async def save_usage(
        self,
        session_id: str,
        user_id: Optional[str],
        summary: Dict[str, Any],
        processing_time_ms: float,
        user_question: Optional[str] = None
    ) -> Optional[int]:
        """
        Save token usage to database.

        Args:
            session_id (str): Conversation session ID
            user_id (Optional[str]): User identifier
            summary (Dict[str, Any]): Token summary from get_token_summary()
            processing_time_ms (float): Request processing time
            user_queestion (Optional[str], optional): User's question. Defaults to None.

        Returns:
            Optional[int]: usage_id
        """
        try:
            # Extract supervisor tokens
            supervisor = summary["breakdown"]["supervisor"]
            
            # Calculate subagent totals
            subagent_totals = sum(
                agent_data["total"]
                for agent_data in summary["breakdown"]["subagents"].values()
            )
            
            # Insert main record
            result = await self.db_session.execute(
                text(
                    """
                    INSERT INTO token_usage(
                        session_id, user_id, created_at, processing_time_ms,
                        total_tokens,
                        supervisor_input_tokens, supervisor_output_tokens,
                        supervisor_cache_read_tokens, supervisor_reasoning_tokens,
                        user_question
                    ) VALUES (
                        :session_id, :user_id, NOW(), :processing_time_ms,
                        :total_tokens,
                        :supervisor_input, :supervisor_output,
                        :supervisor_cache, :supervisor_reasoning,
                        :user_question
                    )
                    RETURNING id
                    """
                ),
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "processing_time_ms": processing_time_ms,
                    "total_tokens": summary["total_tokens"],
                    "supervisor_input": supervisor["input_tokens"],
                    "supervisor_output": supervisor["output_tokens"],
                    "supervisor_cache": supervisor["cache_read_tokens"],
                    "supervisor_reasoning": supervisor["reasoning_tokens"],
                    "user_question": user_question[:1000] if user_question else None  # Truncate
                }
            )
            
            usage_id = result.scalar_one()
            
            # Insert agent details
            for agent_name, agent_data in summary["breakdown"]["subagents"].items():
                # LLM tokens
                llm_tokens = agent_data["llm_tokens"]
                await self.db_session.execute(
                    text(
                        """
                        INSERT INTO token_usage_agent_detail (
                            usage_id, agent_name, token_type,
                            input_tokens, output_tokens, total_tokens,
                            cache_read_tokens, reasoning_tokens
                        ) VALUES (
                            :usage_id, :agent_name, 'llm',
                            :input, :output, :total,
                            :cache_read, :reasoning
                        )
                        """
                    ),
                    {
                        "usage_id": usage_id,
                        "agent_name": agent_name,
                        "input": llm_tokens["input_tokens"],
                        "output": llm_tokens["output_tokens"],
                        "total": llm_tokens["total_tokens"],
                        "cache_read": llm_tokens["cache_read_tokens"],
                        "reasoning": llm_tokens["reasoning_tokens"]
                    }
                )
                
                # Tool tokens
                tool_tokens = agent_data["tool_tokens"]
                await self.db_session.execute(
                    text(
                        """
                        INSERT INTO token_usage_agent_detail (
                            usage_id, agent_name, token_type,
                            input_tokens, output_tokens, total_tokens,
                            cache_read_tokens, reasoning_tokens
                        ) VALUES (
                            :usage_id, :agent_name, 'tool',
                            :input, :output, :total,
                            :cache_read, :reasoning
                        )
                        """
                    ),
                    {
                        "usage_id": usage_id,
                        "agent_name": agent_name,
                        "input": tool_tokens["input_tokens"],
                        "output": tool_tokens["output_tokens"],
                        "total": tool_tokens["total_tokens"],
                        "cache_read": tool_tokens["cache_read_tokens"],
                        "reasoning": tool_tokens["reasoning_tokens"]
                    }
                )
            
            await self.db_session.commit()
            
            logger.info(f"💾 Token usage saved to DB | session_id: {session_id} | user_id: {user_id}")
            
            return usage_id
        except Exception as e:
            logger.error(f"❌ Failed to save token usage: {e}", exc_info=True)
            await self.db_session.rollback()
            return None
        