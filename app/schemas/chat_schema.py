"""
Chat Request/Response Schemas
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ConversationTurn(BaseModel):
    """Single conversation turn (message)"""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Ada donut rasa coklat?"
            }
        }


class ChatRequest(BaseModel):
    """Chat request from upstream services"""
    message: str = Field(..., min_length=1, description="User message")
    user_id: str = Field(..., description="User ID from upstream service")
    session_id: str = Field(..., description="Conversation session ID")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Berapa total toko yang ada sekarang?",
                "user_id": "user_12345",
                "session_id": "session_abc123",
            }
        }


class ChatResponse(BaseModel):
    """Chat response to upstream services"""
    response: str = Field(..., description="Agent response message")
    session_id: str = Field(..., description="Conversation session ID")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata about processing (agent, LLM, tokens, etc)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Donut coklat kami harganya Rp 15.000 per piece.",
                "session_id": "session_abc123",
                "metadata": {
                    "agent_used": "product_agent",
                    "llm_provider": "gemini",
                    "model": "gemini-2.0-flash-exp",
                    "processing_time_ms": 1234.5,
                    "tokens_used": 150,
                    "sql_queries_executed": 1
                }
            }
        }