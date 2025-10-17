"""
Chat Endpoint - Main AI Agent Processing
"""
from fastapi import APIRouter, status, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import time

from app.schemas.chat import ChatRequest, ChatResponse
from app.utils.logger import logger
from app.database.connection.connection import get_db
from app.agents.supervisor_agent import create_supervisor_agent
from app.utils.cost_calculator import calculate_cost

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    summary="Process Chat Message",
    description="Process user message through AI multi-agent system"
)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Chat with AI Business Intelligence System

    The system uses a supervisor to route your questions to specialized agents:
    (e.g. Product Agent, Sales Agent, Store Agent)
    """
    start_time = time.time()

    try:
        logger.info(
            f"💬 Chat Request | "
            f"user={request.user_id} | "
            f"session={request.session_id}"
        )

        logger.info(f"✍️ Question: {request.message}")

        # Crate supervisor (with all agents)
        supervisor = await create_supervisor_agent(db)

        # Build messages list with conversation history
        messages = []

        # Add conversation history if provided
        for turn in request.conversation_history:
            if turn.role == "user":
                messages.append(("user", turn.content))
            elif turn.role == "assistant":
                messages.append(("assistant", turn.content))
        
        # Add current message
        messages.append(("user", request.message))

        # Invoke supervisor with messages
        logger.info("🚀 Invoking supervisor agent...")
        result = await supervisor.ainvoke({
            "messages": messages
        })

        # Extract final response
        # output_mode="full_history" gives us all messages
        all_messages = result.get("messages", [])

        # Log all messages for debugging
        logger.info(f"📨 Total messages in response: {len(all_messages)}")
        for i, msg in enumerate(all_messages):
            msg_type = getattr(msg, 'type', 'unknown')
            msg_name = getattr(msg, 'name', 'N/A')
            content_preview = str(msg.content)[:100] if hasattr(msg, 'content') else 'N/A'
            logger.debug(
                f"  [{i}] Type={msg_type} | Name={msg_name} | Content={content_preview}..."
            )

        # Get last assistant message
        response_text = "Maaf, saya tidak dapat menjawab pertanyaan ini."
        for msg in reversed(all_messages):
            if hasattr(msg, 'type') and msg.type == 'ai':
                response_text = msg.content
                logger.info(f"✅ Final AI response extracted (length={len(response_text)} chars)")
                break
        
        processing_time = (time.time() - start_time) * 1000

        logger.info(f"✅ Response generated | time={processing_time:.2f}ms")
        logger.debug(f"Response: {response_text[:100]}...")

        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            metadata={
                "supervisor": "supervisor_agent",
                "processing_time_ms": round(processing_time, 2),
                "message_count": len(all_messages),
                "conversation_turns": len(request.conversation_history) + 1
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing chat | error = {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))