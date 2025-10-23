"""
Chat Endpoint - HTTP Layer
Thin controller that delegates to service layer
"""
from fastapi import APIRouter, status, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.utils.logger import logger
from app.database.connection.connection import get_db
from app.services.chatService import ChatService, ChatStreamService, ChatStreamEventService

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

    Memory is automatically managed by checkpointer using session_id as thread_id
    """
    try:
        # Delegate to service layer
        response = await ChatService.process_message(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            db=db
        )
        return response
        
    except Exception as e:
        logger.error(f"Error processing chat | error={str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/chat/stream",
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    summary="Process Chat Message with Streaming",
    description="Process user message through AI multi-agent system with real-time streaming"
)
async def chat_stream(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Chat with AI Business Intelligence System (Streaming)

    Returns Server-Sent Events (SSE) stream with real-time updates:
    - start: Stream started
    - agent_start: Agent/subagent started working
    - tool_result: Tool execution completed
    - message: AI response chunk
    - end: Stream completed with final response and metadata
    - error: Error occurred
    
    The system uses a supervisor to route your questions to specialized agents:
    (e.g. Product Agent, Sales Agent, Store Agent)

    Memory is automatically managed by checkpointer using session_id as thread_id
    
    Example usage with JavaScript EventSource:
    ```javascript
    const eventSource = new EventSource('/api/chat/stream', {
        method: 'POST',
        body: JSON.stringify({message: "Hello", user_id: "user1", session_id: "sess1"})
    });
    
    eventSource.addEventListener('message', (e) => {
        const data = JSON.parse(e.data);
        console.log('AI response:', data.content);
    });
    
    eventSource.addEventListener('end', (e) => {
        const data = JSON.parse(e.data);
        console.log('Final response:', data.response);
        eventSource.close();
    });
    ```
    """
    try:
        # Create async generator from service
        async def event_generator():
            try:
                async for sse_message in ChatStreamService.process_message_stream(
                    message=request.message,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    db=db
                ):
                    yield sse_message
            except Exception as e:
                logger.error(f"Error in stream generator: {str(e)}", exc_info=True)
                # Send error event
                yield f"event: error\ndata: {{\"error\": \"{str(e)}\"}}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable buffering in nginx
            }
        )
        
    except Exception as e:
        logger.error(f"Error setting up stream | error={str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/chat/stream-events",
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    summary="Process Chat Message with Token-by-Token Streaming",
    description="Process user message with real-time token-by-token streaming (like ChatGPT)"
)
async def chat_stream_events(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Chat with AI Business Intelligence System (Token-by-Token Streaming)

    Returns Server-Sent Events (SSE) stream with REAL-TIME token-level updates:
    - start: Stream started
    - agent_start: Agent/subagent started working  
    - token: Individual token from AI (creates typing effect)
    - message_complete: Full message when agent finishes
    - tool_start: Tool execution started
    - tool_end: Tool execution completed
    - agent_end: Agent/subagent completed
    - end: Stream completed with final response and metadata
    - error: Error occurred
    
    This provides the smoothest user experience with real-time typing effect,
    similar to ChatGPT interface.
    
    The system uses a supervisor to route your questions to specialized agents:
    (e.g. Product Agent, Sales Agent, Store Agent)

    Memory is automatically managed by checkpointer using session_id as thread_id
    
    Example usage with JavaScript EventSource:
    ```javascript
    const eventSource = new EventSource('/api/chat/stream-events', {
        method: 'POST',
        body: JSON.stringify({
            message: "Berapa harga donut?",
            user_id: "user1",
            session_id: "sess1"
        })
    });
    
    // Real-time tokens (typing effect)
    eventSource.addEventListener('token', (e) => {
        const data = JSON.parse(e.data);
        appendToken(data.token);  // Append to UI as it arrives
    });
    
    // Complete message
    eventSource.addEventListener('message_complete', (e) => {
        const data = JSON.parse(e.data);
        console.log('Complete:', data.content);
    });
    
    // Stream end
    eventSource.addEventListener('end', (e) => {
        const data = JSON.parse(e.data);
        console.log('Done:', data.response);
        eventSource.close();
    });
    ```
    """
    try:
        # Create async generator from service
        async def event_generator():
            try:
                async for sse_message in ChatStreamEventService.process_message_stream_events(
                    message=request.message,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    db=db
                ):
                    yield sse_message
            except Exception as e:
                logger.error(f"Error in event stream generator: {str(e)}", exc_info=True)
                # Send error event
                yield f"event: error\ndata: {{\"error\": \"{str(e)}\"}}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable buffering in nginx
            }
        )
        
    except Exception as e:
        logger.error(f"Error setting up event stream | error={str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))