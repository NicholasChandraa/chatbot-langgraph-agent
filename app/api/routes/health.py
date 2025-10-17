"""
Health Check Endpoint
"""
from fastapi import APIRouter, status
from datetime import datetime, UTC

from app.schemas.health import HealthResponse
from app.config.settings.settings import get_settings
from app.utils.logger import logger
from app.database.connection.connection import db_manager

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Health Check",
    description="Check if the Agent Service is running and healthy"
)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers

    Returns:
        HealthResponse: Service health status and metadata
    """
    settings = get_settings()

    logger.debug("Health check requested")

    return HealthResponse(
        status="healthy",
        service=settings.APP_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(UTC)
    )

@router.get(
    "/health-db",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Health Check",
    description="Check if database connection is healthy"
)
async def health_db_check():
    """
     Health check endpoint for monitoring and load balancers.
    
    Returns:
        - Service status
        - Service metadata
        - Database connectivity status
    """
    settings = get_settings()

    logger.debug("Health check requested")

    db_healthy = await db_manager.health_check()

    # Overall status
    overall_status = "healthy" if db_healthy else 'degraded'

    return HealthResponse(
        status=overall_status,
        service=settings.APP_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(UTC),
        database_connected=db_healthy
    )