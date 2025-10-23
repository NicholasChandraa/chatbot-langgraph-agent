"""
Health Check Schema
"""
from pydantic import BaseModel, Field
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Environment (dev/staging/prod)")
    timestamp: datetime = Field(..., description="Current server timestamp")
    database_connected: bool = Field(..., description="Database Connectivity Status")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "Agent Service",
                "version": "1.0.0",
                "environment": "development",
                "timestamp": "2025-01-15T10:30:00Z",
                "database_connected": True
            }
        }