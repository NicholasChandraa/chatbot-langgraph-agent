"""
FastAPI Application Entry Point
AI Chatbot Multi-Agent System
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes import health, chat, config, agents as agents_routes
from core.orchestrator import Orchestrator
from config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    settings = get_settings()
    print(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    print(f"Environment: {settings.ENVIRONMENT}")

    # Initialize core components
    orchestrator = Orchestrator()
    await orchestrator.initialize()

    app.state.orchestrator = orchestrator

    yield

    # Shutdown
    print("Shutting down application...")
    await orchestrator.shutdown()

app = FastAPI()