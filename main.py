"""
FastAPI Application Entry Point
AI Chatbot Multi-Agent System - Agent Service
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import health_routes, chat_routes, admin_routes
from app.config.settings.settings import get_settings
from app.utils.logger import logger
from app.database.connection.connection import db_manager
from app.config.agent_config.agent_config_manager import get_agent_config
from app.database.memory.checkpointer import checkpointer_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    settings = get_settings()
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"📍 Environment: {settings.ENVIRONMENT}")
    logger.info(f"🤖 Default LLM Provider: {settings.DEFAULT_LLM_PROVIDER}")
    logger.info(f"🔧 Debug Mode: {settings.DEBUG}")

    # Initialize database connection pool
    try:
        db_manager.init()

        # Test database connectivity
        is_healthy = await db_manager.health_check()

        if is_healthy:
            logger.info("✅ Database connection verified")

            # Initialize checkpointer AFTER database is ready
            try:
                await checkpointer_manager.init()
            except Exception as e:
                logger.error(f"❌ Checkpointer init failed: {e}")
                logger.warning("⚠️ Continuing without persistent memory")

            # Warm up cache for common agents
            async for db in db_manager.get_session():
                common_agents = ["sql_agent", "product_agent", "sales_agent", "report_agent", "store_agent"]
                for agent_name in common_agents:
                    try:
                        await get_agent_config(agent_name, db)
                        logger.info(f"✅ Warmed cache for {agent_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to warm cache for {agent_name}: {e}")
                break  # Only need one session for warming
        else:
            logger.warning("⚠️ Database connection failed - service will start but DB operations may fail")

    except Exception as e:
        logger.error(f"❌ Database initialization error: {str(e)}", exc_info=True)
        logger.warning("⚠️ Service will start but DB operations may fail")

    # TODO: Initialize LangGraph supervisor
    # TODO: Load database schema for vector search

    logger.info("=" * 60)
    logger.info("✅ Startup complete - Ready to accept requests")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("=" * 60)
    logger.info("🛑 Shutting down Agent Service...")
    logger.info("=" * 60)

    # Close checkpointer first
    try:
        await checkpointer_manager.close()
    except Exception as e:
        logger.error(f"Error closing checkpointer: {e}")
    
    # Close database connections
    try:
        await db_manager.close()
    except Exception as e:
        logger.error(f"Error during database shutdown: {str(e)}", exc_info=True)

    logger.info("✅ Shutdown Complete")
    logger.info("=" * 60)
    
    # TODO: Cleanup resources


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application

    Returns:
        FastAPI: Configured application instance
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description="AI-powered multi-agent system for donut shop business intelligence",
        version=settings.VERSION,
        lifespan=lifespan,
        debug=settings.DEBUG,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Configure proper origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(health_routes.router, prefix="/api", tags=["Health"])
    app.include_router(chat_routes.router, prefix="/api", tags=["Chat"])
    app.include_router(admin_routes.router, prefix="/api", tags=["Admin"])
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )