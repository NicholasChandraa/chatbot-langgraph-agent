from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.pool import NullPool
from app.config.settings.settings import get_settings
from app.utils.logger import logger

class DatabaseManager:
    """
    Async Database Connection Manager with connection pooling.
    Singleton pattern - initialized once during app startup.
    """

    def __init__(self):
        self.settings = get_settings()
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
    
    def init(self):
        """
        Initialize database engine and session factory.
        Called during FastAPI lifespan startup.
        """
        if self._engine is not None:
            logger.warning("Database already initialized, skipping...")
            return
        
        try:
            logger.info("Initializing database connection pool...")

            # Create async engine with connection pooling
            self._engine = create_async_engine(
                self.settings.DATABASE_URL,
                pool_size=self.settings.DB_POOL_SIZE,
                max_overflow=self.settings.DB_MAX_OVERFLOW,
                pool_timeout=self.settings.DB_POOL_TIMEOUT,
                pool_pre_ping=True,  # Verify connections before using
                echo=self.settings.DB_ECHO_SQL,
                # For production, consider using QueuePool (default)
                # For testing, can use NullPool to avoid connection issues
                # poolclass=NullPool if self.settings.ENVIRONMENT == "test" else None,
            )

            # Create session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,  # Don't expire object after commit
                autoflush=False,  # Manual control over flushing
                autocommit=False,  # Explicit transaction control
            )

            logger.info("✅ Database connection pool initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {str(e)}", exc_info=True)
            raise
    
    async def close(self):
        """
        Close database engine and connection pool.
        Called during FastAPI lifespan shutdown.
        """
        if self._engine is None:
            logger.warning("Database not initialized, skipping close...")
            return
        
        try:
            logger.info("Closing database connection pool...")

            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

            logger.info("✅ Database connection pool closed")
        except Exception as e:
            logger.error(f"❌ Eror closing database: {str(e)}", exc_info=True)
            raise

    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with automatic transaction management.

        Usage with FastAPI dependency injection:
            @router.get("/products")
            async def get_products(db: AsyncSession = Depends(get_db)):
                result = await db.execute(select(Product))
                return result.scalars().all()
        """
        if self._session_factory is None:
            raise RuntimeError(
                "Database not initialized. Call db_manager.init() first."
            )
        
        async with self._session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()  # Auto-rollback on error
                raise
            finally:
                await session.close()
    

    async def health_check(self) -> bool:
        """
        Check database connectivity.
        Used by health endpoint.
        """
        if self._engine is None:
            return False
        
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False

# Singleton Instance
db_manager = DatabaseManager()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database session

    Usage:
        @router.get("/products")
        async def get_products(db: AsyncSession = Depends(get_db)):
            # Use db here
            pass
    """
    async for session in db_manager.get_session():
        yield session