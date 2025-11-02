"""
Application Settings & Configuration
Loads from environment variables with defaults
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""

    # ==============================================
    # APP CONFIGURATION
    # ==============================================
    APP_NAME: str = Field(default="Agent Service", env="APP_NAME")
    VERSION: str = Field(default="1.0.0", env="VERSION")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")

    # Server Config
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    LOG_LEVEL: str = Field(default="DEBUG", env="LOG_LEVEL")

    # ==============================================
    # LLM PROVIDER CONFIGURATION
    # ==============================================
    DEFAULT_LLM_PROVIDER: str = Field(default="gemini", env="DEFAULT_LLM_PROVIDER")

    # OpenAI
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(default="gpt-5-nano", env="OPENAI_MODEL")

    # Anthropic
    ANTHROPIC_API_KEY: str = Field(default="", env="ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = Field(default="claude-4-5-20250929", env="ANTHROPIC_MODEL")

    # Google Gemini
    # GEMINI_API_KEY: str = Field(default="", env="GEMINI_API_KEY")
    GOOGLE_API_KEY: str = Field(default="", env="GOOGLE_API_KEY")
    GEMINI_MODEL: str = Field(default="gemini-2.5-pro", env="GEMINI_MODEL")

    # Ollama (Local LLM)
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = Field(default="qwen3:8b", env="OLLAMA_MODEL")

    # ==============================================
    # DATABASE CONFIGURATION
    # ==============================================
    DB_HOST: str = Field(default="localhost", env="DB_HOST")
    DB_PORT: int = Field(default=5432, env="DB_PORT")
    DB_USER: str = Field(default="postgres", env="DB_USER")
    DB_PASSWORD: str = Field(default="postgres", env="DB_PASSWORD")
    DB_NAME: str = Field(default="chatbot_chat_db", env="DB_NAME")
    
    # Database URL (auto-generated)
    @property
    def DATABASE_URL(self) -> str:
        """Generate database URL from components"""
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # SQL Agent Database URL
    @property
    def SQLAGENT_DATABASE_URL(self) -> str:
        """Generate SQL Agent URL from components"""
        return f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @property
    def POSTGRES_URL(self) -> str:
        """Generate PURE POSTGRE SQL URL from components"""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # Connection Pool Settings
    DB_POOL_SIZE: int = Field(default=20, env="DB_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(default=10, env="DB_MAX_OVERFLOW")
    DB_POOL_TIMEOUT: int = Field(default=30, env="DB_POOL_TIMEOUT")
    DB_POOL_RECYCLE: int = Field(default=3600, env="DB_POOL_RECYCLE")
    DB_ECHO_SQL: bool = Field(default=False, env="DB_ECHO_SQL")

    # ==============================================
    # PGVECTOR CONFIGURATION
    # ==============================================
    VECTOR_DIMENSION: int = Field(default=768, env="VECTOR_DIMENSION")
    EMBEDDING_MODEL: str = Field(default="models/text-embedding-004", env="EMBEDDING_MODEL")

    # ==============================================
    # LANGGRAPH CONFIGURATION
    # ==============================================
    MAX_ITERATIONS: int = Field(default=10, env="MAX_ITERATIONS")
    AGENT_TIMEOUT_SECONDS: int = Field(default=60, env="AGENT_TIMEOUT_SECONDS")

    # ==============================================
    # TOOLS CONFIGURATION
    # ==============================================
    SQL_GENERATION_MAX_RETRIES: int = Field(default=3, env="SQL_GENERATION_MAX_RETRIES")
    SQL_VALIDATION_ENABLED: bool = Field(default=True, env="SQL_VALIDATION_ENABLED")
    QUERY_TIMEOUT_SECONDS: int = Field(default=30, env="QUERY_TIMEOUT_SECONDS")
    MAX_QUERY_RESULTS: int = Field(default=1000, env="MAX_QUERY_RESULTS")

    # ==============================================
    # AGENT-SPECIFIC SETTINGS
    # ==============================================
    # Product Agent
    PRODUCT_AGENT_MODEL: str = Field(default="gemini-2.5-pro", env="PRODUCT_AGENT_MODEL")
    PRODUCT_AGENT_TEMPERATURE: float = Field(default=0.3, env="PRODUCT_AGENT_TEMPERATURE")

    # Sales Agent
    SALES_AGENT_MODEL: str = Field(default="gemini-2.5-pro", env="SALES_AGENT_MODEL")
    SALES_AGENT_TEMPERATURE: float = Field(default=0.3, env="SALES_AGENT_TEMPERATURE")

    # Stock Agent
    STOCK_AGENT_MODEL: str = Field(default="gemini-2.5-pro", env="STOCK_AGENT_MODEL")
    STOCK_AGENT_TEMPERATURE: float = Field(default=0.3, env="STOCK_AGENT_TEMPERATURE")

    # Store Agent
    STORE_AGENT_MODEL: str = Field(default="gemini-2.5-pro", env="STORE_AGENT_MODEL")
    STORE_AGENT_TEMPERATURE: float = Field(default=0.3, env="STORE_AGENT_TEMPERATURE")

    # Customer Agent
    CUSTOMER_AGENT_MODEL: str = Field(default="gemini-2.5-pro", env="CUSTOMER_AGENT_MODEL")
    CUSTOMER_AGENT_TEMPERATURE: float = Field(default=0.3, env="CUSTOMER_AGENT_TEMPERATURE")

    # Report Agent
    REPORT_AGENT_MODEL: str = Field(default="gemini-2.5-pro", env="REPORT_AGENT_MODEL")
    REPORT_AGENT_TEMPERATURE: float = Field(default=0.5, env="REPORT_AGENT_TEMPERATURE")

    # ==============================================
    # REDIS CONFIGURATION
    # ==============================================
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_PASSWORD: str = Field(default="", env="REDIS_PASSWORD")
    CACHE_TTL_SECONDS: int = Field(default=3600, env="CACHE_TTL_SECONDS")

    # ==============================================
    # API SECURITY
    # ==============================================
    INTERNAL_API_KEY: str = Field(default="", env="INTERNAL_API_KEY")

    # ==============================================
    # MONITORING & OBSERVABILITY
    # ==============================================
    SENTRY_DSN: str = Field(default="", env="SENTRY_DSN")
    ENABLE_METRICS: bool = Field(default=False, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")

    class Config:
        """Pydantic config"""
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()