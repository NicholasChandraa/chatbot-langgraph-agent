"""
Application Settings & Configuration
Loads from environment variables with defaults
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""

    # App info
    APP_NAME: str = "AI Chatbot Multi-Agent System"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")

    # Server Config
    HOST: str = Field(default="0.0.0.0", env = "HOST")
    PORT: int = Field(default=8000, env="PORT")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")