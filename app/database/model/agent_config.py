from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.database.model.base_model import Base

class AgentConfig(Base):
    """
    Agent LLM configuration model.
    Stores provider, model, and parameters for each agent.
    """

    __tablename__ = 'agent_configs'

    # primary key
    id = Column(Integer, primary_key=True)

    # Agent identification
    agent_name = Column(String(50), unique=True, nullable=False, index=True)
    agent_description = Column(Text, nullable=True)

    # LLM configuration
    llm_provider = Column(String(20), nullable=False)
    model_name = Column(String(50), nullable=False)
    temperature = Column(Float, nullable=False, default=0.7)
    max_tokens = Column(Integer, nullable=True)

    # Status and metadata
    is_active = Column(Boolean, nullable=False, default=True)
    config_metadata = Column(JSONB, nullable=True, default=dict)

    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<AgentConfig(agent_name={self.agent_name}, provider={self.llm_provider}, model={self.model_name})>"
    
    def to_dict(self):
        """Convert to dictionary for easy access"""
        return {
            "id": self.id,
            "agent_name": self.agent_name,
            "agent_description": self.agent_description,
            "llm_provider": self.llm_provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "is_active": self.is_active,
            "config_metadata": self.config_metadata or {},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
