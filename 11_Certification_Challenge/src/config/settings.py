""" Configuration settings loaded from the environment variables """

from pydantic_settings import BaseSettings
from typing import Optional
 

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str | None = None
    LANGSMITH_API_KEY: str | None = None
    COHERE_API_KEY: str | None = None
    TAVILY_API_KEY: str | None = None
    
    # Model Configuration (Cheapest options as defaults)
    LLM_MODEL: str = "gpt-4.1-mini"  # gpt-4o-mini Cheapest OpenAI model
    LLM_TEMPERATURE: float = 0.0
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # RAG Configuration
    CHUNK_SIZE: int = 750
    CHUNK_OVERLAP: int = 100
    RETRIEVAL_K: int = 5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()


