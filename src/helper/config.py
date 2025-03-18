from pydantic_settings import BaseSettings
import os
from typing import Literal, Optional

class Settings(BaseSettings):
    # Application settings
    APP_TITLE: str = 'Helper App'
    APP_VERSION: str = '0.1.0'
    DEBUG: bool = True
    HOST: str = '127.0.0.1'
    PORT: int = 8000

    # Logging settings
    LOG_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'
    LOG_TO_FILE: bool = True
    LOG_RETENTION_DAYS: int = 30
    LOG_MAX_SIZE_MB: int = 10

    # Memory service settings
    MEMORY_STORE_DIR: str = './memory_store'
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    EMBEDDING_MODEL: str = 'all-MiniLM-L6-v2'
    EMBEDDING_BATCH_SIZE: int = 32  # Batch size for embedding operations

    # Caching settings
    ENABLE_CACHE: bool = True
    CACHE_TYPE: Literal['memory', 'redis'] = 'memory'
    CACHE_TTL_SEARCH: int = 300  # 5 minutes for search results
    CACHE_TTL_EMBEDDINGS: int = 3600  # 1 hour for embeddings
    REDIS_URL: Optional[str] = None  # Example: "redis://localhost:6379/0"

    # Performance settings
    MAX_WORKERS: int = 4  # Max worker threads for concurrent operations
    USE_ASYNC_CHROMA: bool = True  # Use async version of Chroma when available
    BATCH_SIZE: int = 100  # Batch size for bulk operations

    class Config:
        env_file = '.env'
        case_sensitive = True

settings = Settings()