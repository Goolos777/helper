"""Data models for the memory service."""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class MemoryInput(BaseModel):
    """Input model for adding a new memory."""
    content: str = Field(..., description="The text content to remember")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata for the memory")

class Memory(BaseModel):
    """Model representing a retrieved memory."""
    content: str = Field(..., description="The text content of the memory")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata associated with the memory")
    relevance: float = Field(..., description="Relevance score of the memory to the query")

class MemorySearchResults(BaseModel):
    """Model representing search results."""
    query: str = Field(..., description="The original search query")
    results: List[Memory] = Field(..., description="List of matching memories")