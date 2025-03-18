"""Data models for the memory service."""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

class MemoryInput(BaseModel):
    """Input model for adding a new memory."""
    content: str = Field(...,
                         description="The text content to remember",
                         min_length=1,
                         max_length=100000)
    metadata: Optional[Dict[str, Any]] = Field(default=None,
                                               description="Optional metadata for the memory")

    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('Content cannot be empty')
        return v

    @model_validator(mode='after')
    def validate_metadata(self) -> 'MemoryInput':
        if self.metadata:
            # Check for any invalid metadata values
            for key, value in self.metadata.items():
                if not isinstance(key, str):
                    raise ValueError(f"Metadata keys must be strings, got {type(key)}")

                # Special validation for any known metadata fields if needed
                if key == "priority" and not (isinstance(value, int) and 1 <= value <= 10):
                    raise ValueError("Priority must be an integer between 1 and 10")
        return self

class MemoryUpdateInput(BaseModel):
    """Input model for updating an existing memory."""
    content: str = Field(...,
                         description="The new text content",
                         min_length=1,
                         max_length=100000)
    metadata: Optional[Dict[str, Any]] = Field(default=None,
                                               description="Optional metadata for the memory")

    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('Content cannot be empty')
        return v

    @model_validator(mode='after')
    def validate_metadata(self) -> 'MemoryUpdateInput':
        if self.metadata:
            # Check for any invalid metadata values
            for key, value in self.metadata.items():
                if not isinstance(key, str):
                    raise ValueError(f"Metadata keys must be strings, got {type(key)}")

                # Special validation for any known metadata fields if needed
                if key == "priority" and not (isinstance(value, int) and 1 <= value <= 10):
                    raise ValueError("Priority must be an integer between 1 and 10")
        return self

class Memory(BaseModel):
    """Model representing a retrieved memory."""
    content: str = Field(..., description="The text content of the memory")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata associated with the memory")
    relevance: float = Field(..., description="Relevance score of the memory to the query")

class MemoryItem(BaseModel):
    """Model representing a stored memory."""
    id: str = Field(..., description="Unique identifier for the memory")
    content: str = Field(..., description="The text content of the memory")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata associated with the memory")

class MemorySearchResults(BaseModel):
    """Model representing search results."""
    query: str = Field(..., description="The original search query")
    results: List[Memory] = Field(..., description="List of matching memories")