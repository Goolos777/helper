from fastapi import APIRouter, Depends, HTTPException
from ..services.memory.service import MemoryService
from ..services.memory.models import MemoryInput, MemorySearchResults, Memory

router = APIRouter()

# Initialize memory service
memory_service = MemoryService()

from fastapi import APIRouter, HTTPException
from typing import List, Dict  # Добавим импорт List

from ..services.memory.service import MemoryService
from ..services.memory.models import MemoryInput, MemorySearchResults, Memory

router = APIRouter()

# Initialize memory service
memory_service = MemoryService()

@router.post("/memory", response_model=List[str])
async def add_memory(memory_input: MemoryInput):
    """Add a new memory to the system."""
    try:
        ids = memory_service.add_memory(
            text=memory_input.content,
            metadata=memory_input.metadata
        )
        return ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add memory: {str(e)}")

# Остальной код остается без изменений

@router.get("/memory/search", response_model=MemorySearchResults)
async def search_memories(query: str, limit: int = 5):
    """Search for memories using semantic similarity."""
    try:
        results = memory_service.search_memories(query=query, k=limit)
        memories = [Memory(**result) for result in results]
        return MemorySearchResults(query=query, results=memories)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search memories: {str(e)}")

@router.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a specific memory by ID."""
    try:
        memory_service.delete_memory(memory_id)
        return {"message": f"Memory {memory_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {str(e)}")

@router.delete("/memory")
async def clear_all_memories():
    """Delete all memories in the system."""
    try:
        memory_service.clear_all_memories()
        return {"message": "All memories cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear memories: {str(e)}")