from fastapi import APIRouter, HTTPException, Request
from typing import List, Dict, Optional
import time

from ..services.memory.service import MemoryService
from ..services.memory.models import MemoryInput, MemorySearchResults, Memory, MemoryUpdateInput
from ..utils.logger import get_logger

# Initialize logger
logger = get_logger("api")

router = APIRouter()

# Initialize memory service
memory_service = MemoryService()

@router.post("/memory", response_model=List[str])
async def add_memory(memory_input: MemoryInput, request: Request):
    """Add a new memory to the system."""
    request_id = id(request)
    client_ip = request.client.host if request.client else "unknown"

    logger.info(f"Request #{request_id} from {client_ip}: Adding new memory")
    logger.debug(f"Request #{request_id}: Input metadata: {memory_input.metadata}")

    start_time = time.time()
    try:
        ids = memory_service.add_memory(
            text=memory_input.content,
            metadata=memory_input.metadata
        )

        process_time = time.time() - start_time
        logger.info(f"Request #{request_id}: Memory added successfully in {process_time:.2f}s (IDs: {ids})")
        return ids
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request #{request_id}: Failed to add memory in {process_time:.2f}s: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add memory: {str(e)}")

@router.get("/memory/search", response_model=MemorySearchResults)
async def search_memories(request: Request, query: str, limit: int = 5):
    """Search for memories using semantic similarity."""
    request_id = id(request)
    client_ip = request.client.host if request.client else "unknown"

    logger.info(f"Request #{request_id} from {client_ip}: Searching memories with query: '{query}', limit: {limit}")

    start_time = time.time()
    try:
        results = memory_service.search_memories(query=query, k=limit)
        memories = [Memory(**result) for result in results]

        process_time = time.time() - start_time
        logger.info(f"Request #{request_id}: Search completed in {process_time:.2f}s, found {len(memories)} results")
        return MemorySearchResults(query=query, results=memories)
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request #{request_id}: Failed to search memories in {process_time:.2f}s: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to search memories: {str(e)}")

@router.get("/memory", response_model=List[Dict])
async def get_all_memories(request: Request):
    """Get all memories stored in the system."""
    request_id = id(request)
    client_ip = request.client.host if request.client else "unknown"

    logger.info(f"Request #{request_id} from {client_ip}: Getting all memories")

    start_time = time.time()
    try:
        memories = memory_service.get_all_memories()

        process_time = time.time() - start_time
        logger.info(f"Request #{request_id}: Retrieved {len(memories)} memories in {process_time:.2f}s")
        return memories
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request #{request_id}: Failed to retrieve memories in {process_time:.2f}s: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memories: {str(e)}")

@router.put("/memory/{memory_id}")
async def update_memory(memory_id: str, update_input: MemoryUpdateInput, request: Request):
    """Update an existing memory by ID."""
    request_id = id(request)
    client_ip = request.client.host if request.client else "unknown"

    logger.info(f"Request #{request_id} from {client_ip}: Updating memory ID: {memory_id}")
    logger.debug(f"Request #{request_id}: Update metadata: {update_input.metadata}")

    start_time = time.time()
    try:
        memory_service.update_memory(
            memory_id=memory_id,
            text=update_input.content,
            metadata=update_input.metadata
        )

        process_time = time.time() - start_time
        logger.info(f"Request #{request_id}: Memory {memory_id} updated successfully in {process_time:.2f}s")
        return {"message": f"Memory {memory_id} updated successfully"}
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request #{request_id}: Failed to update memory in {process_time:.2f}s: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update memory: {str(e)}")

@router.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str, request: Request):
    """Delete a specific memory by ID."""
    request_id = id(request)
    client_ip = request.client.host if request.client else "unknown"

    logger.info(f"Request #{request_id} from {client_ip}: Deleting memory ID: {memory_id}")

    start_time = time.time()
    try:
        memory_service.delete_memory(memory_id)

        process_time = time.time() - start_time
        logger.info(f"Request #{request_id}: Memory {memory_id} deleted successfully in {process_time:.2f}s")
        return {"message": f"Memory {memory_id} deleted successfully"}
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request #{request_id}: Failed to delete memory in {process_time:.2f}s: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {str(e)}")

@router.delete("/memory")
async def clear_all_memories(request: Request):
    """Delete all memories in the system."""
    request_id = id(request)
    client_ip = request.client.host if request.client else "unknown"

    logger.info(f"Request #{request_id} from {client_ip}: Clearing all memories")

    start_time = time.time()
    try:
        memory_service.clear_all_memories()

        process_time = time.time() - start_time
        logger.info(f"Request #{request_id}: All memories cleared successfully in {process_time:.2f}s")
        return {"message": "All memories cleared successfully"}
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request #{request_id}: Failed to clear memories in {process_time:.2f}s: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear memories: {str(e)}")