"""
Memory service based on langchain and Chroma.
Provides functionality for storing and retrieving memories using embeddings.
"""
import os
import time
from typing import List, Optional, Dict, Any
from threading import RLock

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from fastapi import HTTPException

from ...config import settings
from ...utils.logger import get_logger

# Initialize logger
logger = get_logger("memory_service")

# Singleton for embedding model to avoid reloading
_embedding_model = None

def get_embedding_model():
    """Get or initialize the embedding model (singleton pattern)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Initializing embedding model ({settings.EMBEDDING_MODEL})")
        start_time = time.time()
        try:
            _embedding_model = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': settings.EMBEDDING_BATCH_SIZE
                }
            )
            logger.info(f"Embedding model loaded successfully in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise
    return _embedding_model

class MemoryService:
    def __init__(self, persist_directory: str = None):
        """Initialize the memory service with a vector store."""
        logger.info(f"Initializing MemoryService")
        self.persist_directory = persist_directory or settings.MEMORY_STORE_DIR
        os.makedirs(self.persist_directory, exist_ok=True)

        # Add lock for thread safety
        self._lock = RLock()

        try:
            # Get embedding model using singleton pattern
            self.embeddings = get_embedding_model()

            # Initialize vector store
            logger.info(f"Initializing Chroma vector store at {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

            # Text splitter for chunking documents using settings
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            logger.info(f"MemoryService initialized with chunk_size={settings.CHUNK_SIZE}, "
                        f"chunk_overlap={settings.CHUNK_OVERLAP}")
        except Exception as e:
            logger.error(f"Failed to initialize MemoryService: {e}", exc_info=True)
            raise

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add new memory text to the vector store.

        Args:
            text: The text content to store
            metadata: Optional metadata to store with the text

        Returns:
            List of IDs for the stored chunks
        """
        if not text or not text.strip():
            logger.warning("Attempted to add empty text memory")
            raise ValueError("Cannot add empty text as memory")

        logger.info(f"Adding new memory, text length: {len(text)}, metadata: {metadata}")
        start_time = time.time()

        with self._lock:  # Добавляем блокировку для потокобезопасности
            try:
                chunks = self.text_splitter.split_text(text)
                logger.debug(f"Split text into {len(chunks)} chunks")

                # Prepare metadata for each chunk if provided
                metadatas = [metadata] * len(chunks) if metadata else None

                # Add documents to the vector store
                ids = self.vectorstore.add_texts(
                    texts=chunks,
                    metadatas=metadatas
                )

                # Ensure persistence if needed
                if hasattr(self.vectorstore, '_persist'):
                    self.vectorstore._persist()

                logger.info(f"Memory added successfully with {len(ids)} chunks in {time.time() - start_time:.2f} seconds. IDs: {ids}")
                return ids
            except Exception as e:
                logger.error(f"Failed to add memory: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to add memory: {str(e)}")

    def search_memories(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant memories based on a query.

        Args:
            query: The search query
            k: Number of results to return

        Returns:
            List of documents with their content and metadata
        """
        if not query or not query.strip():
            logger.warning("Attempted to search with empty query")
            raise ValueError("Cannot search with empty query")

        logger.info(f"Searching memories with query: '{query}', k={k}")
        start_time = time.time()

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            # Format results
            memories = []
            for doc, score in results:
                memories.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance": float(score)
                })

            logger.info(f"Found {len(memories)} results in {time.time() - start_time:.2f} seconds")
            logger.debug(f"Search results: {memories}")
            return memories
        except Exception as e:
            logger.error(f"Failed to search memories: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to search memories: {str(e)}")

    def delete_memory(self, memory_id: str) -> None:
        """
        Delete a specific memory by ID.

        Args:
            memory_id: The ID of the memory to delete

        Raises:
            HTTPException: If the memory ID doesn't exist
        """
        if not memory_id or not memory_id.strip():
            logger.warning("Attempted to delete with empty ID")
            raise ValueError("Cannot delete with empty ID")

        logger.info(f"Deleting memory with ID: {memory_id}")

        with self._lock:  # Thread safety
            try:
                # Check if memory exists first
                all_ids = self.vectorstore.get()["ids"]
                if memory_id not in all_ids:
                    logger.warning(f"Memory with ID {memory_id} not found")
                    raise HTTPException(status_code=404, detail=f"Memory with ID {memory_id} not found")

                self.vectorstore.delete([memory_id])

                # Ensure persistence if needed
                if hasattr(self.vectorstore, '_persist'):
                    self.vectorstore._persist()

                logger.info(f"Memory {memory_id} deleted successfully")
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise e
                logger.error(f"Error deleting memory: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error deleting memory: {str(e)}")

    def clear_all_memories(self) -> None:
        """Delete all memories in the vector store."""
        logger.info("Clearing all memories")

        with self._lock:  # Thread safety
            try:
                # Get all IDs
                all_ids = self.vectorstore.get()["ids"]
                if all_ids:
                    logger.info(f"Found {len(all_ids)} memories to delete")

                    # Delete in batches for better performance
                    batch_size = settings.BATCH_SIZE
                    for i in range(0, len(all_ids), batch_size):
                        batch = all_ids[i:i+batch_size]
                        logger.debug(f"Deleting batch {i//batch_size + 1} with {len(batch)} memories")
                        self.vectorstore.delete(batch)
                else:
                    logger.info("No memories to delete")

                # Ensure persistence if needed
                if hasattr(self.vectorstore, '_persist'):
                    self.vectorstore._persist()

                logger.info("All memories cleared successfully")
            except Exception as e:
                logger.error(f"Error clearing memories: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error clearing memories: {str(e)}")

    def get_all_memories(self, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """
        Retrieve all stored memories with pagination.

        Args:
            skip: Number of items to skip
            limit: Maximum number of items to return

        Returns:
            Dictionary with pagination information and memory items
        """
        logger.info(f"Retrieving all memories (skip={skip}, limit={limit})")

        try:
            results = self.vectorstore.get()

            memories = []
            if results and 'documents' in results and results['documents']:
                for i, doc in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if 'metadatas' in results and results['metadatas'] else {}
                    memory_id = results['ids'][i] if 'ids' in results and results['ids'] else f"memory_{i}"

                    memories.append({
                        "id": memory_id,
                        "content": doc,
                        "metadata": metadata
                    })

            # Apply pagination
            paginated_memories = memories[skip:skip+limit]
            logger.info(f"Retrieved {len(paginated_memories)} memories (from total {len(memories)})")

            return {
                "total": len(memories),
                "items": paginated_memories,
                "page": skip // limit + 1 if limit > 0 else 1,
                "page_size": limit,
                "pages": (len(memories) + limit - 1) // limit if limit > 0 else 1
            }
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error retrieving memories: {str(e)}")

    def update_memory(self, memory_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update an existing memory.

        Args:
            memory_id: The ID of the memory to update
            text: The new text content
            metadata: The new metadata (optional)

        Raises:
            HTTPException: If the memory ID doesn't exist
        """
        if not memory_id or not memory_id.strip():
            logger.warning("Attempted to update with empty ID")
            raise ValueError("Cannot update memory with empty ID")

        if not text or not text.strip():
            logger.warning("Attempted to update with empty text")
            raise ValueError("Cannot update memory with empty text")

        logger.info(f"Updating memory with ID: {memory_id}")

        with self._lock:  # Thread safety
            try:
                # Check if memory exists first
                all_ids = self.vectorstore.get()["ids"]
                if memory_id not in all_ids:
                    logger.warning(f"Memory with ID {memory_id} not found")
                    raise HTTPException(status_code=404, detail=f"Memory with ID {memory_id} not found")

                # First delete the old memory
                self.vectorstore.delete([memory_id])
                logger.debug(f"Deleted original memory {memory_id}")

                # Find and delete any related chunk IDs
                related_chunks = [id for id in all_ids if id.startswith(f"{memory_id}_")]
                if related_chunks:
                    logger.debug(f"Deleting {len(related_chunks)} related chunks: {related_chunks}")
                    self.vectorstore.delete(related_chunks)

                # Then add the new version
                chunks = self.text_splitter.split_text(text)
                logger.debug(f"Split updated text into {len(chunks)} chunks")
                metadatas = [metadata] * len(chunks) if metadata else None

                new_ids = [memory_id] + [f"{memory_id}_{i}" for i in range(1, len(chunks))]
                self.vectorstore.add_texts(
                    texts=chunks,
                    metadatas=metadatas,
                    ids=new_ids
                )

                # Ensure persistence if needed
                if hasattr(self.vectorstore, '_persist'):
                    self.vectorstore._persist()

                logger.info(f"Memory {memory_id} updated successfully with {len(chunks)} chunks")
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise e
                logger.error(f"Error updating memory: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error updating memory: {str(e)}")

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the memory service configuration.

        Returns:
            Dictionary with service information
        """
        return {
            "store_directory": self.persist_directory,
            "embedding_model": settings.EMBEDDING_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "batch_size": settings.BATCH_SIZE,
            "cache_enabled": settings.ENABLE_CACHE
        }