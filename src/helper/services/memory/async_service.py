"""
Asynchronous Memory service based on langchain and Chroma.
Provides functionality for storing and retrieving memories using embeddings with async support.
"""
import os
import time
from typing import List, Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from fastapi import HTTPException

from ...config import settings
from ...utils.logger import get_logger
from ...utils.cache import async_cache, cache

# Initialize logger
logger = get_logger("async_memory_service")

# Shared thread pool executor
_executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)

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
                encode_kwargs={'normalize_embeddings': True, 'batch_size': settings.EMBEDDING_BATCH_SIZE}
            )
            logger.info(f"Embedding model loaded successfully in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise
    return _embedding_model

class AsyncMemoryService:
    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize the memory service with a vector store."""
        self.persist_directory = persist_directory or settings.MEMORY_STORE_DIR
        os.makedirs(self.persist_directory, exist_ok=True)

        try:
            # Get embedding model using singleton pattern
            self.embeddings = get_embedding_model()

            # Initialize vector store
            logger.info(f"Initializing Chroma vector store at {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

            # Text splitter for chunking documents
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            logger.info(f"AsyncMemoryService initialized with chunk_size={settings.CHUNK_SIZE}, "
                        f"chunk_overlap={settings.CHUNK_OVERLAP}")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncMemoryService: {e}", exc_info=True)
            raise

    async def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add new memory text to the vector store asynchronously.

        Args:
            text: The text content to store
            metadata: Optional metadata to store with the text

        Returns:
            List of IDs for the stored chunks
        """
        logger.info(f"Adding new memory asynchronously, text length: {len(text)}")
        start_time = time.time()

        try:
            # Run text splitting in executor
            chunks = await asyncio.get_event_loop().run_in_executor(
                _executor,
                self.text_splitter.split_text,
                text
            )
            logger.debug(f"Split text into {len(chunks)} chunks")

            # Prepare metadata for each chunk if provided
            metadatas = [metadata] * len(chunks) if metadata else None

            # Execute vector store operations in thread pool
            ids = await asyncio.get_event_loop().run_in_executor(
                _executor,
                self._add_texts_to_vectorstore,
                chunks,
                metadatas
            )

            logger.info(f"Memory added successfully with {len(ids)} chunks in {time.time() - start_time:.2f} seconds")
            return ids
        except Exception as e:
            logger.error(f"Failed to add memory: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to add memory: {str(e)}")

    def _add_texts_to_vectorstore(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]]) -> List[str]:
        """Helper method to add texts to vectorstore in a thread."""
        ids = self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        self.vectorstore.persist()
        return ids

    @async_cache(ttl=lambda: settings.CACHE_TTL_SEARCH, prefix="memory_search")
    async def search_memories(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant memories based on a query asynchronously.

        Args:
            query: The search query
            k: Number of results to return

        Returns:
            List of documents with their content and metadata
        """
        logger.info(f"Searching memories asynchronously with query: '{query}', k={k}")
        start_time = time.time()

        try:
            # Execute similarity search in thread pool
            results = await asyncio.get_event_loop().run_in_executor(
                _executor,
                self._search_in_vectorstore,
                query,
                k
            )

            # Format results
            memories = []
            for doc, score in results:
                memories.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance": float(score)
                })

            logger.info(f"Found {len(memories)} results in {time.time() - start_time:.2f} seconds")
            return memories
        except Exception as e:
            logger.error(f"Failed to search memories: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to search memories: {str(e)}")

    def _search_in_vectorstore(self, query: str, k: int) -> List[tuple]:
        """Helper method to perform search in vectorstore in a thread."""
        return self.vectorstore.similarity_search_with_score(query, k=k)

    async def delete_memory(self, memory_id: str) -> None:
        """
        Delete a specific memory by ID asynchronously.

        Args:
            memory_id: The ID of the memory to delete

        Raises:
            HTTPException: If the memory ID doesn't exist
        """
        logger.info(f"Deleting memory asynchronously with ID: {memory_id}")

        try:
            # Check if memory exists first - run in thread
            all_ids = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: self.vectorstore.get()["ids"]
            )

            if memory_id not in all_ids:
                logger.warning(f"Memory with ID {memory_id} not found")
                raise HTTPException(status_code=404, detail=f"Memory with ID {memory_id} not found")

            # Delete in thread
            await asyncio.get_event_loop().run_in_executor(
                _executor,
                self._delete_from_vectorstore,
                [memory_id]
            )

            logger.info(f"Memory {memory_id} deleted successfully")
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error deleting memory: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error deleting memory: {str(e)}")

    def _delete_from_vectorstore(self, ids: List[str]) -> None:
        """Helper method to delete from vectorstore in a thread."""
        self.vectorstore.delete(ids)
        self.vectorstore.persist()

    async def clear_all_memories(self) -> None:
        """Delete all memories in the vector store asynchronously."""
        logger.info("Clearing all memories asynchronously")

        try:
            # Get all IDs in thread
            all_ids = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: self.vectorstore.get()["ids"]
            )

            if all_ids:
                logger.info(f"Found {len(all_ids)} memories to delete")
                # Delete in batches for better performance
                batch_size = settings.BATCH_SIZE
                for i in range(0, len(all_ids), batch_size):
                    batch = all_ids[i:i+batch_size]
                    logger.debug(f"Deleting batch {i//batch_size + 1} with {len(batch)} memories")

                    await asyncio.get_event_loop().run_in_executor(
                        _executor,
                        self._delete_from_vectorstore,
                        batch
                    )
            else:
                logger.info("No memories to delete")

            logger.info("All memories cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing memories: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error clearing memories: {str(e)}")

    async def get_all_memories(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve all stored memories asynchronously with pagination.

        Args:
            skip: Number of items to skip
            limit: Maximum number of items to return

        Returns:
            List of all documents in the vector store
        """
        logger.info(f"Retrieving all memories asynchronously (skip={skip}, limit={limit})")

        try:
            # Get all documents in thread
            results = await asyncio.get_event_loop().run_in_executor(
                _executor,
                self.vectorstore.get
            )

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

    async def update_memory(self, memory_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update an existing memory asynchronously.

        Args:
            memory_id: The ID of the memory to update
            text: The new text content
            metadata: The new metadata (optional)

        Raises:
            HTTPException: If the memory ID doesn't exist
        """
        logger.info(f"Updating memory asynchronously with ID: {memory_id}")

        try:
            # Check if memory exists first - run in thread
            all_ids = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: self.vectorstore.get()["ids"]
            )

            if memory_id not in all_ids:
                logger.warning(f"Memory with ID {memory_id} not found")
                raise HTTPException(status_code=404, detail=f"Memory with ID {memory_id} not found")

            # Delete old memory and related chunks
            related_chunks = [id for id in all_ids if id.startswith(f"{memory_id}_")]
            ids_to_delete = [memory_id] + related_chunks

            await asyncio.get_event_loop().run_in_executor(
                _executor,
                self._delete_from_vectorstore,
                ids_to_delete
            )

            logger.debug(f"Deleted original memory {memory_id} and {len(related_chunks)} related chunks")

            # Split text in thread
            chunks = await asyncio.get_event_loop().run_in_executor(
                _executor,
                self.text_splitter.split_text,
                text
            )

            logger.debug(f"Split updated text into {len(chunks)} chunks")
            metadatas = [metadata] * len(chunks) if metadata else None
            new_ids = [memory_id] + [f"{memory_id}_{i}" for i in range(1, len(chunks))]

            # Add new chunks in thread
            await asyncio.get_event_loop().run_in_executor(
                _executor,
                self._add_texts_with_ids,
                chunks,
                metadatas,
                new_ids
            )

            logger.info(f"Memory {memory_id} updated successfully with {len(chunks)} chunks")
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error updating memory: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error updating memory: {str(e)}")

    def _add_texts_with_ids(self, texts: List[str], metadatas: Optional[List[Dict]], ids: List[str]) -> None:
        """Helper method to add texts with specific IDs in a thread."""
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        self.vectorstore.persist()

    @cache(ttl=lambda: settings.CACHE_TTL_EMBEDDINGS, prefix="embedding")
    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding vector for a text.
        This is cached to avoid recomputing embeddings for the same text.

        Args:
            text: The text to embed

        Returns:
            The embedding vector as a list of floats
        """
        return self.embeddings.embed_query(text)

    async def get_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts asynchronously.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Split into batches
        batch_size = settings.EMBEDDING_BATCH_SIZE
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

        # Process batches in parallel
        results = await asyncio.gather(*[
            asyncio.get_event_loop().run_in_executor(
                _executor,
                self.embeddings.embed_documents,
                batch
            ) for batch in batches
        ])

        # Flatten results
        return [embedding for batch_result in results for embedding in batch_result]

    async def get_service_info(self) -> Dict[str, Any]:
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
            "async_enabled": True,
            "cache_enabled": settings.ENABLE_CACHE,
            "batch_size": settings.BATCH_SIZE
        }