"""
Memory service based on langchain and Chroma.
Provides functionality for storing and retrieving memories using embeddings.
"""
import os
from typing import List, Optional, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from fastapi import HTTPException

# Singleton for embedding model to avoid reloading
_embedding_model = None

def get_embedding_model():
    """Get or initialize the embedding model (singleton pattern)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embedding_model

class MemoryService:
    def __init__(self, persist_directory: str = "./memory_store"):
        """Initialize the memory service with a vector store."""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # Get embedding model using singleton pattern
        self.embeddings = get_embedding_model()

        # Initialize vector store
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add new memory text to the vector store.

        Args:
            text: The text content to store
            metadata: Optional metadata to store with the text

        Returns:
            List of IDs for the stored chunks
        """
        chunks = self.text_splitter.split_text(text)

        # Prepare metadata for each chunk if provided
        metadatas = [metadata] * len(chunks) if metadata else None

        # Add documents to the vector store
        ids = self.vectorstore.add_texts(
            texts=chunks,
            metadatas=metadatas
        )

        # Persist the vector store
        self.vectorstore.persist()

        return ids

    def search_memories(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant memories based on a query.

        Args:
            query: The search query
            k: Number of results to return

        Returns:
            List of documents with their content and metadata
        """
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        # Format results
        memories = []
        for doc, score in results:
            memories.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance": float(score)
            })

        return memories

    def delete_memory(self, memory_id: str) -> None:
        """
        Delete a specific memory by ID.

        Args:
            memory_id: The ID of the memory to delete

        Raises:
            HTTPException: If the memory ID doesn't exist
        """
        try:
            # Check if memory exists first
            all_ids = self.vectorstore.get()["ids"]
            if memory_id not in all_ids:
                raise HTTPException(status_code=404, detail=f"Memory with ID {memory_id} not found")

            self.vectorstore.delete([memory_id])
            self.vectorstore.persist()
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Error deleting memory: {str(e)}")

    def clear_all_memories(self) -> None:
        """Delete all memories in the vector store."""
        try:
            # Get all IDs
            all_ids = self.vectorstore.get()["ids"]
            if all_ids:
                # Delete all documents
                self.vectorstore.delete(all_ids)
            self.vectorstore.persist()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error clearing memories: {str(e)}")

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """
        Retrieve all stored memories.

        Returns:
            List of all documents in the vector store
        """
        # Handle the case when the collection is empty
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

            return memories
        except Exception as e:
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
        try:
            # Check if memory exists first
            all_ids = self.vectorstore.get()["ids"]
            if memory_id not in all_ids:
                raise HTTPException(status_code=404, detail=f"Memory with ID {memory_id} not found")

            # First delete the old memory
            self.vectorstore.delete([memory_id])

            # Find and delete any related chunk IDs
            related_chunks = [id for id in all_ids if id.startswith(f"{memory_id}_")]
            if related_chunks:
                self.vectorstore.delete(related_chunks)

            # Then add the new version
            chunks = self.text_splitter.split_text(text)
            metadatas = [metadata] * len(chunks) if metadata else None

            self.vectorstore.add_texts(
                texts=chunks,
                metadatas=metadatas,
                ids=[memory_id] + [f"{memory_id}_{i}" for i in range(1, len(chunks))]
            )

            self.vectorstore.persist()
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Error updating memory: {str(e)}")