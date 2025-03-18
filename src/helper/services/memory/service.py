"""
Memory service based on mem0ai/mem0.
Provides functionality for storing and retrieving memories using embeddings.
"""
import os
from typing import List, Optional, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class MemoryService:
    def __init__(self, persist_directory: str = "./memory_store"):
        """Initialize the memory service with a vector store."""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

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
        """
        self.vectorstore.delete([memory_id])
        self.vectorstore.persist()

    def clear_all_memories(self) -> None:
        """Delete all memories in the vector store."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        self.vectorstore.persist()

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """
        Retrieve all stored memories.

        Returns:
            List of all documents in the vector store
        """
        # This is a simplified implementation that might not be efficient for large collections
        # For a real implementation, you would need pagination
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

    def update_memory(self, memory_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update an existing memory.

        Args:
            memory_id: The ID of the memory to update
            text: The new text content
            metadata: The new metadata (optional)
        """
        # First delete the old memory
        self.delete_memory(memory_id)

        # Then add the new version
        chunks = self.text_splitter.split_text(text)
        metadatas = [metadata] * len(chunks) if metadata else None

        self.vectorstore.add_texts(
            texts=chunks,
            metadatas=metadatas,
            ids=[memory_id] + [f"{memory_id}_{i}" for i in range(1, len(chunks))]
        )

        self.vectorstore.persist()