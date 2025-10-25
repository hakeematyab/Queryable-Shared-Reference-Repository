"""
ChromaDB Vector Store module for document storage and retrieval
"""
import yaml
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import os


class ChromaVectorStore:
    """
    ChromaDB vector store for storing and retrieving document embeddings.
    Replaces FAISS with persistent ChromaDB storage.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ChromaDB vector store.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.vector_config = self.config['vector_store']
        self.collection_name = self.vector_config['collection_name']
        self.persist_directory = self.vector_config['persist_directory']
        self.distance_metric = self.vector_config['distance_metric']
        

        os.makedirs(self.persist_directory, exist_ok=True)
        

        print(f"Initializing ChromaDB at: {self.persist_directory}")
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        self._initialize_collection()
        print(f"ChromaDB collection '{self.collection_name}' ready")
    
    def _initialize_collection(self):
        """Initialize or get existing collection."""
        # Map distance metric to ChromaDB format
        metric_mapping = {
            "cosine": "cosine",
            "l2": "l2",
            "ip": "ip"
        }
        
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
            print(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": metric_mapping[self.distance_metric]}
            )
            print(f"Created new collection: {self.collection_name}")
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents with their embeddings to the vector store.
        
        Args:
            texts: List of document texts
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs
        """
        if not texts or not embeddings:
            print("No documents to add")
            return
        
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{"text": text} for text in texts]
        else:
            for i, meta in enumerate(metadatas):
                if "text" not in meta:
                    meta["text"] = texts[i]
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(texts)} documents to ChromaDB")
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Dictionary containing ids, documents, metadatas, and distances
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "count": count,
            "persist_directory": self.persist_directory
        }
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(name=self.collection_name)
        print(f"Deleted collection: {self.collection_name}")
    
    def reset_collection(self) -> None:
        """Reset the collection by deleting and recreating it."""
        try:
            self.delete_collection()
        except:
            pass
        self._initialize_collection()
        print(f"Reset collection: {self.collection_name}")
    
    def __repr__(self) -> str:
        stats = self.get_collection_stats()
        return f"ChromaVectorStore(collection={stats['name']}, documents={stats['count']})"
