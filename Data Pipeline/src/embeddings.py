"""
Embedding module for SciNCL (Scientific Document Embeddings)
"""
import yaml
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np


class SciNCLEmbeddings:
    """
    Wrapper for SciNCL embedding model using sentence-transformers.
    SciNCL is specifically designed for scientific document embeddings.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the SciNCL embedding model.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.embedding_config = self.config['embedding']
        self.model_name = self.embedding_config['model_name']
        self.device = self.embedding_config['device']
        self.batch_size = self.embedding_config['batch_size']
        
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print(f"Model loaded successfully on {self.device}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True
        )
        
        return embedding.tolist()
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
    
    def __repr__(self) -> str:
        return f"SciNCLEmbeddings(model={self.model_name}, device={self.device})"
