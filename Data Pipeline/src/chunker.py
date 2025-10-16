"""
Text chunking module using recursive character text splitting
"""
import yaml
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """
    Chunk documents using recursive character text splitting.
    Configured for chunk_size=512 and chunk_overlap=0.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the document chunker.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.chunking_config = self.config['chunking']
        self.chunk_size = self.chunking_config['chunk_size']
        self.chunk_overlap = self.chunking_config['chunk_overlap']
        self.separators = self.chunking_config['separators']
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )
        
        print(f"Chunker initialized: size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of documents into smaller pieces.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            
        Returns:
            List of chunked documents with updated metadata
        """
        chunked_docs = []
        
        for doc_idx, document in enumerate(documents):
            content = document['content']
            metadata = document.get('metadata', {})
            
            # Split the text into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create a document for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                chunked_doc = {
                    'content': chunk,
                    'metadata': {
                        **metadata,
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'doc_index': doc_idx,
                        'chunk_size': len(chunk)
                    }
                }
                chunked_docs.append(chunked_doc)
            
            print(f"Document {doc_idx + 1}: {len(chunks)} chunks created")
        
        print(f"\nTotal chunks created: {len(chunked_docs)}")
        return chunked_docs
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk a single text string.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        return self.text_splitter.split_text(text)
    
    def get_chunk_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about chunking for a list of documents.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            Dictionary with chunking statistics
        """
        total_chars = sum(len(doc['content']) for doc in documents)
        estimated_chunks = total_chars // self.chunk_size
        
        return {
            'total_documents': len(documents),
            'total_characters': total_chars,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'estimated_chunks': estimated_chunks
        }
    
    def __repr__(self) -> str:
        return f"DocumentChunker(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
