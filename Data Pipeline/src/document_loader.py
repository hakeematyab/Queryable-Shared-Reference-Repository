"""
Document loader module for loading text files from a directory
"""
import os
import yaml
from typing import List, Dict, Any
from pathlib import Path


class DocumentLoader:
    """
    Load documents from a specified directory.
    Supports multiple file formats including .txt, .pdf, .md
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the document loader.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.loader_config = self.config['document_loader']
        self.source_directory = self.loader_config['source_directory']
        self.file_extensions = self.loader_config['file_extensions']
        self.encoding = self.loader_config['encoding']
        
        print(f"Document loader initialized for: {self.source_directory}")
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Load all documents from the source directory.
        
        Returns:
            List of document dictionaries with 'content', 'metadata', and 'source'
        """
        documents = []
        source_path = Path(self.source_directory)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_directory}")
        
        # Find all files with specified extensions
        for ext in self.file_extensions:
            files = list(source_path.glob(f"*{ext}"))
            
            for file_path in files:
                try:
                    content = self._load_file(file_path)
                    
                    document = {
                        'content': content,
                        'metadata': {
                            'source': str(file_path),
                            'filename': file_path.name,
                            'extension': file_path.suffix,
                            'size': file_path.stat().st_size
                        },
                        'source': str(file_path)
                    }
                    
                    documents.append(document)
                    print(f"Loaded: {file_path.name} ({len(content)} chars)")
                    
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
                    continue
        
        print(f"\nTotal documents loaded: {len(documents)}")
        return documents
    
    def _load_file(self, file_path: Path) -> str:
        """
        Load content from a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        with open(file_path, 'r', encoding=self.encoding, errors='ignore') as f:
            content = f.read()
        
        return content
    
    def get_document_count(self) -> int:
        """
        Get the count of documents in the source directory.
        
        Returns:
            Number of documents
        """
        count = 0
        source_path = Path(self.source_directory)
        
        for ext in self.file_extensions:
            count += len(list(source_path.glob(f"*{ext}")))
        
        return count
    
    def __repr__(self) -> str:
        return f"DocumentLoader(source={self.source_directory}, extensions={self.file_extensions})"
