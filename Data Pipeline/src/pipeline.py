"""
Document processing pipeline integrating all components
"""
import yaml
from typing import List, Dict, Any, Optional
from src.document_loader import DocumentLoader
from src.chunker import DocumentChunker
from src.embeddings import SciNCLEmbeddings
from src.vector_store import ChromaVectorStore


class DocumentPipeline:
    """
    End-to-end pipeline: Load → Chunk → Embed → Store
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the document processing pipeline.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.pipeline_config = self.config['pipeline']
        
        # Initialize all components
        print("=" * 60)
        print("Initializing Document Processing Pipeline")
        print("=" * 60)
        
        self.loader = DocumentLoader(config_path)
        self.chunker = DocumentChunker(config_path)
        self.embeddings = SciNCLEmbeddings(config_path)
        self.vector_store = ChromaVectorStore(config_path)
        
        print("=" * 60)
        print("Pipeline initialization complete!")
        print("=" * 60)
    
    def process_documents(self, reset_collection: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline: load, chunk, embed, and store documents.
        
        Args:
            reset_collection: Whether to reset the vector store before processing
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'documents_loaded': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'documents_stored': 0
        }
        
        try:
            # Step 1: Reset collection if requested
            if reset_collection:
                print("\n[Step 0] Resetting vector store...")
                self.vector_store.reset_collection()
            
            # Step 1: Load documents
            print("\n[Step 1] Loading documents...")
            documents = self.loader.load_documents()
            stats['documents_loaded'] = len(documents)
            
            if not documents:
                print("No documents found to process!")
                return stats
            
            # Step 2: Chunk documents
            print("\n[Step 2] Chunking documents...")
            chunked_docs = self.chunker.chunk_documents(documents)
            stats['chunks_created'] = len(chunked_docs)
            
            # Step 3: Generate embeddings
            print("\n[Step 3] Generating embeddings...")
            texts = [doc['content'] for doc in chunked_docs]
            embeddings = self.embeddings.embed_documents(texts)
            stats['embeddings_generated'] = len(embeddings)
            
            # Step 4: Store in vector database
            print("\n[Step 4] Storing in ChromaDB...")
            metadatas = [doc['metadata'] for doc in chunked_docs]
            
            self.vector_store.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            stats['documents_stored'] = len(texts)
            
            # Print summary
            print("\n" + "=" * 60)
            print("PIPELINE EXECUTION COMPLETE")
            print("=" * 60)
            print(f"Documents loaded:      {stats['documents_loaded']}")
            print(f"Chunks created:        {stats['chunks_created']}")
            print(f"Embeddings generated:  {stats['embeddings_generated']}")
            print(f"Documents stored:      {stats['documents_stored']}")
            print("=" * 60)
            
            return stats
            
        except Exception as e:
            print(f"\nError in pipeline execution: {e}")
            if self.pipeline_config['error_handling'] == 'raise':
                raise
            return stats
    
    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            Query results with documents and metadata
        """
        print(f"\nQuerying: '{query_text}'")
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query_text)
        
        # Search vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=n_results
        )
        
        print(f"Found {len(results['documents'])} results")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary with pipeline statistics
        """
        collection_stats = self.vector_store.get_collection_stats()
        
        return {
            'vector_store': collection_stats,
            'embedding_model': str(self.embeddings),
            'chunker': str(self.chunker),
            'loader': str(self.loader)
        }
    
    def __repr__(self) -> str:
        return "DocumentPipeline(Load → Chunk → Embed → Store)"
