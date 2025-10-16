"""
LangGraph orchestration for the document processing pipeline
"""
import yaml
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from src.document_loader import DocumentLoader
from src.chunker import DocumentChunker
from src.embeddings import SciNCLEmbeddings
from src.vector_store import ChromaVectorStore


# Define the state schema
class PipelineState(TypedDict):
    """State for the document processing pipeline."""
    config_path: str
    reset_collection: bool
    documents: List[Dict[str, Any]]
    chunked_documents: List[Dict[str, Any]]
    embeddings: List[List[float]]
    texts: List[str]
    metadatas: List[Dict[str, Any]]
    stats: Dict[str, Any]
    error: str


class DocumentProcessingGraph:
    """
    LangGraph-based document processing workflow.
    Orchestrates: Load → Chunk → Embed → Store
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the LangGraph workflow.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.loader = DocumentLoader(config_path)
        self.chunker = DocumentChunker(config_path)
        self.embeddings = SciNCLEmbeddings(config_path)
        self.vector_store = ChromaVectorStore(config_path)
        
        # Build the graph
        self.graph = self._build_graph()
        
        print("LangGraph workflow initialized!")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node("reset_store", self._reset_store)
        workflow.add_node("load_documents", self._load_documents)
        workflow.add_node("chunk_documents", self._chunk_documents)
        workflow.add_node("generate_embeddings", self._generate_embeddings)
        workflow.add_node("store_documents", self._store_documents)
        
        # Define edges
        workflow.set_entry_point("reset_store")
        workflow.add_edge("reset_store", "load_documents")
        workflow.add_edge("load_documents", "chunk_documents")
        workflow.add_edge("chunk_documents", "generate_embeddings")
        workflow.add_edge("generate_embeddings", "store_documents")
        workflow.add_edge("store_documents", END)
        
        return workflow.compile()
    
    def _reset_store(self, state: PipelineState) -> PipelineState:
        """Reset the vector store if requested."""
        print("\n[Node: Reset Store]")
        
        if state.get("reset_collection", False):
            print("Resetting vector store...")
            self.vector_store.reset_collection()
        else:
            print("Skipping reset (reset_collection=False)")
        
        return state
    
    def _load_documents(self, state: PipelineState) -> PipelineState:
        """Load documents from the source directory."""
        print("\n[Node: Load Documents]")
        
        try:
            documents = self.loader.load_documents()
            state["documents"] = documents
            
            if not state.get("stats"):
                state["stats"] = {}
            state["stats"]["documents_loaded"] = len(documents)
            
        except Exception as e:
            state["error"] = f"Error loading documents: {e}"
            print(f"ERROR: {state['error']}")
        
        return state
    
    def _chunk_documents(self, state: PipelineState) -> PipelineState:
        """Chunk documents into smaller pieces."""
        print("\n[Node: Chunk Documents]")
        
        try:
            documents = state.get("documents", [])
            
            if not documents:
                state["error"] = "No documents to chunk"
                return state
            
            chunked_docs = self.chunker.chunk_documents(documents)
            state["chunked_documents"] = chunked_docs
            state["stats"]["chunks_created"] = len(chunked_docs)
            
        except Exception as e:
            state["error"] = f"Error chunking documents: {e}"
            print(f"ERROR: {state['error']}")
        
        return state
    
    def _generate_embeddings(self, state: PipelineState) -> PipelineState:
        """Generate embeddings for document chunks."""
        print("\n[Node: Generate Embeddings]")
        
        try:
            chunked_docs = state.get("chunked_documents", [])
            
            if not chunked_docs:
                state["error"] = "No chunks to embed"
                return state
            
            texts = [doc['content'] for doc in chunked_docs]
            metadatas = [doc['metadata'] for doc in chunked_docs]
            
            embeddings = self.embeddings.embed_documents(texts)
            
            state["texts"] = texts
            state["metadatas"] = metadatas
            state["embeddings"] = embeddings
            state["stats"]["embeddings_generated"] = len(embeddings)
            
        except Exception as e:
            state["error"] = f"Error generating embeddings: {e}"
            print(f"ERROR: {state['error']}")
        
        return state
    
    def _store_documents(self, state: PipelineState) -> PipelineState:
        """Store documents in the vector database."""
        print("\n[Node: Store Documents]")
        
        try:
            texts = state.get("texts", [])
            embeddings = state.get("embeddings", [])
            metadatas = state.get("metadatas", [])
            
            if not texts or not embeddings:
                state["error"] = "No documents to store"
                return state
            
            self.vector_store.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            state["stats"]["documents_stored"] = len(texts)
            
        except Exception as e:
            state["error"] = f"Error storing documents: {e}"
            print(f"ERROR: {state['error']}")
        
        return state
    
    def run(self, reset_collection: bool = False) -> Dict[str, Any]:
        """
        Execute the complete pipeline workflow.
        
        Args:
            reset_collection: Whether to reset the vector store
            
        Returns:
            Final state with statistics
        """
        print("=" * 60)
        print("Starting LangGraph Document Processing Workflow")
        print("=" * 60)
        
        # Initialize state
        initial_state = {
            "config_path": self.config_path,
            "reset_collection": reset_collection,
            "documents": [],
            "chunked_documents": [],
            "embeddings": [],
            "texts": [],
            "metadatas": [],
            "stats": {},
            "error": ""
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Print summary
        print("\n" + "=" * 60)
        print("WORKFLOW EXECUTION COMPLETE")
        print("=" * 60)
        
        if final_state.get("error"):
            print(f"ERROR: {final_state['error']}")
        else:
            stats = final_state.get("stats", {})
            print(f"Documents loaded:      {stats.get('documents_loaded', 0)}")
            print(f"Chunks created:        {stats.get('chunks_created', 0)}")
            print(f"Embeddings generated:  {stats.get('embeddings_generated', 0)}")
            print(f"Documents stored:      {stats.get('documents_stored', 0)}")
        
        print("=" * 60)
        
        return final_state
    
    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the vector store.
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            Query results
        """
        print(f"\nQuerying: '{query_text}'")
        
        query_embedding = self.embeddings.embed_query(query_text)
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=n_results
        )
        
        print(f"Found {len(results['documents'])} results")
        
        return results
