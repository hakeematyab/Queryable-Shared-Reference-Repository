"""
Test script for the document processing pipeline
Demonstrates the complete workflow: Load → Chunk → Embed → Store → Query
"""
from src.graph import DocumentProcessingGraph


def main():
    """Run the document processing pipeline and test with sample queries."""
    
    print("\n" + "=" * 70)
    print(" DOCUMENT INDEXING PIPELINE - TEST RUN")
    print("=" * 70)
    
    graph = DocumentProcessingGraph(config_path="config.yaml")

    print("\n🚀 Starting pipeline execution...")
    final_state = graph.run(reset_collection=True)
    
    # Check for errors
    if final_state.get("error"):
        print(f"\n❌ Pipeline failed: {final_state['error']}")
        return
    
    # Test queries
    print("\n" + "=" * 70)
    print(" TESTING QUERY FUNCTIONALITY")
    print("=" * 70)
    
    # Example queries 
    test_queries = [
        "machine learning algorithms",
        "neural networks",
        "data processing techniques"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i} ---")
        results = graph.query(query, n_results=3)
        
        print(f"\nQuery: '{query}'")
        print(f"Results found: {len(results['documents'])}\n")
        
        # Display results
        for j, (doc, metadata, distance) in enumerate(zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        ), 1):
            print(f"Result {j}:")
            print(f"  Source: {metadata.get('filename', 'Unknown')}")
            print(f"  Chunk: {metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}")
            print(f"  Distance: {distance:.4f}")
            print(f"  Content preview: {doc[:150]}...")
            print()
    
    print("=" * 70)
    print(" PIPELINE STATISTICS")
    print("=" * 70)
    
    stats = final_state.get("stats", {})
    print(f"✓ Documents loaded:      {stats.get('documents_loaded', 0)}")
    print(f"✓ Chunks created:        {stats.get('chunks_created', 0)}")
    print(f"✓ Embeddings generated:  {stats.get('embeddings_generated', 0)}")
    print(f"✓ Documents stored:      {stats.get('documents_stored', 0)}")
    
    print("\n" + "=" * 70)
    print(" TEST COMPLETE ✓")
    print("=" * 70)
    print("\nThe pipeline is working correctly!")
    print("You can now use this system to index and search your documents.")
    print("\nTo run again: python run.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
