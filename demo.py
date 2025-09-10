"""
Example usage of the Semantic Search System
This script demonstrates how to use the semantic search system with sample data.
"""

def demo_usage():
    """Demonstrate the semantic search system usage."""
    print("Semantic Search System Demo")
    print("=" * 40)
    
    # Example code that would run with proper API keys
    code_example = '''
from semantic_search import SemanticSearchSystem

# Initialize the system
search_system = SemanticSearchSystem()

# Add a document
uuid = search_system.add_document(
    content="Python is a versatile programming language used for web development, data science, and AI.",
    title="Python Programming",
    metadata={"category": "programming", "language": "python"}
)

# Search for documents
results = search_system.search("programming languages", limit=5)

# Print results
for i, result in enumerate(results, 1):
    print(f"{i}. {result['title']}")
    print(f"   Score: {result['similarity_score']:.4f}")
    print(f"   Content: {result['content'][:100]}...")
    print()

# Close connection
search_system.close()
'''
    
    print("Example Usage:")
    print(code_example)
    
    print("\nSetup Requirements:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set up .env file with your API keys:")
    print("   - GEMINI_API_KEY=your_gemini_api_key")
    print("   - WEAVIATE_URL=http://localhost:8080 (or your Weaviate cloud URL)")
    print("3. Start Weaviate (local Docker or use cloud)")
    print("4. Run: python semantic_search.py")
    
    print("\nKey Features:")
    print("✓ Gemini embeddings for high-quality semantic understanding")
    print("✓ Weaviate vector database for efficient similarity search")
    print("✓ Configurable top-K results (default: 10)")
    print("✓ Metadata support for enhanced document management")
    print("✓ Similarity scoring for ranking results")

if __name__ == "__main__":
    demo_usage()