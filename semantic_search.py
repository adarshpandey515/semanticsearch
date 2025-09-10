"""
Semantic Search System using Gemini for embeddings and Weaviate for storage
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
import weaviate
import weaviate.classes as wvc

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticSearchSystem:
    """
    A semantic search system that uses Gemini for embeddings and Weaviate for storage.
    """
    
    def __init__(self, collection_name: str = "Documents"):
        """
        Initialize the semantic search system.
        
        Args:
            collection_name: Name of the Weaviate collection to store documents
        """
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._setup_gemini()
        self._setup_weaviate()
    
    def _setup_gemini(self):
        """Setup Gemini API for embeddings."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        logger.info("Gemini API configured successfully")
    
    def _setup_weaviate(self):
        """Setup Weaviate client and create collection if it doesn't exist."""
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        
        try:
            # Connect to Weaviate
            if weaviate_api_key:
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=weaviate_url,
                    auth_credentials=wvc.init.Auth.api_key(weaviate_api_key)
                )
            else:
                self.client = weaviate.connect_to_local(host=weaviate_url.replace("http://", "").replace("https://", ""))
            
            # Create collection if it doesn't exist
            if not self.client.collections.exists(self.collection_name):
                self.collection = self.client.collections.create(
                    name=self.collection_name,
                    vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                    properties=[
                        wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="metadata", data_type=wvc.config.DataType.OBJECT),
                    ]
                )
            else:
                self.collection = self.client.collections.get(self.collection_name)
            
            logger.info(f"Connected to Weaviate and initialized collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings using Gemini.
        
        Args:
            text: Text to generate embeddings for
            
        Returns:
            List of embedding values
        """
        try:
            # Use Gemini's text embedding model
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def add_document(self, content: str, title: str = "", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document to the Weaviate database with embeddings.
        
        Args:
            content: Document content
            title: Document title (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Document UUID
        """
        try:
            # Generate embedding for the content
            embedding = self.generate_embedding(content)
            
            # Prepare document data
            document_data = {
                "content": content,
                "title": title,
                "metadata": metadata or {}
            }
            
            # Insert document with vector
            uuid = self.collection.data.insert(
                properties=document_data,
                vector=embedding
            )
            
            logger.info(f"Added document with UUID: {uuid}")
            return str(uuid)
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search and return top results.
        
        Args:
            query: Search query
            limit: Number of results to return (default: 10)
            
        Returns:
            List of search results with content, title, metadata, and similarity score
        """
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)
            
            # Perform vector search
            response = self.collection.query.near_vector(
                near_vector=query_embedding,
                limit=limit,
                return_metadata=wvc.query.MetadataQuery(distance=True)
            )
            
            # Format results
            results = []
            for obj in response.objects:
                result = {
                    "uuid": str(obj.uuid),
                    "content": obj.properties.get("content", ""),
                    "title": obj.properties.get("title", ""),
                    "metadata": obj.properties.get("metadata", {}),
                    "similarity_score": 1 - obj.metadata.distance  # Convert distance to similarity
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} results for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform search: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the collection."""
        try:
            response = self.collection.aggregate.over_all(total_count=True)
            return response.total_count
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def close(self):
        """Close the Weaviate client connection."""
        if self.client:
            self.client.close()
            logger.info("Weaviate client connection closed")


def main():
    """
    Example usage of the Semantic Search System.
    """
    try:
        # Initialize the semantic search system
        search_system = SemanticSearchSystem()
        
        # Sample documents to add
        sample_documents = [
            {
                "content": "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, and artificial intelligence.",
                "title": "Introduction to Python Programming",
                "metadata": {"category": "programming", "difficulty": "beginner"}
            },
            {
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
                "title": "Machine Learning Basics",
                "metadata": {"category": "AI", "difficulty": "intermediate"}
            },
            {
                "content": "Web development involves creating websites and web applications. It includes frontend development with HTML, CSS, and JavaScript, and backend development with various programming languages.",
                "title": "Web Development Overview",
                "metadata": {"category": "web", "difficulty": "beginner"}
            },
            {
                "content": "Data science combines statistics, programming, and domain expertise to extract insights from data. It involves data collection, cleaning, analysis, and visualization.",
                "title": "Data Science Fundamentals",
                "metadata": {"category": "data", "difficulty": "intermediate"}
            },
            {
                "content": "Cloud computing provides on-demand access to computing resources over the internet. It offers scalability, cost-effectiveness, and flexibility for businesses and developers.",
                "title": "Cloud Computing Introduction",
                "metadata": {"category": "cloud", "difficulty": "beginner"}
            }
        ]
        
        # Check if documents already exist
        doc_count = search_system.get_document_count()
        if doc_count == 0:
            print("Adding sample documents...")
            for doc in sample_documents:
                uuid = search_system.add_document(
                    content=doc["content"],
                    title=doc["title"],
                    metadata=doc["metadata"]
                )
                print(f"Added document: {doc['title']} (UUID: {uuid})")
        else:
            print(f"Found {doc_count} existing documents in the database")
        
        # Perform semantic searches
        test_queries = [
            "programming languages for beginners",
            "artificial intelligence and machine learning",
            "building websites and web apps",
            "analyzing data and statistics"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            results = search_system.search(query, limit=10)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']}")
                print(f"   Similarity Score: {result['similarity_score']:.4f}")
                print(f"   Content: {result['content'][:100]}...")
                print(f"   Category: {result['metadata'].get('category', 'N/A')}")
        
        # Close the connection
        search_system.close()
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()