# Semantic Search System

A Python implementation of semantic search using Google's Gemini API for embeddings and Weaviate for vector database storage.

## Features

- **Gemini Embeddings**: Uses Google's Gemini API to generate high-quality embeddings for text content
- **Weaviate Storage**: Stores embeddings in Weaviate vector database for efficient similarity search
- **Semantic Search**: Performs semantic search to find the most relevant documents based on query similarity
- **Top-K Results**: Returns configurable number of top results (default: 10)
- **Metadata Support**: Supports storing and retrieving additional metadata with documents

## Installation

1. Clone the repository:
```bash
git clone https://github.com/adarshpandey515/semanticsearch.git
cd semanticsearch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file with the following variables:

```
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Weaviate Configuration
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your_weaviate_api_key_here_if_using_cloud
```

### Setting up Weaviate

#### Option 1: Local Weaviate (Docker)
```bash
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  semitechnologies/weaviate:latest
```

#### Option 2: Weaviate Cloud
Sign up for [Weaviate Cloud](https://console.weaviate.cloud/) and use your cluster URL and API key.

### Getting Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

## Usage

### Basic Usage

```python
from semantic_search import SemanticSearchSystem

# Initialize the system
search_system = SemanticSearchSystem()

# Add documents
uuid = search_system.add_document(
    content="Your document content here",
    title="Document Title",
    metadata={"category": "example", "tags": ["search", "ai"]}
)

# Perform semantic search
results = search_system.search("your search query", limit=10)

# Process results
for result in results:
    print(f"Title: {result['title']}")
    print(f"Content: {result['content']}")
    print(f"Similarity: {result['similarity_score']}")
    print(f"Metadata: {result['metadata']}")

# Close connection
search_system.close()
```

### Running the Example

The repository includes a complete example with sample documents:

```bash
python semantic_search.py
```

This will:
1. Initialize the semantic search system
2. Add sample documents about programming, AI, web development, etc.
3. Perform several test queries
4. Display the top 10 results for each query

## API Reference

### SemanticSearchSystem Class

#### Methods

- `__init__(collection_name="Documents")`: Initialize the system
- `add_document(content, title="", metadata=None)`: Add a document with embeddings
- `search(query, limit=10)`: Perform semantic search
- `generate_embedding(text)`: Generate embeddings for text
- `get_document_count()`: Get total number of documents
- `close()`: Close database connection

#### Return Format

Search results are returned as a list of dictionaries:

```python
{
    "uuid": "document-uuid",
    "content": "document content",
    "title": "document title",
    "metadata": {"key": "value"},
    "similarity_score": 0.95  # Similarity score (0-1)
}
```

## Requirements

- Python 3.7+
- Google Gemini API key
- Weaviate database (local or cloud)

## Dependencies

- `google-generativeai`: Google's Gemini API client
- `weaviate-client`: Weaviate Python client
- `python-dotenv`: Environment variable management

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
