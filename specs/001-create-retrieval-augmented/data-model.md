# Data Model: RAG Backend System

## Entity: Document Embedding
- **Fields**:
  - id: string (unique identifier for the embedding)
  - content: string (the text content of the .md file or chunk)
  - metadata: dict (file path, filename, creation date, etc.)
  - vector: list[float] (the embedding vector representation)
- **Relationships**: Stored in Qdrant vector database collection
- **Validation**: Content must not be empty, vector must match embedding model dimensions

## Entity: Query Request
- **Fields**:
  - query: string (the user's question or input)
  - user_id: string (optional, for tracking purposes)
  - timestamp: datetime (when the query was submitted)
- **Relationships**: No direct relationships, processed independently
- **Validation**: Query must not be empty, length should be within reasonable limits

## Entity: Query Response
- **Fields**:
  - response: string (the system's answer to the query)
  - sources: list[DocumentEmbedding] (references to source documents used)
  - query_type: enum (greeting, book_related, no_match, inappropriate)
  - confidence: float (confidence score for the response)
- **Relationships**: Related to original Query Request
- **Validation**: Response must not be empty, query_type must be valid enum value

## Entity: Configuration Parameters
- **Fields**:
  - qdrant_api: string (API key for Qdrant)
  - qdrant_url: string (URL for Qdrant instance)
  - qdrant_collection_name: string (name of the collection to use)
  - embedding_model: string (name of the embedding model to use)
  - google_api_key: string (Google API key if needed)
  - openrouter_api: string (OpenRouter API key)
  - openrouter_baseurl: string (OpenRouter base URL)
  - openrouter_model_name: string (OpenRouter model name)
- **Relationships**: Global configuration, no relationships
- **Validation**: All API keys and URLs must be properly formatted

## Entity: Directory Path
- **Fields**:
  - path: string (the directory path containing .md files)
  - recursive: boolean (whether to process subdirectories)
- **Relationships**: Used to generate Document Embeddings
- **Validation**: Path must exist and be readable, must contain .md files