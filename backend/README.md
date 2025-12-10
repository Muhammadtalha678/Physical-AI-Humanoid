# RAG Backend System

A Retrieval-Augmented Generation (RAG) backend system using FastAPI, Qdrant vector database, and OpenAI Agent SDK.

## Overview

This system provides two main API endpoints:
1. **Embeddings API**: Processes .md files from a specified directory and stores their embeddings in Qdrant vector database
2. **Query API**: Handles user queries with intelligent response generation based on document content

## Features

- Process all .md and .txt files in a directory and its subdirectories
- Store document embeddings in Qdrant vector database
- Intelligent query handling with greeting detection
- Inappropriate query filtering
- Semantic search capabilities

## API Endpoints

### `/api/embeddings` (POST)
Process documents and store embeddings.

Request body:
```json
{
  "folder_path": "/path/to/docs"
}
```

Response:
```json
{
  "status": "success",
  "files_processed": 42
}
```

### `/api/query` (POST)
Query the system for information.

Request body:
```json
{
  "query": "Your question here"
}
```

Response:
```json
{
  "final_output": "Response to the user's query"
}
```

### `/health` (GET)
Health check endpoint.

Response:
```json
{
  "status": "healthy",
  "service": "RAG Backend System"
}
```

## Setup

1. Install dependencies: `uv sync`
2. Create `.env` file with required configuration
3. Start the server: `uvicorn main:app --reload`

## Environment Variables

- `QDRANT_URL`: Qdrant instance URL
- `QDRANT_API_KEY`: Qdrant API key
- `QDRANT_COLLECTION_NAME`: Name of the collection to use
- `OPENROUTER_API`: OpenRouter API key
- `OPENROUTER_URL`: OpenRouter base URL
- `OPENROUTER_MODEL_NAME`: OpenRouter model name
- `GOOGLE_API_KEY`: Google API key for embeddings
- `EMBEDDING_MODEL`: Embedding model name