# API Contracts: RAG Backend System

## Embedding API Contract

### POST /embeddings
Process all .md files in a specified directory and store their embeddings in Qdrant vector database.

**Request Body:**
```json
{
  "folder_path": "/path/to/docs"
}
```

**Success Response (200):**
```json
{
  "status": "success",
  "files_processed": 42
}
```

**Error Response (500):**
```json
{
  "detail": "Error: description of the error"
}
```

**Validation:**
- folder_path must be a valid directory path containing .md files
- Only .md and .txt files will be processed

## Query API Contract

### POST /query
Process user query and return relevant response based on document embeddings.

**Request Body:**
```json
{
  "query": "What is the main topic of the book?"
}
```

**Success Response (200):**
```json
{
  "final_output": "Response to the user's query based on document content"
}
```

**Error Response (500):**
```json
{
  "detail": "Error message"
}
```

**Validation:**
- query must be a non-empty string
- Query will be processed through OpenAI Agent SDK with greeting detection