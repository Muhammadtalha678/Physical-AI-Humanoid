# Quickstart Guide: RAG Backend System

## Prerequisites
- Python 3.11+
- uv package manager
- Qdrant vector database instance
- OpenRouter API key
- Google API key (if using Google embeddings)

## Setup

1. **Initialize the project:**
   ```bash
   uv init backend
   ```

2. **Create virtual environment:**
   ```bash
   uv venv
   ```

3. **Activate virtual environment:**
   ```bash
   .venv\Scripts\activate  # On Windows
   # or
   source .venv/bin/activate  # On Linux/Mac
   ```

4. **Install dependencies:**
   ```bash
   uv add fastapi uvicorn qdrant-client python-dotenv pydantic openai-agents
   ```

5. **Create the project structure:**
   ```bash
   mkdir -p backend/src/{routers,controllers,lib}
   ```

## Configuration

1. **Create .env file in backend/ directory:**
   ```
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_COLLECTION_NAME=your_collection_name
   OPENROUTER_API=your_openrouter_base_url
   OPENROUTER_URL=your_openrouter_url
   OPENROUTER_MODEL_NAME=your_model_name
   GOOGLE_API_KEY=your_google_api_key
   EMBEDDING_MODEL=your_embedding_model
   API_KEY=your_api_key  # This seems to be used for both Qdrant and OpenRouter in the templates
   ```

## Running the Application

1. **Start the server:**
   ```bash
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## Usage

1. **Process documents:**
   ```bash
   curl -X POST http://localhost:8000/embeddings \
     -H "Content-Type: application/json" \
     -d '{"folder_path": "/path/to/your/docs"}'
   ```

2. **Query the system:**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Your question here"}'
   ```

## Expected Response Times
- Embedding processing: Depends on number of files, typically under 5 minutes for 100 files
- Query responses: Should respond within 10 seconds for 95% of requests