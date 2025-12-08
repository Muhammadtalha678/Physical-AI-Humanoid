# Backend Project Structure

```
backend/
 ├── main.py
 ├── .env
 └── src/
      ├── routers/
      │     ├── embeddings_router.py
      │     └── query_router.py
      ├── controllers/
      │     ├── embeddings_controller.py
      │     └── agent_handler.py
      └── lib/
            ├── configs.py
            ├── vectordb_connection.py
            └── openaiagentsdk_config.py
```

## Description
- **main.py** → FastAPI app entrypoint with lifespan for DB & AgentSDK
- **.env** → Environment variables for Qdrant, API keys, model names, book path, etc.
- **src/routers/** → API routes for embeddings and query
- **src/controllers/** → Controller functions to handle embedding storage & agent query
- **src/lib/** → Configs, vector DB connection, OpenAIAgentSDK configuration

