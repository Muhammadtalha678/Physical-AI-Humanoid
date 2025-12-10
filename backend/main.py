from fastapi import FastAPI
from src.routers.embeddings_router import router as embeddings_router
from src.routers.query_router import router as query_router
from contextlib import asynccontextmanager
from src.lib.configs import settings
from src.lib.openaiagentsdk_config import AgentConfig
from src.lib.vectordb_connection import get_qdrant


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("started!")
    # Initialize and store agent configuration in app state
    config_agent = AgentConfig(
        base_url=settings.openrouter_url,  # API base URL (e.g., https://openrouter.ai/api/v1)
        api_key=settings.openrouter_api,   # API key
        model_name=settings.openrouter_model_name
    )
    app.state.agent_config = config_agent

    # Initialize and store Qdrant client in app state to avoid repeated connections
    qdrant_client = get_qdrant()
    app.state.qdrant_client = qdrant_client
    print("Qdrant client initialized and stored in app state")

    yield

    print("closed!")


app = FastAPI(lifespan=lifespan)
app.include_router(embeddings_router, prefix="/api")
app.include_router(query_router, prefix="/api")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RAG Backend System"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
