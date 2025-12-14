from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers.embeddings_router import router as embeddings_router
from src.routers.query_router import router as query_router
from contextlib import asynccontextmanager
from src.lib.configs import settings
from src.lib.openaiagentsdk_config import AgentConfig
from src.lib.vectordb_connection import get_qdrant

import httpx
import asyncio
async def invoke_render(client:httpx.AsyncClient):
    while True:
        print("request send")
        await client.get("https://rag-backend-22uh.onrender.com") 
        await asyncio.sleep(300)  

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

    client = httpx.AsyncClient()
    task = asyncio.create_task(invoke_render(client))
    yield
    task.cancel()
    await client.aclose()
    print("closed!")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(embeddings_router, prefix="/api")
app.include_router(query_router, prefix="/api")



@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "RAG Backend System"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
