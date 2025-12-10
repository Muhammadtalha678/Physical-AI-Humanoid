from fastapi import APIRouter, Request
from src.controllers.agent_handler import handle_query_operation

router = APIRouter()


@router.post("/query")
async def query_rag(data: dict, request: Request):
    agent_config = request.app.state.agent_config
    qdrant_client = request.app.state.qdrant_client
    return await handle_query_operation(agent_config, data["query"], qdrant_client)
    # return await run_query(query=data["query"],qdant_client= qdrant_client)