from fastapi import APIRouter, Request
from src.controllers.embeddings_controller import process_documents

router = APIRouter()


@router.post("/embeddings")
async def create_embeddings(data: dict, request: Request):
    folder_path = data["folder_path"]
    print(folder_path)
    qdrant_client = request.app.state.qdrant_client
    return await process_documents(folder_path, qdrant_client)