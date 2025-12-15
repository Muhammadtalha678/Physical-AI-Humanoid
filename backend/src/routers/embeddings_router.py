from fastapi import APIRouter, Request
from src.controllers.embeddings_controller import process_documents
import os
router = APIRouter()


@router.post("/embeddings")
async def create_embeddings(data: dict, request: Request):
    print("Current Working Directory (CWD):", os.getcwd())
    print("Files in CWD:", os.listdir())
    folder_path = data["folder_path"]
    print(folder_path)
    qdrant_client = request.app.state.qdrant_client
    return await process_documents(folder_path, qdrant_client)