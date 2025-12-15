from fastapi import APIRouter, Request
from src.controllers.embeddings_controller import process_documents
import os
router = APIRouter()

print("Current Working Directory (CWD):", os.getcwd())
print("Files in CWD:", os.listdir())

@router.post("/embeddings")
async def create_embeddings(data: dict, request: Request):
    folder_path = data["folder_path"]
    print(folder_path)
    qdrant_client = request.app.state.qdrant_client
    return await process_documents(folder_path, qdrant_client)