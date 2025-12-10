import os
import uuid
from fastapi import HTTPException
from qdrant_client.http import models
from src.lib.configs import settings
from src.lib.google_embedding import embed_text
from src.lib.chunker import chunk_markdown


async def process_documents(folder_path: str, qdrant_client=None):
    try:
        # Validate that the directory exists
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=400, detail="Directory does not exist")

        if not os.path.isdir(folder_path):
            raise HTTPException(status_code=400, detail="Path is not a directory")

        # Use provided Qdrant client from app state, or get a new one as fallback
        client = qdrant_client
        if client is None:
            from src.lib.vectordb_connection import get_qdrant
            client = get_qdrant()

        all_files = []

        # Recursively get all .md and .txt files up to reasonable depth
        for root, dirs, files in os.walk(folder_path):
            # print(root)
            # print(dirs)
            # print(files)
            # Calculate depth to enforce reasonable limit
            current_depth = root.replace(folder_path, '').count(os.sep)
            if current_depth > 10:  # Reasonable depth limit
                continue

            for f in files:
                if f.lower().endswith((".md", ".txt")):  # Validate file extension
                    all_files.append(os.path.join(root, f))

        print(len(all_files))
        
        if not all_files:
            return {"status": "error", "message": "No markdown or text files found in the specified directory."}

        for file_path in all_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    print(len(text))

                chunks = chunk_markdown(text)
                # print(len(chunks))
                # 
                for i, chunk in enumerate(chunks):
                    embedding = embed_text(chunk)
                    # 
                    print(i)
                    client.upsert(
                        collection_name=settings.qdrant_collection_name,
                        points=[
                            models.PointStruct(
                                id=str(uuid.uuid4()),
                                vector=embedding,
                                payload={
                                    "file_path": file_path,
                                    "chunk_index": i,
                                    "chunk_text": chunk
                                }
                            )
                        ]
                    )
                #  
            except Exception as file_error:
                # Log the error but continue processing other files
                print(f"Error processing file {file_path}: {str(file_error)}")
                continue

        return {"status": "success", "files_processed": len(all_files)}

    # except HTTPException:
    #     # Re-raise HTTP exceptions as they are
    #     raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")