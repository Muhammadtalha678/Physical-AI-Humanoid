from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.lib.configs import settings
from fastapi import HTTPException


_qdrant_client = None


def get_qdrant():
    global _qdrant_client
    if not _qdrant_client:
        try:
            _qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
            # Test the connection by getting collections
            _qdrant_client.get_collections()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error connecting to Qdrant: {str(e)}")

    # Check if collection exists, create if it doesn't
    try:
        collections = _qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        print(collection_names)
        if settings.qdrant_collection_name not in collection_names:
            # Create the collection with appropriate vector size for the embedding model
            _qdrant_client.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)  # Standard for text-embedding-3-small
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error managing Qdrant collection: {str(e)}")

    return _qdrant_client
