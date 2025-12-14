from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str
    openrouter_api: str
    openrouter_url: str
    openrouter_model_name: str
    google_api_key: str
    embedding_model: str
    backend_url: str
    frontend_url: str
    localhost_url: str

    class Config:
        env_file = ".env"


settings = Settings()