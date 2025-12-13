import google.generativeai as genai
from src.lib.configs import settings

# Configure the API key
genai.configure(api_key=settings.google_api_key)


def embed_text(text: str):
    try:
        # Use the embedding model specified in settings
        result = genai.embed_content(
            model=settings.embedding_model,
            content=text
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Fallback: return a simple representation if embedding fails
        return [0.0] * 1536  # Standard embedding dimension