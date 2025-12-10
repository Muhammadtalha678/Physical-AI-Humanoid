---
name: rag-backend
description: "A backend-focused skill for RAG system using FastAPI, Qdrant vector DB, and OpenAIAgentSDK. Provides full backend templates, step-by-step guides, and examples."
---

# Purpose
Execute backend RAG workflows including project initialization, vector DB connection, embedding storage, and semantic search.

# Key Responsibilities
1. Provide templates for `main.py`, controllers, routers, configs, and vector DB connection.
2. Provide step-by-step workflow guides for project initialization, Qdrant setup, embedding storage, and semantic search.
3. Provide example `.env` and API request/response sample outputs.
4. Ensure modular and high-level responses suitable for frontend integration.

# Templates
All backend templates are included in:

- **Main application template** → [templates/main_py_template.txt](templates/main_py_template.txt)  
- **Controller logic template** → [templates/controller_template.txt](templates/controller_template.txt)  
- **Router template** → [templates/router_template.txt](templates/router_template.txt)  
- **Environment config loader**, **OpenAIAgentSDK configuration**, **Qdrant vector DB connection**  → [templates/config_template.txt](templates/config_template.txt)  

# Project Structure Reference
A complete FastAPI backend layout is documented under:

- [project-structure/backend-structure.md](project-structure/backend-structure.md)

# Step-by-Step Guides
To understand how this RAG backend works internally, follow:

- **Initialization Guide** → [steps/initialization.md](steps/initialization.md)  
- **Qdrant Setup Guide** → [steps/qdrant-setup.md](steps/qdrant-setup.md)  
- **Embedding Storage Flow** → [steps/embeddings-save.md](steps/embeddings-save.md)  
- **Semantic Search Flow** → [steps/search-flow.md](steps/search-flow.md)

# Examples
Practical examples are included here:

- Sample `.env` file → [examples/example_env.txt](examples/example_env.txt)  
- Example output (embedding storage) → [examples/store_embeddings_output.json](examples/store_embeddings_output.json)  
- Example output (semantic search response) → [examples/search_output.json](examples/search_output.json)

# Constraints
- Do not embed heavy code directly; use templates.
- Keep steps modular.
- Provide high-level JSON-friendly responses.

# Usage Examples
1. Initialize a new FastAPI RAG backend using UV and provided templates.  
2. Store embeddings into Qdrant using the controller template.  
3. Execute semantic search and return vector results using OpenAIAgentSDK.

