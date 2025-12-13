from fastapi import HTTPException
from src.lib.configs import settings
from src.lib.vectordb_connection import get_qdrant
from src.lib.google_embedding import embed_text
from src.lib.agent_instructions import instructions
from agents import Agent, Runner, RunConfig, FunctionTool, function_tool, RunContextWrapper
from dataclasses import dataclass
import re
from qdrant_client.http import models



@dataclass
class QdrantContext:
    client: object  # The Qdrant client object

@function_tool()
async def run_query_with_context(ctx: RunContextWrapper[QdrantContext], query: str):
    print(query)
    try:
        client = ctx.context.client
        # print(client)
        embedding = embed_text("Waht is Physical AI")
        # print("embedding",embedding)

        search_result = client.query_points(
            collection_name=settings.qdrant_collection_name,
            query=embedding,
            limit=5,
          
        )
        # print(search_result)
        return search_result
        return {
            "results": [
                {
                    "score": r.score,
                    "file_path": r.payload.get("file_path"),
                    "chunk_text": r.payload.get("chunk_text")
                }
                for r in search_result
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def handle_query_operation(agent_config, query, qdrant_client=None):
    try:
        # Create run configuration
        runConfig = RunConfig(
            model=agent_config.model(),
            model_provider=agent_config.client(),
        )

        # Create Qdrant context with the client from app state
        qdrant_context = QdrantContext(client=qdrant_client)

        # Create the agent with specific instructions based on the specification
        starting_agent = Agent(
            name="RAG Query Agent",
            instructions= instructions,
            tools=[run_query_with_context]
        )

        # Run the agent with the user's query and context
        result = await Runner.run(
            starting_agent=starting_agent,
            input=query,
            run_config=runConfig,
            context=qdrant_context
        )

        return {"status": "success", "query": query, "result": result.final_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")