from fastapi import HTTPException
from src.lib.configs import settings
from src.lib.vectordb_connection import get_qdrant
from src.lib.google_embedding import embed_text
from src.lib.agent_instructions import instructions
from agents import Agent, Runner, RunConfig,  function_tool, RunContextWrapper,ModelSettings
from dataclasses import dataclass
import re
from qdrant_client.http import models



@dataclass
class QdrantContext:
    client: object  # The Qdrant client object

@function_tool()
async def run_query_with_context(ctx: RunContextWrapper[QdrantContext], query: str):
    try:
        print("tool called")
        print("client",ctx.context.client)
        client = ctx.context.client
        embedding = embed_text(query)
        
        search_result = client.query_points(
            collection_name=settings.qdrant_collection_name,
            query=embedding,
            limit=5,
          
        )
        # print("result_lists",search_result)
        result_lists = []
        # if search_result:
            # print("result_lists",search_result[0].score)
            # [result_lists.append(r.payload.get("chunk_text")) for r in search_result]
        return search_result
      
    except Exception as e:
        print("e",e)
        raise HTTPException(status_code=500, detail=str(e))


async def handle_query_operation(agent_config, query, qdrant_client=None):
    try:
        # Create run configuration
        runConfig = RunConfig(
            model=agent_config.model(),
            model_provider=agent_config.client(),
            model_settings= ModelSettings(
                tool_choice="required",
                
            )
        )

        # Create Qdrant context with the client from app state
        qdrant_context = QdrantContext(client=qdrant_client)

        # Create the agent with specific instructions based on the specification
        starting_agent = Agent(
            name="RAG Query Agent",
            instructions= instructions,
            tools=[run_query_with_context],
            # tool_use_behavior="stop_on_first_tool"
        )

        # Run the agent with the user's query and context
        result = await Runner.run(
            starting_agent=starting_agent,
            input=query,
            run_config=runConfig,
            context=qdrant_context
            
        )
        # print("Final Output:", result.final_output)
        return {"status": "success", "query": query, "result": result.final_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")