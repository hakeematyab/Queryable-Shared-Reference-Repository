from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from orchestration.prompts import format_context

class SearchSchema(BaseModel):
    query: str = Field("A keyword enriched search query")

def create_tools(searcher):
    def hybrid_search(query: str) -> list[dict]:
        results =  searcher.search_sync(query)
        return format_context(results.get("results",[{}]))
    
    return [
        StructuredTool.from_function(
            func=hybrid_search,
            name = "hybrid_search",
            description="Search papers using hybrid retrieval",
            args_schema=SearchSchema
        )
    ]