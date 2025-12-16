from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from orchestration.prompts import format_context
import json
import logging

logger = logging.getLogger(__name__)

class SearchSchema(BaseModel):
    query: str = Field("A keyword enriched search query")

def create_tools(searcher, lock=None):
    def hybrid_search(query: str) -> str:
        logger.info(f"[TOOL: hybrid_search] Query: {query}")
        if lock:
            with lock:
                results = searcher.search_sync(query)
        else:
            results = searcher.search_sync(query)
        docs = results.get("results", [])
        logger.info(f"[TOOL: hybrid_search] Retrieved {len(docs)} documents")
        citations = []
        for doc in docs:
            meta = doc.get("metadata", {})
            name = meta.get("doc_name", "Unknown")
            pages = meta.get("page_numbers", [])
            if len(pages) > 1:
                cite = f"{name}, Pages {pages[0]}-{pages[-1]}"
            elif pages:
                cite = f"{name}, Page {pages[0]}"
            else:
                cite = name
            if cite not in citations:
                citations.append(cite)
        logger.info(f"[TOOL: hybrid_search] Citations extracted: {citations}")
        return json.dumps({
            "retrieved_docs": format_context(docs),
            "citations": citations
        })
    
    return [
        StructuredTool.from_function(
            func=hybrid_search,
            name = "hybrid_search",
            description="Search papers using hybrid retrieval",
            args_schema=SearchSchema
        )
    ]