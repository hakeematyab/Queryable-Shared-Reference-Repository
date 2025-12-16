from typing import TypedDict, Literal, Optional
from langchain_core.messages import BaseMessage

class GenerationState(TypedDict):
    query: str
    is_data_valid: bool
    full_history: list[BaseMessage]
    current_turn: list[BaseMessage]
    messages: list[BaseMessage]
    used_retrieval: bool
    citations: Optional[list[str]]
    response: str
    hallucination_score: Optional[int]
    retrieved_docs: Optional[list[str]]
    # Per-turn metadata lists (correspond to AI messages in full_history)
    citations_history: list[list[str]]
    hallucination_scores: list[Optional[int]]
    error: Optional[str]