from typing import TypedDict, Literal, Optional
from langchain_core.messages import BaseMessage

class ChatStream(TypedDict):
    status: Literal["streaming", "completed"]
    token: str

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
    chat_stream: ChatStream

    error: Optional[str]