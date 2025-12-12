from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage, trim_messages, HumanMessage
from orchestration.state import GenerationState

class RAGPipeline:
    def __init__(self,
                 system_prompt,
                 llm,
                 tools,
                 hallucination_detector,
                 max_query_len = 3200,
                 max_history = 15000,
                 ):
        self.agent_executor = create_react_agent(model=llm,
                                                 tools=tools,
                                                 prompt=system_prompt
                                                 )
        self.hallucination_detector = hallucination_detector
        self.llm = llm
        self.max_query_len = max_query_len
        self.max_history = max_history


    def _estimate_tokens(self, text: str) -> int:
        char_estimate = len(text) * 0.25
        word_estimate = len(text.split()) * 1.6
        return int((char_estimate + word_estimate) // 2)

    def data_validation(self, state: dict):
        total_tokens = self._estimate_tokens(state['query'])
        state["is_data_valid"] = total_tokens <= self.max_query_len 
        return state

    async def react_loop(self, state: dict):
        state["chat_stream"] = {"token":""}
        response = ''
        retrieval_result = []
        state["current_turn"] = []
        state["messages"].append(HumanMessage(content=state["query"]))
        state["current_turn"].append(HumanMessage(content=state["query"]))
        async for chunk in self.agent_executor.astream({"messages": state["messages"]}):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                state["chat_stream"]["token"] = chunk.content
                response += chunk.content
            elif isinstance(chunk, ToolMessage) and chunk.name == 'hybrid_search':
                retrieval_result.append(chunk.content)
        state["used_retrieval"] = len(retrieval_result) > 0
        state["citations"] = retrieval_result
        state["response"] = response
        state["current_turn"].append(AIMessage(content=response))
        return state

    async def hallucination_check(self, state: dict):
        claims = [state["query"]+state["response"]]
        sources = "\n\n\n---\n\n\n".join(state["citations"])
        hallucination_score = await self.hallucination_detector.detect(claims=claims,
                                                                      sources=sources)
        state["hallucination_score"] = hallucination_score
        return state

    def context_management(self, state: dict):
        if "full_history" not in state:
            state["full_history"] = []
        state["full_history"].extend(state.get("current_turn", []))
        state["messages"] = trim_messages(
            state["messages"],
            max_tokens=self.max_history,
            strategy="last",
            token_counter=self.llm,
        )
        return state


def get_rag_graph(system_prompt, llm, tools, hallucination_detector, checkpointer=None):
    def denial_node(state: dict):
        DENIAL_RESPONSE = "Your query is too long. Please shorten it and try again."
        state["error"] = DENIAL_RESPONSE
        state["response"] = DENIAL_RESPONSE
        return state
    def route_after_validation(state: dict):
        return "denial" if not state["is_data_valid"] else "react_loop"
    def route_after_react(state: dict):
        return "hallucination_check" if state["used_retrieval"] else "context_management"

    pipeline = RAGPipeline(system_prompt, llm, tools, hallucination_detector)
    
    graph = StateGraph(GenerationState)
    
    graph.add_node("data_validation", pipeline.data_validation)
    graph.add_node("denial", denial_node)
    graph.add_node("context_management", pipeline.context_management)
    graph.add_node("react_loop", pipeline.react_loop)
    graph.add_node("hallucination_check", pipeline.hallucination_check)
    
    graph.add_edge(START, "data_validation")
    graph.add_conditional_edges("data_validation", 
                                route_after_validation,
                                {"denial": END, 
                                "react_loop": "react_loop"}
                                )
    graph.add_conditional_edges("react_loop", route_after_react,
                                {"hallucination_check": "hallucination_check", 
                                "context_management": "context_management"}
                                )
    graph.add_edge("hallucination_check", "context_management")
    graph.add_edge("context_management", END)
    
    return graph.compile(checkpointer=checkpointer)


async def create_rag_graph_with_memory(
    system_prompt, 
    llm, 
    tools, 
    hallucination_detector,
    db_path: str = "./data/checkpoints.db"
):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    
    checkpointer = AsyncSqliteSaver.from_conn_string(db_path)
    await checkpointer.setup()
    
    return get_rag_graph(
        system_prompt, 
        llm, 
        tools, 
        hallucination_detector,
        checkpointer=checkpointer
    )