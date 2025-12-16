import re
import logging
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, ToolMessage, trim_messages, HumanMessage
from orchestration.state import GenerationState

logger = logging.getLogger(__name__)

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
        logger.info("[NODE: data_validation] Starting validation")
        total_tokens = self._estimate_tokens(state['query'])
        state["is_data_valid"] = total_tokens <= self.max_query_len
        logger.info(f"[NODE: data_validation] Query tokens: {total_tokens}, max: {self.max_query_len}, valid: {state['is_data_valid']}")
        return state

    async def react_loop(self, state: dict):
        import json
        from langchain_core.messages import AIMessageChunk
        
        logger.info("[NODE: react_loop] Starting agent execution")
        logger.info(f"[NODE: react_loop] Query: {state['query'][:100]}..." if len(state.get('query','')) > 100 else f"[NODE: react_loop] Query: {state.get('query','')}")
        
        response = ''
        state["citations"] = []
        state["hallucination_score"] = None
        state["retrieved_docs"] = []
        retrieval_results = []
        state["current_turn"] = []
        messages = state.get("messages",[])
        messages.append(HumanMessage(content=state["query"]))
        state["current_turn"].append(HumanMessage(content=state["query"]))
        
        # Use astream with combined modes: messages for tokens, values for tool results
        async for stream_mode, event in self.agent_executor.astream(
            {"messages": messages}, 
            stream_mode=["messages", "values"],
            config={"recursion_limit": 10}
        ):
            if stream_mode == "values":
                for msg in event.get("messages", []):
                    if isinstance(msg, ToolMessage) and msg.name == 'hybrid_search':
                        if msg.content not in retrieval_results:
                            logger.info(f"[NODE: react_loop] Tool call: hybrid_search returned {len(msg.content)} chars")
                            retrieval_results.append(msg.content)
                    # Capture the final AI response (last AIMessage without tool_calls)
                    elif isinstance(msg, AIMessage) and msg.content:
                        if not getattr(msg, 'tool_calls', None):
                            response = msg.content
        
        state["used_retrieval"] = len(retrieval_results) > 0
        logger.info(f"[NODE: react_loop] Used retrieval: {state['used_retrieval']}, tool calls: {len(retrieval_results)}")
        
        citations = []
        retrieved_docs = []
        for result_str in retrieval_results:
            try:
                data = json.loads(result_str) if isinstance(result_str, str) else result_str
                retrieved_docs.extend(data.get("retrieved_docs", []))
                citations.extend(data.get("citations", []))
            except:
                pass
        
        state["retrieved_docs"] = retrieved_docs
        state["citations"] = citations
        state["response"] = response
        state["current_turn"].append(AIMessage(content=response))
        messages.append(AIMessage(content=response))
        state["messages"] = messages
        logger.info(f"[NODE: react_loop] Complete. Citations: {len(citations)}, Response length: {len(response)}")
        return state

    async def hallucination_check(self, state: dict):
        logger.info("[NODE: hallucination_check] Starting hallucination detection")
        claims = [state["query"]+state["response"]]
        sources = "\n\n---\n\n".join(state["retrieved_docs"])
        logger.info(f"[NODE: hallucination_check] Claims: {len(claims)}, Sources length: {len(sources)}")
        hallucination_score = await self.hallucination_detector.detect(claims=claims,
                                                                      sources=sources)
        state["hallucination_score"] = hallucination_score
        logger.info(f"[NODE: hallucination_check] Score: {hallucination_score}")
        return state

    def context_management(self, state: dict):
        logger.info("[NODE: context_management] Managing conversation context")
        if "full_history" not in state:
            state["full_history"] = []
        if "citations_history" not in state:
            state["citations_history"] = []
        if "hallucination_scores" not in state:
            state["hallucination_scores"] = []
        
        state["full_history"].extend(state.get("current_turn", []))
        
        # Append per-turn metadata (empty if no retrieval)
        state["citations_history"].append(state.get("citations", []) or [])
        state["hallucination_scores"].append(state.get("hallucination_score"))
        
        pre_trim_count = len(state["messages"])
        state["messages"] = trim_messages(
            state["messages"],
            max_tokens=self.max_history,
            strategy="last",
            token_counter=self.llm,
        )
        logger.info(f"[NODE: context_management] Messages: {pre_trim_count} -> {len(state['messages'])}, History: {len(state['full_history'])}")
        return state


def get_rag_graph(system_prompt, llm, tools, hallucination_detector, checkpointer=None):    
    # Patterns indicating the model couldn't answer (condensed from common non-answer templates)
    DATA_VALID_DENIAL_RESPONSE = "Your query is too long. Please shorten it and try again."
    
    NON_ANSWER_PATTERN = re.compile(
        r"(cannot\s+(find|determine|answer|provide)|"
        r"not\s+(mentioned|provided|specified|covered|addressed|present)|"
        r"(insufficient|no|without)\s+(information|evidence|details|context)|"
        r"don'?t\s+have\s+(enough|sufficient)|"
        r"unable\s+to\s+(find|provide|answer)|"
        r"(unclear|unavailable)\s+based\s+on)",
        re.IGNORECASE
    )
    def denial_node(state: dict):
        logger.info("[NODE: denial] Query denied due to validation failure")
        state["error"] = DATA_VALID_DENIAL_RESPONSE
        state["response"] = DATA_VALID_DENIAL_RESPONSE
        return state
    
    def route_after_validation(state: dict):
        next_node = "denial" if not state["is_data_valid"] else "react_loop"
        logger.info(f"[ROUTE: after_validation] is_valid={state['is_data_valid']} -> {next_node}")
        return next_node
    
    def route_after_react(state: dict):
        if not state["used_retrieval"]:
            logger.info("[ROUTE: after_react] No retrieval used -> context_management")
            return "context_management"
        # Check if response is a non-answer (model refused to answer)
        response = state.get("response", "")
        if NON_ANSWER_PATTERN.search(response):
            # Reset citations/retrieved_docs since the model didn't actually use them
            logger.info("[ROUTE: after_react] Non-answer detected, clearing citations -> context_management")
            state["citations"] = []
            state["retrieved_docs"] = []
            return "context_management"
        logger.info("[ROUTE: after_react] Retrieval used -> hallucination_check")
        return "hallucination_check"

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