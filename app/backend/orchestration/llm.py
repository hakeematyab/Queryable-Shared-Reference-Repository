from langchain_ollama import ChatOllama


class OllamaClient:
    def __init__(
        self,
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.7,
        max_tokens=2048,
    ):
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def astream(self, messages):
        async for chunk in self.llm.astream(messages):
            yield chunk

    def bind_tools(self, tools):
        return self.llm.bind_tools(tools)
    