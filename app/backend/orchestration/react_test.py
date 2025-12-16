import random
import os

import langchain

# langchain.debug = True
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessageChunk, ToolMessage, SystemMessage, HumanMessage



class WeatherSchema(BaseModel):
    city: str = Field("Name of the city to get weather from")

def get_weather(city: str):
    """Get the current weather for a specific city."""
    # Returns random values as requested
    temp = random.randint(0, 60)
    condition = random.choice(["sunny", "raining", "snow", "overcast"])
    return f"The weather in {city} is {temp}°C and {condition}."
    

tools = [StructuredTool.from_function(
    func=get_weather,
    name="weather_tool",
    description="Get weather for the provided city",
    args_schema=WeatherSchema
)]
system_prompt = "You are a weather reporter who speaks exactly like a cat person. Always use the nya slang."
system_prompt = SystemMessage(content=system_prompt)
llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
)

agent_executor = create_react_agent(llm, 
                                    prompt=system_prompt,
                                    tools=tools)


query = "What is the weather in Boston right now?"
# response = agent_executor.invoke({"messages": [("user",query)]})

# print(response["messages"][-1].content)

async def main():
    tool_result = []
    output = ''
    msgs = [HumanMessage(content=query)]
    async for chunk, _ in agent_executor.astream({"messages": msgs}, stream_mode="messages"):
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            output+=chunk.content
            print(output, flush=True)
        elif isinstance(chunk, ToolMessage) and chunk.name=='weather_tool':
            tool_result.append(chunk.content)

    print(tool_result)
if __name__=='__main__':
    import asyncio
    asyncio.run(main())