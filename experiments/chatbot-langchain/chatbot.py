# Langchain's agent chatbot
# Using Sreaming interface (v2)

import asyncio
import os
import uuid
import random
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import AIMessageChunk, AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
import contextvars

load_dotenv(Path(__file__).parent.parent.parent / ".env")
MAX_TOKENS = 2000
DEPLOYMENT_NAME = "gpt-4o"
API_VERSION = "2024-02-01"


# Utilities
# ================================================================
def make_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"{timestamp}-{suffix}"


def generate_weather_conditions() -> tuple[int, int, str]:
    """Returns (temperature_degC, humidity_pct, condition)."""
    temperature = random.randint(10, 30)
    humidity = random.randint(50, 90)
    condition = random.choice(["cloudy", "sunny", "partially cloudy", "rainy"])
    return temperature, humidity, condition


# def save_history(session_id: str, history: list[dict]):
#     os.makedirs(HISTORY_DIR, exist_ok=True)
#     path = os.path.join(HISTORY_DIR, f"{session_id}.json")
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(history, f, indent=2, ensure_ascii=False)
#     print(f"\nHistory saved to {path}")


# ================================================================


# Agent model and harness
# ================================================================
client = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_deployment=DEPLOYMENT_NAME,
    api_version=API_VERSION,
)


# update the agent and its tools below
@tool()
def get_weather(location: str) -> str:
    """
    get weather data at the specified location
    """
    weather_conditions = generate_weather_conditions()
    weather = f"""
    Weather in {location}:
    temperature: {weather_conditions[0]} degC
    humidity: {weather_conditions[1]}%
    {weather_conditions[2]}"
    """
    return weather


@tool()
def generate_itinerary(location: str, days: str, interest: str = "general") -> str:
    """
    Generate a short n-days tourist itinerary for a given location and user's interest.
    """
    prompt = (
        f"Create a concise {days}-day tourist itinerary for {location} with the weather of, the user has {interest} interest. "
        "For each day list 2-3 activities or attractions with a one-sentence description each."
        "Keep the total response under 250 words."
    )
    ctx = contextvars.Context()
    response = ctx.run(client.invoke, [{"role": "user", "content": prompt}])
    return response.content


SYSTEM_PROMPT = """
You are a tour guide assistant that provides weather update and other useful information about the user's question on the location.
"Make your advice suitable with the weather condition, or at least advice the user accordingly."
If you dont know the answer, or the question is not about your expertise, politely decline.
"""


tools = [get_weather, generate_itinerary]
agent = create_agent(
    client, tools=tools, system_prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver()
)


#
# async def llm_stream(user_input: str, thread_id: str):
#     config = {"configurable": {"thread_id": thread_id}}
#     async for chunk in agent.astream(
#         input={"messages": [{"role": "user", "content": user_input}]},
#         config=config,
#         stream_mode=["messages", "updates"],
#         version="v2",
#     ):
#         yield chunk

async def llm_stream(messages: list[dict], thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    async for chunk in agent.astream(
        input={"messages": messages},
        config=config,
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        yield chunk


async def main():
    thread_id = make_session_id()  # use session ID as thread ID for now

    print(
        f"Chat with the model (each response is capped at ~{int(MAX_TOKENS * 0.75)} words). Ctrl+C or 'quit' to exit.\n"
    )

    try:
        while True:
            user_input = await asyncio.to_thread(input, "You: ")
            if user_input.strip().lower() in ("quit", "exit"):
                print("\n\nExited, saving...")
                break

            # update the streaming below
            # =====================================================================
            current_node = None
            print("Assistant: ", end="", flush=True)
            # async for chunk in llm_stream(user_input=user_input, thread_id=thread_id):
            async for chunk in llm_stream(messages=[{"role": "user", "content": user_input}], thread_id=thread_id):
                chunk_type = chunk["type"]

                if chunk_type == "messages":  # if the chunk is from LLLM
                    token, metadata = chunk["data"]
                    node = metadata.get("langgraph_node")

                    if node != current_node:
                        print(f"\n[node: {node}]")
                        current_node = node

                    if isinstance(token, AIMessageChunk):
                        # Printing fine-grained mode: individual token by token by LLM
                        for block in token.content_blocks:
                            if block["type"] == "text":
                                print(block["text"], end="", flush=True)

                elif (
                    chunk_type == "updates"
                ):  # if the chunk is from the node upon completion
                    # Coarse-grained: full state after a node completes
                    for node_name, state_update in chunk["data"].items():

                        # The AI message:
                        if node_name == "model":
                            msg = state_update["messages"][-1]
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                print(
                                    f"\n[TOOL CALLS]: {[tc['name'] for tc in msg.tool_calls]}"
                                )
                                for tc in msg.tool_calls:
                                    print(f"  - {tc['name']}({tc['args']})")
                        
                        # The tool message
                        elif node_name == "tools":
                            msg = state_update["messages"][-1]
                            if isinstance(msg, ToolMessage):
                                print(
                                    f"\n[TOOL RESULT from {msg.name}]: {msg.content}\n"
                                )

                        elif node_name == "__interupt__":
                            # Human in the loop breakpoints
                            print(f"\n[INTERRUPT]: {state_update}")

            print("\n")
            # =====================================================================

    except KeyboardInterrupt:
        print("\n\nInterrupted, exitting...")

    # finally:
    #     # The general principle: cleanup-that-must-always-happen belongs in finally
    #     save_history(session_id, history)


if __name__ == "__main__":
    asyncio.run(main())
