# LangChain agent streaming (v2) compatible server

import os
import asyncio
import json
from langchain_openai import AzureChatOpenAI
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential, get_bearer_token_provider
import contextvars
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
import random
import mlflow
from contextlib import contextmanager


mlflow.set_tracking_uri("databricks://joshuale-common")
mlflow.set_experiment("/Shared/chatbot-langchain-server")
mlflow.langchain.autolog()

load_dotenv(Path(__file__).parent.parent.parent / ".env")

MAX_TOKENS = 2000
DEPLOYMENT_NAME = "gpt-4o"
API_VERSION = "2024-02-01"

credential = ClientSecretCredential(
    tenant_id=os.environ.get("AZURE_TENANT_ID"),
    client_id=os.environ.get("AZURE_CLIENT_ID"),
    client_secret=os.environ.get("AZURE_CLIENT_SECRET"),
)

token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)


client = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_deployment=DEPLOYMENT_NAME,
    api_version=API_VERSION,
)


def generate_weather_conditions() -> tuple[int, int, str]:
    """Returns (temperature_degC, humidity_pct, condition)."""
    temperature = random.randint(10, 30)
    humidity = random.randint(50, 90)
    condition = random.choice(["cloudy", "sunny", "partially cloudy", "rainy"])
    return temperature, humidity, condition

@contextmanager
def disable_nested_tracing():
    """Temporarily disable autologging for nested LLM calls within tools"""
    mlflow.langchain.autolog(disable=True)
    try:
        yield
    finally:
        mlflow.langchain.autolog()

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
    with disable_nested_tracing():
        prompt = (
            f"Create a concise {days}-day tourist itinerary for {location} with the weather of, the user has {interest} interest. "
            "For each day list 2-3 activities or attractions with a one-sentence description each."
            "Keep the total response under 250 words."
        )
        ctx = contextvars.Context()
        response = ctx.run(client.invoke, [{"role": "user", "content": prompt}])
        result = response.content
    return result


SYSTEM_PROMPT = """
You are a tour guide assistant that provides weather update and other useful information about the user's question on the location.
"Make your advice suitable with the weather condition, or at least advice the user accordingly."
If you dont know the answer, or the question is not about your expertise, politely decline.
"""


tools = [get_weather, generate_itinerary]
agent = create_agent(
    client, tools=tools, system_prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver()
)

async def llm_stream_langchain(messages: list[dict], thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    async for chunk in agent.astream(
        input={"messages": messages},
        config=config,
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        chunk_type = chunk["type"]

        if chunk_type == "messages":
            token, _metadata = chunk["data"]          

            if isinstance(token, AIMessageChunk):
                for block in token.content_blocks:
                    if block["type"] == "text":
                        yield block["text"]

def sse_event(data: dict, event: str | None = None) -> str:
    """
    Given a payload and an optional event name, produce a syntactically correct SSE event block
    Note the mandatory trailing blank line.
    Avoids emitting raw token in `data: token` event, but wraps the token content inside json.
    This prevents any newline '\n' characters as part of the token from breaking the SSE format.

    Args:
        data: dictionary of the actual message
        event: the name of the event type, one of [token, error, done]
    
    Returns:
        SSE event block strings. E.g.,
        ```
        event: token
        data:  {"token": str}

        event: error
        data:  {"message": str}

        event: done
        data:  {}
        ```
    """
    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data)}")

    return "\n".join(lines) + "\n\n"


async def event_stream_langchain(messages: list[dict], thread_id: str, request: Request):
    """
    Reformat the LLM token stream into SSE-formatted bytes.

    This generator yields Server-Sent Events for each token produced by the
    language model and emits a final done event when streaming completes.
    It also checks whether the client connection is still alive and stops
    streaming if the request is disconnected.

    Args:
        messages (list[dict]): Conversation history with 'role' and 'content'
            keys to send to the LLM.
        thread_id (str): the unique thread of the current conversation, for short-term memory.
        request (Request): ASGI request object used to detect client
            disconnection.

    Yields:
        str: SSE-formatted event blocks.
    """
    try:
        async for token in llm_stream_langchain(messages, thread_id):
            if await request.is_disconnected():
                break
            yield sse_event(data={"token": token}, event="token")
        yield sse_event(data={}, event="done")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        yield sse_event(data={"message": str(e)}, event="error")



class ChatRequest(BaseModel):
    messages: list[dict]
    thread_id: str

app = FastAPI()

@app.post("/chat")
async def chat(chat_request: ChatRequest, request: Request):
    return StreamingResponse(
        event_stream_langchain(chat_request.messages, chat_request.thread_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx buffering if behind a proxy
        }
    )