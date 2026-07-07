# LangChain agent streaming (v2) compatible server
import os
from langchain_openai import AzureChatOpenAI
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential, get_bearer_token_provider
import contextvars
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import random
import mlflow
from contextlib import contextmanager

from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from utils import (
    get_session_id,
    process_agent_astream_events,
)
from typing import AsyncGenerator


load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Databricks MLflow Connection via U2M Profile
mlflow.set_tracking_uri(uri=f"databricks://{os.environ.get("DATABRICKS_CONFIG_PROFILE")}")
mlflow.set_experiment(experiment_name=os.environ.get("DATABRICKS_EXPERIMENT_NAME"))
mlflow.langchain.autolog()


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

# Agent and harness
# =========================================================================================================
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

# implementing the ResponsesAgent interface to LangGraph agent:
# =========================================================================================================
@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    messages = {"messages": to_chat_completions_input([i.model_dump() for i in request.input])}

    async for event in process_agent_astream_events(
        agent.astream(
            input=messages,
            stream_mode=["updates", "messages"],
            config={"configurable": {"thread_id": session_id}},
        )
    ):
        yield event