import os
import asyncio
import json
from openai import AsyncAzureOpenAI
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv


load_dotenv(Path(__file__).parent.parent.parent / ".env")

MAX_TOKENS = 2000
deployment_name = "gpt-4o"
endpoint = "https://local-rag-resource.services.ai.azure.com/"

client = AsyncAzureOpenAI(
    azure_endpoint=endpoint,
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01"
)

app = FastAPI()

# ---------------------------------------------------------------------------
# LLM: streaming client (OpenAI/Anthropic SDK etc)
# ---------------------------------------------------------------------------
async def llm_stream(messages: list[dict]):
    stream = await client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        max_tokens=MAX_TOKENS,
        stream=True
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content
        if token:
            yield token

# ---------------------------------------------------------------------------
# SSE relay
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    messages: list[dict]

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


async def event_stream(messages: list[dict], request: Request):
    """
    Reformat the LLM token stream into SSE-formatted bytes.
    Detects the connection status and stops streaming once it's closed.

    Args:
        prompt: The user's input. Passed through to the LLM call.
        request: The live ASGI Request for this connection. Carries no
            data we need — it exists so we can poll
            `request.is_disconnected()` each iteration and check whether
            the client (browser tab, curl, etc.) is still on the other
            end of the socket.
    """
    try:
        async for token in llm_stream(messages):
            if await request.is_disconnected():
                break
            yield sse_event(data={"token": token}, event="token")
        yield sse_event(data={}, event="done") # using named event for completion flag
    except asyncio.CancelledError:
        # Propagates if the server itself cancels the task (e.g. shutdown).
        # Re-raise so the ASGI server can clean up properly.
        raise
    except Exception as e:
        # Surface upstream errors as a typed SSE event rather than
        # silently dying — the client can branch on event type.
        yield sse_event(data={"message": str(e)}, event="error")


@app.post("/chat")
async def chat(chat_request: ChatRequest, request: Request):
    return StreamingResponse(
        event_stream(chat_request.messages, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx buffering if behind a proxy
        }
    )