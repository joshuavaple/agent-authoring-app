import os
import asyncio
import json
from openai import AsyncAzureOpenAI
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential, get_bearer_token_provider


load_dotenv(Path(__file__).parent.parent.parent / ".env")

MAX_TOKENS = 2000
DEPLOYMENT_NAME = "gpt-4o"
API_VERSION = "2024-02-01"

# ---------------------------------------------------------------------------
# MS Foundry Authentication
# ---------------------------------------------------------------------------
credential = ClientSecretCredential(
    tenant_id=os.environ.get("AZURE_TENANT_ID"),
    client_id=os.environ.get("AZURE_CLIENT_ID"),
    client_secret=os.environ.get("AZURE_CLIENT_SECRET"),
)

# the function below is sync in nature - ok for small app
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)

# Use Foundry API key
# client = AsyncAzureOpenAI(
#     azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
#     api_version=API_VERSION
# )

# Use OAuth2 with EntraID:
client = AsyncAzureOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=token_provider,
    api_version=API_VERSION,
)

app = FastAPI()

# ---------------------------------------------------------------------------
# LLM: streaming client (OpenAI/Anthropic SDK etc)
# ---------------------------------------------------------------------------
async def llm_stream(messages: list[dict]):
    """Stream tokens from the LLM API.

    Makes an asynchronous request to the Azure OpenAI API and yields
    individual tokens as they arrive from the stream. Filters out empty
    tokens and handles malformed chunks gracefully.

    Args:
        messages: A list of message dictionaries with 'role' and 'content' keys,
            representing the conversation history to send to the LLM.

    Yields:
        str: Individual tokens from the LLM response.
    """
    stream = await client.chat.completions.create(
        model=DEPLOYMENT_NAME,
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

    This generator yields Server-Sent Events for each token produced by the
    language model and emits a final done event when streaming completes.
    It also checks whether the client connection is still alive and stops
    streaming if the request is disconnected.

    Args:
        messages (list[dict]): Conversation history with 'role' and 'content'
            keys to send to the LLM.
        request (Request): ASGI request object used to detect client
            disconnection.

    Yields:
        str: SSE-formatted event blocks.
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

# ---------------------------------------------------------------------------
# Helper endpoint to generate chat session summary
# ---------------------------------------------------------------------------
 
class TitleRequest(BaseModel):
    messages: list[dict]
 
 
@app.post("/title")
async def title(title_request: TitleRequest):
    """
    One-shot, non-streaming completion that summarizes a conversation
    into a short session title. Deliberately separate from /chat:
    titling doesn't need token-by-token delivery, so SSE framing would
    just be overhead here — a plain JSON response is the right shape.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Summarize this conversation in 3-10 words as a short title. "
                "Respond with only the title text — no punctuation, no quotes, no preamble."
            ),
        }
    ] + [m for m in title_request.messages if m["role"] in ("user", "assistant")]
 
    response = await client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=messages,
        max_tokens=20,
        stream=False,
    )
    title_text = response.choices[0].message.content.strip()
    return {"title": title_text}