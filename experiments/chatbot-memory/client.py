import asyncio
import json
import uuid
from datetime import datetime
import httpx
from httpx_sse import aconnect_sse


HISTORY_DIR = "chat_logs"
SERVER_URL = "http://localhost:8000/responses"  # ResponsesAgent route exposed by AgentServer
MAX_TOKENS = 2000
SYSTEM_PROMPT = "You are a witty, helpful assistant. Keep your answer brief, preferably less than 3 sentences, unless asked for details."


def make_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"{timestamp}-{suffix}"


async def stream_chat_langchain(message: list[dict], conversation_id: str) -> str:
    """
    POST message to the server for short-term memory
    Print tokens as they arrive
    Raises RuntimeError if the server reports an upstream error mid-stream.

    Args:
        message: a user message of {"role": role, "content": content}, which follow OpenAI message format.
        conversation_id: a unique ID of the conversation

    Returns:
        The full assistant response (str)
    """
    full_response = []

    payload = {
        "input": message,
        "stream": True,
        "context": {"conversation_id": conversation_id},
    }

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        async with aconnect_sse(
            client=http_client,
            method="POST",
            url=SERVER_URL,
            json=payload,
        ) as event_source:
            async for sse in event_source.aiter_sse():
                if sse.data == "[DONE]":
                    break

                chunk = json.loads(sse.data)
                if "error" in chunk:
                    raise RuntimeError(f"Server error mid-stream: {chunk['error']}")

                chunk_type = chunk.get("type")

                if chunk_type == "response.output_text.delta":
                    token = chunk["delta"]
                    print(token, end="", flush=True)
                    full_response.append(token)

    return "".join(full_response)


async def main():
    conversation_id = make_session_id()
    print(f"Chat with the model (each response is capped at ~{int(MAX_TOKENS * 0.75)} words). Ctrl+C or 'quit' to exit.\n")

    try:
        while True:
            user_input = await asyncio.to_thread(input, "You: ")
            if user_input.strip().lower() in ("quit", "exit"):
                print("\n\nExited, saving...")
                break

            print("Assistant: ", end="", flush=True)

            try:
                _full_response = await stream_chat_langchain(
                    message=[{"role": "user", "content": user_input}],
                    conversation_id=conversation_id,
                )
            except RuntimeError as e:
                print(f"Server Error: {e}")
                continue

            print("\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted, saving...")


if __name__ == "__main__":
    asyncio.run(main())