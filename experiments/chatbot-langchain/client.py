import asyncio
import httpx
import json
import os
import uuid
from datetime import datetime
import httpx
from httpx_sse import aconnect_sse


HISTORY_DIR = "chat_logs"
SERVER_URL = "http://localhost:8000/chat"
MAX_TOKENS = 2000
SYSTEM_PROMPT = "You are a witty, helpful assistant. Keep your answer brief, preferably less than 3 sentences, unless asked for details."

def make_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"{timestamp}-{suffix}"

async def stream_chat_langchain(message: list[dict], thread_id: str) -> str:
    """
    POST message to the server for short-term memory
    Print tokens as they arrive
    Raises RuntimeError if the server reports an upstream error mid-stream.

    Args:
        message: a user message of {"role": role, "content": content}, which follow OpenAI message format.
        thread_id: a unique ID of the thread
    
    Returns:
        The full assistant response (str)

    """
    full_response = []
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        async with aconnect_sse(
            client=http_client,
            method="POST",
            url=SERVER_URL,
            json={"messages": message, "thread_id": thread_id}
        ) as event_source:
            # sse object handles the sse string blocks and their delimiter:
            async for sse in event_source.aiter_sse(): 
                if sse.event == "token":                # same contract-level logic
                    payload = sse.json()                # .json() parses sse.data for you
                    token = payload["token"]
                    print(token, end="", flush=True)
                    full_response.append(token)
                elif sse.event == "done":               # same contract-level logic
                    break
                elif sse.event == "error":              # same contract-level logic
                    message = sse.json().get("message", "unknown server error")
                    raise RuntimeError(f"Server error mid-stream: {message}")

    return "".join(full_response)

async def main():
    thread_id = make_session_id()
    # history = [
    #     {"role": "system", "content": SYSTEM_PROMPT}
    # ]
    print(f"Chat with the model (each response is capped at ~{int(MAX_TOKENS * 0.75)} words). Ctrl+C or 'quit' to exit.\n")

    try:
        while True:
            user_input = await asyncio.to_thread(input, "You: ")
            if user_input.strip().lower() in ("quit", "exit"):
                print("\n\nExited, saving...")
                break

            # history.append({"role": "user", "content": user_input})

            print("Assistant: ", end="", flush=True)
            
            try:
                _full_response = await stream_chat_langchain(
                    message=[{"role": "user", "content": user_input}],
                    thread_id=thread_id
                )
            # except RuntimeError as e:
            #     print(f"\n[{e}]")
            #     # don't append a partial/failed assistant turn to history
            #     history.pop()  # remove the user turn too, so the failed exchange isn't half-recorded
            #     continue
            except RuntimeError as e:
                print(f"Server Error: {e}")
                continue

            print("\n")

            # history = [
            #     # {"role": "system", "content": SYSTEM_PROMPT}
            # ]
            # history.append({"role": "assistant", "content": _full_response})

    except KeyboardInterrupt:
        print("\n\nInterrupted, saving...")

    # finally:
    #     save_history(session_id, history)

if __name__ == "__main__":
    asyncio.run(main())