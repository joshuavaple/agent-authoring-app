import asyncio
import httpx
import json
import os
import uuid
from datetime import datetime

HISTORY_DIR = "chat_logs"
SERVER_URL = "http://localhost:8000/chat"
MAX_TOKENS = 2000

def make_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"{timestamp}-{suffix}"

def save_history(session_id: str, history: list[dict]):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"\nHistory saved to {path}")

def parse_sse_blocks(raw_lines: str):
    """
    Group raw SSE lines into (event, data) blocks.
    """
    event = None
    data_line = None
    for line in raw_lines:
        if line.startswith("event: "):
            event = line[len("event: "):]
        if line.startswith("data: "):
            data_line = line[len("data: "):]
        elif line == "":
            # blank line = end of one SSE block
            if event is not None and data_line is not None:
                yield event, data_line
            event, data_line = None, None
        # ignoring any other line for now

async def stream_chat(history: list[dict]) -> str:
    """
    POST history to the server, print tokens as they arrive, return the
    full assistant response. Raises RuntimeError if the server reports
    an upstream error mid-stream.
    """
    full_response = []
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        async with http_client.stream(
            "POST",
            SERVER_URL,
            json={"messages": history}
        ) as response:
            raw_lines = response.aiter_lines()
            event = None
            data_line = None
            async for line in raw_lines:
                if line.startswith("event: "):
                    event = line[len("event: "):]
                elif line.startswith("data: "):
                    data_line = line[len("data: "):]
                elif line == "":
                    if event == "token" and data_line is not None:
                        payload = json.loads(data_line)
                        token = payload["token"]
                        print(token, end="", flush=True)
                        full_response.append(token)
                    elif event == "done":
                        break
                    elif event == "error":
                        payload = json.loads(data_line) if data_line else {}
                        message = payload.get("message", "unknown server error")
                        raise RuntimeError(f"Server error mid-stream: {message}")
                    event, data_line = None, None

    return "".join(full_response)

async def main():
    session_id = make_session_id()
    history = [
        {"role": "system", "content": "You are a witty, helpful assistant. Keep your answer brief, preferably less than 3 sentences, unless asked for details."}
    ]
    print(f"Chat with the model (each response is capped at ~{int(MAX_TOKENS * 0.75)} words). Ctrl+C or 'quit' to exit.\n")

    try:
        while True:
            user_input = await asyncio.to_thread(input, "You: ")
            if user_input.strip().lower() in ("quit", "exit"):
                print("\n\nExited, saving...")
                break

            history.append({"role": "user", "content": user_input})

            print("Assistant: ", end="", flush=True)
            
            try:
                full_response = await stream_chat(history)
            except RuntimeError as e:
                print(f"\n[{e}]")
                # don't append a partial/failed assistant turn to history
                history.pop()  # remove the user turn too, so the failed exchange isn't half-recorded
                continue

            print("\n")

            history.append({"role": "assistant", "content": full_response})

    except KeyboardInterrupt:
        print("\n\nInterrupted, saving...")

    finally:
        save_history(session_id, history)

if __name__ == "__main__":
    asyncio.run(main())