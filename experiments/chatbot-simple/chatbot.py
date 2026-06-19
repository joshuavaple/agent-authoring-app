# A simple chatbot operating in a while loop
# with short term memory and saving of each session to a file
# vanilla OpenAI API

import asyncio
import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv


load_dotenv(Path(__file__).parent.parent.parent / ".env")

MAX_TOKENS = 2000
HISTORY_DIR = "chat_logs"
deployment_name = "gpt-4o" # make sure you have this deployment in your MS Foundry.
endpoint = "https://local-rag-resource.services.ai.azure.com/" #Use your own MS Foundry endpoint name.

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

client = AsyncAzureOpenAI(
    azure_endpoint=endpoint,
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01"
)

async def llm_stream(messages: list[dict]):
    stream = await client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        max_tokens=MAX_TOKENS, # put a hard limit to control cost
        stream=True
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content
        if token:
            yield token



async def main():
    session_id = make_session_id()
    history = [
        {"role": "system", "content":"You are a witty, helpful assistant. Keep your answer brief, prefereably less than 3 sentences, unless asked for details."}
    ]
    print(f"Chat with the model (each response is capped at ~{int(MAX_TOKENS * 0.75)} words). Ctrl+C or 'quit' to exit.\n")
    
    try:
        while True:
            user_input = await asyncio.to_thread(input, "You: ")
            if user_input.strip().lower() in ("quit", "exit"):
                print("\n\nExited, saving...")
                break
            
            history.append({"role":"user", "content": user_input})

            print("Assistant: ", end="", flush=True)
            full_response = []

            # invoke the LLM streaming
            async for token in llm_stream(history):
                # output tokens to user:
                print(token, end="", flush=True)

                # collect the full response
                full_response.append(token)
            
            print("\n")

            # store the full response from LLM to chat history
            history.append({"role": "assistant", "content": "".join(full_response)})
    
    except KeyboardInterrupt:
        print("\n\nInterrupted, saving...")   
    
    finally:
        # The general principle: cleanup-that-must-always-happen belongs in finally 
        save_history(session_id, history)

if __name__ == "__main__":
    asyncio.run(main())