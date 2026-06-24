# A simple chatbot operating in a while loop
# with short term memory and saving of each session to a file
# vanilla OpenAI API

import asyncio
import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
import trafilatura
from langchain.messages import AIMessageChunk, AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

# load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv("../.env")
HISTORY_DIR = "chat_logs"
MAX_TOKENS = 2000
DEPLOYMENT_NAME = "gpt-4o"
API_VERSION = "2024-02-01"


# Utilities
# ================================================================
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


# ================================================================


# Agent model and harness
# ================================================================
# Use Foundry API key
client = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_deployment=DEPLOYMENT_NAME,
    api_version=API_VERSION,
)


async def afetch_link_content(url: str) -> str:
    """
    Runs trafilatura's blocking fetch+extract in a thread pool,
    freeing the event loop while the HTTP request is in flight.
    """

    def _blocking():  # uses 1 thread for both fetching and extracting
        downloaded = trafilatura.fetch_url(url)
        return trafilatura.extract(
            downloaded,
            output_format="markdown",
            include_links=True,
            include_tables=True,
            include_images=False,
        )

    return await asyncio.to_thread(_blocking)


@tool("afetch_express_entry_intro_tool")
async def afetch_express_entry_intro_tool() -> str:
    """
    Scrapes the IRCC website for the latest introduction on the Express Entry system.
    """
    return await afetch_link_content(
        url="https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry.html"
    )


@tool("afetch_express_entry_documents_tool")
async def afetch_express_entry_documents_tool() -> str:
    """
    Scrapes the IRCC website for the latest list of documents to create a profile Express Entry system.
    """
    return await afetch_link_content(
        url="https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/documents.html"
    )


SYSTEM_PROMPT = """
You are an expert in Canada Express Entry program to assist potential permanent residence applicants.
Keep your answer to the point without too much information, unless asked. 
You are equipped with tools. Always use them. Only answer based on the tool results. 
Do not make up answers if there is no information returned from the tools.
If you dont know the answer, or the question is not about your expertise, politely decline."""


atools = [afetch_express_entry_intro_tool, afetch_express_entry_documents_tool]
agent = create_agent(
    model=client,
    tools=atools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(), # CRUCIAL for short-term memory
)


async def llm_stream(user_input: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    async for chunk in agent.astream(
        input={"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        yield chunk


async def main():
    thread_id = make_session_id()  # use session ID as thread ID for now
    # history = [
    #     {"role": "system", "content":"You are a witty, helpful assistant. Keep your answer brief, prefereably less than 3 sentences, unless asked for details."}
    # ]
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
            async for chunk in llm_stream(user_input=user_input, thread_id=thread_id):
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

                            elif block["type"] == "tool_call_chunk":
                                # stream partial JSON for tool args
                                print(f"[tool_chunk: {block['args']}]", end="")

                elif (
                    chunk_type == "updates"
                ):  # if the chunk is from the node upon completion
                    # Coarse-grained: full state after a node completes
                    for node_name, state_update in chunk["data"].items():
                        if node_name == "model":
                            msg = state_update["messages"][-1]
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                print(
                                    f"\n[TOOL CALLS]: {[tc['name'] for tc in msg.tool_calls]}"
                                )

                        elif node_name == "tools":
                            msg = state_update["messages"][-1]
                            if isinstance(msg, ToolMessage):
                                # print(f"\n[TOOL RESULT from {msg.name} (truncated)]: {msg.content[:100]} \n...")
                                print(
                                    f"\n[TOOL RESULT from {msg.name}]: {msg.content} \n..."
                                )

                        elif node_name == "__interupt__":
                            # Human in the loop breakpoints
                            print(f"\n[INTERRUPT]: {state_update}")

            print("\n")
            # =====================================================================

    except KeyboardInterrupt:
        print("\n\nInterrupted, saving...")

    # finally:
    #     # The general principle: cleanup-that-must-always-happen belongs in finally
    #     save_history(session_id, history)


if __name__ == "__main__":
    asyncio.run(main())
