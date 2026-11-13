"""
Demo of AgentRunner: one agent, one conversation, streaming then non-streaming.

Run from backend/:  uv run python scripts/demo_runner.py
"""

import asyncio

from agent.core.graph import init_agent
from agent.serving.runner import AgentRunner


def make_request(text: str, conversation_id: str = "123") -> dict:
    return {
        "input": [{"role": "user", "content": text}],
        "context": {"conversation_id": conversation_id, "user_id": "456"},
    }


async def main():
    # Build the agent once so both turns share the same checkpointer (conversation memory).
    runner = AgentRunner(init_agent())

    # Turn 1: streaming. Print each event as it arrives.
    request_1 = make_request("What's the monthly payment on a $500k mortgage at 5% over 25 years?")
    async for event in runner.run_stream(request_1):
        if event.type == "response.output_text.delta":
            print(event.type, "\t", event.delta)
        else:
            print(event.type, "\t", getattr(event, "item", None))

    print("=" * 20)

    # Turn 2: non-streaming, same conversation. Returns one complete response.
    request_2 = make_request("What's the amount you just calculated?")
    response = await runner.run(request_2)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
