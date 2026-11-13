from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from agent.core.graph import init_agent
from agent.serving.translate_stateful import (
    request_to_graph_input,
    get_thread_id,
    STREAM_MODES,
    process_agent_astream_events,
)


def create_graph_stream(agent, request_content):
    # Convert from mlflow request to langgraph input
    request_mlflow = ResponsesAgentRequest(**request_content)
    graph_input = request_to_graph_input(request_mlflow)
    thread_id = get_thread_id(request_mlflow)

    # Create the stream object
    langgraph_stream = agent.astream(
        input=graph_input,
        config={"configurable": {"thread_id": thread_id}},
        stream_mode=STREAM_MODES,
    )
    return langgraph_stream


async def print_output(agent, request_content, streaming=True):
    langgraph_stream = create_graph_stream(agent, request_content)

    # For streaming mode, print the events as they are generated
    if streaming:
        async for event in process_agent_astream_events(langgraph_stream):
            # Text deltas print the chunk
            if event.type == "response.output_text.delta":
                print(event.type, "\t", event.delta)
            # every other event type prints its item (tool calls, tool outputs, final messages)
            else:
                print(event.type, "\t", getattr(event, "item", None))
    
    # For non-streaming mode, collect all events and get the final ".done"
    else:
        events = [e async for e in process_agent_astream_events(langgraph_stream)]
        items = [e.item for e in events if e.type == "response.output_item.done"]
        response = ResponsesAgentResponse(output=items)
        print(response.model_dump_json(indent=2))


async def main():
    agent = init_agent()

    request_1 = {
        "input": [
            {
                "role": "user",
                "content": "What's the monthly payment on a $500k mortgage at 5% over 25 years?",
            }
        ],
        "context": {"conversation_id": "123", "user_id": "456"},
    }

    request_2 = {
        "input": [
            {
                "role": "user",
                "content": "What's the amount you just calculated?",
            }
        ],
        "context": {"conversation_id": "123", "user_id": "456"},
    }

    await print_output(agent, request_1, streaming=True)
    print("=" * 20)
    await print_output(agent, request_2, streaming=False)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
