## Langgraph Streaming (v2)
- To be differentiated from the new Event Streaming
- The built-in ReAct agent has the default state of `AgentState`. [link](https://reference.langchain.com/python/langchain/agents/middleware/types/AgentState)
    ```python
    class AgentState(TypedDict, Generic[ResponseT]):
        """State schema for the agent."""
        messages: Required[Annotated[list[AnyMessage], add_messages]]
        ...
    ```
- Every chunk is a `StreamPart` dict with a consistent shape — regardless of stream mode, number of modes, or subgraph settings:
    ```python
    {
    "type": "values" | "updates" | "messages" | "custom" | "checkpoints" | "tasks" | "debug",
    "ns": (),           # namespace tuple, populated for subgraph events
    "data": ...,        # the actual payload (type varies by stream mode)
    }
    ```
- Each stream mode has a corresponding TypedDict.
- `updates` mode - `UpdatesStreamPart`:
    - the `data` field contains `{"<node_name>": <state update by node>}`.
    - Because the agent's state is an instance of the class `AgentState` above, the state update essentially contains this `{"messages": [<list of messages>]}` (usually). The last update is added to the end of the message list.
    - Note that the state update is what the node function returned, i.e., the node's raw return dict pre-reducer. It is not the post-reducer, full graph state that contains all the messages from start to end.
    - For a vanilla agent, there are only 2 nodes with names of "model" and "tools". they have different raw values in the "messages" field as seen below. You can choose which part of the messages to return to the caller.

        ```python
        # model node completes — LLM generated a response
        chunk["data"] = {
            "model": {
                "messages": [AIMessage(
                    content="Let me search for that.",
                    tool_calls=[{"name": "search_web", "args": {"query": "..."}, "id": "call_abc"}]
                )]
            }
        }

        # tools node completes — tool(s) executed
        chunk["data"] = {
            "tools": {
                "messages": [ToolMessage(
                    content="Results for '...': [result1, result2]",
                    tool_call_id="call_abc",
                    name="search_web"
                )]
            }
        }
        ```