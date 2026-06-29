## Architecture Changes: Naive OpenAI → LangChain Agent

### 1. LLM Client

| Naive | LangChain |
|---|---|
| `AsyncAzureOpenAI` (OpenAI SDK) | `AzureChatOpenAI` (LangChain wrapper) |
| Direct async HTTP client | LangChain abstraction that handles agent execution loop |

LangChain needs its own wrapper so it can control the LLM internally (e.g., for tool call / re-prompt cycles) — you can't pass a raw OpenAI client into the agent.

---

### 2. History Management: Client-owned → Server-owned

**Naive:** Client sends the full `messages: list[dict]` history on every request. The server is stateless — it just forwards whatever the client sends.

**LangChain:** Client sends only the **current user message** (still as `messages: list[dict]`, but just the new turn). The server holds state via `InMemorySaver` (LangGraph's checkpointer). The `thread_id` is the key that routes each request to the correct in-memory conversation history.

```
# Naive: client owns all history
POST /chat  →  messages: [sys, user1, ai1, user2, ai2, user3]

# LangChain: server owns history, client just sends new turn + thread key
POST /chat  →  { messages: [user3], thread_id: "abc-123" }
```

The `ChatRequest` model gains `thread_id: str` to support this.

---

### 3. System Prompt: Client-side → Server-side

**Naive:** The first message in the `messages` list sent by the client is the system prompt — it's baked into the conversation history the client manages.

**LangChain:** `SYSTEM_PROMPT` is defined on the server and injected into the agent at construction time via `create_agent(..., system_prompt=SYSTEM_PROMPT)`. The client never sees or sends it.

This is the natural outcome of server-owned history: once the server controls state, the system prompt belongs there too.

---

### 4. Token Extraction: SDK delta → LangChain v2 chunk unwrapping

**Naive:**
```python
token = chunk.choices[0].delta.content   # raw OpenAI chunk
```

**LangChain (stream_mode v2):**
```python
# chunk is {"type": "messages"|"updates", "data": ...}
chunk_type = chunk["type"]
if chunk_type == "messages":
    token, _metadata = chunk["data"]          # unpack (msg, metadata) tuple
    if isinstance(token, AIMessageChunk):
        for block in token.content_blocks:    # iterate content blocks
            if block["type"] == "text":
                yield block["text"]
```

LangChain's v2 streaming emits two event types — `messages` (token-level) and `updates` (node-level state). You filter to `messages` and then check the chunk is `AIMessageChunk` (not a tool result or internal state object) before extracting text from `content_blocks`.

---

### 5. Tool Support (new capability)

The LangChain version adds tools (`get_weather`, `generate_itinerary`) declared with `@tool()` and passed to `create_agent`. The agent runs an internal ReAct-style loop — it can call tools, receive results, and continue reasoning — all before yielding tokens to your SSE stream. The naive server has no concept of tools.

---

### 6. SSE Protocol & `event_stream` — Unchanged

`sse_event()` and the `event_stream` wrapper are identical in structure. The SSE wire format (`event: token`, `event: done`, `event: error`) is preserved — the client doesn't need to change at all. This was the deliberate design choice: swap the internals, keep the contract.

---

## MLflow Tracing
- The server holds the authentication information to the MLflow tracking server - the U2M profile name (or M2M service principal), the experiment name
- The client sends the thread ID as usual to the server for short-term memory, and this thread ID is being automatically mapped to MLflow's session ID (mlflow.trace.session), which appears in the "Session" column of the MLflow tracing UI.

## Reference: Langgraph Streaming (v2)
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