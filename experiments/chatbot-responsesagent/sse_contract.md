# SSE Contract — MLflow `AgentServer` (ResponsesAgent)

Supersedes the hand-rolled `event:`-typed contract (`token`/`done`/`error`). This
server's SSE shape is dictated by `AgentServer._generate` (MLflow source), not
by us — it's a narrow subset of SSE matching what OpenAI's own Responses API
streaming wire format looks like, so `openai`-SDK-shaped clients work against
it without translation.

## Endpoint

```
POST /responses
Content-Type: application/json
```

## Request body

```json
{
  "input": [
    {"role": "user", "content": "..."}
  ],
  "stream": true,
  "context": {
    "conversation_id": "<thread_id>"
  }
}
```

- `input` — OpenAI-role/content message list. Maps to LangGraph's `messages` via
  `to_chat_completions_input`.
- `stream` — required for the SSE path; `AgentServer` pops this key (`STREAM_KEY`)
  before dispatch to decide `_handle_stream_request` vs `_handle_invoke_request`.
- `context.conversation_id` — **this is the thread/session id.** Resolved
  server-side by `get_session_id(request)`, which checks `context.conversation_id`
  first, falling back to `custom_inputs["session_id"]` if absent. This value is
  threaded into `agent.astream(..., config={"configurable": {"thread_id": ...}})`
  — omit it and the checkpointer raises `ValueError: Checkpointer requires ...
  thread_id`.

## Response frames

Every frame is `data:`-only — **no `event:` field**. SSE clients see every
frame under the default event type (`"message"`); discrimination happens on a
`type` key *inside* the JSON payload, not the SSE envelope.

```
data: {"type": "response.output_item.added", "item": {...}}\n\n
data: {"type": "response.output_text.delta", "delta": "Hel", "item_id": "..."}\n\n
data: {"type": "response.output_text.delta", "delta": "lo", "item_id": "..."}\n\n
data: {"type": "response.output_item.done", "item": {...}}\n\n
data: [DONE]\n\n
```

| Frame | When | Client handling |
|---|---|---|
| `response.output_item.added` | New output item (message, tool call, reasoning) starts | Usually ignorable for a plain terminal client |
| `response.output_text.delta` | Streamed token(s) of assistant text | `chunk["delta"]` — append/print |
| `response.output_item.done` | An item (message or resolved tool call) completes | Full item available; `invoke_handler` collects these for non-streaming responses |
| `{"error": "..."}` | Exception mid-stream (`_generate`'s except block) | **Not a typed event** — no `"type"` key. Check for `"error"` key before dispatching on `type`. Always immediately followed by `[DONE]`. |
| `[DONE]` | Terminal sentinel | Literal string, not JSON — check `sse.data == "[DONE]"` before `json.loads`. Stream ends here regardless of success/failure. |

## Client parse loop (reference)

```python
async for sse in event_source.aiter_sse():
    if sse.data == "[DONE]":
        break
    chunk = json.loads(sse.data)
    if "error" in chunk:
        raise RuntimeError(f"Server error mid-stream: {chunk['error']}")
    if chunk.get("type") == "response.output_text.delta":
        token = chunk["delta"]
        print(token, end="", flush=True)
```

## What we deliberately gave up vs. the old contract

- **No `event:`-level typing.** Old contract: `sse.event == "token"`. New:
  `chunk["type"] == "response.output_text.delta"`. Type discriminator moved
  from SSE protocol layer to JSON payload layer.
- **No SSE `id:` / resume support.** Reconnection after a drop means re-POSTing
  `/responses` from scratch — replay safety depends on the agent being
  idempotent/checkpointed via `thread_id`, not on SSE-native resumption.
- **Tool calls are pre-resolved, not interactive.** `response.output_item.done`
  items for tool calls represent *already-executed* history (server ran the
  LangGraph tool loop internally). A downstream OpenAI-SDK-shaped client should
  **not** attempt to re-execute them — there's no pending work to hand back.

## Known process-scope caveat (not a wire-contract issue, but adjacent)

`thread_id`/`conversation_id` continuity depends on `CHECKPOINTER` (`InMemorySaver`)
being a **module-level singleton**, not reconstructed per request. Fine for
single-worker local dev; breaks silently across `--workers > 1` (separate
processes, separate memory) — revisit with a shared backing store
(`AsyncSqliteSaver` or similar) before scaling out.