# `/chat` SSE Contract

## Request

`POST /chat`

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

Caller sends the full message history every request. Server is stateless between requests.

## Response

`Content-Type: text/event-stream`

Three named event types. Clients **must** branch on `event:`, not assume every event is a token.

### `event: token`
Emitted once per LLM token. Payload is JSON-wrapped — never raw text — specifically so a
token containing `\n` cannot be mistaken for the SSE blank-line delimiter.

```
event: token
data: {"token": "Hello"}

```

Client must `json.loads(data)` and read `["token"]`. Accumulate these to build the full response.

### `event: done`
Emitted exactly once, after the last `token` event, on normal completion. Signals "stop reading,
generation finished successfully."

```
event: done
data: {}

```

### `event: error`
Emitted if the LLM call raises an exception mid-stream (upstream API error, rate limit, etc.).
Replaces the `done` event for that request — **only one of `done` or `error` will appear, never
both.** No further `token` events follow an `error` event.

```
event: error
data: {"message": "<exception string>"}

```

## Required client behavior

1. Parse SSE blocks by `event:`/`data:` line pairs, not by raw text matching.
2. On `token`: parse JSON, extract `token`, display/accumulate it.
3. On `done`: stop reading, treat the accumulated text as the final assistant turn.
4. On `error`: stop reading, surface the message, do **not** treat partial accumulated tokens
   as a complete/saved assistant turn (decide explicitly whether to keep or discard the partial text).
5. Stream may end with no `done`/`error` event if the connection drops or server process dies —
   client should not assume one of these always arrives; guard with a connection-level timeout too.

## Notes for the client implementer

- The server checks `request.is_disconnected()` between tokens, so closing the client connection
  early (e.g. user hits Ctrl+C mid-stream) is the **correct** way to cancel generation — no
  separate "stop" message/event exists yet.
- `data` is always valid JSON, even for `done` (`{}`), so the client's parser path is uniform
  across all three event types — no special-casing an empty/non-JSON payload.