# A. QUICK START
## Intro
Architecture
- The `server.py`: 
    - async generator wrapper around the `openai.AsyncAzureOpenAI` client class to stream only the assistant's text tokens to the user.
    - FastAPI implementation of SSE protocol to enable long-running HTTP connection and continuous streaming to the user from the server. 
    - This is a BFF with 2 purposes: an SSE client to the LLM service (MS Foundry) and an SSE server to the downstream client — it proxies the token stream through while managing state on both sides of the request.
    - The SSE contract is stored in the `SSE-CONTRACT.md` file for client reference.
    - To end the server, use Ctrl+C

- The `client.py`: 
    - A simple terminal program
    - Manages chat messages short-term memory
    - Logs each session chat history to a local folder upon exit.
    - Uses httpx-sse to handle SSE events reliably.

## Setup
- In the repo root folder, use the sample.env template to create a new .env file and store your API key.
- For simplicity, other configs for the client class is hard-coded in the server codes.
- Install and activate the conda environment in the `environment.yml` file: `conda env create -f environment.yml` -> `conda activate a3`

## Startup
- cd to the `chatbot-sse/` folder
- In a terminal, start the SSE server by running `uvicorn server:app --reload`
- In another terminal, start the terminal client by running `python client.py`
- Start chatting with your bot!
- You can have concurrent client sessions chatting with the same server to test the server async process.
- To end a client, use Ctrl+C or type "exit" or "quit".

-----
# B. PROJECT EXPLAINER

## Objective

Build a working chatbot — streaming, multi-turn, persisted to disk — without investing in
any frontend UI. The terminal *is* the UI. This deliberately isolates the part of an LLM
chat app that's genuinely hard to get right (async streaming, stateless server design,
SSE protocol correctness) from the part that's a separate skillset entirely (browser/React
UI work). The result is a small, correct backbone that a real frontend can be bolted onto
later without changing the server at all.

## End Product

Two processes, one Azure OpenAI deployment:

- `server.py` — a FastAPI app (`uvicorn server:app --reload`), holds the LLM credentials,
  streams tokens back over SSE.
- `client.py` — a terminal script (`python client.py`), owns the conversation history,
  renders tokens as they arrive, saves each session to a JSON file on exit.

Multiple `client.py` processes can talk to one `server.py` concurrently — used during
development to sanity-check the server's async handling under concurrent load.

---

## Architecture

### The pieces and how they relate

```
Terminal (client.py)  ←—SSE/HTTP—→  FastAPI server (server.py)  ←—HTTPS—→  Azure OpenAI
   owns history                       stateless, no memory              hosted LLM
```

**LLM provider — Azure OpenAI.** Accessed via `AsyncAzureOpenAI`, the async variant of OpenAI's SDK. 
"Async" here means I/O-bound waiting (the network round-trip to Azure) doesn't block other work on the same process 
— relevant once a server handles multiple users at once.

`stream=True` turns the API call into an async generator yielding tokens as the model produces them, 
instead of waiting for the full response.

**Server — FastAPI, stateless.** Every `/chat` request carries the *entire* conversation so far; 
the server holds nothing in memory between requests. 
This is the single most important design decision in the whole project (see Decision Log below) — 
it's what makes the server horizontally scalable and  frontend-agnostic. 
This implementation reflects how LLM chat models work - inherently, they do not know your chat history. 
The illusion of short term memory during a chat session is achieved by 
sending the new message appended with previous messages, and clever client-side implementation. 
For simplicity, this server only yield the actual incremental chunks from the LLM to the client 
and discard other metadata (see `llm_stream()`).

**Client — plain Python script, stateful.** Owns the `history` list, the only place
conversation state lives between turns. Talks to the server purely over HTTP; has no special
relationship to it beyond knowing the contract.

**Transport — Server-Sent Events (SSE).** A text-based streaming format over plain HTTP,
not a new protocol. Chosen over WebSockets because the data only flows one direction
(server → client) — no mid-stream input from the user is needed, and SSE survives ordinary
HTTP infrastructure (proxies, load balancers) better than WebSockets do. The wire contract
(`event: token|done|error`, JSON-wrapped payloads) is documented separately in
`SSE-CONTRACT.md` — treat that as the source of truth for anyone implementing a new client.

### Key decision points

| Decision | Why |
|---|---|
| Server is stateless; client resends full history each turn | Any server instance can serve any request — no session pinned to one process. Required for horizontal scaling and for supporting multiple client types (terminal today, browser later) against the same server unchanged. |
| SSE over WebSockets | Token streaming is one-directional. SSE works over plain HTTP (easier to deploy behind standard infra), at the cost of not supporting mid-stream client→server messages (e.g. a "stop generating" button) — acceptable for this scope. |
| Tokens wrapped in JSON, not sent raw | A raw token containing `\n` would be indistinguishable from the SSE blank-line event delimiter, corrupting the stream. JSON-wrapping (`{"token": "..."}`) sidesteps this entirely. |
| Named SSE events (`token` / `done` / `error`) instead of one generic event | Lets the client branch on what kind of message it received, instead of inferring meaning from payload shape. `error` explicitly replaces `done` for that request — never both. |
| `request.is_disconnected()` checked per token | If the client closes its connection (Ctrl+C, crash), the server stops calling the LLM rather than burning tokens/cost generating a response nobody will read. |
| `/title` is a separate non-streaming endpoint | Title generation is a one-shot summarization, not a token-by-token UX — SSE framing would be pure overhead for it. Different problem shape, different endpoint shape. |
| History saved as JSON, not flat text | Preserves structure (`role`/`content` per turn) so it can be reloaded directly into a future session, replayed, or analyzed — a flattened transcript string would need re-parsing to recover that structure. |

---

## Server Design (`server.py`)

Two responsibilities, kept in separate functions on purpose:

- **`llm_stream`** — the only code that talks to Azure OpenAI. Yields plain text tokens.
  If you ever swapped providers (Anthropic, a different Azure deployment), this is the only
  function that would need to change.
- **`event_stream` / `sse_event`** — the only code that knows about the SSE wire format.
  Takes plain tokens in, emits correctly-framed `event:`/`data:` blocks out. This is what
  enforces the contract documented in `SSE-CONTRACT.md`.

Splitting these matters because they're two genuinely different concerns: one is "how do I
get text out of an LLM," the other is "how do I deliver text over HTTP in a streaming-safe
way." Mixing them into one function would make either one harder to change without touching
the other.

Error handling: the `try/except` in `event_stream` distinguishes a server-initiated shutdown
(`asyncio.CancelledError`, re-raised — the ASGI server needs to see this to clean up properly)
from an upstream failure (any other exception, converted into an `error` SSE event so the
client gets a typed, parseable signal instead of the connection just dying silently).

---

## Client Design (`client.py`)

Three responsibilities:

- **History management** — a single list, system prompt at index 0, growing by one user
  turn + one assistant turn per exchange. This *is* short-term memory; nothing more elaborate
  is happening here.
- **Streaming/parsing (`stream_chat`)** — uses `httpx-sse` (`aconnect_sse` +
  `event_source.aiter_sse()`) rather than hand-parsing raw lines. The library already knows
  how to detect SSE block boundaries; the only app-specific code left is branching on
  `sse.event` and unwrapping `sse.json()`. (An earlier hand-rolled version, `stream_chat_manual`,
  is kept in the file for reference/comparison — not used by `main()`.)
- **Session lifecycle** — `try/except KeyboardInterrupt/finally` ensures `save_history` runs
  on *every* exit path (normal `quit`, Ctrl+C, or an unhandled error), not just the happy path.
  On a server-reported `error` event, the failed exchange is fully rolled back
  (`history.pop()` removes the orphaned user turn) rather than left half-recorded.

---

## Path Forward — What a Full-Fledged Chatbot Would Still Need

This project deliberately stopped short of several things. Listed roughly in the order
they'd start to matter:

- **A real frontend.** The server doesn't change for this — it's already a generic SSE
  endpoint. A browser client would replace `client.py`'s role, using `EventSource`/`fetch`
  instead of `httpx-sse`, and would not hold `history` in JS memory long-term (see next point).
- **Server-side persistent storage (DB) for history**, replacing client-held lists. Needed
  the moment there's a browser client — you don't want conversation state living only in a
  tab that can be closed. This also enables multi-device access to the same conversation.
- **Authentication / `user_id` resolution**, so history can be scoped per person instead of
  per process.
- **Long-term memory** — summarizing/condensing older turns instead of resending full history
  forever. Sketched conceptually earlier in this project's design discussion (rolling summary
  buffer, ideally computed offline/in a background job rather than synchronously at session
  start) but not implemented here.
- **A stop-generation control.** Currently the only way to cancel a response is closing the
  connection (Ctrl+C). A real "stop" button needs a channel for the client to signal the
  server mid-stream — this is the point where SSE's one-directional limitation would actually
  start to bite, and where WebSockets (or a side-channel cancellation endpoint) would become
  relevant.
- **Multi-user concurrency at scale** — this project's "two terminals talking to one server"
  test confirms basic async correctness, but not real load behavior (connection pool limits,
  rate limiting against Azure, backpressure).
- **Automated tests** — `pytest` cases for server request validation and client error-handling
  paths (e.g. malformed payload → 422, server `error` event → client raises) were discussed
  but not yet written into a `tests/` suite.
- **Fixing the relative import** between `tests/` and the root-level modules, deferred to a
  separate session.