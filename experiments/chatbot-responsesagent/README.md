## Summary
- Build a terminal chatbot with server-client separation
- Agent framework: `langchain.agents.create_agent` (LangChain ≥1.0), executing on the LangGraph runtime
- Agent interface: MLflow `ResponsesAgent`, decorator-based (`@invoke()` / `@stream()`)
- Transport: async SSE (MLflow `AgentServer` ↔ client), fed by an inner async LangGraph stream (`agent.astream()`), see `sse_contract.md` .
- Agent short-term memory: server-side, in-memory, single-process checkpoint (`InMemorySaver`, module-level singleton keyed by `thread_id`)
- Agent long-term memory: none
- Agent tools: code-based, in-process (no MCP/remote execution)
- Session/thread addressing: single `conversation_id` → dual-purpose (MLflow trace session tag + LangGraph `thread_id`)
- Observability: MLflow (on Dababricks)

## Prerequisite
- Create an .env file using the sample .env file from the repo root.
- Populate the .env file with your own account values.
- Install and activate the conda environment in the `environment.yml` file: `conda env create -f environment.yml` -> `conda activate a3`

## Start the agent server
- run `python start_server.py`

## Start the client chatbot
- run `python client.py`

## The `start_server.py`
- A thin uvicorn-based server from mlflow `AgentServer`
- Needs to import the main module that defines the agent (`agent.py`) to register the functions with the server.
- Needs `app_import_string` with value of `module_name:attribute_name`
    - e.g., the module name is `start_server.py`, and the app object variable is `app`, then "start_server:app" is used.

## The `agent.py`
- The main file that constructs the agent.
- In this implementation, a simple LangChain ReAct agent with tools is used.
- To maintain agent short term memory, it is defined at the module level instead of initializing new agent instance inside the @stream function.
- Initializing of other backend connections to LLM provider and remote MLflow are also done at the module level.
- For simplicity, the following designs are made:
    - a single LLM client is used for both the agent "brain" and any LLM-based tools.
    - tools are code-based and stored in the same agent module.
    - system prompt is also stored as code in the same agent module.

## Client - Server Working:
- Client sends:
    ```python
    # client.py
    payload = {
        "input": message,       # list[{"role":..,"content":..}]
        "stream": True,
        "context": {"conversation_id": conversation_id},
    }
    ```
- Server sees:
    - "stream" = True: routes the POST body to the `@stream` decorated function.
    - "input": reshapes Responses input items to ChatCompletion dictionaries using `to_chat_completions_input`.
    - "context.conversation_id": the server-side utility function `get_session_id()` parses these exact named fields to get the conversation ID value, which is then assigned to the `session_id` variable in the `@stream` function. As this utility function is copied as-is from Databricks online example, no change to the required dict keys `context` and `conversation_id`.
    - the duo-purpose of the conversation ID:
        - purpose 1: mlflow uses this ID as the session ID to group client's chat requests - which belong to a trace (`mlflow.update_current_trace()`).
        - purpose 2: langchain uses this ID as the thread ID to maintain short-term memory in `agent.astream(..., config())`
    - as the client does not need to know the 2 purposes in the server, it only sees the friendly named variable `conversation_id`