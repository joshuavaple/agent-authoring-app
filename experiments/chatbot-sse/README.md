## Intro
Architecture
- The `server.py`: 
    - async generator wrapper around the `openai.AsyncAzureOpenAI` client class to stream only the assistant's text tokens to the user.
    - FastAPI implementation of SSE protocol to enable long-running HTTP connection and continuous streaming to the user from the server. 
    - This is a BFF with 2 purposes: an SSE client to the LLM service (MS Foundry) and an SSE server to the downstream client — it proxies the token stream through while managing state on both sides of the request.
    - The SSE contract is stored in the `SSE-CONTRACT.md` file for client reference.
    
- The `client.py`: 
    - A simple terminal program
    - Manages chat messages short-term memory
    - Logs each session chat history to a local folder.
    - Uses httpx-sse to handle SSE events reliably.

## Startup
- cd to the chatbot/ folder
- In a terminal, start the SSE server by running `uvicorn server:app --reload`
- In another terminal, start the terminal client by running `python client.py`. You can start as many clients as you want.
