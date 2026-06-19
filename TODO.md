## To-dos
- [ ] Handle exception `422 Unprocessable Entity` when the client code sends malformed messages to the SSE server.

## Issues
### [OPEN] KeyboardInterrupt exception is still raised
- Date: 2026-06-20
- Symptom: when using Ctl+C in the middle of LLM token streaming, `KeyboardInterrupt` was raised, despite this exception being handled in the try-except loop by printing to the terminal.
- Context: `experiments/chatbot-sse/client.py/main()`


