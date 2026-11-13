"""
This layer sits between the LangGraph agent and the MLflow serving infrastructure.
    Request ---> MLflow ---> LangGraph ---> MLflow ---> Response
It translates between the MLflow ResponsesAgent schema and the LangGraph agent.
Two directions:
  request side -> MLFflow to LangGraph: ResponsesAgentRequest -> (thread_id, graph input)
  response side -> LangGraph to MLflow: LangGraph astream chunks -> ResponsesAgentStreamEvent

ResponsesAgentRequest:
    input: list[Message(content, role)]
    context: ChatContext(conversational_id, user_id)

ResponsesAgentResponse(Response):


The graph is expected to be streamed with stream_mode=["updates", "messages"]:
  - "messages" chunks carry token deltas        -> response.output_text.delta
  - "updates"  chunks carry completed messages   -> response.output_item.done
    (assistant text, function_call, function_call_output)
"""

import json
import logging
from typing import Any, AsyncGenerator, AsyncIterator

from langchain.messages import AIMessageChunk, HumanMessage, ToolMessage
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentStreamEvent,
    create_text_delta,
    output_to_responses_items_stream,
)

logger = logging.getLogger(__name__)

# Pass this to graph.astream(stream_mode=...) so the chunks match what we translate below.
STREAM_MODES = ["updates", "messages"]


# Request side
# =========================================================================================================
def get_thread_id(request: ResponsesAgentRequest) -> str | None:
    """Conversation id used as the LangGraph checkpointer thread_id."""
    if request.context and request.context.conversation_id:
        return request.context.conversation_id
    if isinstance(request.custom_inputs, dict):
        return request.custom_inputs.get("session_id")
    return None


def _input_item_text(item: Any) -> str:
    """Extract plain text from a Responses input message item."""
    content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if isinstance(part, dict):
            text = part.get("text")
        else:
            text = getattr(part, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def _input_item_role(item: Any) -> str | None:
    """
    Extract the role from a Responses input message item.
    """
    return item.get("role") if isinstance(item, dict) else getattr(item, "role", None)


def request_to_graph_input(request: ResponsesAgentRequest) -> dict[str, Any]:
    """
    Buils the user's human input from MLFlow request for a STATEFUL agent.

    The checkpointer already holds the conversation, and the graph state appends messages
    (operator.add). Only forward the latest user message, as sending the whole request history would duplicate it every turn.
    History lives behind the thread_id.

    Args:
        request: the request that follows MLflow ResponsesAgentRequest schema. The input follows oldest to newest order.

    Returns:
        a human message dictionary that follows LangGraph schema of {"messages": [HumanMessage(content=...)]}
    
    """
    for item in reversed(request.input):
        if _input_item_role(item) == "user":
            return {"messages": [HumanMessage(content=_input_item_text(item))]}
    raise ValueError("ResponsesAgentRequest.input contains no user message")


# Response side
# =========================================================================================================
def _ai_item_text(chunk: AIMessageChunk) -> str:
    """Text of a LangGraph streamed AI message chunk.
    Content is a str, or a list of parts for some providers.
    See LangGraph BaseMesssage.
    """
    content = chunk.content
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _stringify_tool_content(msg: Any) -> Any:
    """Responses function_call_output must be a string; JSON-encode structured tool results."""
    if isinstance(msg, ToolMessage) and not isinstance(msg.content, str):
        return msg.model_copy(update={"content": json.dumps(msg.content)})
    return msg


async def process_agent_astream_events(
    graph_stream: AsyncIterator[Any],
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """
    Convert a LangGraph `astream(stream_mode=STREAM_MODES)` iterator into Responses events.

    Args:
        graph_stream: the async iterator returned by compiled_graph.astream(...)
    """
    async for mode, payload in graph_stream:
        if mode == "updates":
            # payload: {node_name: {"messages": [...], ...} | None}
            for node_update in payload.values():
                messages = (node_update or {}).get("messages") or []
                if not messages:
                    continue
                messages = [_stringify_tool_content(m) for m in messages]
                for event in output_to_responses_items_stream(messages):
                    yield event
        elif mode == "messages":
            # payload: (message_chunk, metadata)
            # A malformed chunk should not kill the whole response stream: log and keep going.
            try:
                chunk = payload[0]
                if isinstance(chunk, AIMessageChunk) and (text := _ai_item_text(chunk)):
                    yield ResponsesAgentStreamEvent(
                        **create_text_delta(delta=text, item_id=chunk.id)
                    )
            except Exception:
                logger.exception("Error processing agent stream event")
        else:
            logger.debug("Ignoring unexpected stream mode %r", mode)