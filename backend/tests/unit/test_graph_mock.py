"""Unit tests for the hand-built ReAct graph in agent.core.graph.

No network: the real Azure model is replaced by a scripted fake. Importing
`agent.core.graph` calls get_llm() at module level, so dummy Azure settings
are set before the import (constructing the client makes no HTTP call).
"""
import asyncio
import os

for _k, _v in {
    "AZURE_OPENAI_ENDPOINT": "https://example.invalid",
    "AZURE_OPENAI_API_KEY": "dummy",
    "AZURE_OPENAI_DEPLOYMENT": "dummy",
    "AZURE_OPENAI_API_VERSION": "2024-02-01",
}.items():
    os.environ.setdefault(_k, _v)

import pytest
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END

from agent.core import graph as g


class FakeModel:
    """Stands in for model_with_tools: returns scripted replies, records inputs."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def pmt_call(call_id="call_1"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "pmt",
                "args": {"rate": 0.005, "nper": 360, "pv": 200000},
                "id": call_id,
            }
        ],
    )


@pytest.fixture
def fake_model(monkeypatch):
    def install(replies):
        fake = FakeModel(replies)
        monkeypatch.setattr(g, "model_with_tools", fake)
        return fake

    return install


@pytest.fixture
def agent():
    # return asyncio.run(g.init_agent())
    return g.init_agent()


def cfg(thread_id):
    return {"configurable": {"thread_id": thread_id}}


# --- should_continue -------------------------------------------------------

def test_should_continue_routes_to_tools_when_tool_calls():
    assert g.should_continue({"messages": [pmt_call()]}) == "tool_node"


def test_should_continue_ends_without_tool_calls():
    assert g.should_continue({"messages": [AIMessage(content="hi")]}) == END


# --- tool_node -------------------------------------------------------------

def test_tool_node_returns_tool_message_with_matching_id():
    out = g.tool_node({"messages": [pmt_call("abc")]})

    (msg,) = out["messages"]
    assert isinstance(msg, ToolMessage)
    assert msg.tool_call_id == "abc"
    assert float(msg.content) == pytest.approx(-1199.10, abs=0.01)


def test_tool_node_handles_parallel_tool_calls():
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "pmt", "args": {"rate": 0.005, "nper": 360, "pv": 100000}, "id": "a"},
            {"name": "pmt", "args": {"rate": 0.005, "nper": 180, "pv": 100000}, "id": "b"},
        ],
    )
    out = g.tool_node({"messages": [ai]})
    assert [m.tool_call_id for m in out["messages"]] == ["a", "b"]


# --- llm_call --------------------------------------------------------------

def test_llm_call_prepends_system_prompt_and_counts(fake_model):
    fake = fake_model([AIMessage(content="ok")])
    user = HumanMessage("hello")

    out = g.llm_call({"messages": [user]})

    sent = fake.calls[0]
    assert isinstance(sent[0], SystemMessage)
    assert sent[0].content == g.SYSTEM_PROMPT
    assert sent[1:] == [user]
    assert out["llm_calls"] == 1
    assert out["messages"][0].content == "ok"


def test_llm_call_increments_existing_count(fake_model):
    fake_model([AIMessage(content="ok")])
    out = g.llm_call({"messages": [HumanMessage("hi")], "llm_calls": 3})
    assert out["llm_calls"] == 4


# --- full graph ------------------------------------------------------------

def test_graph_without_tool_call_ends_after_one_llm_call(agent, fake_model):
    fake_model([AIMessage(content="I only do mortgages.")])

    result = agent.invoke({"messages": [HumanMessage("weather?")]}, cfg("t1"))

    assert result["llm_calls"] == 1
    assert [type(m) for m in result["messages"]] == [HumanMessage, AIMessage]


def test_graph_tool_loop_runs_tool_then_answers(agent, fake_model):
    fake = fake_model([pmt_call(), AIMessage(content="About $1,199 per month.")])

    result = agent.invoke({"messages": [HumanMessage("payment?")]}, cfg("t1"))

    assert [type(m) for m in result["messages"]] == [
        HumanMessage, AIMessage, ToolMessage, AIMessage,
    ]
    assert result["llm_calls"] == 2
    # the 2nd model call must have seen the tool result
    assert isinstance(fake.calls[1][-1], ToolMessage)


def test_checkpointer_remembers_within_a_thread(agent, fake_model):
    fake = fake_model([AIMessage(content="first"), AIMessage(content="second")])

    agent.invoke({"messages": [HumanMessage("my loan is 200k")]}, cfg("t1"))
    agent.invoke({"messages": [HumanMessage("and the rate?")]}, cfg("t1"))

    second_input = fake.calls[1]
    contents = [m.content for m in second_input]
    assert "my loan is 200k" in contents
    assert "first" in contents


def test_checkpointer_isolates_threads(agent, fake_model):
    fake = fake_model([AIMessage(content="first"), AIMessage(content="second")])

    agent.invoke({"messages": [HumanMessage("my loan is 200k")]}, cfg("t1"))
    agent.invoke({"messages": [HumanMessage("hello")]}, cfg("t2"))

    contents = [m.content for m in fake.calls[1]]
    assert "my loan is 200k" not in contents
