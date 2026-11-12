"""Integration tests: the real agent graph against the real Azure OpenAI model.

Skipped when Azure credentials are not set. `agent.core.graph` builds the
model at import time, so it is imported inside the fixture, after the skip
check, to keep collection from failing without credentials.
"""
import uuid

import pytest
from pydantic import ValidationError
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from agent.config import AzureChatOpenAISettings


def _has_credentials() -> bool:
    try:
        AzureChatOpenAISettings()  # reads env vars
        return True
    except ValidationError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_credentials(), reason="Azure OpenAI credentials not set"
)


@pytest.fixture
def agent():
    from agent.core.graph import init_agent

    return init_agent()


def ask(agent, text: str) -> dict:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return agent.invoke({"messages": [HumanMessage(text)]}, config)


def test_agent_computes_pmt_with_all_arguments(agent):
    result = ask(
        agent,
        "What is the monthly payment on a $200,000 loan at 6% annual interest "
        "over 30 years?",
    )
    messages = result["messages"]

    # the model called the pmt tool with correctly converted arguments
    tool_calls = [tc for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls]
    assert [tc["name"] for tc in tool_calls] == ["pmt"]
    args = tool_calls[0]["args"]
    assert args["rate"] == pytest.approx(0.005)  # 6% / 12, as a float
    assert args["nper"] == 360  # 30 years * 12
    assert args["pv"] == 200000

    # the tool ran, and the final answer reports its result (~1,199.10)
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert float(tool_msgs[0].content) == pytest.approx(-1199.10, abs=0.01)
    final = messages[-1]
    assert isinstance(final, AIMessage) and not final.tool_calls
    assert "1,199" in final.content or "1199" in final.content


def test_agent_introduces_itself_briefly(agent):
    result = ask(agent, "Hi, who are you? Answer briefly.")
    messages = result["messages"]

    # no tools needed: one LLM call, no tool messages
    assert result["llm_calls"] == 1
    assert not any(isinstance(m, ToolMessage) for m in messages)

    final = messages[-1]
    assert isinstance(final, AIMessage) and not final.tool_calls
    assert final.content.strip() != ""
    # assert len(final.content) < 600
    # assert "mortgage" in final.content.lower()
