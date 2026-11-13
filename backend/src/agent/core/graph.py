from agent.adapters.llm import get_llm
from agent.tools.tools import (
    pmt,
    calculate_monthly_rate,
    amortization_schedule,
    fetch_canada_property_tax_rate,
    estimate_canada_property_tax_rate,
    property_tax_amount,
    estimate_canada_utilities_amount,
    gross_debt_service
)

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from langchain.messages import SystemMessage, ToolMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver


# 1. Define tools and model
TOOLS = [
    pmt,
    calculate_monthly_rate,
    amortization_schedule,
    fetch_canada_property_tax_rate,
    estimate_canada_property_tax_rate,
    property_tax_amount,
    estimate_canada_utilities_amount,
    gross_debt_service]

tools_by_name = {tool.name: tool for tool in TOOLS}


model = get_llm()
model_with_tools = model.bind_tools(TOOLS)

# 2. Define state
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# 3. Define LLM node
SYSTEM_PROMPT = """
You are an expert in mortgage calculation. Only base your answer on tool execution results.
If you don't know the answer or the question is not relevant to your role, politely decline the user.
Be careful when handling rates. Note that the tax rate given by the user can be in percentage. In this case, convert it to float.
E.g., given rate is 5%, rate for calculation is 0.05
Be careful when handling interest rate, differentiate annual rate, which is usually given, and the rate to use in monthly calculation, which is by dividing annual rate by 12.
If the user does not give a mortgage interest rate, use the rate estimation tool. Present the estimated range with its
benchmark and date, say it is indicative and not a lender offer, and confirm the rate with the user before calculating.
The estimation tool returns percentages (4.5 means 4.5%), so convert to a float for calculations.
"""

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content= SYSTEM_PROMPT
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


# 4. Define tool node
def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# 5. Define end logic
# The conditional edge function routes to the tool node or to the end, 
# depending on whether the LLM made a tool call.
def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


# 6. Build and compile the agent
def init_agent() -> CompiledStateGraph:
    # Build workflow
    agent_builder = StateGraph(MessagesState)

    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")

    # Add checkpointer
    checkpointer = InMemorySaver()

    # Compile the agent
    agent = agent_builder.compile(checkpointer=checkpointer)
    return agent
