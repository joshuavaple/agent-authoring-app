"""
A demo script to test out the overall agent.
"""

if __name__ == "__main__":
    from langchain.messages import HumanMessage
    from agent.core.graph import init_agent

    agent = init_agent()
    config = {"configurable": {"thread_id": "demo"}}

    result = agent.invoke({"messages": [HumanMessage("Hi, who are you?")]}, config)
    result["messages"][-1].pretty_print()