"""
The orchestrator layer.
It coordinates the steps 
    - translating the MLflow request to LangGraph input, 
    - run the graph, 
    - translate the graph output to MLflow response

"""

from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from agent.serving.translate_stateful import (
    request_to_graph_input,
    get_thread_id,
    STREAM_MODES,
    process_agent_astream_events,
)

class AgentRunner:
    def __init__(self, agent):
        self.agent = agent

    def _create_graph_stream(self, request):  # your create_graph_stream, private
        # Convert from mlflow request to langgraph input
        request_mlflow = ResponsesAgentRequest(**request)
        graph_input = request_to_graph_input(request_mlflow)
        thread_id = get_thread_id(request_mlflow)

        # Create the stream object
        langgraph_stream = self.agent.astream(
            input=graph_input,
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=STREAM_MODES,
        )
        return langgraph_stream

    
    def run_stream(self, request):
        langgraph_stream = self._create_graph_stream(request)
        return process_agent_astream_events(langgraph_stream)
    
    async def run(self, request):
        events = [e async for e in self.run_stream(request)]
        items = [e.item for e in events if e.type == "response.output_item.done"]
        response = ResponsesAgentResponse(output=items)
        return response