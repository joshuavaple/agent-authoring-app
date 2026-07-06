from mlflow.genai.agent_server import AgentServer
import agent

agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=True)

# Define the app as a module level variable to enable multiple workers
app = agent_server.app


def main():

    agent_server.run(app_import_string="start_server:app")


if __name__ == "__main__":
    main()