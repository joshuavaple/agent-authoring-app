from langchain_openai import AzureChatOpenAI
from agent.config import AzureChatOpenAISettings


def get_llm() -> AzureChatOpenAI:
    s = AzureChatOpenAISettings()
    llm_client = AzureChatOpenAI(
        azure_endpoint=s.endpoint,
        api_key=s.api_key,
        azure_deployment=s.deployment,
        api_version=s.api_version
    )
    return llm_client