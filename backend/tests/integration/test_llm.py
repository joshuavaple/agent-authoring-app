import pytest
from pydantic import ValidationError
from agent.config import AzureChatOpenAISettings
from agent.adapters.llm import get_llm


def _has_credentials() -> bool:
    try:
        AzureChatOpenAISettings() # reads env vars
        return True
    except ValidationError:
        return False

# @pytest.mark.integration    # arbitrary label to test separately
@pytest.mark.skipif(not _has_credentials(), reason="Azure OpenAI credentials not set")
def test_get_llm_real_call():
    client = get_llm()
    response = client.invoke("say hello in one short sentence.")
    assert isinstance(response.content, str)
    assert response.content.strip() != ""