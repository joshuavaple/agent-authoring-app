import pytest
from pydantic import ValidationError

from agent.config import AzureChatOpenAISettings

VARS = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
]

@pytest.fixture
def clean_env(monkeypatch):
    """
    Deletes the Azure env vars, then returns monkeypatch.
    So every test that uses it starts with those variables unset
    """
    for v in VARS:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch

def test_missing_settings_rase(clean_env):
    """
    tests that without an env file, the error is loud.
    """
    with pytest.raises(ValidationError) as exc:
        AzureChatOpenAISettings(_env_file=None)
    missing = {e["loc"][0] for e in exc.value.errors()}
    assert missing == {"endpoint", "api_key", "deployment", "api_version"}

def test_env_vars_map_to_attributes(clean_env):
    # set the env vars with fake values
    clean_env.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    clean_env.setenv("AZURE_OPENAI_API_KEY", "fake-key")
    clean_env.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")
    clean_env.setenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    s = AzureChatOpenAISettings(_env_file=None)

    assert s.endpoint == "https://example.openai.azure.com/"
    assert s.api_key.get_secret_value() == "fake-key"
    assert s.deployment == "my-deployment"
    assert s.api_version == "2024-02-01"

def test_api_key_not_leaked_in_repr(clean_env):
    clean_env.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    clean_env.setenv("AZURE_OPENAI_API_KEY", "super-secret")
    clean_env.setenv("AZURE_OPENAI_DEPLOYMENT", "d")
    clean_env.setenv("AZURE_OPENAI_API_VERSION", "v")

    s = AzureChatOpenAISettings(_env_file=None)

    assert "super-secret" not in repr(s)
    assert "super-secret" not in str(s)