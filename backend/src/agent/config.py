from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# For each attribute, pydantic-settings builds the env var name as env_prefix + attribute name, 
# and the lookup ignores case.
class AzureChatOpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AZURE_OPENAI_")
    
    endpoint: str # maps to "AZURE_OPENAI_ENDPOINT"
    api_key: SecretStr
    deployment: str
    api_version: str