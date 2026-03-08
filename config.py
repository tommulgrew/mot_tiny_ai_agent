from pydantic import BaseModel

class ConfigModel(BaseModel):
    """LLM service configuration"""
    name: str                               # Model name
    url: str = "http://localhost:1234/v1"   # URL to chat completions service. (Default is LM Studio default endpoint)
    api_key: str = "[not used]"
    context_limit: int = 4096               # Token context limit. Must match limit configured at server end.

class Config(BaseModel):
    """Main configuration"""
    model: ConfigModel                      # LLM model configuration

def load_config(path: str) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate_json(f.read())
