from typing import Literal
from pydantic import BaseModel

# LLM model config
class ModelConfig(BaseModel):
    """LLM service configuration"""
    name: str                               # Model name
    url: str = "http://localhost:1234/v1"   # URL to chat completions service. (Default is LM Studio default endpoint)
    api_key: str = "[not used]"
    context_limit: int = 4096               # Token context limit. Must match limit configured at server end.

# File tools config
class FolderConfig(BaseModel):
    alias: str
    path: str
    access: Literal['rw', 'ro']

class FileToolsConfig(BaseModel):
    folders: list[FolderConfig]
    trash_path: str
    max_read_chars: int = 8192

class MemoryConfig(BaseModel):
    storage_path: str = "memory.json"
    users_name: str | None = None
    agents_name: str | None = None
    extra_info: list[str] | None = None

class SpeechToTextConfig(BaseModel):
    enabled: bool = False
    model_size: str = "small"
    trigger_key: str = "ctrl_r"
    sample_rate: int = 16000
    language: str | None = None

class AppConfig(BaseModel):
    name: str
    path: str
    description: str | None = None

class Config(BaseModel):
    """Main configuration"""
    model: ModelConfig                      # LLM model configuration
    file_tools: FileToolsConfig
    memory: MemoryConfig
    speech_to_text: SpeechToTextConfig
    allowed_apps: list[AppConfig] | None = None

def load_config(path: str) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate_json(f.read())
