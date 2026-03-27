from typing import Literal
from pydantic import BaseModel

# LLM model config
class ModelConfig(BaseModel):
    """LLM service configuration"""
    name: str                               # Model name
    url: str = "http://localhost:1234/v1"   # URL to chat completions service. (Default is LM Studio default endpoint)
    api_key: str = "[not used]"
    prompt_token_limit: int = 11000
    prompt_token_target: int | None = None

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
    retrieve_memory_limit: int = 10
    working_memory_storage_path: str = "working_memory.json"
    working_memory_limit: int = 15

class AgentConfig(BaseModel):
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

# Speak tools config
class SpeakConfig(BaseModel):
    enabled: bool = True

# Reminder tools config
class ReminderConfig(BaseModel):
    storage_path: str = "reminders.json"

# Todo tools config
class TodoConfig(BaseModel):
    storage_path: str = "tasks.json"
    completed_log_path: str = "tasks_completed.jsonl"
    list_limit: int = 50

# Web search tools config
class WebSearchConfig(BaseModel):
    enabled: bool = True

# Email tools config
class ImapConfig(BaseModel):
    imap_host: str
    imap_port: int = 993
    username: str
    password: str | None = None     # If omitted, resolved from OS keyring at startup

class NamedImapConfig(ImapConfig):
    name: str                       # Short account alias used in email IDs (e.g. "slingshot")

class AgentInboxConfig(ImapConfig):
    smtp_host: str
    smtp_port: int = 587
    send_whitelist: list[str] = []

class EmailConfig(BaseModel):
    user_inboxes: list[NamedImapConfig] = []
    agent_inbox: AgentInboxConfig | None = None
    storage_path: str = "email_state.json"
    max_body_chars: int = 4000
    poll_interval_seconds: int = 300

class Config(BaseModel):
    """Main configuration"""
    model: ModelConfig                      # LLM model configuration
    file_tools: FileToolsConfig
    memory: MemoryConfig
    agent: AgentConfig
    speech_to_text: SpeechToTextConfig
    allowed_apps: list[AppConfig] | None = None
    email: EmailConfig | None = None
    todo: TodoConfig | None = None
    reminders: ReminderConfig | None = None
    speak: SpeakConfig | None = None
    web_search: WebSearchConfig | None = None

def load_config(path: str) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate_json(f.read())
