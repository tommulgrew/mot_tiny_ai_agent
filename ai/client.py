
from abc import ABC, abstractmethod
from asyncio import Event
from typing import Callable
from ai.message_accessor import AIMessageAccessor
from ai.tools import AITools
from ai.message_history import AIChatMessageHistory

class AIChatResponse:
    def __init__(self, new_messages: list, history: AIChatMessageHistory):
        self.new_messages = new_messages
        self.history = history

class AIChatClient(ABC):
    """A chat completions client, with tool resolution"""

    @abstractmethod
    async def chat(self, 
            system_prompt: str, 
            user_prompt: str | list[str],
            history: AIChatMessageHistory | None = None,                        # Previous chat history
            tools: AITools | None = None,                                       # Tool calls
            additional_context_prompt: str | list[str] | None = None,           # Additional context added as user messages that are NOT persisted in the history
            strip_think: bool = True,                                           # Whether to strip think blocks from response
            retry_on_context_full: bool = False,                                # Whether to retry on context overflow
            output_callback: Callable[[str], None] | None = None,               # Callback for text output
            is_system_info_callback: Callable[[str], bool] | None = None,       # Callback to check if text corresponds to a "system_info" message
            cancel_event: Event | None = None) -> AIChatResponse:               # Optional cancellation event
        ...

    @abstractmethod
    def get_message_accessor(self) -> AIMessageAccessor:
        ...
