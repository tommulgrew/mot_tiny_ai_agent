import json
from abc import ABC, abstractmethod
from typing import Any

class AIMessageAccessor(ABC):

    @abstractmethod
    def get_content(self, message) -> str:
        ...

    @abstractmethod
    def is_user_message(self, message) -> bool:
        ...

    @abstractmethod
    def is_assistant_message(self, message) -> bool:
        ...

    @abstractmethod
    def is_tool_message(self, message) -> bool:
        ...

    def to_jsonl(self, message) -> str:
        """
        Convert message to JSON string.
        JSON must be formatted on a single line (i.e. jsonl format)
        """
        return json.dumps(message)

    def from_jsonl(self, s) -> Any:
        """Convert JSON string to message"""
        return json.loads(s)
