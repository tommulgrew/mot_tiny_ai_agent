from abc import ABC, abstractmethod

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

