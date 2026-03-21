from pydantic import BaseModel
from ai.message_accessor import AIMessageAccessor

class AIChatMessageGroup(BaseModel):
    start_index: int
    message_count: int
    user_message_count: int

class AIChatMessageHistory:
    """Stores chat message history, and manages trimming to recover tokens"""

    def __init__(self, message_accessor: AIMessageAccessor):
        self.message_accessor = message_accessor
        self.messages = []
        self.token_count = 0

    def set_token_count(self, token_count: int):
        self.token_count = token_count

    def add(self, messages: list):
        self.messages.extend(messages)

        # Adjust estimated token count
        self.token_count += estimate_tokens_list(messages, self.message_accessor)

    def remove(self, messages: list):
        for msg in messages:
            self.messages.remove(msg)

            # Adjust estimated token count
            self.token_count -= estimate_tokens(msg, self.message_accessor)

    def set_content(self, idx: int, content: str):
        message = self.messages[idx]
        prev_message_tokens = estimate_tokens(message, self.message_accessor)
        message["content"] = content
        new_message_tokens = estimate_tokens(message, self.message_accessor)
        self.token_count = self.token_count - prev_message_tokens + new_message_tokens

    def get_groups(self) -> list[AIChatMessageGroup]:
        groups = []
        i = 0
        user_message_count = 0
        while i < len(self.messages):
            start = i

            # User messages
            while i < len(self.messages) and self.message_accessor.is_user_message(self.messages[i]):
                i += 1
                user_message_count += 1

            # Assistant and tool messages
            while i < len(self.messages) and not self.message_accessor.is_user_message(self.messages[i]):
                i += 1
            
            groups.append(AIChatMessageGroup(start_index=start, user_message_count=user_message_count, message_count=i - start))
        
        return groups

    def group_messages(self, group: AIChatMessageGroup) -> list:
        return self.messages[group.start_index:group.start_index + group.message_count]

def estimate_tokens(message, accessor: AIMessageAccessor) -> int:
    content = accessor.get_content(message)
    return len(content) // 4

def estimate_tokens_list(messages: list, accessor: AIMessageAccessor) -> int:
    return sum(estimate_tokens(m, accessor) for m in messages)
