from openai.types.chat import ChatCompletionMessageParam
from ai.message_accessor import AIMessageAccessor

class OpenAIMessageAccessor(AIMessageAccessor):
    """Message accessor for OpenAI ChatCompletionMessageParam message type"""

    def get_content(self, message: ChatCompletionMessageParam) -> str:
        content = message.get('content')
        return content if isinstance(content, str) else ""

    def is_user_message(self, message) -> bool:
        return message.get('role', '') == "user"

    def is_assistant_message(self, message) -> bool:
        return message.get('role', '') == "assistant"

    def is_tool_message(self, message) -> bool:
        return message.get('role', '') == "tool"

