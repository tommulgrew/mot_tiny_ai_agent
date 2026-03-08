from pydantic import BaseModel
from typing import Awaitable, Callable, Iterable
from config import ConfigModel
from openai import AsyncClient
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionSystemMessageParam

class AIToolParam(BaseModel):
    """Parameter definition for an AI tool"""
    name: str
    type: str
    description: str
    optional: bool = False

class AITool(BaseModel):
    """An AI tool, including definition and callback function"""
    name: str
    description: str
    params: list[AIToolParam]
    async_callback: Callable[..., Awaitable[str]]
    single_use: bool = False

class AIClient:
    """Basic client for OpenAI chat completions API, with tool callbacks"""
    
    def __init__(self, settings: ConfigModel):
        self.settings = settings
        self.client = AsyncClient(
            api_key=settings.api_key,
            base_url=settings.url
        )

    async def chat(self, system_message: ChatCompletionSystemMessageParam, messages: Iterable[ChatCompletionMessageParam], strip_think: bool = True) -> list[ChatCompletionMessageParam]:
        """Call the chat completions API, resolve any tool calls and return the new messages generated"""

        # Collect new messages
        new_messages: list[ChatCompletionMessageParam] = []

        # Create list so we can append new messages
        message_list = [*messages]

        # Track whether we are in a think block.
        # (Think blocks can span multiple messages in Qwen3.5).
        think_open = False        

        # Repeat until all tool calls resolved
        while True:

            # Call chat completion service
            response = await self.client.chat.completions.create(
                messages=[ system_message, *message_list], 
                model=self.settings.name
            )

            # Extract response message and content
            choice = response.choices[0]
            msg = choice.message
            content = msg.content

            # Add assistant message to conversation
            assistant_message = ChatCompletionAssistantMessageParam(
                role="assistant",
                content=content,
                # tool_calls=msg.tool_calls     # TODO
            )
            message_list.append(assistant_message)            

            # Remove <think> blocks if necessary
            if strip_think and content != None:
                # Create a version of the assistant message with think blocks stripped out
                stripped_content, think_open = strip_think_block(content, think_open)
                stripped_assistant_message = ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=stripped_content,
                    # tool_calls=msg.tool_calls     # TODO
                )
                new_messages.append(stripped_assistant_message)     # Return version without think block. "message_list" retains thinking so LLMs though chain isn't broken during tool resolution.
                callback_content=stripped_content
            else:
                new_messages.append(assistant_message)              # Return unmodified response
                callback_content=content

            # TODO: Display callback

            if choice.finish_reason == 'tool_calls':
                # TODO: Tool calls
                assert(False)

            else:
                # Not a tool call. Return.
                return new_messages

def strip_think_block(text: str, think_open: bool) -> tuple[str, bool]:
    offset = 0
    result = ""
    while True:
        # Find next <think> or </think> tag
        tag = "</think>" if think_open else "<think>"
        next_offset = text.find(tag, offset)

        # ... or end of string
        if next_offset == -1:
            next_offset = len(text)

        # Append any non-think content
        if not think_open:
            result += text[offset:next_offset]
        
        # Exit if found end of string
        if next_offset == len(text):
            return result, think_open

        # Toggle think open and skip tag
        think_open = not think_open
        offset = next_offset + len(tag)
