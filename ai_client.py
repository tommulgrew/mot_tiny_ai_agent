import logging
import asyncio
import json
from openai.types.chat.chat_completion_tool_param import FunctionDefinition
from typing import Callable, Iterable, cast
from ai_tools import AIToolError, AITools, AITool
from config import ModelConfig
from openai import AsyncClient
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionSystemMessageParam, ChatCompletionMessageToolCallParam, ChatCompletionFunctionToolParam, ChatCompletionToolMessageParam, ChatCompletionUserMessageParam
from util import log_dump

class AIClientError(Exception):
    """General AI chat completion client error"""

class AIClient:
    """Basic client for OpenAI chat completions API, with tool callbacks"""
    
    def __init__(self, settings: ModelConfig):
        self.settings = settings
        self.client = AsyncClient(
            api_key=settings.api_key,
            base_url=settings.url
        )

    async def chat(self, 
            system_prompt: str, 
            user_prompt: str | list[str],
            message_history: Iterable[ChatCompletionMessageParam] = [],
            tools: AITools | None = None,
            strip_think: bool = True,
            output_callback: Callable[[str], None] | None = None) -> list[ChatCompletionMessageParam]:
        """Call the chat completions API, resolve any tool calls and return the new messages generated"""

        # Normalise user_prompt into an array
        user_prompts = [ user_prompt ] if isinstance(user_prompt, str) else user_prompt

        # Create message objects
        system_message = ChatCompletionSystemMessageParam(
            role="system",
            content=system_prompt
        )
        user_messages = [ 
            ChatCompletionUserMessageParam(
                role="user",
                content=p
            )
            for p in user_prompts
        ]

        # Working message list
        messages = [
            *message_history,
            *user_messages
        ]

        # New messages
        new_messages: list[ChatCompletionMessageParam] = [ *user_messages ]

        # Describe tools to chat completion
        if tools:            
            completion_tools = [ make_chat_completion_tool(t) for t in tools.tools ]
            tools.clear_called_tools()
        else:
            completion_tools = []

        # Track whether we are in a think block.
        # (Think blocks can span multiple messages in Qwen3.5).
        think_open = False        

        # Repeat until all tool calls resolved
        while True:

            logging.debug("Chat API request: %s", log_dump([ system_message, *messages ]))            

            # Call chat completion service
            response = await self.client.chat.completions.create(
                    messages=[ system_message, *messages], 
                    model=self.settings.name,
                    tools=completion_tools
                )

            logging.debug("Chat API response: %s", log_dump(response))            

            # Extract response message and content
            choice = response.choices[0]
            msg = choice.message
            content = msg.content

            # Cast tool calls to param tool calls (to keep Pyright happy - objects have identical fields)
            tool_call_params = cast(list[ChatCompletionMessageToolCallParam], msg.tool_calls) if msg else []

            # Add assistant message to conversation (*with* thinking, so thought chain isn't broken across tool calls)
            assistant_message = ChatCompletionAssistantMessageParam(
                role="assistant",
                content=content,
                tool_calls=tool_call_params
            )
            messages.append(assistant_message)            

            # Determine new message to return and callback content
            if strip_think and content != None:
                # Create copy of assistant message with think blocks stripped out
                # Track whether the think block is open/closed at the end of the message
                # as think blocks can span multiple messages during tool calls
                # (at least in Qwen3.5).
                stripped_content, think_open = strip_think_block(content, think_open)
                stripped_assistant_message = ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=stripped_content,
                    tool_calls=tool_call_params
                )
                new_messages.append(stripped_assistant_message)     # Return version without think block. "message_list" retains thinking so LLMs though chain isn't broken during tool resolution.
                callback_content=stripped_content
            else:
                new_messages.append(assistant_message)              # Return unmodified response
                callback_content=content

            # Pass current output to callback
            if output_callback:
                callback_content = (callback_content or "").strip()
                
                # Append any tool calls (just tool name)
                if msg.tool_calls:
                    tool_names = [tool_call.function.name for tool_call in msg.tool_calls if tool_call.type == "function"]
                    if tool_names:
                        tool_desc = f"[{','.join(tool_names)}]"
                        callback_content = callback_content + "\n" + tool_desc if callback_content else tool_desc
                
                # Call callback (if any content to report)
                if callback_content:
                    output_callback(callback_content)

            # If not a tool call, we are finished
            if choice.finish_reason != 'tool_calls':
                return new_messages

            # Call tools
            if not tools:
                raise AIClientError("Received tool call when no tools are available")

            for tool_call in msg.tool_calls or []:
                if tool_call.type != "function":
                    raise AIClientError("Tool type must be 'function'")

                # Parse JSON parameters
                tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                # Call tool
                try:
                    tool_result = await tools.call_tool(tool_call.function.name, tool_args)
                except AIToolError as exc:
                    tool_result = f"ERROR: {exc}"

                # Add tool result message
                tool_message = ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tool_call.id,
                    content=tool_result
                )
                messages.append(tool_message)
                new_messages.append(tool_message)

    def flatten_messages(self, messages: Iterable[ChatCompletionMessageParam]) -> str:
        return "\n\n".join(f"[{m['role'].upper()}]:\n{m.get('content') or ''}" for m in messages)

    def is_user_message(self, message: ChatCompletionMessageParam) -> bool:
        return message['role'] == "user"

    def get_message_content(self, message: ChatCompletionMessageParam) -> str:
        content = message.get('content')
        return content if isinstance(content, str) else ""

def make_chat_completion_tool(tool: AITool) -> ChatCompletionFunctionToolParam:
    
    # Convert parameters to dictionary
    properties = {
        p.name: {
            "type": p.type,
            "description": p.description
        }
        for p in tool.params
    }
    
    # Build tool for chat completion
    return ChatCompletionFunctionToolParam(
        type="function",
        function=FunctionDefinition(
            name=tool.name,
            description=tool.description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": [p.name for p in tool.params if not p.optional]
            }
        )
    )

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
