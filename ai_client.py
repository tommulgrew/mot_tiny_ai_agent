import json
from openai.types.chat.chat_completion_tool_param import FunctionDefinition
from pydantic import BaseModel
from typing import Awaitable, Callable, Iterable, cast
from config import ConfigModel
from openai import AsyncClient
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionSystemMessageParam, ChatCompletionMessageToolCallParam, ChatCompletionFunctionToolParam, ChatCompletionToolMessageParam

class AIToolError(Exception):
    """Raised when an error occurs calling a tool. Message will be passed back to the LLM as the tool response."""

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

    async def chat(self, 
            system_message: ChatCompletionSystemMessageParam, 
            messages: Iterable[ChatCompletionMessageParam], 
            tools: Iterable[AITool] | None = None,
            strip_think: bool = True,
            output_callback: Callable[[str], None] | None = None) -> list[ChatCompletionMessageParam]:
        """Call the chat completions API, resolve any tool calls and return the new messages generated"""

        # Collect new messages
        new_messages: list[ChatCompletionMessageParam] = []

        # Create list so we can append new messages
        message_list = [*messages]

        # Describe tools to chat completion
        completion_tools = [ make_chat_completion_tool(t) for t in tools ] if tools else []

        # Track whether we are in a think block.
        # (Think blocks can span multiple messages in Qwen3.5).
        think_open = False        

        # Repeat until all tool calls resolved
        while True:

            # Call chat completion service
            response = await self.client.chat.completions.create(
                messages=[ system_message, *message_list], 
                model=self.settings.name,
                tools=completion_tools
            )

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
            message_list.append(assistant_message)            

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
            for tool_call in msg.tool_calls or []:
                if tool_call.type != "function":
                    raise AIToolError("Tool type must be 'function'")

                # Call the tool
                tool_result = await self.call_tool(tool_call, tools)

                # Add tool result message
                tool_message = ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tool_call.id,
                    content=tool_result
                )
                message_list.append(tool_message)
                new_messages.append(tool_message)

    async def call_tool(self, tool_call, tools: Iterable[AITool] | None) -> str:

        # Find tool
        tool = next((tool for tool in tools or [] if tool.name == tool_call.function.name), None)
        if tool is None:
            raise AIToolError(f"Tool '{tool_call.function.name}' not found")

        # Collect parameters
        args_json = tool_call.function.arguments
        args_dict = json.loads(args_json) if args_json else {}
        params = [
            args_dict[p.name] if not p.optional else args_dict.get(p.name)
            for p in tool.params
        ]

        # Call tool callback
        return await tool.async_callback(*params)


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
