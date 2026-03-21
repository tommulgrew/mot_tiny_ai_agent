from asyncio import Event
import json
from typing import Callable, Iterable, cast
from ai.client import AIChatClient, AIChatResponse
from ai.context_manager import AIContextManager
from ai.tools import AIToolError, AITools, AITool
from ai.message_accessor import AIMessageAccessor
from ai.message_history import AIChatMessageHistory
from ai.errors import AIClientError, AIClientTokenOverflowError
from ai.util import strip_think_block
from config import ModelConfig
from util import create_logger, log_dump
from openai_impl.openai_message_accessor import OpenAIMessageAccessor
from openai import AsyncClient, BadRequestError
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionSystemMessageParam, ChatCompletionMessageToolCallParam, ChatCompletionFunctionToolParam, ChatCompletionToolMessageParam, ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_tool_param import FunctionDefinition

class OpenAIChatClient(AIChatClient):
    """Basic client for OpenAI chat completions API, with tool callbacks"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = AsyncClient(
            api_key=config.api_key,
            base_url=config.url
        )
        self.context_manager = AIContextManager(self.get_message_accessor())

        # Create separate logger for chat completions dumps
        self.chat_logger = create_logger("tinyagent.completions", "completions_log.txt", propagate=False)

    async def chat(self, 
            system_prompt: str, 
            user_prompt: str | list[str],
            history: AIChatMessageHistory | None = None,
            tools: AITools | None = None,
            strip_think: bool = True,
            retry_on_context_full: bool = False,
            output_callback: Callable[[str], None] | None = None,
            is_system_info_callback: Callable[[str], bool] | None = None,
            cancel_event: Event | None = None) -> AIChatResponse:
        """Call the chat completions API, resolve any tool calls and return the new messages generated"""

        # Default to empty history
        history = history or AIChatMessageHistory(message_accessor=self.get_message_accessor())

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

        # Add to history
        history.add(user_messages)

        # Track new messages
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

            # Check for cancellation.
            if cancel_event and cancel_event.is_set():

                # Add a fake assistant message to terminate the message group
                assistant_message = ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content="[INTERRUPTED]"
                )
                history.add([assistant_message])
                return AIChatResponse(history=history, new_messages=new_messages)

            # Trim history if estimated token size exceeds threshold
            self.context_manager.trim_to_limit(history, self.config.prompt_token_limit, is_system_info_callback)

            self.chat_logger.debug("Chat API request: %s", log_dump([ system_message, *history.messages ]))            

            # Call chat completion service
            response = None
            while not response:
                try:
                    response = await self.client.chat.completions.create(
                            messages=[ system_message, *history.messages], 
                            model=self.config.name,
                            tools=completion_tools
                        )
                except BadRequestError as e:
                    if "Context size has been exceeded" not in str(e) and "The number of tokens to keep from the initial prompt is greater than the context length" not in str(e):
                        raise

                    # Trim history and retry
                    if not retry_on_context_full or not self.context_manager.trim(history, is_system_info_callback):
                        raise AIClientTokenOverflowError("Token context overflowed.")

            self.chat_logger.debug("Chat API response: %s", log_dump(response))            

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
            history.add([assistant_message])

            # Use actual token count if available
            if response.usage:
                history.token_count = response.usage.total_tokens

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
                return AIChatResponse(history=history, new_messages=new_messages)

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
                    if output_callback:
                        output_callback(f"[Tool error: {exc}]")
                    tool_result = f"ERROR: {exc}"

                # Add tool result message
                tool_message = ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tool_call.id,
                    content=tool_result
                )
                history.add([tool_message])
                new_messages.append(tool_message)

    def get_message_accessor(self) -> AIMessageAccessor:
        return OpenAIMessageAccessor()

    def flatten_messages(self, messages: Iterable[ChatCompletionMessageParam]) -> str:
        return "\n\n".join(f"[{m['role'].upper()}]:\n{m.get('content') or ''}" for m in messages)

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

