from enum import Enum
import json
from dataclasses import field
from openai.types.chat.chat_completion_tool_param import FunctionDefinition
from typing import Callable, Iterable, cast
from ai_tools import AIToolError, AITools, AITool
from config import ModelConfig
from openai import AsyncClient, BadRequestError, BaseModel
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionSystemMessageParam, ChatCompletionMessageToolCallParam, ChatCompletionFunctionToolParam, ChatCompletionToolMessageParam, ChatCompletionUserMessageParam
from util import create_logger, log_dump

class AIClientError(Exception):
    """General AI chat completion client error"""

class TrimTokensOpType(int, Enum):
    STRIP_THINK = 0
    REMOVE_SYS_INFO = 1
    TRUNCATE_TOOL_RESULTS = 2
    REMOVE_GROUP = 3

_recover_tokens_op_weights = [0.1, 0.25, 0.4, 1.0]
_recover_tokens_age_weights = [1.0, 0.6, 0.4, 0.3, 0.2, 0.1]

class AIChatMessageGroup(BaseModel):
    start_index: int
    message_count: int
    user_message_count: int

class TrimTokensOp(BaseModel):
    group: AIChatMessageGroup
    age: int                    # 0 = newest
    op_type: TrimTokensOpType

    def get_weight(self) -> float:
        op_weight = _recover_tokens_op_weights[self.op_type]
        age_weight = (
            _recover_tokens_age_weights[self.age] 
            if self.age < len(_recover_tokens_age_weights) 
            else _recover_tokens_age_weights[-1]
        )
        return op_weight * age_weight

class AIChatMessageHistory(BaseModel):
    """Stores chat message history, and manages trimming to recover tokens"""
    messages: list[ChatCompletionMessageParam] = field(default_factory=list)
    token_count: int = 0

    def add(self, messages: Iterable[ChatCompletionMessageParam]):
        self.messages.extend(messages)

        # Adjust estimated token count
        self.token_count += estimate_tokens_list(messages)

    def trim_to_limit(self, token_limit: int) -> bool:
        while self.token_count > token_limit:
            if not self.trim():
                return False
        return True

    def trim(self) -> bool:

        # Split messages into groups. Exclude last group, as we cannot trim tokens
        # from the current active group.
        groups = self._get_groups()[:-1]
        if not groups:
            return False

        # Create token trim operations for each group
        ops = [ 
            TrimTokensOp(
                group=g, 
                age=len(groups) - idx - 1, 
                op_type=self._get_trim_tokens_op_type(g)
            ) 
            for idx, g in enumerate(groups)
        ]

        # Sort by weight
        ops.sort(key=lambda x: x.get_weight())

        # Apply lowest weighted operation
        self._do_trim_op(ops[0])
        return True

    def _get_groups(self) -> list[AIChatMessageGroup]:
        groups = []
        i = 0
        user_message_count = 0
        while i < len(self.messages):
            start = i

            # User messages
            while i < len(self.messages) and _is_user_message(self.messages[i]):
                i += 1
                user_message_count += 1

            # Assistant and tool messages
            while i < len(self.messages) and not _is_user_message(self.messages[i]):
                i += 1
            
            groups.append(AIChatMessageGroup(start_index=start, user_message_count=user_message_count, message_count=i - start))
        
        return groups

    def _get_trim_tokens_op_type(self, group: AIChatMessageGroup) -> TrimTokensOpType:
        group_messages = self.messages[group.start_index:group.start_index + group.message_count]

        if self._group_has_think_block(group_messages):
            return TrimTokensOpType.STRIP_THINK
        
        elif self._group_has_sys_info(group_messages):
            return TrimTokensOpType.REMOVE_SYS_INFO

        elif self._group_has_untruncated_tools(group_messages):
            return TrimTokensOpType.TRUNCATE_TOOL_RESULTS

        else:
            return TrimTokensOpType.REMOVE_GROUP

    def _do_trim_op(self, op: TrimTokensOp):

        if (op.op_type == TrimTokensOpType.STRIP_THINK):
            self._strip_think_blocks(op.group)

        elif (op.op_type == TrimTokensOpType.REMOVE_SYS_INFO):
            self._remove_sys_info(op.group)

        elif (op.op_type == TrimTokensOpType.TRUNCATE_TOOL_RESULTS):
            self._truncate_tool_results(op.group)

        elif (op.op_type == TrimTokensOpType.REMOVE_GROUP):
            self._remove_group(op.group)

    def _group_has_think_block(self, messages: list[ChatCompletionMessageParam]) -> bool:
        return any(
            _is_assistant_message(m) and "<think>" in _get_message_content(m) 
            for m in messages
        )

    def _group_has_sys_info(self, messages: list[ChatCompletionMessageParam]) -> bool:
        return False            # TODO

    def _group_has_untruncated_tools(self, messages: list[ChatCompletionMessageParam]) -> bool:
        return False            # TODO

    def _strip_think_blocks(self, group: AIChatMessageGroup):
        raise Exception("Not implemented")      # TODO

    def _remove_sys_info(self, group: AIChatMessageGroup):
        raise Exception("Not implemented")      # TODO

    def _truncate_tool_results(self, group: AIChatMessageGroup):
        raise Exception("Not implemented")      # TODO

    def _remove_group(self, group: AIChatMessageGroup):
        raise Exception("Not implemented")      # TODO

class AIChatResponse(BaseModel):
    new_messages: list[ChatCompletionMessageParam]
    history: AIChatMessageHistory        

class AIClient:
    """Basic client for OpenAI chat completions API, with tool callbacks"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = AsyncClient(
            api_key=config.api_key,
            base_url=config.url
        )

        # Create separate logger for chat completions dumps
        self.chat_logger = create_logger("tinyagent.completions", "completions_log.txt", propagate=False)

    async def chat(self, 
            system_prompt: str, 
            user_prompt: str | list[str],
            history: AIChatMessageHistory | None = None,
            tools: AITools | None = None,
            strip_think: bool = True,
            retry_on_context_full: bool = False,
            output_callback: Callable[[str], None] | None = None) -> AIChatResponse:
        """Call the chat completions API, resolve any tool calls and return the new messages generated"""

        # Default to empty history
        history = history or AIChatMessageHistory()

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

            # Trim history if estimated token size exceeds threshold
            history.trim_to_limit(self.config.prompt_token_limit)

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
                    if not retry_on_context_full or not history.trim():
                        raise AIClientError("Token context overflowed.")

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

    def flatten_messages(self, messages: Iterable[ChatCompletionMessageParam]) -> str:
        return "\n\n".join(f"[{m['role'].upper()}]:\n{m.get('content') or ''}" for m in messages)

    def is_user_message(self, message: ChatCompletionMessageParam) -> bool:
        return _is_user_message(message)

    def get_message_content(self, message: ChatCompletionMessageParam) -> str:
        return _get_message_content(message)

def _is_user_message(message: ChatCompletionMessageParam) -> bool:
    return message.get('role', '') == "user"

def _is_assistant_message(message: ChatCompletionMessageParam) -> bool:
    return message.get('role', '') == "assistant"

def _get_message_content(message: ChatCompletionMessageParam) -> str:
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

def estimate_tokens(message: ChatCompletionMessageParam) -> int:
    content = str(message.get("content", ""))
    return len(content) // 4

def estimate_tokens_list(messages: Iterable[ChatCompletionMessageParam]) -> int:
    return sum(estimate_tokens(m) for m in messages)