import json
from collections import deque
from typing import Callable
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from ai_client import AIClient
from ai_tools import AITools
from ai_memory import AIMemory

class AIAgentPrompts(BaseModel):
    main: str           # Main agent system prompt

class AIAgent:
    """A basic autonomous AI agent"""
    def __init__(self, client: AIClient, memory: AIMemory, tools: AITools | None, output_callback: Callable[[str], None] | None = None):
        self.client = client
        self.memory = memory
        self.tools = tools
        self.output_callback = output_callback
        self.prompts = create_ai_prompts()
        self.message_history = deque(maxlen=20)

    async def process_user_message(self, message: str):
        await self._process_event(message)

    async def process_system_event(self, data):
        await self._process_event(json.dumps(data))

    async def _process_event(self, content: str):

        # Retrieve related memories
        memories = self.memory.retrieve(content)
        if memories:
            # Serialize the memories as JSON, to (hopefully) help the LLM 
            # understand it is not user input.
            memory_text = f"{{ relevant_memories: {json.dumps(memories)} }}"

            # Insert a memories message immediately before the main user input
            user_prompt = [ memory_text, content ]
        else:
            user_prompt = content

        # Call chat client
        new_messages = await self.client.chat(
            system_prompt=self.prompts.main,
            user_prompt=user_prompt,
            message_history=self.message_history,
            tools=self.tools,
            output_callback=self._filter_output
        )

        # Add to history
        self.message_history.extend(new_messages)

        # Record memories
        # Just use user input, not assistant replies, otherwise LLM can get 
        # confused about who said what.
        self.memory.create_memories(f"[USER]\n{content}")

    def _filter_output(self, output: str):
        if self.output_callback and output: # and output != "NO_OUTPUT":
            self.output_callback(output)

    def _is_relevant_memories_msg(self, msg) -> bool:
        return (
            self.client.is_user_message(msg) and 
            self.client.get_message_content(msg).strip().startswith("{ relevant_memories:")
        )


def create_ai_prompts() -> AIAgentPrompts:
    return AIAgentPrompts(
        main="""\
You are an helpful and proactive AI agent that acts like a secretary for the user.

Perform tasks when requested by the user, using the available tools calls.
The system will notify you of events like new emails or scheduled tasks. Respond to events in a productive and proactive way, taking into account the user's goals and requests. The user may not be online when a system event is received.

Output: You can respond in three ways:
- Text - Displayed directly to the user.
- Tool calls - Invoke tools to perform actions.
- No output - When you determine no action or response is required, respond with only the text: NO_OUTPUT
  This response will be filtered out and never shown to the user.
""")