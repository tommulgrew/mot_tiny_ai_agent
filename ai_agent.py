import logging
import json
from collections import deque
from typing import Callable
from openai import BadRequestError
from pydantic import BaseModel
from ai_client import AIClient
from ai_tools import AITool, AITools, AIToolParam
from ai_memory import AIMemory
from config import MemoryConfig

class AIAgentPrompts(BaseModel):
    main: str           # Main agent system prompt

class AIAgent:
    """A basic autonomous AI agent"""
    def __init__(self, config: MemoryConfig, client: AIClient, memory: AIMemory, tools: AITools | None, output_callback: Callable[[str], None] | None = None):
        self.client = client
        self.memory = memory
        self.tools = tools
        if self.tools:
            self.tools.add([self._make_recall_memories_tool()])
        self.output_callback = output_callback
        self.prompts = create_ai_prompts(users_name=config.users_name, agents_name=config.agents_name, extra_info=config.extra_info)
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
        new_messages = await self._call_chat_client(user_prompt)

        # Record memories
        memory_messages = [m for m in new_messages if not self._is_relevant_memories_msg(m)]        
        self.memory.create_memories(memory_messages, memories)

    async def _call_chat_client(self, user_prompt: list[str] | str) -> list:
        # Retry until call completes without context overflowing.
        # (Removing history messages as necessary)
        while True:
            try:
                # Call client
                new_messages, stats = await self.client.chat(
                    system_prompt=self.prompts.main,
                    user_prompt=user_prompt,
                    message_history=self.message_history,
                    tools=self.tools,
                    output_callback=self._filter_output
                )

                # Add to history
                self.message_history.extend(new_messages)

                # Trim history if prompt size exceeds limit
                if stats and stats.prompt_tokens > 2000:
                    self._trim_message_history()

                return new_messages
            
            except BadRequestError as e:
                if "Context size has been exceeded" not in str(e):
                    raise

                # Context size exceeded - Trim message history and try again
                if self.message_history:
                    logging.warning("Context sizes exceeded. Trimming history and retrying.")
                    self._trim_message_history()
                else:
                    raise                           # No history left to trim!

    def _trim_message_history(self):
        # Remove 3 oldest messages
        for _ in range(0, 3):
            if self.message_history:
                self.message_history.pop()

    def _filter_output(self, output: str):
        if self.output_callback and output: # and output != "NO_OUTPUT":
            self.output_callback(output)

    def _is_relevant_memories_msg(self, msg) -> bool:
        return (
            self.client.is_user_message(msg) and 
            self.client.get_message_content(msg).strip().startswith("{ relevant_memories:")
        )

    def _make_recall_memories_tool(self) -> AITool:
        return AITool(
            name="recall_memories",
            description="Lookup previously stored memories related to one or more keywords",
            params=[
                AIToolParam(
                    name="keywords",
                    type="string",
                    description="Comma separated list of keywords"                    
                )
            ],
            async_callback=self._recall_memories_tool
        )

    async def _recall_memories_tool(self, keywords: str) -> str:
        memories = self.memory.retrieve(keywords, housekeeping=False)
        return json.dumps(memories) if memories else "No memories found"

def create_ai_prompts(users_name: str | None, agents_name: str | None, extra_info: list[str] | None) -> AIAgentPrompts:

    extra_info_lines = []
    if agents_name:
        extra_info_lines.append(f"Your name is {agents_name}.")
    if users_name:
        extra_info_lines.append(f"The user's name is {users_name}.")
    if extra_info:
        extra_info_lines.extend(extra_info)
    extra_info_text = "\n".join(extra_info_lines)

    return AIAgentPrompts(
        main=f"""\
You are an helpful and proactive AI agent that acts like a secretary for the user.

Perform tasks when requested by the user, using the available tools calls.
The system will notify you of events like new emails or scheduled tasks. Respond to events in a productive and proactive way, taking into account the user's goals and requests. The user may not be online when a system event is received.

Output: You can respond in three ways:
- Text - Displayed directly to the user.
- Tool calls - Invoke tools to perform actions.
- No output - When you determine no action or response is required, respond with only the text: NO_OUTPUT

{extra_info_text if extra_info_text else ""}\
""")