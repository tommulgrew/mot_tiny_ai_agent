import logging
import json
from datetime import datetime
from collections import deque
from typing import Callable
import humanize
from openai import BadRequestError
from pydantic import BaseModel
from ai_client import AIChatMessageHistory, AIClient, AIClientError
from ai_tools import AITool, AITools, AIToolParam
from ai_memory import AIMemory
from config import AgentConfig, MemoryConfig

class AIAgentPrompts(BaseModel):
    main: str           # Main agent system prompt

class AgentSystemInfo(BaseModel):
    current_time: str
    user_last_active: str
    relevant_memories: list[str] | None

class AgentSystemInfoMessage(BaseModel):
    system_info: AgentSystemInfo

class AIAgent:
    """A basic autonomous AI agent"""
    def __init__(self, config: AgentConfig, client: AIClient, memory: AIMemory, tools: AITools | None, output_callback: Callable[[str], None] | None = None):
        self.logger = logging.getLogger("tinyagent.agent")
        self.config = config
        self.client = client
        self.memory = memory
        self.tools = tools
        if self.tools:
            self.tools.add([self._make_recall_memories_tool()])
        self.output_callback = output_callback
        self.prompts = create_ai_prompts(users_name=config.users_name, agents_name=config.agents_name, extra_info=config.extra_info)
        self.message_history = AIChatMessageHistory()
        self.user_last_active = datetime.now()

    async def process_user_message(self, message: str):
        self.user_last_active = datetime.now()
        await self._process_event(message)

    async def process_system_event(self, data):
        await self._process_event(json.dumps(data))

    async def _process_event(self, content: str):

        # Add system info message with the current time and any relevant memories.
        memories = self.memory.retrieve(content)
        system_info = AgentSystemInfoMessage(
            system_info=AgentSystemInfo(
                current_time=datetime.now().strftime('%A %d %B %Y %I:%M %p'),
                user_last_active=humanize.naturaltime(datetime.now() - self.user_last_active),
                relevant_memories=memories
            )
        )

        # Call chat client
        new_messages = await self._call_chat_client([ system_info.model_dump_json(), content ])

        # Record memories
        memory_messages = [m for m in new_messages if not self._is_system_info_msg(m)]        
        self.memory.create_memories(memory_messages, memories)

    async def _call_chat_client(self, user_prompt: list[str] | str) -> list:

        # Call client
        try:
            chat_response = await self.client.chat(
                system_prompt=self.prompts.main,
                user_prompt=user_prompt,
                history=self.message_history,
                tools=self.tools,
                output_callback=self._filter_output,
                is_system_info_callback=self._is_system_info_content,
                retry_on_context_full=True
            )

        except AIClientError as e:
            if self.output_callback:
                self.output_callback(f"AI Client error: {str(e)}")
            raise

        # Preserve history
        self.message_history = chat_response.history

        return chat_response.new_messages
    

    def _filter_output(self, output: str):
        if self.output_callback and output: # and output != "NO_OUTPUT":
            self.output_callback(output)

    def _is_system_info_msg(self, msg) -> bool:
        return(
            self.client.is_user_message(msg) and
            self._is_system_info_content(self.client.get_message_content(msg))
        )

    def _is_system_info_content(self, content: str) -> bool:
        return content.strip().startswith("{\"system_info\":")

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
You are a helpful and proactive AI agent acting as a personal secretary for the user.

Perform tasks when requested using the available tool calls.
The system will notify you of events such as new emails or scheduled reminders. Respond proactively, taking into account the user's goals. The user may not be online when a system event is received.

Output - respond in one of three ways:
- Text - displayed directly to the user.
- Tool calls - invoke tools to perform actions.
- NO_OUTPUT - when no action or response is required, respond with only: NO_OUTPUT

"speak" tool:
- Use to read text aloud to the user.
- Always use when the user's message is prefixed with [Voice input].
- Always use when the user asks you to read something aloud.
- Use to notify the user of completed tasks or incoming events, but only if the user has not been active for 10+ minutes.
- Keep spoken messages short — one or two sentences — unless reading content the user has requested.
- Always include the same information as text output, as speech may not be heard.

"create_reminder" tool:
- Use to notify the user of upcoming events, appointments and tasks.
- Reminder messages must not contain relative time references such as "tomorrow", "next week", or "on Tuesday".
- Always use the resolved date and time in the message instead.
- Example: for "remind me to put the bins out tomorrow at 6PM", create the message "Put the bins out" scheduled for the actual date — NOT "Put the bins out tomorrow".

"timer" system event:
- You will receive this every 15 minutes when the user is inactive.
- This event is for you, as the agent.
- Do not notify the user of this event.

{extra_info_text if extra_info_text else ""}\
""")
