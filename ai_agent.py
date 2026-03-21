import asyncio
from dataclasses import dataclass, field
import logging
import json
from datetime import datetime
from typing import Any, Callable, Literal, Union
import humanize
from pydantic import BaseModel
from ai.client import AIChatClient, AIChatMessageHistory
from ai.errors import AIClientError
from ai.tools import AITool, AITools, AIToolParam
from ai_memory import AIMemory
from config import AgentConfig

class UserInputEvent(BaseModel):
    type: Literal["user"]
    input: str

class SystemEvent(BaseModel):
    type: Literal["system"]
    data: Any       # Must be serializable to JSON

AgentEvent = Union[UserInputEvent, SystemEvent]

@dataclass(order=True)
class PrioritisedEvent:
    priority: int
    event: AgentEvent = field(compare=False)

# Priorities
PRIORITY_USER = 0
PRIORITY_SYSTEM = 1
PRIORITY_SYSTEM_LOWER = 2

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
    def __init__(self, config: AgentConfig, client: AIChatClient, memory: AIMemory, tools: AITools | None, output_callback: Callable[[str], None] | None = None):
        self.logger = logging.getLogger("tinyagent.agent")
        self.config = config
        self.client = client
        self.message_accessor = client.get_message_accessor()
        self.memory = memory
        self.tools = tools
        if self.tools:
            self.tools.add([self._make_recall_memories_tool()])
        self.output_callback = output_callback
        self.prompts = create_ai_prompts(users_name=config.users_name, agents_name=config.agents_name, extra_info=config.extra_info)
        self.message_history = AIChatMessageHistory(message_accessor=self.message_accessor)
        self.user_last_active = datetime.now()

        # Event queue
        self.cancel_chat_completion_event = asyncio.Event()
        self.event_queue = asyncio.PriorityQueue()
        asyncio.create_task(self._queue_worker())

    def user_input(self, input: str):
        # Cancel any existing chat completion, and submit with PRIORITY_USER
        # Message will be processed ahead of any system events etc.
        self.cancel_chat_completion_event.set()
        self._queue_event(UserInputEvent(type="user", input=input), PRIORITY_USER)

    def voice_event(self, text: str):
        if self.output_callback:
            self.output_callback(f"VOICE INPUT: {text}")
        self.user_input(f"[Voice input]: {text}")

    def system_event(self, data, priority: int = PRIORITY_SYSTEM):
        self._queue_event(SystemEvent(type="system", data=data), priority)

    def _queue_event(self, event: AgentEvent, priority: int):
        self.event_queue.put_nowait(PrioritisedEvent(priority=priority, event=event))

    async def _queue_worker(self):
        """Main queue worker loop"""
        while True:
            prioritised_event: PrioritisedEvent = await self.event_queue.get()
            event = prioritised_event.event
            
            try:
                # Process event
                if event.type == "user":
                    await self._process_user_message(event.input)

                elif event.type == "system":
                    await self._process_system_event(event.data)
            
            except AIClientError as e:
                self.logger.error(f"AI client error: {str(e)}")

            finally:
                # Event done
                self.event_queue.task_done()            

    async def _process_user_message(self, message: str):
        self.user_last_active = datetime.now()
        await self._process_event(message)

    async def _process_system_event(self, data):
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
            self.cancel_chat_completion_event.clear()
            chat_response = await self.client.chat(
                system_prompt=self.prompts.main,
                user_prompt=user_prompt,
                history=self.message_history,
                tools=self.tools,
                output_callback=self._filter_output,
                is_system_info_callback=self._is_system_info_content,
                retry_on_context_full=True,
                cancel_event=self.cancel_chat_completion_event
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
            self.message_accessor.is_user_message(msg) and
            self._is_system_info_content(self.message_accessor.get_content(msg))
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

"todo" tools:
- Use to manage the user's to-do list.
- Only use "complete_todo_item" after the user has confirmed they are complete.
- Likewise only use "delete_todo_item" when the user agrees the item is incorrect.
- If the to-do list fills up and no more items can be added, consult with the user about whether existing items can be deleted to make room.

{extra_info_text if extra_info_text else ""}\
""")
