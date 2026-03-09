from typing import Any, Literal, Callable
from openai import BaseModel
from config import load_config
import asyncio
from ai_client import AIClient
from ai_agent import AIAgent
from ai_tools import AITools
from tools.browser_tools import BrowserTools
from tools.file_tools import FileTools
from tools.speak_tools import SpeakTools

class UserInputEvent(BaseModel):
    type: Literal["user"]
    input: str

class SystemEvent(BaseModel):
    type: Literal["system"]
    data: Any       # Must be serializable to JSON

class App:
    """Tiny agent main application class. Implements the main wireup and event queue"""

    def __init__(self, args, output_callback: Callable[[str], None] | None = None):
        
        # Parse configuration
        self.config = load_config(args.config)

        # Wire up agent
        client = AIClient(self.config.model)
        tools = self._create_ai_tools()
        self.agent = AIAgent(client, tools, output_callback)

        # Create event queue
        self.event_queue = asyncio.Queue()
        asyncio.create_task(self._process_queue())

    def user_input(self, input: str):        
        self._queue_event(UserInputEvent(type="user", input=input))

    def system_event(self, data):
        self._queue_event(SystemEvent(type="system", data=data))

    def _queue_event(self, event: UserInputEvent | SystemEvent):
        self.event_queue.put_nowait(event)

    async def _process_queue(self):
        while True:
            event: UserInputEvent | SystemEvent = await self.event_queue.get()
            
            try:
                # Process event
                if event.type == "user":
                    await self.agent.process_user_message(event.input)

                elif event.type == "system":
                    await self.agent.process_system_event(event.data)
            
            finally:
                # Event done
                self.event_queue.task_done()

    def _create_ai_tools(self) -> AITools:
        tools = AITools()

        tools.add(BrowserTools().make_tools())
        if self.config.file_tools:
            tools.add(FileTools(self.config.file_tools).make_tools())
        tools.add(SpeakTools().make_tools())

        return tools
