from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Callable
from ai_memory import AIMemory
from ai_working_memory import AIWorkingMemory
from config import load_config
import asyncio
from openai_impl.openai_client import OpenAIChatClient
from ai_agent import AIAgent
from ai.tools import AITools
from speech_input import SpeechToTextInput
from tools.app_tools import AppTools
from tools.browser_tools import BrowserTools
from tools.email_tools import EmailTools
from tools.file_tools import FileTools
from tools.speak_tools import SpeakTools
from tools.reminder_tools import ReminderTools
from tools.todo_tools import TodoTools
from tools.search_tools import SearchTools

class App:
    """Tiny agent main application class. Implements the main wireup and event queue"""

    def __init__(self, args, output_callback: Callable[[str], None] | None = None):
        self.logger = logging.getLogger("tinyagent")
        
        # Parse configuration
        self.config = load_config(args.config)
        self.output_callback = output_callback

        # Wire up agent
        client = OpenAIChatClient(config=self.config.model)
        memory = AIMemory(
            client=client, 
            config=self.config.memory, 
            agent_config=self.config.agent
        )
        working_memory = AIWorkingMemory(config=self.config.memory)
        tools = self._create_ai_tools()
        tools.add(working_memory.make_tools())
        self.agent = AIAgent(
            config=self.config.agent,
            client=client, 
            memory=memory,
            working_memory=working_memory, 
            tools=tools, 
            output_callback=output_callback
        )
        memory.user_last_active_callback = self._get_user_last_active

        # Init speech-to-text
        self.speech_to_text: SpeechToTextInput | None = None
        if self.config.speech_to_text.enabled:
            self.speech_to_text = SpeechToTextInput(self.config.speech_to_text, inject_callback=self.agent.voice_event)
            self.speech_to_text.start()

        # Periodic timer event
        asyncio.create_task(self._timer_worker())

    def _create_ai_tools(self) -> AITools:
        tools = AITools()

        if self.config.reminders:
            reminder_tools = ReminderTools(Path(self.config.reminders.storage_path), self._reminder_callback)
            asyncio.create_task(reminder_tools.check_task())
            tools.add(reminder_tools.make_tools())

        tools.add(BrowserTools().make_tools())
        if self.config.file_tools:
            tools.add(FileTools(self.config.file_tools).make_tools())
        if self.config.speak and self.config.speak.enabled:
            tools.add(SpeakTools().make_tools())
        if self.config.allowed_apps:
            tools.add(AppTools(self.config.allowed_apps).make_tools())
        if self.config.email:
            email_tools = EmailTools(self.config.email, self._email_event_callback)
            asyncio.create_task(email_tools.check_task())
            tools.add(email_tools.make_tools())
        if self.config.todo:
            tools.add(TodoTools(self.config.todo).make_tools())
        if self.config.web_search and self.config.web_search.enabled:
            tools.add(SearchTools().make_tools())

        return tools

    def _email_event_callback(self, data: dict):
        self.agent.system_event(data)

    def _reminder_callback(self, message: str):
        self.agent.system_event({
            "system_event": {
                "type": "scheduled reminder",
                "message" : message
            }
        })

    def _get_user_last_active(self) -> datetime:
        return self.agent.user_last_active

    async def _timer_worker(self):
        while True:
            # Timer event fires every 15 minutes, if the user has been inactive
            # for at least 10 minutes
            await asyncio.sleep(900)
            if datetime.now() - self.agent.user_last_active >= timedelta(minutes=10):
                self.agent.system_event({
                    "system_event": {
                        "type": "timer",
                        "message" : "This is a 15-minute periodic timer event"
                    }
                })
