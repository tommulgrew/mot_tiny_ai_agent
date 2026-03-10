import asyncio
from dataclasses import field
import json
import re
import random
from pathlib import Path
from openai import BadRequestError
from pydantic import BaseModel
import Stemmer
from ai_client import AIClient
from ai_tools import AITools, AITool, AIToolParam
from config import MemoryConfig

class AIMemoryPrompts(BaseModel):
    create_memories: str        # Create memories from conversation snippet

class AISavedMemory(BaseModel):
    fact: str
    keywords: list[str]

class AIMemoryFile(BaseModel):
    memories: list[AISavedMemory] = field(default_factory=list)

# TODO: 
# - Log/report tool calls, and other LLM output somewhere
# - Save/load memory to file
# - Stem keywords

class AIMemory:
    """Basic AI memory service, for extracting and retrieving memories during conversation"""
    
    def __init__(self, client: AIClient, config: MemoryConfig):
        self.client = client
        self.storage_path = Path(config.storage_path)
        self.memories: list[AISavedMemory] = []
        self.prompts = create_ai_prompts()        
        self.tools = self._make_ai_tools()
        self.dirty = False                  # True if memories need to be saved

        # Load memory file
        self._load()

        # Create stemmer
        self.stemmer = Stemmer.Stemmer('english')

        # Create task queue
        self.task_queue = asyncio.Queue()
        asyncio.create_task(self._process_queue())

    def create_memories(self, conversation: str):
        """Create memories from a snippet of conversation"""
        self.task_queue.put_nowait(conversation)

    def retrieve(self, conversation: str) -> list[str]:
        """Retrieve memories based on snippet of conversation"""
        keywords = set(self.get_keywords(conversation))
        memories = [ 
            m.fact 
            for m in self.memories 
            if bool(keywords & set(m.keywords))     # Sets intersect
        ]
        random.shuffle(memories)
        if len(memories) > 8:
            memories = memories[:8]

        return memories

    async def _process_queue(self):
        """Main task queue loop"""
        while True:
            conversation: str = await self.task_queue.get()

            try:
                # Process conversation
                await self._create_memories(conversation)

            finally:
                # Task done
                self.task_queue.task_done()

    async def _create_memories(self, conversation: str):
        """Create memories"""
        # Call chat client. Memories will be created via tool calls
        try:
            response = await self.client.chat(
                system_prompt=self.prompts.create_memories,
                user_prompt=conversation,
                tools=self.tools,
                lock_service=False      # Allow agent to interrupt chat API calls
            )
        except BadRequestError as e:
            if "Context size has been exceeded" not in str(e):
                raise
            # Otherwise ignore context size exceeded. This is non-critical 
            # background process. Often a few memories will have been written
            # anyway, so just carry on and save them.
        if self.dirty:
            self._save()

    def _make_ai_tools(self) -> AITools:
        tools = AITools()
        save_memory_tool = AITool(
            name="save_memory",
            description="Save a fact to the memory store",
            params=[
                AIToolParam(name="memory", type="string", description="A statement of the fact to store"),
                AIToolParam(name="keywords", type="string", description="A comma separated list of related keywords that will trigger the memory in future chat sessions")
            ],
            async_callback=self._save_memory_tool
        )
        tools.add([ save_memory_tool ])
        return tools

    async def _save_memory_tool(self, memory: str, keywords: str) -> str:
        """save_memory tool call"""
        keyword_list = self.get_keywords(keywords)
        self.memories.append(
            AISavedMemory(
                fact=memory,
                keywords=keyword_list
            )
        )
        self.dirty = True
        return "Memory saved"

    def get_keywords(self, text: str) -> list[str]:

        # Strip out punctuation etc
        text = re.sub(r"[^\w-]", " ", text)

        # Process words
        words = [self.process_keyword(w) for w in text.split() if w]

        # Return distinct words
        return list(set(words))

    def process_keyword(self, keyword: str) -> str:
        keyword = keyword.strip().lower()
        keyword = self.stemmer.stemWord(keyword)
        return keyword

    def _load(self):
        if self.storage_path.exists():
            with open(self.storage_path, encoding="utf-8") as f:
                file = AIMemoryFile.model_validate_json(f.read())
                self.memories = file.memories

    def _save(self):
        file = AIMemoryFile(memories=self.memories)
        json_text = file.model_dump_json(indent=2)
        self.storage_path.write_text(json_text, encoding="utf-8")
        self.dirty = False
        

def create_ai_prompts() -> AIMemoryPrompts:
    return AIMemoryPrompts(
        create_memories="""\
You are a memory service for an AI chatbot.

Scan the recent chat conversation for facts and information that the chatbot
needs to remember. Include:
- Facts about the user
- Instructions for the chatbot to follow
- Other information that may be useful later on

Use the save_memory tool to save these memories in the memory store.
For each memory, include a set of appropriate keywords that will trigger the 
memory when present in future chat conversation.

Examples:
"[USER]: Hi! My name is John."
Memories: [ { memory: "User's name is John", keywords: "user,name,john" } ]

"[USER]: Can you open netflix for me. The URL is: https://www.netflix.com/browse
Memories: [ { memory: "The Netflix URL is https://www.netflix.com/browse", keywords: "netflix,movie" } ]

Important:
- Sections that start with [USER] are input from the user.
- Sections that start with [ASSISTANT] are the AI chatbot's response.

Once you have finished, respond with: DONE
""")        