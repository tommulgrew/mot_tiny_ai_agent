import asyncio
from dataclasses import field
import re
import random
from datetime import datetime
from pathlib import Path
from typing import Literal
from openai import BadRequestError
from pydantic import BaseModel
import Stemmer
from ai_client import AIClient
from ai_tools import AITools, AITool, AIToolParam
from config import MemoryConfig

class AIMemoryPrompts(BaseModel):
    create_memories: str        # Create memories from conversation snippet
    classify_memories: str      # Detect duplicate, conflicting and expired memories

class AISavedMemory(BaseModel):
    id: int
    fact: str
    keywords: list[str]
    when_created: datetime

class CreateMemoriesTask(BaseModel):
    type: Literal["create"]
    conversation: str

class ClassifyMemoriesTask(BaseModel):
    type: Literal["classify"]
    conversation: str
    keywords: set[str]
    memories: list[AISavedMemory]

class AIMemoryFile(BaseModel):
    memories: list[AISavedMemory] = field(default_factory=list)
    id_generator: int

# TODO: 
# - Log/report tool calls, and other LLM output somewhere
# - Save/load memory to file
# - Stem keywords

class AIMemory:
    """Basic AI memory service, for extracting and retrieving memories during conversation"""
    
    def __init__(self, client: AIClient, config: MemoryConfig):
        self.client = client
        self.storage_path = Path(config.storage_path)
        self.id_generator = 0
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
        self.task_queue.put_nowait(CreateMemoriesTask(type="create", conversation=conversation))

    def retrieve(self, conversation: str) -> list[str]:
        """Retrieve memories based on snippet of conversation"""
        keywords = set(self.get_keywords(conversation))
        memories = [ 
            m
            for m in self.memories 
            if bool(keywords & set(m.keywords))     # Sets intersect
        ]

        # Queue a task to classify memories
        self.task_queue.put_nowait(ClassifyMemoriesTask(type="classify", memories=memories, keywords=keywords, conversation=conversation))

        random.shuffle(memories)
        if len(memories) > 8:
            memories = memories[:8]

        # Return facts
        return [m.fact for m in memories]

    async def _process_queue(self):
        """Main task queue loop"""
        while True:
            task: CreateMemoriesTask | ClassifyMemoriesTask = await self.task_queue.get()

            try:
                # Perform task
                if task.type == "create":
                    await self._create_memories(task.conversation)
                elif task.type == "classify":
                    await self._classify_memories(task.memories, task.conversation, task.keywords)

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
                tools=self.tools
            )
        except BadRequestError as e:
            if "Context size has been exceeded" not in str(e):
                raise
            # Otherwise ignore context size exceeded. This is non-critical 
            # background process. Often a few memories will have been written
            # anyway, so just carry on and save them.
        if self.dirty:
            self._save()

    async def _classify_memories(self, memories: list[AISavedMemory], convesation: str, keywords: set[str]):
        """Classify memories for relevance, redundancy and correctness"""
        # TO DO

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
                when_created=datetime.now(),
                id=self._generate_id(),
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

    def _generate_id(self) -> int:
        self.id_generator += 1
        return self.id_generator

    def _load(self):
        if self.storage_path.exists():
            with open(self.storage_path, encoding="utf-8") as f:
                file = AIMemoryFile.model_validate_json(f.read())
                self.memories = file.memories
                self.id_generator = file.id_generator

    def _save(self):
        file = AIMemoryFile(memories=self.memories, id_generator=self.id_generator)
        json_text = file.model_dump_json(indent=2)
        self.storage_path.write_text(json_text, encoding="utf-8")
        self.dirty = False
        

def create_ai_prompts() -> AIMemoryPrompts:
    return AIMemoryPrompts(
        create_memories="""\
You are a memory service for an AI chatbot.

Scan the recent chat conversation for facts and information that the chatbot needs to remember. Include:
- Facts about the user
- Instructions for the chatbot to follow
- Other information that may be useful later on

Use the save_memory tool to save these memories in the memory store.
For each memory, include a set of appropriate keywords that will trigger the memory when present in future chat conversation. Peoples' names should always be included in the keywords

Examples:
"[USER]: Hi! My name is John."
Memories: [ { memory: "User's name is John", keywords: "user,name,john" } ]

"[USER]: Can you open netflix for me. The URL is: https://www.netflix.com/browse
Memories: [ { memory: "The Netflix URL is https://www.netflix.com/browse", keywords: "netflix,movie" } ]

Important:
- Sections that start with [USER] are input from the user.
- Sections that start with [ASSISTANT] are the AI chatbot's response.

Once you have finished, respond with: DONE
""",
        classify_memories="""\
You are a memory service for an AI chatbot.

Process the supplied memories and classify each one as either:
"contradicts" - Contradicts another memory
"duplicate" - Duplicate of another memory
"relevant" - Relevant to the conversation
"not_relevant" - Not relevant to the conversation

Use the supplied conversation snippet to determine relevance.

Return the results as JSON in the following format:

[
    {{ "id": 123, "class": "contradicts", "otherid": 157 }},
    {{ "id": 157, "class": "contradicts", "otherid": 123 }},
    {{ "id": 45, "class": "duplicate", "otherid": 371 }},
    {{ "id": 371, "class": "duplicate", "otherid": 45 }},
    {{ "id": 72, "class": "not_relevant" }},
    {{ "id": 94, "class": "relevant" }},
    {{ "id": 32, "class": "relevant" }},
    {{ "id": 33, "class": "not_relevant" }}
]

Note: "otherid" is required for contradictions and duplicates.

Respond only with JSON in the format described. No preamble or explanation.
""")        


# Add option once memory age is tracked
# "expired" - Memory is no longer relevant
