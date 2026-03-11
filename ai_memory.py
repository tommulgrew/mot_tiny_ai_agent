import logging
import asyncio
import json
from dataclasses import field
import re
import random
from datetime import datetime
from pathlib import Path
from typing import Literal
from openai import BadRequestError
from pydantic import BaseModel, Field, ValidationError
import Stemmer
import humanize
from ai_client import AIClient
from ai_tools import AITools, AITool, AIToolParam
from config import MemoryConfig

class AIMemoryPrompts(BaseModel):
    create_memories: str        # Create memories from conversation snippet
    delete_memories: str        # Delete incorrect, duplicate or no-longer required memories

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

class DeletableMemory(BaseModel):
    id: int
    fact: str
    when_created: str            # Description of when created

class DeletableMemories(BaseModel):
    memories: list[DeletableMemory]
    conversation_snippet: str   

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
        self.save_tools = self._make_save_memory_tools()
        self.delete_tools = self._make_delete_memory_tools()
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

    async def _do_chat_tool_operation(self, system_prompt: str, user_prompt: str, tools: AITools):
        try:
            response = await self.client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tools
            )
        except BadRequestError as e:
            if "Context size has been exceeded" not in str(e):
                raise
            # Otherwise ignore context size exceeded. This is non-critical 
            # background process. Often a few memories will have been written
            # anyway, so just carry on and save them.
        if self.dirty:
            self._save()

    async def _create_memories(self, conversation: str):
        """Create memories"""
        await self._do_chat_tool_operation(
            system_prompt=self.prompts.create_memories, 
            user_prompt=conversation,
            tools=self.save_tools
        )

    async def _classify_memories(self, memories: list[AISavedMemory], conversation: str, keywords: set[str]):
        """Classify memories for relevance, redundancy and correctness"""

        # Sort newer memories first
        memories_by_age = sorted(memories, key=lambda x: x.when_created, reverse=True)

        # Convert to JSON
        deletable = DeletableMemories(
            memories=[
                DeletableMemory(
                    id=m.id, 
                    fact=m.fact,
                    when_created=humanize.naturaltime(datetime.now() - m.when_created)
                ) 
                for m in memories_by_age
            ],
            conversation_snippet=conversation
        )
        deletable_json = deletable.model_dump_json()

        # Delete memories
        await self._do_chat_tool_operation(
            system_prompt=self.prompts.delete_memories,
            user_prompt=deletable_json,
            tools=self.delete_tools
        )

    def _make_save_memory_tools(self) -> AITools:
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

    def _make_delete_memory_tools(self) -> AITools:
        tools = AITools()
        delete_memory_tool = AITool(
            name="delete_memory",
            description="Delete a memory from the memory store",
            params=[
                AIToolParam(name="id", type="number", description="The memory's unique ID number")
            ],
            async_callback=self._delete_memory_tool
        )
        tools.add([ delete_memory_tool ])
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

    async def _delete_memory_tool(self, id: int) -> str:
        """delete_memory tool call"""
        memory = next((m for m in self.memories if m.id == id), None)
        if memory:
            self.memories.remove(memory)
            self.dirty = True
            return f"Memory {id} deleted"
        else:
            return f"ERROR: Memory {id} not found"

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
        delete_memories="""\
You are a memory service for an AI chatbot.

Consider the list of memories in relation to each other and to the conversation
snippet, to determine which memories should be deleted, using the "delete_memory" tool.

Examine each memory *in order*. Evaluate:
1. Does information in the conversation snippet *specifically* contradict the memory? Note: It's okay if the snippet does not mention the memory - the memory could be from an earlier part of the conversation.
2. Does the memory contradict a *previous* memory in the list?
3. Is the memory a duplicate of a *previous* memory in the list?
4. Is the memory no longer relevant or useful?

If the answer to any of the questions is yes, then delete the memory.

Once you have finished, respond with: DONE
""")        


# Add option once memory age is tracked
# "expired" - Memory is no longer relevant
