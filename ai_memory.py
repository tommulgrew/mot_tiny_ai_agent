import logging
import asyncio
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
    housekeep_memories: str     # Detect duplicate and conflicting memories

class AISavedMemory(BaseModel):
    id: int
    fact: str
    keywords: list[str]
    when_created: datetime

class CreateMemoriesTask(BaseModel):
    type: Literal["create"]
    conversation: str

class MemoryHousekeepingTask(BaseModel):
    type: Literal["housekeeping"]
    memories: list[AISavedMemory]

class AIMemoryFile(BaseModel):
    memories: list[AISavedMemory] = field(default_factory=list)
    id_generator: int

class HousekeepingMemory(BaseModel):
    """Memory representation to pass to the AI model for housekeeping tasks"""
    id: int
    fact: str
    when_created: str            # Description of when created

class HousekeepingMemories(BaseModel):
    memories: list[HousekeepingMemory]

class MemoryHousekeepingAction(BaseModel):
    type: Literal["duplicate", "conflict"]
    memories: list[AISavedMemory]

class KeywordMapping(BaseModel):
    from_keywords: set[str]
    to_keywords: set[str]

# TODO: 
# - Log/report tool calls, and other LLM output somewhere
# - Save/load memory to file
# - Stem keywords

class AIMemory:
    """Basic AI memory service, for extracting and retrieving memories during conversation"""
    
    def __init__(self, client: AIClient, config: MemoryConfig):
        self.client = client
        self.storage_path = Path(config.storage_path)
        self.users_name = config.users_name
        self.agents_name = config.agents_name
        self.id_generator = 0
        self.memories: list[AISavedMemory] = []
        self.housekeeping_actions: list[MemoryHousekeepingAction] = []
        self.prompts = create_ai_prompts()        
        self.save_tools = self._make_save_memory_tools()
        self.housekeeping_tools = self._make_housekeeping_tools()
        self.dirty = False                  # True if memories need to be saved

        # Load memory file
        self._load()

        # Create stemmer
        self.stemmer = Stemmer.Stemmer('english')

        # Create keyword mappings (requires stemmer)
        self.keyword_mappings: list[KeywordMapping] = self._make_keyword_mappings()

        # Create task queue
        self.task_queue = asyncio.Queue()
        asyncio.create_task(self._process_queue())

    def create_memories(self, conversation: str):
        """Create memories from a snippet of conversation"""
        self.task_queue.put_nowait(CreateMemoriesTask(type="create", conversation=conversation))

    def retrieve(self, conversation: str, housekeeping: bool = True) -> list[str]:
        """Retrieve memories based on snippet of conversation"""

        # Add mapped keywords
        keywords = set(self.get_keywords(conversation))
        for mapping in self.keyword_mappings:
            if bool(keywords & mapping.from_keywords):
                keywords.update(mapping.to_keywords)

        memories = [ 
            m
            for m in self.memories 
            if bool(keywords & set(m.keywords))     # Sets intersect
        ]

        # Queue a memory housekeeping task
        if housekeeping:
            self.task_queue.put_nowait(MemoryHousekeepingTask(type="housekeeping", memories=memories))

        random.shuffle(memories)
        if len(memories) > 8:
            memories = memories[:8]

        # Return facts
        return [m.fact for m in memories]

    async def _process_queue(self):
        """Main task queue loop"""
        while True:
            task: CreateMemoriesTask | MemoryHousekeepingTask = await self.task_queue.get()

            try:
                # Perform task
                if task.type == "create":
                    await self._create_memories(task.conversation)
                elif task.type == "housekeeping":
                    await self._housekeep_memories(task.memories)

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

    async def _housekeep_memories(self, memories: list[AISavedMemory]):
        """Detect and resolve memory conflicts and duplicates"""

        # Convert to JSON
        housekeeping = HousekeepingMemories(
            memories=[
                HousekeepingMemory(
                    id=m.id, 
                    fact=m.fact,
                    when_created=humanize.naturaltime(datetime.now() - m.when_created)
                ) 
                for m in memories
            ]
        )
        housekeeping_json = housekeeping.model_dump_json()

        # Detect duplicate and conflicting memories
        self.housekeeping_actions = []
        await self._do_chat_tool_operation(
            system_prompt=self.prompts.housekeep_memories,
            user_prompt=housekeeping_json,
            tools=self.housekeeping_tools
        )

        # Resolve reported duplicates and conflicts
        # TO DO
        logging.info(f"{len(self.housekeeping_actions)} housekeeping actions found")

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

    def _make_housekeeping_tools(self) -> AITools:
        tools = AITools()
        report_duplicate_tool = AITool(
            name="report_duplicate",
            description="Report two memories that are duplicated of each other",
            params=[
                AIToolParam(name="id1", type="number", description="The first duplicate memory's unique ID number"),
                AIToolParam(name="id2", type="number", description="The second duplicate memory's unique ID number")
            ],
            async_callback=self._report_duplicate_tool
        )
        report_conflict_tool = AITool(
            name="report_conflict",
            description="Report two memories that conflict with each other",
            params=[
                AIToolParam(name="id1", type="number", description="The first conflicting memory's unique ID number"),
                AIToolParam(name="id2", type="number", description="The second conflicting memory's unique ID number")
            ],
            async_callback=self._report_conflict_tool
        )
        tools.add([ report_duplicate_tool, report_conflict_tool ])
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
        logging.info(f"Saved memory: {memory}")
        return "Memory saved"

    def _get_memory_by_id(self, id: int) -> AISavedMemory | None:
        return next((m for m in self.memories if m.id == id), None)

    async def _do_housekeeping_action(self, type: Literal["duplicate", "conflict"], id1: int, id2: int) -> str:
        if id1 == id2:
            return "ERROR: Memory IDs must be different"

        # Find memories
        memory1 = self._get_memory_by_id(id1)
        if not memory1:
            return f"ERROR: Memory {id1} not found"
        memory2 = self._get_memory_by_id(id2)
        if not memory2:
            return f"ERROR: Memory {id2} not found"

        # Check for existing action
        existing_action = next((
            a 
            for a in self.housekeeping_actions 
            if a.type == type and memory1 in a.memories and memory2 in a.memories
        ), None)
        if existing_action:
            return f"ERROR: {type} has already been reported"

        # Add housekeeping action
        self.housekeeping_actions.append(MemoryHousekeepingAction(type=type, memories=[memory1, memory2]))
        logging.info(f"Found {type} memories: {memory1.fact}({memory1.id}),{memory2.fact}({memory2.id})")
        return f"{type} reported succesfully"

    async def _report_duplicate_tool(self, id1: int, id2: int) -> str:
        """Report duplicate memories to be merged"""
        return await self._do_housekeeping_action("duplicate", id1, id2)

    async def _report_conflict_tool(self, id1: int, id2: int) -> str:
        """Report conflicting memories to be resolved"""
        return await self._do_housekeeping_action("conflict", id1, id2)

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

    def _make_keyword_mappings(self) -> list[KeywordMapping]:
        data = [
            [f"I Im Ill me my myself mine {self.users_name or ''}", f"user {self.users_name or ''}"],
            [f"you youll youre your yours {self.agents_name or ''}", f"assistant ai agent {self.agents_name or ''}"]
        ]
        return [
            KeywordMapping(
                from_keywords=set(self.get_keywords(d[0])),
                to_keywords=set(self.get_keywords(d[1]))
            )
            for d in data 
        ]
        

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
        housekeep_memories="""\
You are a memory service for an AI chatbot.

Examine the provided memories and check whether:

1) Any two distinct memories contradict each other. I.e. they *cannot* both be true. 

Example 1.a: "User's name is Bob" conflicts with "User's name is Mary".
Example 1.b: "Bob likes cats" conflicts with "Bob can't stand cats"

Use the report_conflict tool to report any pairs of conflicting memories (if any).

2) Any two distinct memories are duplicates of each other. I.e. they both state the *same* thing.

Example 2.a: "User's name is George" is a duplicate of "The user prefers to be called George"
Example 2.b: "George likes programming" is a duplicate of "George's hobbies include programming"

Use the report_duplicate tool to report any pairs of duplicate memories (if any).

Once you have finished, respond with: DONE
""")        
