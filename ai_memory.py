import asyncio
from pydantic import BaseModel
from ai_client import AIClient
from ai_tools import AITools, AITool, AIToolParam

class AIMemoryPrompts(BaseModel):
    create_memories: str        # Create memories from conversation snippet

class AISavedMemory(BaseModel):
    fact: str
    keywords: list[str]

# TODO: 
# - Log/report tool calls, and other LLM output somewhere
# - Save/load memory to file
# - Stem keywords

class AIMemory:
    """Basic AI memory service, for extracting and retrieving memories during conversation"""
    
    def __init__(self, client: AIClient):
        self.client = client
        self.memories: list[AISavedMemory] = []
        self.prompts = create_ai_prompts()        
        self.tools = self._make_ai_tools()

        # Create task queue
        self.task_queue = asyncio.Queue()
        asyncio.create_task(self._process_queue())

    def create_memories(self, conversation: str):
        """Create memories from a snippet of conversation"""
        self.task_queue.put_nowait(conversation)

    def retrieve(self, conversation: str) -> list[str]:
        """Retrieve memories based on snippet of conversation"""
        # TO DO
        return [ "Assistant's name is Alfred", "User's name is Tom" ]

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
        response = await self.client.chat(
            system_prompt=self.prompts.create_memories,
            user_prompt=conversation,
            tools=self.tools
        )

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
        self.memories.append(
            AISavedMemory(
                fact=memory,
                keywords=[ kw.strip().lower() for kw in keywords.split() ]
            )
        )
        return "Memory saved"


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

Once you have finished, respond with: DONE
""")        