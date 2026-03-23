from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from ai.tools import AITool, AIToolError, AIToolParam
from config import MemoryConfig

VALID_DURATIONS = ("today", "hours", "days", "permanent")

@dataclass
class AISavedWorkingMemory:
    id: int
    memory: str
    when_created: datetime
    when_expires: datetime | None

class AIWorkingMemory:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger("tinyagent.memory.working")
        self._memories: list[AISavedWorkingMemory] = []
        self._id_generator = 0
        self._load()

    def make_tools(self) -> list[AITool]:
        return [
            AITool(
                name="record_working_memory",
                description=(
                    "Record a working memory. This memory will remain in the chat context window "
                    "until it's duration expires, or is removed via `remove_working_memory`."
                ),
                params=[
                    AIToolParam(name="memory", type="string", description="The information to remember. E.g. 'Never use the speak tool after 10pm.', 'User has been notified of invoice email.'"),
                    AIToolParam(name="duration", type="string", description='Duration: "today", "hours", "days", "permanent".'),
                    AIToolParam(name="count", type="number", description='The number of "days" or "hours" to set the duration to.', optional=True)
                ],
                async_callback=self._record_memory
            ),
            AITool(
                name="delete_working_memory",
                description="Delete a working memory.",
                params=[
                    AIToolParam(name="id", type="integer", description="Working memory ID")
                ],
                async_callback=self._delete_memory
            )
        ]

    def _load(self):
        # To do
        pass

    def _save(self):
        # To do
        pass

    async def _record_memory(self, memory: str, duration: str, count: int | None) -> str:
        if len(self._memories) > self.config.working_memory_limit:
            raise AIToolError(
                f"The working memory is full ({self.config.working_memory_limit} memories)."
                "Delete an existing working memory before recording a new one."
            )

        duration = _validate_duration(duration)
        created = datetime.now()
        expiry = _get_expiry_datetime(created, duration, count or 1)

        saved_memory = AISavedWorkingMemory(
            id=self._next_id(),
            memory=memory,
            when_created=created,
            when_expires=expiry
        )
        self._memories.append(saved_memory)
        self._save()
        self.logger.info(f"Added working memory [{saved_memory.id}]: {saved_memory.memory}")
        return f"Working memory added (ID: {saved_memory.id})."

    async def _delete_memory(self, id: int) -> str:
        m = self._find_memory(id)
        self._memories.remove(m)
        self._save()
        self.logger.info(f"Deleted working memory [{m.id}]: {m.memory}")
        return f"Working memory [{id}] deleted."

    def _next_id(self) -> int:
        self._id_generator += 1
        return self._id_generator

    def _find_memory(self, id: int) -> AISavedWorkingMemory:
        m = next((m for m in self._memories if m.id == id), None)
        if m is None:
            raise AIToolError(f"No working memory with ID {id}.")
        return m

def _validate_duration(duration: str) -> str:
    d = duration.lower().strip()
    if d not in VALID_DURATIONS:
        raise AIToolError(f'Invalid duration "{duration}". Use "hour", "day", "week", "month", or "permanent"')
    return d

def _get_expiry_datetime(dt: datetime, duration: str, count: int) -> datetime | None:
    if duration == "today":
        return (dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    if duration == "hour":
        return dt + timedelta(hours=count)
    if duration == "day":
        return dt + timedelta(days=count)
    if duration == "permanent":
        return None
    assert(False)       # Should never happen
