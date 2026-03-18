import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from ai_tools import AITool, AIToolError, AIToolParam
from config import TodoConfig

VALID_PRIORITIES = ("high", "medium", "low")
PRIORITY_ORDER   = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Task:
    id: int
    description: str
    priority: str           # "high" | "medium" | "low"
    due_by: str | None      # "YYYY-MM-DD" or None
    created_at: str         # ISO datetime string


class TodoTools:
    def __init__(self, config: TodoConfig):
        self.config = config
        self.logger = logging.getLogger("tinyagent.todo")
        self._tasks: list[Task] = []
        self._id_generator = 0
        self._load()

    def make_tools(self) -> list[AITool]:
        return [
            AITool(
                name="add_todo_item",
                description=(
                    f"Add an item to the user's to-do list (max {self.config.list_limit} items). "
                    "Returns an error if the list is full."
                ),
                params=[
                    AIToolParam(name="description", type="string", description="Brief description of the task"),
                    AIToolParam(name="priority",    type="string", description='Priority: "high", "medium" (default), or "low"', optional=True),
                    AIToolParam(name="due_by",      type="string", description='Optional due date in "YYYY-MM-DD" format', optional=True),
                ],
                async_callback=self._add_task,
            ),
            AITool(
                name="list_todo_items",
                description="List all active items on the user's to-do list, sorted by priority then due date.",
                params=[],
                async_callback=self._list_tasks,
            ),
            AITool(
                name="complete_todo_item",
                description="Mark a to-do item as completed. It is removed from the active list and appended to the completed log.",
                params=[
                    AIToolParam(name="id", type="integer", description="Item ID"),
                ],
                async_callback=self._complete_task,
            ),
            AITool(
                name="delete_todo_item",
                description="Delete a to-do item without marking it as completed (use for items added in error).",
                params=[
                    AIToolParam(name="id", type="integer", description="Item ID"),
                ],
                async_callback=self._delete_task,
            ),
            AITool(
                name="update_todo_item",
                description=(
                    "Update a to-do item's description, priority, or due date. "
                    "Only the parameters you provide are changed. "
                    'To clear the due date, pass due_by as "".'
                ),
                params=[
                    AIToolParam(name="id",          type="integer", description="Item ID"),
                    AIToolParam(name="description", type="string",  description="New description", optional=True),
                    AIToolParam(name="priority",    type="string",  description='New priority: "high", "medium", or "low"', optional=True),
                    AIToolParam(name="due_by",      type="string",  description='New due date "YYYY-MM-DD", or "" to clear', optional=True),
                ],
                async_callback=self._update_task,
            ),
        ]

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    async def _add_task(
        self,
        description: str,
        priority: str | None = None,
        due_by: str | None = None,
    ) -> str:
        if len(self._tasks) >= self.config.list_limit:
            raise AIToolError(
                f"The to-do list is full ({self.config.list_limit} items). "
                "Complete or delete an item before adding a new one."
            )

        priority = _validate_priority(priority or "medium")
        due_by   = _validate_due_by(due_by) if due_by else None

        task = Task(
            id=self._next_id(),
            description=description,
            priority=priority,
            due_by=due_by,
            created_at=datetime.now().isoformat(),
        )
        self._tasks.append(task)
        self._save()
        self.logger.info(f"Added todo [{task.id}]: {description}")
        return f"Item added (ID: {task.id})."

    async def _list_tasks(self) -> str:
        if not self._tasks:
            return "The to-do list is empty."

        sorted_tasks = sorted(self._tasks, key=_sort_key)
        lines = [f"To-do ({len(self._tasks)}/{self.config.list_limit}):"]
        for t in sorted_tasks:
            due = t.due_by or "-"
            lines.append(f"[{t.id}] {t.priority.upper():<6}  {due:<12}  {t.description}")
        return "\n".join(lines)

    async def _complete_task(self, id: int) -> str:
        task = self._find_task(id)
        self._tasks.remove(task)
        self._append_completed_log(task, completed_at=datetime.now().isoformat())
        self._save()
        self.logger.info(f"Completed todo [{id}]: {task.description}")
        return f"Item [{id}] marked as completed."

    async def _delete_task(self, id: int) -> str:
        task = self._find_task(id)
        self._tasks.remove(task)
        self._save()
        self.logger.info(f"Deleted todo [{id}]: {task.description}")
        return f"Item [{id}] deleted."

    async def _update_task(
        self,
        id: int,
        description: str | None = None,
        priority: str | None = None,
        due_by: str | None = None,
    ) -> str:
        task = self._find_task(id)

        if description is not None:
            task.description = description
        if priority is not None:
            task.priority = _validate_priority(priority)
        if due_by is not None:
            task.due_by = _validate_due_by(due_by) if due_by else None

        self._save()
        return f"Item [{id}] updated."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_task(self, id: int) -> Task:
        task = next((t for t in self._tasks if t.id == id), None)
        if task is None:
            raise AIToolError(f"No to-do item with ID {id}.")
        return task

    def _next_id(self) -> int:
        self._id_generator += 1
        return self._id_generator

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        data = {
            "id_generator": self._id_generator,
            "tasks": [asdict(t) for t in self._tasks],
        }
        Path(self.config.storage_path).write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        path = Path(self.config.storage_path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._id_generator = data.get("id_generator", 0)
            # Support both old dict-of-lists format and new flat list format
            tasks_data = data.get("tasks", [])
            if isinstance(tasks_data, dict):
                tasks_data = tasks_data.get("user", [])
            self._tasks = [Task(**t) for t in tasks_data]
        except Exception as e:
            self.logger.warning(f"Could not load tasks: {e}")

    def _append_completed_log(self, task: Task, completed_at: str) -> None:
        entry = {**asdict(task), "completed_at": completed_at}
        with open(self.config.completed_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _validate_priority(priority: str) -> str:
    p = priority.lower()
    if p not in VALID_PRIORITIES:
        raise AIToolError(f'Invalid priority "{priority}". Use "high", "medium", or "low".')
    return p


def _validate_due_by(due_by: str) -> str:
    try:
        datetime.strptime(due_by, "%Y-%m-%d")
    except ValueError:
        raise AIToolError(f'Invalid due date "{due_by}". Use "YYYY-MM-DD" format.')
    return due_by


def _sort_key(task: Task):
    priority_rank = PRIORITY_ORDER.get(task.priority, 1)
    due_rank = task.due_by or "9999-99-99"
    return (priority_rank, due_rank, task.id)
