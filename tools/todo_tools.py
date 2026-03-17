import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from ai_tools import AITool, AIToolError, AIToolParam
from config import TodoConfig

VALID_PRIORITIES = ("high", "medium", "low")
PRIORITY_ORDER   = {"high": 0, "medium": 1, "low": 2}
LIST_LIMITS      = {"agent": "agent_list_limit", "user": "user_list_limit"}


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
        self._tasks: dict[str, list[Task]] = {"agent": [], "user": []}
        self._id_generator = 0
        self._load()

    def make_tools(self) -> list[AITool]:
        list_desc = 'Which to-do list: "user" for the user\'s tasks, "agent" for your own pending objectives'
        return [
            AITool(
                name="add_todo_item",
                description=(
                    "Add an item to a to-do list. "
                    "Use the 'user' list for tasks the user needs to do. "
                    "Use the 'agent' list to track your own objectives that cannot be completed right now "
                    "(e.g. a task that requires waiting for an event, or that you want to revisit later)."
                ),
                params=[
                    AIToolParam(name="list_name",   type="string", description=list_desc),
                    AIToolParam(name="description", type="string", description="Brief description of the item"),
                    AIToolParam(name="priority",    type="string", description='Priority: "high", "medium" (default), or "low"', optional=True),
                    AIToolParam(name="due_by",      type="string", description='Optional due date in "YYYY-MM-DD" format', optional=True),
                ],
                async_callback=self._add_task,
            ),
            AITool(
                name="list_todo_items",
                description="List all active items in a to-do list, sorted by priority then due date.",
                params=[
                    AIToolParam(name="list_name", type="string", description=list_desc),
                ],
                async_callback=self._list_tasks,
            ),
            AITool(
                name="complete_todo_item",
                description="Mark a todo item as completed. The item is removed from the active list and appended to the completed log.",
                params=[
                    AIToolParam(name="list_name",   type="string", description=list_desc),
                    AIToolParam(name="id",          type="integer", description="Item ID"),
                ],
                async_callback=self._complete_task,
            ),
            AITool(
                name="delete_todo_item",
                description="Delete a todo item without marking it as completed (use for items added in error).",
                params=[
                    AIToolParam(name="list_name",   type="string", description=list_desc),
                    AIToolParam(name="id",          type="integer", description="Item ID"),
                ],
                async_callback=self._delete_task,
            ),
            AITool(
                name="update_todo_item",
                description=(
                    "Update a todo item's description, priority, or due date. "
                    "Only the parameters you provide are changed. "
                    'To clear the due date, pass due_by as "".'
                ),
                params=[
                    AIToolParam(name="list_name",   type="string",  description=list_desc),
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
        list: str,
        description: str,
        priority: str | None = None,
        due_by: str | None = None,
    ) -> str:
        tasks = self._get_list(list)
        limit = self._get_limit(list)

        if len(tasks) >= limit:
            raise AIToolError(
                f"The {list} task list is full ({limit} tasks). "
                "Complete or delete a task before adding a new one."
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
        tasks.append(task)
        self._save()
        self.logger.info(f"Added {list} task [{task.id}]: {description}")
        return f"Task added (ID: {task.id})."

    async def _list_tasks(self, list: str) -> str:
        tasks = self._get_list(list)
        limit = self._get_limit(list)

        if not tasks:
            return f"No {list} tasks."

        sorted_tasks = sorted(tasks, key=_sort_key)
        lines = [f"{list.capitalize()} tasks ({len(tasks)}/{limit}):"]
        for t in sorted_tasks:
            due = t.due_by or "-"
            lines.append(f"[{t.id}] {t.priority.upper():<6}  {due:<12}  {t.description}")
        return "\n".join(lines)

    async def _complete_task(self, list: str, id: int) -> str:
        task = self._find_task(list, id)
        self._get_list(list).remove(task)
        self._append_completed_log(list, task, completed_at=datetime.now().isoformat())
        self._save()
        self.logger.info(f"Completed {list} task [{id}]: {task.description}")
        return f"Task [{id}] marked as completed."

    async def _delete_task(self, list: str, id: int) -> str:
        task = self._find_task(list, id)
        self._get_list(list).remove(task)
        self._save()
        self.logger.info(f"Deleted {list} task [{id}]: {task.description}")
        return f"Task [{id}] deleted."

    async def _update_task(
        self,
        list: str,
        id: int,
        description: str | None = None,
        priority: str | None = None,
        due_by: str | None = None,
    ) -> str:
        task = self._find_task(list, id)

        if description is not None:
            task.description = description
        if priority is not None:
            task.priority = _validate_priority(priority)
        if due_by is not None:
            task.due_by = _validate_due_by(due_by) if due_by else None

        self._save()
        return f"Task [{id}] updated."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_list(self, list_name: str) -> list[Task]:
        if list_name not in self._tasks:
            raise AIToolError(f'Unknown list "{list_name}". Use "agent" or "user".')
        return self._tasks[list_name]

    def _get_limit(self, list_name: str) -> int:
        return getattr(self.config, LIST_LIMITS[list_name])

    def _find_task(self, list_name: str, id: int) -> Task:
        task = next((t for t in self._get_list(list_name) if t.id == id), None)
        if task is None:
            raise AIToolError(f"No task with ID {id} in the {list_name} list.")
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
            "tasks": {
                list_name: [asdict(t) for t in tasks]
                for list_name, tasks in self._tasks.items()
            },
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
            for list_name, task_list in data.get("tasks", {}).items():
                if list_name in self._tasks:
                    self._tasks[list_name] = [Task(**t) for t in task_list]
        except Exception as e:
            self.logger.warning(f"Could not load tasks: {e}")

    def _append_completed_log(self, list_name: str, task: Task, completed_at: str) -> None:
        entry = {
            **asdict(task),
            "list": list_name,
            "completed_at": completed_at,
        }
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
    # Tasks with no due date sort after those with a date
    due_rank = task.due_by or "9999-99-99"
    return (priority_rank, due_rank, task.id)
