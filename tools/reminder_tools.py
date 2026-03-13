import asyncio
import calendar
import json
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from ai_tools import AIToolParam, AITool, AIToolError

VALID_RECURRENCES = ("daily", "weekly", "monthly")


@dataclass
class Reminder:
    id: str
    message: str
    trigger_time: datetime
    recurrence: str | None
    created_at: datetime


class ReminderTools:
    def __init__(self, storage_path: Path, event_callback: Callable[[str], None]):
        self.storage_path = storage_path
        self.event_callback = event_callback
        self.reminders: dict[str, Reminder] = {}
        self._load()

    def make_tools(self) -> list[AITool]:
        return [
            AITool(
                name="create_reminder",
                description=(
                    "Create a reminder that fires an event at a specified date and time, with a message you provide."
                    "You, the AI agent, will receive a notification when the reminder fires, which you can pass on to the user, or act upon as appropriate. "
                    "Optionally repeats on a schedule. "
                    "Time format: 'YYYY-MM-DD HH:MM' (e.g. '2026-03-04 09:00'). "
                    "Valid recurrence values: 'daily', 'weekly', 'monthly'."
                ),
                params=[
                    AIToolParam(name="message", type="string", description="The reminder message to deliver when triggered"),
                    AIToolParam(name="trigger_time", type="string", description="When to trigger, in 'YYYY-MM-DD HH:MM' format"),
                    AIToolParam(name="recurrence", type="string", description="Optional repeat schedule: 'daily', 'weekly', or 'monthly'", optional=True),
                ],
                async_callback=self._create_reminder,
            ),
            AITool(
                name="list_reminders",
                description="List all active reminders with their IDs, next trigger times, recurrence, and messages.",
                params=[],
                async_callback=self._list_reminders,
            ),
            AITool(
                name="delete_reminder",
                description="Cancel and delete a reminder by its ID. Use list_reminders to find IDs.",
                params=[
                    AIToolParam(name="id", type="string", description="The reminder ID to delete"),
                ],
                async_callback=self._delete_reminder,
            ),
        ]

    async def check_task(self):
        """Background task: checks for due reminders every 30 seconds."""
        while True:
            await asyncio.sleep(30)
            try:
                now = datetime.now()
                changed = False
                for reminder in list(self.reminders.values()):
                    if reminder.trigger_time <= now:
                        self.event_callback(reminder.message)
                        if reminder.recurrence:
                            # Advance past all missed occurrences to the next future time
                            next_time = reminder.trigger_time
                            while next_time <= now:
                                next_time = _advance(next_time, reminder.recurrence)
                            reminder.trigger_time = next_time
                        else:
                            del self.reminders[reminder.id]
                        changed = True
                if changed:
                    self._save()
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"Warning: reminder check error: {exc}\n{tb}")

    async def _create_reminder(self, message: str, trigger_time: str, recurrence: str | None = None) -> str:
        dt = _parse_datetime(trigger_time)

        if dt <= datetime.now():
            raise AIToolError("Trigger time must be in the future.")

        if recurrence is not None and recurrence not in VALID_RECURRENCES:
            raise AIToolError(
                f"Invalid recurrence '{recurrence}'. Choose from: {', '.join(VALID_RECURRENCES)}."
            )

        reminder_id = uuid.uuid4().hex[:8]
        self.reminders[reminder_id] = Reminder(
            id=reminder_id,
            message=message,
            trigger_time=dt,
            recurrence=recurrence,
            created_at=datetime.now(),
        )
        self._save()

        recurrence_str = f", repeating {recurrence}" if recurrence else ""
        return f"Reminder created (ID: {reminder_id}) for {dt.strftime('%Y-%m-%d %H:%M')}{recurrence_str}."

    async def _list_reminders(self) -> str:
        if not self.reminders:
            return "No active reminders."
        lines = []
        for r in sorted(self.reminders.values(), key=lambda r: r.trigger_time):
            recurrence_str = f" [{r.recurrence}]" if r.recurrence else ""
            lines.append(f"{r.id}  {r.trigger_time.strftime('%Y-%m-%d %H:%M')}{recurrence_str}  {r.message}")
        return "\n".join(lines)

    async def _delete_reminder(self, id: str) -> str:
        if id not in self.reminders:
            raise AIToolError(f"No reminder found with ID '{id}'.")
        del self.reminders[id]
        self._save()
        return f"Reminder {id} deleted."

    def _load(self):
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            for d in data:
                r = Reminder(
                    id=d["id"],
                    message=d["message"],
                    trigger_time=datetime.fromisoformat(d["trigger_time"]),
                    recurrence=d.get("recurrence"),
                    created_at=datetime.fromisoformat(d["created_at"]),
                )
                self.reminders[r.id] = r
        except Exception as exc:
            print(f"Warning: could not load reminders from {self.storage_path}: {exc}")

    def _save(self):
        try:
            data = [
                {
                    "id": r.id,
                    "message": r.message,
                    "trigger_time": r.trigger_time.isoformat(),
                    "recurrence": r.recurrence,
                    "created_at": r.created_at.isoformat(),
                }
                for r in self.reminders.values()
            ]
            self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"Warning: could not save reminders to {self.storage_path}: {exc}")


def _parse_datetime(s: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise AIToolError(f"Invalid time format '{s}'. Use 'YYYY-MM-DD HH:MM'.")


def _advance(dt: datetime, recurrence: str) -> datetime:
    if recurrence == "daily":
        return dt + timedelta(days=1)
    if recurrence == "weekly":
        return dt + timedelta(weeks=1)
    if recurrence == "monthly":
        month = dt.month % 12 + 1
        year = dt.year + (1 if dt.month == 12 else 0)
        day = min(dt.day, calendar.monthrange(year, month)[1])
        return dt.replace(year=year, month=month, day=day)
    raise ValueError(f"Unknown recurrence: {recurrence}")


def load_reminder_storage_path(config_dict: dict) -> Path:
    rc = config_dict.get("reminders", {})
    return Path(rc.get("storage_path", "reminders.json"))
