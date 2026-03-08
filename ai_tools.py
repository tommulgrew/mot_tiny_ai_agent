from pydantic import BaseModel
from typing import Awaitable, Callable, Iterable

class AIToolError(Exception):
    """Raised when an error occurs calling a tool. Message will be passed back to the LLM as the tool response."""

class AIToolParam(BaseModel):
    """Parameter definition for an AI tool"""
    name: str
    type: str
    description: str
    optional: bool = False

class AITool(BaseModel):
    """An AI tool, including definition and callback function"""
    name: str
    description: str
    params: list[AIToolParam]
    async_callback: Callable[..., Awaitable[str]]
    single_use: bool = False


class AITools:
    """Handles calling AI tools"""
    def __init__(self):
        self.tools: list[AITool] = []
        self.called_tools: set[str] = set()

    def add(self, tools: Iterable[AITool]):
        self.tools.extend(tools)

    def clear_called_tools(self):
        self.called_tools = set()

    async def call_tool(self, name: str, args_dict: dict) -> str:
        
        # Find tool
        tool = next((tool for tool in self.tools if tool.name == name), None)
        if tool is None:
            raise AIToolError(f"Tool '{name}' not found")

        # Handle single use tools
        if tool.single_use and name in self.called_tools:
            raise AIToolError(f"Tool cannot be called again this turn.")

        # Collect parameters
        params = [
            args_dict[p.name] if not p.optional else args_dict.get(p.name)
            for p in tool.params
        ]

        # Call tool callback
        self.called_tools.add(name)
        return await tool.async_callback(*params)
