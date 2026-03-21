import subprocess
from ai.tools import AIToolError, AIToolParam, AITool
from config import AppConfig

class AppTools:

    def __init__(self, apps: list[AppConfig]):
        self.apps = apps

    def make_tools(self) -> list[AITool]:
        return [
            AITool(
                name="list_apps",
                description="List applications that can be opened with the 'open_app' tool",
                params=[],
                async_callback=self._list_apps
            ),
            AITool(
                name="open_app",
                description="Open an application on the user's PC",
                params=[
                    AIToolParam(name="name", type="string", description="The name of the app to launch. Use the 'list_apps' tool to see the available apps.")
                ],
                async_callback=self._open_app,
                single_use=True
            )
        ]
    
    async def _list_apps(self) -> str:
        return "\n".join(f"{app.name}, - {app.description if app.description else ''}" for app in self.apps)

    async def _open_app(self, name: str) -> str:
        app = next((app for app in self.apps if app.name.lower() == name.lower()), None)
        if not app: 
            raise AIToolError(f"App '{name}' not found. Use 'list_apps' to list available apps.")

        try:
            subprocess.Popen([app.path])
        except FileNotFoundError:
            raise AIToolError(f"The app failed to open. Tell the user to check the path in the config file!")
        return f"Opened '{name}'"
