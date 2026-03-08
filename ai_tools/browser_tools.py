import asyncio
import webbrowser

from ai_client import AIToolParam, AITool

class BrowserTools:
    def make_tools(self) -> list[AITool]:
        return [
            AITool(
                name="open_url",
                single_use=True,
                description="Open a URL in the user's default web browser.",
                params=[
                    AIToolParam(name="url", type="string", description="The URL to open."),
                ],
                async_callback=self._open_url,
            )
        ]

    async def _open_url(self, url: str) -> str:
        opened = await asyncio.get_running_loop().run_in_executor(
            None, webbrowser.open, url
        )
        if opened:
            return f"Opened URL in browser: {url}"
        return f"Failed to open URL: {url}"
