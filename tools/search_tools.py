import asyncio
import getpass
import keyring
from tavily import TavilyClient

from ai.tools import AITool, AIToolParam

_KEYRING_SERVICE = "tinyagent"
_KEYRING_KEY = "tavily_api_key"


def _resolve_api_key() -> str:
    """Return the Tavily API key, prompting and storing via keyring if not set."""
    key = keyring.get_password(_KEYRING_SERVICE, _KEYRING_KEY)
    if key:
        return key

    print("No Tavily API key found.")
    key = getpass.getpass("Enter Tavily API key: ")
    keyring.set_password(_KEYRING_SERVICE, _KEYRING_KEY, key)
    print("API key stored in keyring.")
    return key


class SearchTools:
    def __init__(self):
        self._client = TavilyClient(api_key=_resolve_api_key())

    def make_tools(self) -> list[AITool]:
        return [
            AITool(
                name="web_search",
                description=(
                    "Search the web for current information using Tavily. "
                    "Returns relevant text results with titles, URLs, and content snippets. "
                    "Use this for questions about recent events, facts you're unsure of, or anything requiring up-to-date information."
                ),
                params=[
                    AIToolParam(name="query", type="string", description="The search query"),
                    AIToolParam(name="max_results", type="integer", description="Maximum number of results to return (default: 5, max: 10)", optional=True),
                ],
                async_callback=self._web_search,
            )
        ]

    async def _web_search(self, query: str, max_results: int | None = None) -> str:
        max_results = min(max_results or 5, 10)
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.search(query, max_results=max_results, include_raw_content=False)
        )
        results = response.get("results", [])
        if not results:
            return "No results found."
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"{i}. {r['title']}\n   URL: {r['url']}\n   {r.get('content', '')}")
        return "\n\n".join(parts)
