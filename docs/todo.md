[X] - Port reminder tools from POC

[X] - Handle context overflows
    [X] - Remove historical messages and try again
    [X] - Avoid overflows (somewhat) by removing messages when context usage approaches the limit

[X] - Email support
    [X] - Tool calls
    [X] - Email received notification

[ ] - Rules
    [ ] - Extract rules from user speech.
        Allow user to specify rules, e.g. "When XXX happens, do YYY."
        Can we use the memory system to detect rules? E.g. add a special flag to a memory.
        Include rules in memory housekeeping, so that user can correct them.
    [ ] - Fetch rules and insert as a floating message.

[X] - Task lists
    Allow agent to define task lists, that persist as a floating message.
    [X] - User task list. Less autonomous. Allows the AI to maintain a todo list for the user.

[X] - Memory system
    [X] - Rework prompt to help memory service understand user input is not directed to them.
    [X] - Maybe encode text as JSON.
    [X] - Include agent's response.

[ ] - Web search using duckduckgo search?

        ```
        pip install duckduckgo-search
        ```

        ```python
        from duckduckgo_search import DDGS

        def search(query: str, max_results: int = 5) -> list[dict]:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return results
        ```

[ ] - Short term memory

[ ] - Formalise system event.
    Accept event "type" parameter.
    Wrap in "system_event" structure with "type" and "message"

