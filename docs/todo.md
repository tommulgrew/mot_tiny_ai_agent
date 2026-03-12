[ ] - Port reminder tools from POC

[ ] - Handle context overflows
    [ ] - Remove historical messages and try again
    [ ] - Avoid overflows (somewhat) by removing messages when context usage approaches the limit

[ ] - Email support
    [ ] - Tool calls
    [ ] - Email received notification

[ ] - Rules
    [ ] - Extract rules from user speech.
        Allow user to specify rules, e.g. "When XXX happens, do YYY."
        Can we use the memory system to detect rules? E.g. add a special flag to a memory.
        Include rules in memory housekeeping, so that user can correct them.
    [ ] - Fetch rules and insert as a floating message.

[ ] - Memory system
    [ ] - Rework prompt to help memory service understand user input is not directed to them.
    [ ] - Maybe encode text as JSON.
    [ ] - Include agent's response.
