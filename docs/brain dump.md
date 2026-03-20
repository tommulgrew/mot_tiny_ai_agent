Can we encourage LLM to be more autonomous?
    Timer event would give it a chance to take action.
    But would still need some direction. Not sure.

Refactoring

    Create a "message accessor" class for accessing messages. Move "get_message_content" and "is_user_message" etc into it.
    
    Create a "context manager" object.
        Inject the message accessor.
        Move the context management logic out of the message history class into the manager.

    Move the event loop out of the App class and into the AIAgent

Need a mechanism to allow events to interrupt the tool call loop, in case the LLM gets stuck in a loop, or unwanted long running task.

    User messages can interrupt.
        Use asyncio.Event as a cancellation token.
        Created by the agent.
        Passed in to each call (cleared before each call).
        Set to cancel the completions loop.
        Client checks it each time around the loop.

    User messages should be moved to the front of the queue in front of other events.
        Use asyncio.PriorityQueue

    Need a way to signal to the AI that a new user message is pending.
        When user message added:
            1. Add to queue
            2. Signal cancellation token

    Need a way to cancel the current tool resolution.
        Generate a fake assistant message to "close off" the message group, and return immediately.
        New message can then be processed and responded to. LLM will still have the previous group messages for context.

Persisting the agent's token context window would be helpful, particularly when briefly stopping and restarting the application.
Would help keep recent discussion in context.

    Perhaps this could be AI generated?

    Should look into this *after* the proposed refactoring.