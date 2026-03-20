import logging
import asyncio
from app import App
from args import parse_main_args
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from util import create_logger

async def main():

    # Init logging
    logger = create_logger("tinyagent", "log.txt", level=logging.DEBUG, propagate=False)
    logger.info("Tiny agent started")

    # Prompt toolkit session
    session = PromptSession()
    with patch_stdout():

        # Create application
        args = parse_main_args()
        app = App(args, print)

        print("Tiny agent v1.0")
        print(f"Model: {app.config.model.name}")

        # Console REPL loop
        while True:
            user_input = await session.prompt_async("> ")
            user_input = user_input.strip()
            if user_input.lower() in ("exit", "quit"):
                break
            app.agent.user_input(user_input)

if __name__ == "__main__":
    asyncio.run(main())
