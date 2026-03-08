import asyncio
from app import App
from args import parse_main_args
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from ai_client import AIClient
from ai_tools.browser_tools import BrowserTools

async def main():

    # Prompt toolkit session
    session = PromptSession()
    with patch_stdout():

        # Create application
        args = parse_main_args()
        app = App(args)

        client = AIClient(app.config.model)
        tools=BrowserTools().make_tools()
        message_history=[]

        # Console REPL loop
        while True:
            user_input = await session.prompt_async("> ")
            user_input = user_input.strip()
            if user_input.lower() in ("exit", "quit"):
                break

            # Call completions service
            new_msgs = await client.chat(
                system_prompt="You are a helpful, friendly chatbot named Alfred.", 
                user_prompt=user_input,
                message_history=message_history, 
                tools=tools,
                output_callback=print
            )
            message_history.extend(new_msgs)

            # Trim message history
            while len(message_history) > 20:
                message_history.remove(message_history[0])

            # app.user_input(user_input)

if __name__ == "__main__":
    asyncio.run(main())
