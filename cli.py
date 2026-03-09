import asyncio
from app import App
from args import parse_main_args
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from ai_client import AIClient
from ai_tools import AITools
from tools.browser_tools import BrowserTools
from tools.file_tools import FileTools
from tools.speak_tools import SpeakTools

async def main():

    # Prompt toolkit session
    session = PromptSession()
    with patch_stdout():

        # Create application
        args = parse_main_args()
        app = App(args)

        client = AIClient(app.config.model)

        tools = AITools()
        tools.add(BrowserTools().make_tools())
        if app.config.file_tools:
            tools.add(FileTools(app.config.file_tools).make_tools())
        tools.add(SpeakTools().make_tools())

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
