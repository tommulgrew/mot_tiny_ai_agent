import asyncio
from app import App
from args import parse_main_args
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from ai_client import AIClient
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from ai_tools.browser_tools import BrowserTools

async def main():

    # Prompt toolkit session
    session = PromptSession()
    with patch_stdout():

        # Create application
        args = parse_main_args()
        app = App(args)

        client = AIClient(app.config.model)
        msgs=[]
        system_msg = ChatCompletionSystemMessageParam(
            role="system",
            content="You are a helpful, friendly chatbot named Alfred."
        )
        tools=BrowserTools().make_tools()

        # Console REPL loop
        while True:
            user_input = await session.prompt_async("> ")
            user_input = user_input.strip()
            if user_input.lower() in ("exit", "quit"):
                break

            # Add user message to chat
            user_msg = ChatCompletionUserMessageParam(
                role="user",
                content=user_input
            )
            msgs.append(user_msg)

            # Call completions service
            new_msgs = await client.chat(
                system_message=system_msg, 
                messages=msgs, 
                tools=tools,
                output_callback=print
            )
            msgs.extend(new_msgs)

            # Trim message history
            while len(msgs) > 20:
                msgs.remove(msgs[0])

            # app.user_input(user_input)

if __name__ == "__main__":
    asyncio.run(main())
