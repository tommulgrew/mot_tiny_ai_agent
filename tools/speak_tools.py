import asyncio

import pyttsx3

from ai_tools import AIToolParam, AITool


class SpeakTools:
    def make_tools(self) -> list[AITool]:
        return [
            AITool(
                name="speak",
                single_use=True,
                description=(
                    "Speak text aloud through the computer's speakers using text-to-speech. "
                    "Use this to read out important information, reminders, or responses when "
                    "the user may not be looking at the screen."
                ),
                params=[
                    AIToolParam(name="text", type="string", description="The text to speak aloud."),
                ],
                async_callback=self._speak,
            )
        ]

    async def _speak(self, text: str) -> str:
        def _run():
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()

        await asyncio.get_running_loop().run_in_executor(None, _run)
        return f"Spoke: \"{text}\""
