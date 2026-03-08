from config import load_config

class App:
    """Tiny agent main application class."""

    def __init__(self, args):
        self.config = load_config(args.config)

        print("Tiny agent v1.0")
        print(f"Model: {self.config.model.name}")

    def user_input(self, content: str):
        print("TO DO: User input")

    def system_event(self, content: str):
        print("TO DO: system event")
