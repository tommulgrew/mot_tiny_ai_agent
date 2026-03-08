import argparse

def parse_main_args():
    parser = argparse.ArgumentParser(description="Tiny agent")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to JSON config file"
    )
    return parser.parse_args()
