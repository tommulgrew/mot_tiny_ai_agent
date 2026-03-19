"""Simple CLI for managing tinyagent keyring passwords."""
import argparse
import getpass
import sys

import keyring

SERVICE = "tinyagent"


def cmd_delete(username: str) -> None:
    try:
        keyring.delete_password(SERVICE, username)
        print(f"Password deleted for {username}.")
        print("You will be prompted for the new password on next startup.")
    except keyring.errors.PasswordDeleteError:
        print(f"No password found for {username}.", file=sys.stderr)
        sys.exit(1)


def cmd_set(username: str) -> None:
    password = getpass.getpass(f"New password for {username}: ")
    keyring.set_password(SERVICE, username, password)
    print(f"Password updated for {username}.")


parser = argparse.ArgumentParser(description="Manage tinyagent keyring passwords")
sub = parser.add_subparsers(dest="command", required=True)

sub.add_parser("delete", help="Delete a stored password (app will prompt on next startup)") \
    .add_argument("username", help="Username / email address")

sub.add_parser("set", help="Set or update a stored password directly") \
    .add_argument("username", help="Username / email address")

args = parser.parse_args()
if args.command == "delete":
    cmd_delete(args.username)
elif args.command == "set":
    cmd_set(args.username)
