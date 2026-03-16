import asyncio
import email as email_lib
import getpass
import imaplib
import json
import logging
import smtplib
from contextlib import contextmanager
from email.header import decode_header as _decode_header
from email.mime.text import MIMEText
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Generator

import keyring

from ai_tools import AITool, AIToolError, AIToolParam
from config import AgentInboxConfig, EmailConfig, ImapConfig


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

_SUPPRESS_TAGS = {"style", "script"}

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._suppress_depth: int = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SUPPRESS_TAGS:
            self._suppress_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SUPPRESS_TAGS and self._suppress_depth > 0:
            self._suppress_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._suppress_depth > 0:
            return
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(html_text: str) -> str:
    stripper = _HTMLStripper()
    stripper.feed(html_text)
    return stripper.get_text()


# ---------------------------------------------------------------------------
# Email parsing helpers
# ---------------------------------------------------------------------------

def _decode_header_str(value: str | None) -> str:
    if not value:
        return ""
    parts = []
    for fragment, charset in _decode_header(value):
        if isinstance(fragment, bytes):
            parts.append(fragment.decode(charset or "utf-8", errors="replace"))
        else:
            parts.append(fragment)
    return "".join(parts)


def _format_date(date_str: str) -> str:
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return (date_str or "")[:16]


def _extract_body(msg: email_lib.message.Message) -> str:
    plain: str | None = None
    html: str | None = None
    attachments: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in disposition:
                attachments.append(part.get_filename() or "unnamed")
                continue
            charset = part.get_content_charset() or "utf-8"
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            text = payload.decode(charset, errors="replace")
            if content_type == "text/plain" and plain is None:
                plain = text
            elif content_type == "text/html" and html is None:
                html = text
    else:
        charset = msg.get_content_charset() or "utf-8"
        payload = msg.get_payload(decode=True)
        if payload:
            text = payload.decode(charset, errors="replace")
            if msg.get_content_type() == "text/html":
                html = text
            else:
                plain = text

    body = plain if plain is not None else (_strip_html(html) if html else "(No text content)")

    if attachments:
        body += f"\n\n[Attachments: {', '.join(attachments)}]"

    return body


# ---------------------------------------------------------------------------
# IMAP connection
# ---------------------------------------------------------------------------

_KEYRING_SERVICE = "tinyagent"


def _resolve_password(config: ImapConfig, label: str) -> str:
    """Return the password for an inbox, prompting and storing via keyring if not set."""
    if config.password:
        return config.password

    password = keyring.get_password(_KEYRING_SERVICE, config.username)
    if password:
        return password

    # Not in keyring — prompt the user once and store it
    print(f"No password found for {label} ({config.username}).")
    password = getpass.getpass(f"Enter password for {config.username}: ")
    keyring.set_password(_KEYRING_SERVICE, config.username, password)
    print(f"Password stored in keyring for {config.username}.")
    return password


@contextmanager
def _imap_connect(config: ImapConfig, password: str) -> Generator[imaplib.IMAP4_SSL, None, None]:
    imap = imaplib.IMAP4_SSL(config.imap_host, config.imap_port)
    try:
        imap.login(config.username, password)
        imap.select("INBOX")
        yield imap
    finally:
        try:
            imap.logout()
        except Exception:
            pass


def _search_uids(imap: imaplib.IMAP4_SSL) -> list[int]:
    """Return all UIDs in INBOX as a list of integers."""
    typ, data = imap.uid("search", None, "ALL")  # type: ignore[arg-type]
    if typ != "OK" or not data or not data[0]:
        return []
    return [int(u) for u in data[0].decode().split() if u]


# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------

class _EmailState:
    """Tracks read UIDs and highest-seen UIDs per inbox, persisted to JSON."""

    def __init__(self, path: Path):
        self._path = path
        self._read_uids: dict[str, set[int]] = {}
        self._max_uids: dict[str, int] = {}
        self._load()

    def is_read(self, inbox: str, uid: int) -> bool:
        return uid in self._read_uids.get(inbox, set())

    def mark_read(self, inbox: str, uid: int) -> None:
        self._read_uids.setdefault(inbox, set()).add(uid)
        self._save()

    def get_max_uid(self, inbox: str) -> int:
        return self._max_uids.get(inbox, 0)

    def set_max_uid(self, inbox: str, uid: int) -> None:
        self._max_uids[inbox] = uid
        self._save()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._read_uids = {k: set(v) for k, v in data.get("read_uids", {}).items()}
            self._max_uids = data.get("max_uids", {})
        except Exception as e:
            logging.getLogger("tinyagent.email").warning(f"Could not load email state: {e}")

    def _save(self) -> None:
        try:
            data = {
                "read_uids": {k: sorted(v) for k, v in self._read_uids.items()},
                "max_uids": self._max_uids,
            }
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logging.getLogger("tinyagent.email").warning(f"Could not save email state: {e}")


# ---------------------------------------------------------------------------
# EmailTools
# ---------------------------------------------------------------------------

class EmailTools:
    def __init__(self, config: EmailConfig, event_callback: Callable[[dict], None] | None = None):
        self.config = config
        self.event_callback = event_callback
        self.logger = logging.getLogger("tinyagent.email")
        self._state = _EmailState(Path(config.storage_path))

        # Resolve passwords at startup (prompts if not in keyring)
        self._passwords: dict[str, str] = {}
        if config.user_inbox:
            self._passwords["user"] = _resolve_password(config.user_inbox, "user inbox")
        if config.agent_inbox:
            self._passwords["agent"] = _resolve_password(config.agent_inbox, "agent inbox")

    def make_tools(self) -> list[AITool]:
        tools: list[AITool] = []

        inboxes: list[str] = []
        if self.config.user_inbox:
            inboxes.append('"user"')
        if self.config.agent_inbox:
            inboxes.append('"agent"')

        if not inboxes:
            return tools

        inbox_desc = f"Inbox to access: {' or '.join(inboxes)}"

        tools.append(AITool(
            name="list_emails",
            description=(
                "List the 20 most recent emails in an inbox. "
                "Returns UID, date, sender, subject, and read status for each email. "
                "Use the UID with read_email to fetch the full message."
            ),
            params=[
                AIToolParam(name="inbox", type="string", description=inbox_desc),
            ],
            async_callback=self._list_emails,
        ))

        tools.append(AITool(
            name="read_email",
            description=(
                f"Read the full content of an email by UID. "
                f"Body is limited to {self.config.max_body_chars} characters per call. "
                f"If the email is larger, a continuation hint is appended — use the offset parameter to read subsequent chunks. "
                f"HTML emails are converted to plain text. Attachments are listed by name only."
            ),
            params=[
                AIToolParam(name="inbox", type="string", description=inbox_desc),
                AIToolParam(name="id", type="string", description="Email UID from list_emails"),
                AIToolParam(name="offset", type="integer", description="Character offset to start reading from", optional=True),
            ],
            async_callback=self._read_email,
        ))

        if self.config.agent_inbox:
            tools.append(AITool(
                name="send_email",
                description=(
                    "Send an email from the agent inbox. "
                    "The recipient address must be in the configured send whitelist."
                ),
                params=[
                    AIToolParam(name="to", type="string", description="Recipient email address"),
                    AIToolParam(name="subject", type="string", description="Email subject line"),
                    AIToolParam(name="body", type="string", description="Plain text email body"),
                ],
                async_callback=self._send_email,
            ))

        return tools

    async def check_task(self) -> None:
        """Background task: polls inboxes periodically for new emails."""
        while True:
            await asyncio.sleep(self.config.poll_interval_seconds)
            for inbox_name, inbox_config in self._inbox_items():
                try:
                    await self._poll_inbox(inbox_name, inbox_config)
                except Exception as e:
                    self.logger.warning(f"Email poll error ({inbox_name}): {e}")

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    async def _list_emails(self, inbox: str) -> str:
        config = self._get_inbox_config(inbox)

        password = self._passwords[inbox]

        def _run() -> str:
            with _imap_connect(config, password) as imap:
                all_uids = _search_uids(imap)
                if not all_uids:
                    return "No emails found."

                # Most recent 20
                uids = sorted(all_uids, reverse=True)[:20]

                lines: list[str] = []
                for uid in uids:
                    uid_bytes = str(uid).encode()
                    typ, data = imap.uid(  # type: ignore[arg-type]
                        "fetch", uid_bytes,
                        "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"
                    )
                    if typ != "OK" or not data or not isinstance(data[0], tuple):
                        continue
                    msg = email_lib.message_from_bytes(data[0][1])
                    from_addr = _decode_header_str(msg.get("From", ""))[:35]
                    subject   = _decode_header_str(msg.get("Subject", "(no subject)"))[:45]
                    date      = _format_date(_decode_header_str(msg.get("Date", "")))
                    read_flag = " [read]" if self._state.is_read(inbox, uid) else ""
                    lines.append(f"{uid:<6}  {date}  {from_addr:<35}  {subject}{read_flag}")

                return "\n".join(lines)

        try:
            return await asyncio.get_running_loop().run_in_executor(None, _run)
        except imaplib.IMAP4.error as e:
            raise AIToolError(f"IMAP error: {e}")

    async def _read_email(self, inbox: str, id: str, offset: int | None = None) -> str:
        config = self._get_inbox_config(inbox)
        offset = offset or 0

        try:
            uid = int(id)
        except ValueError:
            raise AIToolError(f"Invalid email ID '{id}'.")

        password = self._passwords[inbox]

        def _run() -> str:
            uid_bytes = str(uid).encode()
            with _imap_connect(config, password) as imap:
                typ, data = imap.uid("fetch", uid_bytes, "(BODY.PEEK[])")  # type: ignore[arg-type]
                if typ != "OK" or not data or not isinstance(data[0], tuple):
                    raise AIToolError(f"Email {id} not found.")
                msg = email_lib.message_from_bytes(data[0][1])

            from_addr = _decode_header_str(msg.get("From", ""))
            subject   = _decode_header_str(msg.get("Subject", ""))
            date      = _format_date(_decode_header_str(msg.get("Date", "")))
            body      = _extract_body(msg)
            return f"From: {from_addr}\nDate: {date}\nSubject: {subject}\n\n{body}"

        try:
            full_text = await asyncio.get_running_loop().run_in_executor(None, _run)
        except AIToolError:
            raise
        except imaplib.IMAP4.error as e:
            raise AIToolError(f"IMAP error: {e}")

        self._state.mark_read(inbox, uid)

        total = len(full_text)
        chunk = full_text[offset : offset + self.config.max_body_chars]
        if not chunk:
            return f"(No content at offset {offset}. Email has {total} characters.)"
        end = offset + len(chunk)
        if end < total:
            return (
                chunk
                + f"\n--- Truncated. Read {len(chunk)} chars (offset {offset}–{end} of {total}). "
                f"Use offset={end} to continue. ---"
            )
        return chunk

    async def _send_email(self, to: str, subject: str, body: str) -> str:
        cfg = self.config.agent_inbox
        if not cfg:
            raise AIToolError("Agent inbox is not configured.")

        if to not in cfg.send_whitelist:
            permitted = ", ".join(cfg.send_whitelist) if cfg.send_whitelist else "none configured"
            raise AIToolError(
                f"'{to}' is not in the send whitelist. Permitted recipients: {permitted}."
            )

        password = self._passwords["agent"]

        def _run() -> None:
            msg = MIMEText(body, "plain", "utf-8")
            msg["From"] = cfg.username
            msg["To"] = to
            msg["Subject"] = subject
            with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as smtp:
                smtp.starttls()
                smtp.login(cfg.username, password)
                smtp.send_message(msg)

        try:
            await asyncio.get_running_loop().run_in_executor(None, _run)
        except smtplib.SMTPException as e:
            raise AIToolError(f"Failed to send email: {e}")

        return f"Email sent to {to}."

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_inbox(self, inbox_name: str, inbox_config: ImapConfig) -> None:
        password = self._passwords[inbox_name]

        def _get_uids() -> list[int]:
            with _imap_connect(inbox_config, password) as imap:
                return _search_uids(imap)

        all_uids = await asyncio.get_running_loop().run_in_executor(None, _get_uids)
        if not all_uids:
            return

        current_max = max(all_uids)
        known_max = self._state.get_max_uid(inbox_name)

        if known_max == 0:
            # First poll — record baseline without firing an event
            self._state.set_max_uid(inbox_name, current_max)
            return

        if current_max > known_max:
            new_uids = [u for u in all_uids if u > known_max]
            self._state.set_max_uid(inbox_name, current_max)
            if self.event_callback:
                self.event_callback({
                    "system_event": {
                        "type": "new_email",
                        "inbox": inbox_name,
                        "message": f"{len(new_uids)} new email(s) received in the {inbox_name} inbox.",
                    }
                })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_inbox_config(self, inbox: str) -> ImapConfig:
        if inbox == "user":
            if self.config.user_inbox:
                return self.config.user_inbox
            raise AIToolError("User inbox is not configured.")
        if inbox == "agent":
            if self.config.agent_inbox:
                return self.config.agent_inbox
            raise AIToolError("Agent inbox is not configured.")
        raise AIToolError(f"Unknown inbox '{inbox}'. Use 'user' or 'agent'.")

    def _inbox_items(self) -> Generator[tuple[str, ImapConfig], None, None]:
        if self.config.user_inbox:
            yield "user", self.config.user_inbox
        if self.config.agent_inbox:
            yield "agent", self.config.agent_inbox
