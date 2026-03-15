Email tools design
================================================================================

Overview
--------

Two inboxes are supported:

- "user" inbox  - read-only access to the user's personal email
- "agent" inbox - read/write; the AI agent's own email address, can send and
                  reply to addresses in a configured whitelist

Both inboxes use IMAP for reading (leaving mail on the server untouched) and
the agent inbox uses SMTP for sending.


Protocols
---------

IMAP (port 993, SSL) for reading.
  - Emails are never deleted from the server, so the user's own email client
    (e.g. Thunderbird) is unaffected.
  - Only the INBOX folder is accessed (no multi-folder support).
  - Headers are fetched separately from bodies, so list_emails is efficient.
  - IMAP UIDs are used as email IDs. UIDs are stable integers assigned by the
    server that persist across sessions.

SMTP (port 587, STARTTLS) for sending (agent inbox only).

Both protocols are supported by Python's stdlib (imaplib, smtplib). No extra
dependencies required.


Email IDs
---------

Emails are identified by their IMAP UID — a stable integer assigned by the
server, e.g. 1042. UIDs are per-mailbox; since only INBOX is supported this
is unambiguous.

list_emails returns UIDs in its output, which the agent passes to read_email.


Read tracking
-------------

Read state is tracked locally in a JSON file (storage_path in config), keyed
by inbox name and UID. The server \Seen flag is NOT modified, so the user's
email client always sees its own unread/read state independently.

The local tracker records which UIDs have been fetched via read_email. This
allows list_emails to indicate which emails the agent has already read.


Tool surface
------------

list_emails(inbox)
  - Returns the 20 most recent emails in the specified inbox.
  - Output per email: UID, date, from address, subject, ~100-char body snippet.
  - Indicates which emails the agent has already read (per local tracker).
  - inbox: "user" or "agent"

read_email(inbox, id, offset?)
  - Fetches the full body of an email by UID.
  - Body is truncated to max_body_chars characters. If truncated, a continuation
    hint is appended (same pattern as read_file).
  - offset parameter allows reading subsequent chunks.
  - HTML parts are stripped to plain text. If both plain-text and HTML parts
    are present, the plain-text part is preferred.
  - Attachment names are listed but attachment content is not included.
  - Marks the email as read in the local tracker.
  - inbox: "user" or "agent"

send_email(to, subject, body)
  - Sends an email from the agent inbox.
  - Always uses the agent inbox (no inbox parameter — user inbox is read-only).
  - The "to" address must be in the configured send_whitelist, otherwise an
    AIToolError is raised.

New email notification (background task)
  - A background polling task (similar to ReminderTools.check_task) periodically
    checks both inboxes for new mail.
  - When new mail is detected, a system event is fired so the agent can decide
    whether to notify the user.
  - Poll interval is configurable (default: 5 minutes).


Configuration
-------------

Added to config.json under an "email" key:

    "email": {
        "user_inbox": {
            "imap_host": "imap.example.com",
            "imap_port": 993,
            "username": "user@example.com",
            "password": "..."
        },
        "agent_inbox": {
            "imap_host": "imap.example.com",
            "imap_port": 993,
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "username": "agent@example.com",
            "password": "...",
            "send_whitelist": ["trusted@example.com", "boss@work.com"]
        },
        "storage_path": "email_state.json",
        "max_body_chars": 4000,
        "poll_interval_seconds": 300
    }

Both user_inbox and agent_inbox are optional. If only one is configured, only
the tools relevant to that inbox are registered. send_email is only registered
if agent_inbox is present.

New config models added to config.py:
  - ImapConfig       (imap_host, imap_port, username, password)
  - AgentImapConfig  (extends ImapConfig: smtp_host, smtp_port, send_whitelist)
  - EmailConfig      (user_inbox, agent_inbox, storage_path, max_body_chars,
                      poll_interval_seconds)
  - Config.email: EmailConfig | None = None


Implementation notes
--------------------

- Open and close IMAP connections per operation (do not hold persistent
  sessions). Reconnect on failure.

- Use imaplib.IMAP4_SSL for IMAP, smtplib.SMTP with starttls() for SMTP.

- Fetch UIDs with: typ, data = imap.uid('search', None, 'ALL')
  Sort descending and take the last 20 UIDs for list_emails.

- Fetch headers only for list_emails:
    imap.uid('fetch', uid, '(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])')
  PEEK avoids setting the \Seen flag on the server.

- Fetch full message for read_email:
    imap.uid('fetch', uid, '(BODY.PEEK[])')

- Parse emails with Python's email.message_from_bytes().

- Strip HTML using html.parser (stdlib) — no BeautifulSoup dependency.

- The local state file (email_state.json) stores a set of read UIDs per inbox:
    {
        "user": [1035, 1038],
        "agent": [204, 205, 210]
    }

- Gmail / Outlook may require app-specific passwords or OAuth. Document this
  in config comments; no OAuth implementation in scope.


Files
-----

tools/email_tools.py   - EmailTools class
  - __init__(config: EmailConfig)
  - make_tools() -> list[AITool]
  - check_task()  - background polling coroutine (called from app.py like
                    ReminderTools.check_task)

config.py              - Add ImapConfig, AgentImapConfig, EmailConfig;
                         add email: EmailConfig | None = None to Config

config.json            - Add example "email" section (commented / with
                         placeholder values)

app.py                 - Instantiate EmailTools if config.email is set;
                         register tools and schedule check_task
