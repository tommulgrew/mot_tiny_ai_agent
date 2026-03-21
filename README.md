# tinyagent

An experimental personal AI assistant that runs entirely on your own PC — no 
cloud subscription, no data leaving your machine.

tinyagent is designed around small, locally-hosted language models (8–10B 
parameters) that fit comfortably on a mid-range graphics card. It connects to 
[LM Studio](https://lmstudio.ai/) to run the model, and gives the AI a small,
curated set of practical tools it can use to actually help you day-to-day.

tinyagent does not try to be OpenClaw! It doesn't support MCP. Rather it is an
agent harness tuned for small context windows and sandboxed to allow smaller
less powerful LLMs make mistakes without disasterous consequences.

tinyagent is a little slow, and sometimes gets confused, like offering to 
perform email actions it doesn't have the tools for. But for an AI agent that 
can run on a mid-range 8 GB GPU it may surprise you.

**Tested with:** Qwen3.5-9B · AMD Radeon RX 6600 XT (8 GB VRAM) · 16K context 
window

--------------------------------------------------------------------------------

## What can it do?

You interact with tinyagent through a terminal. You can type to it, or hold a 
key to speak and have your voice transcribed automatically. It understands 
natural language, remembers things you've told it, and can take real actions on 
your behalf.

**Some things you can ask it to do:**

- *"Remind me to call the dentist on Friday at 10am"*
- *"What emails have come in today?"*
- *"Add 'buy milk' to my to-do list"*
- *"Search my documents folder for anything about the Johnson project"* - Note: 
  limited to text format files only (e.g. txt, md, csv).
- *"Open Firefox"*
- *"What do you know about my family?"*

It also works in the background. When a reminder comes due, it can speak up and 
notify you. When new emails arrive, it can summarise them and read them aloud.

--------------------------------------------------------------------------------

## Features

**Talks and listens**
- Chat via the terminal, or use push-to-talk voice input (hold a key to record, 
  release to send)
- Can speak responses aloud via text-to-speech

**Takes action**
- Reads and sends email (multiple IMAP inboxes supported)
- Manages a to-do list — add, complete, prioritise, and update tasks
- Sets reminders that fire at the right time, even when you're not actively 
  using it
- Reads, writes and searches files in folders you've granted it access to
- Launches applications on your PC from a configurable whitelist

**Remembers things**
- Builds up a long-term memory of facts you've shared — your name, your family, 
  your preferences, where your files live
- Surfaces relevant memories automatically based on the conversation
- Memories persist across sessions

**Stays within bounds**
- Only accesses folders you've explicitly configured
    - Folder access can be read-only
    - Files "deleted" from writable folders are moved to a trash folder.
- Only launches apps on your approved list
- Only sends emails to addresses you've explicitly configured
- Passwords are stored in your OS keyring — never written to config files
- Runs entirely locally — your conversations don't leave your machine

--------------------------------------------------------------------------------

## Requirements

- Python 3.11+
- [LM Studio](https://lmstudio.ai/) with a model that supports tool calling (see
  below)
- A GPU with enough VRAM for your chosen model

**Recommended model:** Qwen3.5-9B fits in 8 GB VRAM and handles tool use 
reliably. Any model with solid tool-call support should work — check the LM 
Studio model page for details.

--------------------------------------------------------------------------------

## Getting started

**1. Clone the repo and set up a virtual environment**

```bash
git clone https://github.com/tommulgrew/mot_tiny_ai_agent.git
cd mot_tiny_ai_agent
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure**

Copy `config.json` and edit it for your setup. At minimum you'll need to set:
- `model` — your LM Studio model name and server URL.
    - Also set the `prompt_token_limit` to the token context length you will be 
      using minus about 4-5K. E.g. for a 16000 token context, set it to about 
      11000.
- `agent` — your name and whatever you'd like to call the assistant

Configure any tools you'd like to make available to the agent. See 
[Configuration](#configuration) below for all options.

**4. Start LM Studio**

Load your model and start the local server. The default endpoint is 
`http://localhost:1234/v1`.

Recommended settings for Qwen3.5-9B (select "My models", click the config button
for the model):
- "Load" tab:
    - Context Length = 16000
    - GPU Offload = 32 (full)
- "Inferrence" tab. Based on "Thinking mode for general tasks" recommendations 
  from https://huggingface.co/Qwen/Qwen3.5-9B:    
    - Temperature = 1
    - Top K Sampling = 20
    - Repeat penalty = 1
    - Top P Sampling = 0.95
    - Min P Sampling = 0
- Also turn "Enable Thinking" on

Note: It *is* worth spending the time to configure it correctly, as with the 
default settings it can often fall into a neverending "think" until it 
eventually runs out of context tokens.

**5. Run**

```bash
python cli.py
```

--------------------------------------------------------------------------------

## Configuration

All settings live in `config.json` (validated against `config-schema.json`).

### `model`
```json
"model": {
    "name": "qwen/qwen3.5-9b",
    "url": "http://localhost:1234/v1",
    "prompt_token_limit": 11000
}
```
Set `prompt_token_limit` a little below your model's actual context size — this is when the agent starts trimming older conversation history to make room.

### `agent`
```json
"agent": {
    "users_name": "Tom",
    "agents_name": "Alfred",
    "extra_info": []
}
```
`extra_info` lets you add standing instructions to the agent's system prompt — for example your timezone, preferences, or recurring rules like *"always summarise emails longer than a few paragraphs"*.

### `memory`
```json
"memory": {
    "storage_path": "memory.json"
}
```

### `file_tools`
```json
"file_tools": {
    "folders": [
        { "alias": "ai_files", "path": "C:\\ai\\sandbox", "access": "rw" },
        { "alias": "docs",     "path": "C:\\docs",         "access": "ro" }
    ],
    "trash_path": "C:\\ai\\trash"
}
```
Give each folder a short alias — the agent uses these names rather than full paths. Access is `"rw"` (read/write) or `"ro"` (read-only). Deleted files go to `trash_path` rather than being permanently deleted.

### `todo` *(optional)*
```json
"todo": {
    "storage_path": "tasks.json",
    "completed_log_path": "tasks_completed.jsonl",
    "list_limit": 50
}
```

### `email` *(optional)*
```json
"email": {
    "user_inboxes": [
        { "name": "gmail", "imap_host": "imap.gmail.com", "imap_port": 993, "username": "you@gmail.com" }
    ],
    "storage_path": "email_state.json",
    "max_body_chars": 4000,
    "poll_interval_seconds": 300
}
```
Multiple inboxes are supported. Passwords are stored securely in your OS keyring (Windows Credential Manager on Windows) — you'll be prompted on first run. To update a stored password later:

```bash
python manage_keyring.py set you@gmail.com
```

### `speech_to_text` *(optional)*
```json
"speech_to_text": {
    "enabled": true,
    "model_size": "small",
    "trigger_key": "alt_gr",
    "language": "en"
}
```
Hold the trigger key to record, release to transcribe and send. Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) locally — no audio leaves your machine. `model_size` can be `tiny`, `small`, `medium`, or `large` — `small` is a good balance of speed and accuracy.

If you don't want voice input, set `"enabled": false` or remove the section entirely.

### `allowed_apps` *(optional)*
```json
"allowed_apps": [
    { "name": "Firefox", "path": "C:\\Program Files\\Mozilla Firefox\\firefox.exe", "description": "Web browser" }
]
```
The agent can only launch applications listed here.

--------------------------------------------------------------------------------

## Notes

- The agent works best with models trained for extended thinking and tool use. Qwen3.5 9B is the primary tested model.
- Context window management is handled automatically — older history is trimmed gradually (think blocks first, then older messages) to stay within the configured token limit.
- tinyagent is a personal project and a work in progress. Expect rough edges.
