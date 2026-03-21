import datetime
import shutil
import subprocess
import re
from pathlib import Path
from ai.tools import AIToolParam, AITool, AIToolError
from config import FileToolsConfig, FolderConfig

class AbortFileSearchException(Exception):
    """Raised to abort the file search."""

class FileTools:
    def __init__(self, config: FileToolsConfig):
        self.config = config
        Path(config.trash_path).mkdir(parents=True, exist_ok=True)

    def make_tools(self) -> list[AITool]:
        return [
            AITool(
                name="list_folders",
                description="List available folder aliases and their access permissions (read-only or read/write).",
                params=[],
                async_callback=self._list_folders,
            ),
            AITool(
                name="list_files",
                description=(
                    "List files and subdirectories at a path. "
                    "The path must start with a folder alias (e.g. 'notes' or 'notes/subfolder'). "
                    "Use list_folders to see available aliases."
                ),
                params=[
                    AIToolParam(name="path", type="string", description="Path to list, starting with a folder alias"),
                ],
                async_callback=self._list_files,
            ),
            AITool(
                name="read_file",
                description=(
                    f"Read the text contents of a file. "
                    f"Content is limited to {self.config.max_read_chars} characters per call. "
                    f"If the file is larger, a continuation hint is appended — use the offset parameter to read subsequent chunks."
                ),
                params=[
                    AIToolParam(name="path", type="string", description="File path, starting with a folder alias"),
                    AIToolParam(name="offset", type="integer", description="Character offset to start reading from", optional=True),
                ],
                async_callback=self._read_file,
            ),
            AITool(
                name="append_file",
                description=(
                    "Append text to a file, creating the file (and any parent directories) if it does not exist. "
                    "Only available for writable folder aliases."
                ),
                params=[
                    AIToolParam(name="path", type="string", description="File path, starting with a writable folder alias"),
                    AIToolParam(name="content", type="string", description="Text content to append"),
                ],
                async_callback=self._append_file,
            ),
            AITool(
                name="delete_file",
                description=(
                    "Delete a file by moving it to the trash. "
                    "Only available for writable folder aliases."
                ),
                params=[
                    AIToolParam(name="path", type="string", description="File path, starting with a writable folder alias"),
                ],
                async_callback=self._delete_file,
            ),
            AITool(
                name="show_in_explorer",
                single_use=True,
                description=(
                    "Open Windows Explorer at a folder, or open Explorer with a specific file selected. "
                    "Useful for showing the user the contents of a folder or the location of a file."
                ),
                params=[
                    AIToolParam(name="path", type="string", description="File or folder path, starting with a folder alias"),
                ],
                async_callback=self._show_in_explorer,
            ),
            AITool(
                name="search_files",
                description="Search file contents for lines matching a regex pattern.",
                params=[
                    AIToolParam(name="path", type="string", description="Folder to search, starting with a folder alias."),
                    AIToolParam(name="pattern", type="string", description="Regex pattern to match against file contents, e.g. 'hello' or '(cats|dogs)'. Case insensitive."),
#                    AIToolParam(name="file_wildcard", type="string", description="Filter files by name, e.g. '*.txt', '*.py'. Omit to search all files.", optional=True),
                    AIToolParam(name="recursive", type="boolean", description="True to include subfolders, false for current folder only.")
                ],
                async_callback=self._search_files
            )
        ]

    def _resolve_path(self, virtual_path: str) -> tuple[FolderConfig, Path]:
        parts = Path(virtual_path).parts
        if not parts:
            raise AIToolError("Path cannot be empty")

        alias = parts[0]
        folder = next((f for f in self.config.folders if f.alias == alias), None)
        if folder is None:
            raise AIToolError(f"Unknown folder alias '{alias}'. Use list_folders to see available aliases.")

        folder_real = Path(folder.path).resolve()
        if len(parts) > 1:
            real_path = (folder.path / Path(*parts[1:])).resolve()
        else:
            real_path = folder_real

        # Guard against path traversal
        if not real_path.is_relative_to(folder_real):
            raise AIToolError("Access denied: path is outside the allowed folder")

        return folder, real_path

    async def _list_folders(self) -> str:
        if not self.config.folders:
            return "No folders configured."
        lines = [
            f"{f.alias} ({'read/write' if f.access == 'rw' else 'read-only'})"
            for f in self.config.folders
        ]
        return "\n".join(lines)

    async def _list_files(self, path: str) -> str:
        _, real_path = self._resolve_path(path)

        if not real_path.exists():
            raise AIToolError(f"Path does not exist: {path}")
        if not real_path.is_dir():
            raise AIToolError(f"Path is not a directory: {path}")

        entries = sorted(real_path.iterdir(), key=lambda p: (p.is_file(), p.name))
        if not entries:
            return "(empty)"

        lines = []
        for entry in entries:
            if entry.is_dir():
                lines.append(f"{entry.name}/")
            else:
                lines.append(f"{entry.name} ({entry.stat().st_size} bytes)")
        return "\n".join(lines)

    async def _read_file(self, path: str, offset: int | None = None) -> str:
        _, real_path = self._resolve_path(path)

        if not real_path.exists():
            raise AIToolError(f"File does not exist: {path}")
        if not real_path.is_file():
            raise AIToolError(f"Path is not a file: {path}")

        offset = offset or 0
        content = real_path.read_text(encoding="utf-8", errors="replace")
        total = len(content)
        chunk = content[offset : offset + self.config.max_read_chars]

        if not chunk:
            return f"(No content at offset {offset}. File has {total} characters.)"

        end = offset + len(chunk)
        if end < total:
            return (
                chunk
                + f"\n--- Truncated. Read {len(chunk)} chars (offset {offset}–{end} of {total}). "
                f"Use offset={end} to continue. ---"
            )
        return chunk

    async def _append_file(self, path: str, content: str) -> str:
        folder, real_path = self._resolve_path(path)

        if folder.access == 'ro':
            raise AIToolError(f"Folder '{folder.alias}' is read-only")

        real_path.parent.mkdir(parents=True, exist_ok=True)
        with real_path.open("a", encoding="utf-8") as f:
            f.write(content)

        return f"Appended {len(content)} characters to {path}"

    async def _delete_file(self, path: str) -> str:
        folder, real_path = self._resolve_path(path)

        if folder.access == 'ro':
            raise AIToolError(f"Folder '{folder.alias}' is read-only")
        if not real_path.exists():
            raise AIToolError(f"File does not exist: {path}")
        if not real_path.is_file():
            raise AIToolError(f"Path is not a file: {path}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        trash_dest = Path(self.config.trash_path) / f"{timestamp}_{real_path.name}"
        shutil.move(str(real_path), str(trash_dest))

        return f"Deleted {path} (moved to trash as {trash_dest.name})"


    async def _show_in_explorer(self, path: str) -> str:
        _, real_path = self._resolve_path(path)

        if not real_path.exists():
            raise AIToolError(f"Path does not exist: {path}")

        if real_path.is_file():
            subprocess.Popen(["explorer", f"/select,{real_path}"])
            return f"Opened Explorer with {path} selected"
        else:
            subprocess.Popen(["explorer", str(real_path)])
            return f"Opened Explorer at {path}"

    async def _search_files(self, path: str, pattern: str, recursive: bool | None) -> str:
        _, real_path = self._resolve_path(path)

        if recursive is None:
            recursive = False

        if not real_path.exists():
            raise AIToolError(f"Path does not exist: {path}")

        if real_path.is_file():
            raise AIToolError(f"Not a folder: {path}")

        # Find files
        files = real_path.rglob("*.*") if recursive else real_path.glob("*.*")
        if not files:
            return "No files found"

        # Search each file
        results:list[str] = []
        file_ct = 0
        try:
            for file in files:
                if file.is_file():

                    # Search each line
                    for i, line in enumerate(file.read_text(errors="ignore").splitlines(), 1):
                        line = line.strip()
                        if len(line) <= 200 and re.search(pattern, line, re.IGNORECASE):

                            # Convert file path back to virtual
                            virtual_path = Path(path) / (file.relative_to(real_path))
                            results.append(f"file: {str(virtual_path)}, line: {i}, text: {line}")

                            # Stop after 10 matches
                            if len(results) >= 10:
                                raise AbortFileSearchException("Search stopped after 10 results")
                            break

                # Stop after 1000 files
                file_ct += 1
                if file_ct >= 1000:                         # Stop after 1000 files searched
                    raise AbortFileSearchException("Search stopped after 1000 files")
        except AbortFileSearchException as e:
            results.append(str(e))

        return "\n".join(results) if results else "No matches found"        
