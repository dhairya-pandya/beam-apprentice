"""The six tools an agent gets inside a sandbox. Every call returns a JSON string and never raises."""

from __future__ import annotations

import json
import posixpath
import shlex
from dataclasses import dataclass

from apprentice.patches import strip_protected

WORKSPACE = "/workspace"


@dataclass(frozen=True)
class Limits:
    turns: int = 40
    tool_calls: int = 60
    nudges: int = 3
    cmd_timeout: int = 300
    out_chars: int = 5000
    file_lines: int = 150
    file_chars: int = 10000


def _fn(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


_STR = {"type": "string"}
_INT = {"type": "integer"}
_BOOL = {"type": "boolean"}

SCHEMAS = [
    _fn("run_command", "Run a bash command in /workspace (offline). Output is truncated.",
        {"command": _STR}, ["command"]),
    _fn("read_file", "Read a file, optionally a 1-indexed inclusive line range. Lines come back numbered.",
        {"filepath": _STR, "start_line": _INT, "end_line": _INT}, ["filepath"]),
    _fn("edit_file", "Replace old_string with new_string in an existing file. old_string must match exactly "
        "once unless allow_multiple is true.",
        {"filepath": _STR, "old_string": _STR, "new_string": _STR, "allow_multiple": _BOOL},
        ["filepath", "old_string", "new_string"]),
    _fn("write_file", "Create or overwrite a file, creating parent directories.",
        {"filepath": _STR, "content": _STR}, ["filepath", "content"]),
    _fn("submit_patch", "Submit your changes as the final patch. Call this last, after verifying the fix.",
        {}, []),
    _fn("get_status", "Show remaining tool calls and turns.", {}, []),
]

FREE_TOOLS = {"submit_patch", "get_status"}


class ToolError(Exception):
    pass


class Toolset:
    def __init__(self, sandbox, limits: Limits):
        self.sb = sandbox
        self.limits = limits
        self.tool_calls_used = 0
        self.turns_used = 0
        self.submitted = False
        self.patch = ""

    def schemas(self) -> list[dict]:
        return SCHEMAS

    def call(self, name: str, arguments: str) -> str:
        try:
            handler = getattr(self, f"_t_{name}", None)
            if handler is None or name not in {s["function"]["name"] for s in SCHEMAS}:
                raise ToolError(f"unknown tool {name!r}")
            try:
                args = json.loads(arguments or "{}")
            except json.JSONDecodeError as e:
                raise ToolError(f"arguments are not valid JSON: {e}")
            if not isinstance(args, dict):
                raise ToolError("arguments must be a JSON object")
            if name not in FREE_TOOLS:
                self.tool_calls_used += 1
            try:
                result = handler(**args)
            except TypeError as e:
                raise ToolError(f"bad arguments for {name}: {e}")
            return json.dumps(result)
        except ToolError as e:
            return json.dumps({"status": "error", "error": str(e)})
        except Exception as e:  # tool bugs must not end the episode
            return json.dumps({"status": "error", "error": f"{type(e).__name__}: {e}"})

    def _path(self, filepath: str) -> str:
        p = filepath.strip()
        if p.startswith(WORKSPACE + "/"):
            p = p[len(WORKSPACE) + 1 :]
        norm = posixpath.normpath(p)
        if p.startswith("/") or norm == ".." or norm.startswith("../"):
            raise ToolError(f"path must be inside {WORKSPACE}: {filepath}")
        return norm

    def _clip(self, text: str, limit: int) -> tuple[str, bool]:
        if len(text) <= limit:
            return text, False
        return text[-limit:] + f"\n[output truncated to last {limit} chars]", True

    def _t_run_command(self, command: str) -> dict:
        res = self.sb.exec(command, timeout=self.limits.cmd_timeout)
        out, truncated = self._clip(res.output, self.limits.out_chars)
        return {"exit_code": res.exit_code, "output": out, "truncated": truncated, "timed_out": res.timed_out}

    def _t_read_file(self, filepath: str, start_line: int | None = None, end_line: int | None = None) -> dict:
        path = self._path(filepath)
        try:
            lines = self.sb.read(path).splitlines()
        except FileNotFoundError:
            raise ToolError(f"file not found: {filepath}")
        start = max(1, start_line or 1)
        end = min(len(lines), end_line or len(lines))
        truncated = False
        if end - start + 1 > self.limits.file_lines:
            end = start + self.limits.file_lines - 1
            truncated = True
        body = "\n".join(f"{i}\t{lines[i - 1]}" for i in range(start, end + 1))
        if len(body) > self.limits.file_chars:
            body = body[: self.limits.file_chars]
            truncated = True
        return {"content": body, "total_lines": len(lines), "shown": [start, end], "truncated": truncated}

    def _t_edit_file(self, filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> dict:
        path = self._path(filepath)
        try:
            text = self.sb.read(path)
        except FileNotFoundError:
            raise ToolError(f"file not found: {filepath}")
        if not old_string:
            raise ToolError("old_string must not be empty")
        count = text.count(old_string)
        if count == 0:
            raise ToolError("old_string not found; read the file again and copy the text exactly")
        if count > 1 and not allow_multiple:
            raise ToolError(f"old_string matches {count} times; add context or set allow_multiple")
        self.sb.write(path, text.replace(old_string, new_string))
        return {"status": "ok", "replacements": count}

    def _t_write_file(self, filepath: str, content: str) -> dict:
        path = self._path(filepath)
        self.sb.write(path, content)
        return {"status": "ok", "bytes": len(content.encode())}

    def _t_submit_patch(self) -> dict:
        self.patch = strip_protected(self.sb.diff())
        self.submitted = True
        files = [line[6:] for line in self.patch.splitlines() if line.startswith("+++ b/")]
        return {"status": "submitted", "files": files, "empty": not self.patch.strip()}

    def _t_get_status(self) -> dict:
        return {
            "tool_calls_used": self.tool_calls_used,
            "tool_calls_remaining": max(0, self.limits.tool_calls - self.tool_calls_used),
            "turns_used": self.turns_used,
            "turns_remaining": max(0, self.limits.turns - self.turns_used),
        }

    def workspace_listing(self, max_entries: int = 200) -> str:
        cmd = (
            "find . -mindepth 1 -maxdepth 3 -not -path '*/.*' -not -name '__pycache__' "
            "-not -name '*.pyc' -not -path '*.egg-info*' | sort | head -n " + shlex.quote(str(max_entries))
        )
        return self.sb.exec(cmd, timeout=60).output
