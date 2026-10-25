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
    _fn("replace_lines", "Replace lines start_line..end_line (1-indexed, inclusive, as numbered by read_file) "
        "with new_content. Use this when edit_file cannot match the text exactly.",
        {"filepath": _STR, "start_line": _INT, "end_line": _INT, "new_content": _STR},
        ["filepath", "start_line", "end_line", "new_content"]),
    _fn("write_file", "Create or overwrite a file, creating parent directories.",
        {"filepath": _STR, "content": _STR}, ["filepath", "content"]),
    _fn("submit_patch", "Submit your changes as the final patch. Call this last, after verifying the fix.",
        {"note": {"type": "string", "description": "Optional message to the user who gave you the task, "
                  "like a pull request description."}}, []),
    _fn("get_status", "Show remaining tool calls and turns.", {}, []),
]

FREE_TOOLS = {"submit_patch", "get_status"}


class ToolError(Exception):
    pass


_ESCAPES = {"\\n": "\n", "\\t": "\t", '\\"': '"', "\\\\": "\\"}


def _decode_escapes(text: str) -> str:
    """Turn literal \\n, \\t, \\" and \\\\ (copied from JSON-encoded tool output) into real characters."""
    out, i = [], 0
    while i < len(text):
        pair = text[i : i + 2]
        if pair in _ESCAPES:
            out.append(_ESCAPES[pair])
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _looks_escaped(text: str) -> bool:
    return "\\n" in text and "\n" not in text


def _escape_variants(text: str) -> list[str]:
    """Readings of text an agent may have meant when it copied JSON escapes, most literal first."""
    if "\\" not in text:
        return []
    variants = [_decode_escapes(text), _decode_escapes(text.replace("\\\n", "\n"))]
    return [v for i, v in enumerate(variants) if v != text and v not in variants[:i]]


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _replace_ignoring_indentation(text: str, old: str, new: str) -> str | None:
    """Fallback for edit_file: match old line by line ignoring surrounding whitespace.

    Only a unique match is accepted. The replacement is re-indented by the difference between
    the matched block's indentation and the indentation the caller wrote.
    """
    old_lines = old.split("\n")
    new_lines = new.split("\n")
    while old_lines and not old_lines[-1].strip():
        old_lines.pop()
    while new_lines and not new_lines[-1].strip():
        new_lines.pop()
    while old_lines and not old_lines[0].strip():
        old_lines.pop(0)
    if not old_lines:
        return None
    lines = text.split("\n")
    want = [l.strip() for l in old_lines]
    hits = [i for i in range(len(lines) - len(want) + 1)
            if [l.strip() for l in lines[i : i + len(want)]] == want]
    if len(hits) != 1:
        return None
    start = hits[0]
    have_indent = _indent(lines[start])
    wrote_indent = _indent(old_lines[0])

    def reindent(line: str) -> str:
        if not line.strip():
            return line
        if line.startswith(wrote_indent):
            return have_indent + line[len(wrote_indent):]
        return have_indent + line.lstrip()

    replaced = [reindent(l) for l in new_lines]
    return "\n".join(lines[:start] + replaced + lines[start + len(want):])


class Toolset:
    def __init__(self, sandbox, limits: Limits):
        self.sb = sandbox
        self.limits = limits
        self.tool_calls_used = 0
        self.turns_used = 0
        self.submitted = False
        self.patch = ""
        self.note = ""

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
            fuzzy = _replace_ignoring_indentation(text, old_string, new_string)
            matched = "ignoring indentation"
            if fuzzy is None:
                for old_d, new_d in zip(_escape_variants(old_string), _escape_variants(new_string) or [new_string] * 2):
                    if text.count(old_d) == 1:
                        fuzzy = text.replace(old_d, new_d)
                    else:
                        fuzzy = _replace_ignoring_indentation(text, old_d, new_d)
                    if fuzzy is not None:
                        matched = "after decoding escapes"
                        break
            if fuzzy is None:
                raise ToolError("old_string not found; read the file again and copy the text exactly")
            self.sb.write(path, fuzzy)
            return {"status": "ok", "replacements": 1, "matched": matched}
        if count > 1 and not allow_multiple:
            raise ToolError(f"old_string matches {count} times; add context or set allow_multiple")
        self.sb.write(path, text.replace(old_string, new_string))
        return {"status": "ok", "replacements": count, "matched": "exact"}

    def _t_replace_lines(self, filepath: str, start_line: int, end_line: int, new_content: str) -> dict:
        path = self._path(filepath)
        try:
            text = self.sb.read(path)
        except FileNotFoundError:
            raise ToolError(f"file not found: {filepath}")
        lines = text.split("\n")
        trailing = text.endswith("\n")
        if trailing:
            lines = lines[:-1]
        if not (1 <= start_line <= end_line <= len(lines)):
            raise ToolError(f"need 1 <= start_line <= end_line <= {len(lines)}; got {start_line}..{end_line}")
        if _looks_escaped(new_content):
            new_content = _decode_escapes(new_content)
        new_lines = new_content.split("\n")
        if new_lines and new_lines[-1] == "":
            new_lines = new_lines[:-1]
        out = lines[: start_line - 1] + new_lines + lines[end_line:]
        self.sb.write(path, "\n".join(out) + ("\n" if trailing else ""))
        return {"status": "ok", "lines_replaced": end_line - start_line + 1, "new_total_lines": len(out)}

    def _t_write_file(self, filepath: str, content: str) -> dict:
        path = self._path(filepath)
        if _looks_escaped(content):
            content = _decode_escapes(content)
        self.sb.write(path, content)
        return {"status": "ok", "bytes": len(content.encode())}

    def _t_submit_patch(self, note: str = "") -> dict:
        self.note = note
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

    def workspace_listing(self, max_entries: int = 80, max_depth: int = 2) -> str:
        cmd = (
            f"find . -mindepth 1 -maxdepth {int(max_depth)} -not -path '*/.*' -not -name '__pycache__' "
            "-not -name '*.pyc' -not -path '*.egg-info*' | sort | head -n " + shlex.quote(str(max_entries))
        )
        return self.sb.exec(cmd, timeout=60).output
