"""The task prompt an agent sees, plus the nudges sent when a turn ends without progress."""

from __future__ import annotations

from apprentice.tasks import Task
from apprentice.tools import Limits

SYSTEM = (
    "You are an autonomous software engineer working inside a sandboxed copy of a Python repository. "
    "You act only through the provided tools. Make the smallest correct change that resolves the issue."
)

NUDGE_CUT_TOOL_CALL = (
    "Your last reply was cut off before the tool call finished. Skip the long analysis and send your next "
    "tool call right away. Split big edits into smaller ones."
)
NUDGE_LENGTH = (
    "You ran out of room while thinking, before making a tool call. Keep your reasoning to a few sentences "
    "and send the next tool call now, or call submit_patch if you are done and have verified the fix."
)
NUDGE_CONTINUE = (
    "Please keep going with the tools, or call submit_patch once your change is complete and verified."
)


def build_prompt(task: Task, workspace_listing: str, limits: Limits) -> str:
    parts = [
        f"You are working on a software engineering task for the repository {task.repo}.\n\n"
        f"Problem statement:\n{task.problem_statement.strip()}"
    ]
    if task.hints_text.strip():
        parts.append(f"## Hints\n{task.hints_text.strip()}")
    parts.append(
        "## Budget (the session ends when any of these runs out)\n"
        f"- Tool calls: {limits.tool_calls}\n"
        f"- Turns: {limits.turns}"
    )
    parts.append(
        "## Environment\n"
        f"- One command may run for at most {limits.cmd_timeout} seconds.\n"
        f"- Command output is cut to {limits.out_chars} characters.\n"
        f"- read_file shows at most {limits.file_lines} lines and {limits.file_chars} characters per call.\n"
        "- There is no network. Dependencies are already installed; do not run pip install."
    )
    parts.append(
        "## How to work\n"
        "0. Work only under /workspace.\n"
        "1. Read the relevant code and follow the repository's existing conventions.\n"
        "2. Fix the library code. Hidden tests decide whether the task is solved, and any edits you make to "
        "test files or test configuration are discarded before they run.\n"
        "3. Check your change with targeted tests or quick assertions before submitting.\n"
        "4. Put scratch scripts in /tmp, not /workspace, or they become part of your patch.\n"
        "5. If edit_file cannot match your text, use replace_lines with the line numbers from read_file.\n"
        "6. Call submit_patch as your final action."
    )
    parts.append(f"## Workspace layout\n{workspace_listing.strip()}")
    return "\n\n".join(parts)


def nudge_for(finish_reason: str, content: str) -> str:
    if "<|tool_call" in content and "<tool_call|>" not in content:
        return NUDGE_CUT_TOOL_CALL
    if finish_reason.lower() in ("length", "max_tokens"):
        return NUDGE_LENGTH
    return NUDGE_CONTINUE
