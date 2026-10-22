"""Turn test-verified episodes into supervised fine-tuning examples.

Only resolved episodes from the training split become examples; held-out tasks are refused
outright so the student's evaluation stays honest. Each example keeps the OpenAI-style
message list and tool schemas; the teacher's reasoning stays on each assistant turn under
`reasoning_content`, so the training script can choose how (or whether) to render it.
"""

from __future__ import annotations

import json
from pathlib import Path

from apprentice.tools import SCHEMAS


def _clip_tool_output(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    head = max_chars // 3
    return content[:head] + "\n[... trimmed for training ...]\n" + content[-(max_chars - head):]


def episode_to_example(trace: dict, max_tool_chars: int = 3000) -> dict:
    ep = trace["episode"]
    messages = []
    for m in ep["messages"]:
        m = dict(m)
        if m["role"] == "tool":
            m["content"] = _clip_tool_output(m["content"], max_tool_chars)
        messages.append(m)
    # an episode ends on the submit_patch tool result; nothing after it is useful to imitate
    return {
        "instance_id": ep["instance_id"],
        "repo": trace.get("repo", ""),
        "effort": ep["effort"],
        "turns": ep["turns"],
        "messages": messages,
        "tools": SCHEMAS,
    }


def export_sft(
    trace_dirs: list[Path],
    out_path: Path,
    held_out_ids: set[str],
    max_tool_chars: int = 3000,
    one_per_task: bool = False,
) -> dict:
    """Write resolved training-split episodes as JSONL. Returns counts."""
    examples, seen = [], set()
    skipped = {"unresolved": 0, "held_out": 0, "duplicate": 0}
    for d in trace_dirs:
        for path in sorted(Path(d).glob("traces/*.json")):
            trace = json.loads(path.read_text())
            iid = trace["episode"]["instance_id"]
            if iid in held_out_ids:
                skipped["held_out"] += 1
                continue
            if not trace["verify"]["resolved"]:
                skipped["unresolved"] += 1
                continue
            if one_per_task and iid in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(iid)
            examples.append(episode_to_example(trace, max_tool_chars))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return {"examples": len(examples), "tasks": len(seen), **skipped}
