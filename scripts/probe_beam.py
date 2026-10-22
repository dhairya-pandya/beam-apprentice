"""One cheap live call to Beam: prints the raw response shape so the client's assumptions can be checked.

Usage: uv run python scripts/probe_beam.py
"""

import json
import os

from openai import OpenAI

from apprentice.beam_client import BEAM_MODEL, HF_ROUTER, BeamClient, assistant_message, hf_token
from apprentice.ledger import CostLedger

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the repository.",
        "parameters": {
            "type": "object",
            "properties": {"filepath": {"type": "string"}},
            "required": ["filepath"],
        },
    },
}]

raw = OpenAI(base_url=HF_ROUTER, api_key=hf_token())
ledger = CostLedger("runs/ledger.jsonl", cap_usd=25.0, session_cap_usd=0.05)
client = BeamClient(raw, ledger, meta={"run": "probe"})

messages = [
    {"role": "system", "content": "You are a coding agent. Use tools when you need to see code."},
    {"role": "user", "content": "What does src/app.py define? Look before answering."},
]
first = raw.chat.completions.create(
    model=BEAM_MODEL, messages=messages, tools=TOOLS, reasoning_effort="low", max_completion_tokens=2048
)
print("RAW MESSAGE:", json.dumps(first.choices[0].message.model_dump(), indent=2)[:3000])
print("FINISH:", first.choices[0].finish_reason)
print("USAGE:", json.dumps(first.usage.model_dump(), indent=2))
ledger.record(first.usage.model_dump(), {"run": "probe"})

c = client.complete(messages, TOOLS, "low", 2048)
messages.append(assistant_message(c))
for t in c.tool_calls:
    messages.append({"role": "tool", "tool_call_id": t.id, "content": "def hello():\n    return 'world'\n"})
second = client.complete(messages, TOOLS, "low", 2048)
print("SECOND TURN OK:", second.finish_reason, repr(second.content[:300]))
print(f"SPENT THIS PROBE: ${ledger.session_spent():.5f}")
