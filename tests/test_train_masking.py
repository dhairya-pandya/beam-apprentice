import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("train_sft", Path(__file__).parents[1] / "training" / "train_sft.py")
train_sft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_sft)

TEXT = (
    "<bos><|turn>system\nrules<turn|>\n<|turn>user\nfix the bug<turn|>\n"
    "<|turn>model\n<|channel>thought\nlook<channel|><|tool_call>call:read_file{}<tool_call|>"
    "<|tool_response>response:read_file{SECRET}<tool_response|>"
    "<|tool_call>call:submit_patch{}<tool_call|><turn|>\n"
)


def covered(text, spans):
    return "".join(text[a:b] for a, b in spans)


def test_spans_cover_model_output_but_not_tool_results():
    got = covered(TEXT, train_sft.train_char_spans(TEXT))
    assert "look" in got
    assert "call:read_file" in got and "call:submit_patch" in got
    assert got.endswith("<turn|>")
    assert "SECRET" not in got
    assert "fix the bug" not in got and "rules" not in got


def test_labels_follow_spans():
    text = "aa<|turn>model\nBB<|tool_response>CC<tool_response|>DD<turn|>"
    ids = list(range(len(text)))
    offsets = [(i, i + 1) for i in range(len(text))]
    labels = train_sft.labels_from_offsets(ids, offsets, train_sft.train_char_spans(text))
    kept = "".join(text[i] for i, l in enumerate(labels) if l != -100)
    assert kept == "BBDD<turn|>"


def test_prepare_messages_parses_arguments_and_can_drop_thinking():
    ex = {"messages": [
        {"role": "assistant", "content": "", "reasoning_content": "hmm",
         "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]},
    ]}
    with_t = train_sft.prepare_messages(ex, thinking=True)
    assert with_t[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    assert with_t[0]["reasoning_content"] == "hmm"
    assert "reasoning_content" not in train_sft.prepare_messages(ex, thinking=False)[0]
    assert ex["messages"][0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'  # input untouched
