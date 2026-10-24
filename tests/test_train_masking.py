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


SESSION = (
    "<|turn>user\nTASK<turn|>\n<|turn>model\n"
    + "".join(f"<|tool_call>s{i}<tool_call|><|tool_response>r{i}<tool_response|>" for i in range(6))
    + "<|tool_call>submit<tool_call|>"
)


def test_split_steps_round_trips():
    prefix, steps = train_sft.split_steps(SESSION)
    assert prefix.endswith("<|turn>model\n")
    assert prefix + "".join(steps) == SESSION
    assert len(steps) == 7 and steps[-1] == "<|tool_call>submit<tool_call|>"


def test_windows_fit_and_train_every_step_once():
    prefix, steps = train_sft.split_steps(SESSION)
    n = len  # one "token" per character keeps the test exact
    max_len = len(prefix) + 3 * len(steps[0]) + 64
    windows = list(train_sft.make_windows(prefix, steps, n, max_len))
    assert len(windows) > 1
    trained = []
    for text, label_from in windows:
        assert text.startswith(prefix) and len(text) <= max_len
        spans = train_sft.clip_spans(train_sft.train_char_spans(text), label_from)
        trained.append("".join(text[a:b] for a, b in spans))
    joined = "".join(trained)
    for i in range(6):
        assert joined.count(f"s{i}") == 1
        assert f"r{i}" not in joined
    assert joined.count("submit") == 1


def test_short_session_is_one_window():
    prefix, steps = train_sft.split_steps(SESSION)
    windows = list(train_sft.make_windows(prefix, steps, len, 10_000))
    assert windows == [(SESSION, len(prefix))]


def test_last_step_window_trains_only_final_step():
    prefix, steps = train_sft.split_steps(SESSION)
    text, label_from = train_sft.last_step_window(prefix, steps, len, 10_000)
    assert text == SESSION
    spans = train_sft.clip_spans(train_sft.train_char_spans(text), label_from)
    trained = "".join(text[a:b] for a, b in spans)
    assert trained == "<|tool_call>submit<tool_call|>"


def test_last_step_window_drops_oldest_context_to_fit():
    prefix, steps = train_sft.split_steps(SESSION)
    max_len = len(prefix) + len(steps[-1]) + 2 * len(steps[0]) + 64
    text, label_from = train_sft.last_step_window(prefix, steps, len, max_len)
    assert text.startswith(prefix) and text.endswith(steps[-1]) and len(text) <= max_len
    assert "s0" not in text and "s5" in text


def test_last_step_window_none_when_step_too_big():
    prefix, steps = train_sft.split_steps(SESSION)
    assert train_sft.last_step_window(prefix, steps, len, len(prefix) + 5) is None
