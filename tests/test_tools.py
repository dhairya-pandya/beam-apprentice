import json

from apprentice.tools import Limits, Toolset


def call(ts, name, **args):
    return json.loads(ts.call(name, json.dumps(args)))


def test_schemas_cover_all_tools(local_sandbox):
    names = [s["function"]["name"] for s in Toolset(local_sandbox, Limits()).schemas()]
    assert names == ["run_command", "read_file", "edit_file", "write_file", "submit_patch", "get_status"]


def test_read_file_line_slicing_and_caps(local_sandbox):
    local_sandbox.write("big.py", "".join(f"line{i}\n" for i in range(1, 401)))
    ts = Toolset(local_sandbox, Limits())
    out = call(ts, "read_file", filepath="big.py", start_line=10, end_line=12)
    assert out["content"].splitlines() == ["10\tline10", "11\tline11", "12\tline12"]
    assert out["total_lines"] == 400
    out = call(ts, "read_file", filepath="big.py")
    assert len(out["content"].splitlines()) == 150
    assert out["truncated"] is True
    out = call(ts, "read_file", filepath="/workspace/pkg/core.py")
    assert "return a - b" in out["content"]


def test_read_file_rejects_escape(local_sandbox):
    out = call(Toolset(local_sandbox, Limits()), "read_file", filepath="../../etc/passwd")
    assert out["status"] == "error"


def test_edit_file_requires_unique_match_unless_allow_multiple(local_sandbox):
    local_sandbox.write("m.py", "x = 1\nx = 1\n")
    ts = Toolset(local_sandbox, Limits())
    assert call(ts, "edit_file", filepath="m.py", old_string="x = 1", new_string="x = 2")["status"] == "error"
    assert call(ts, "edit_file", filepath="m.py", old_string="nope", new_string="y")["status"] == "error"
    out = call(ts, "edit_file", filepath="m.py", old_string="x = 1", new_string="x = 2", allow_multiple=True)
    assert out["status"] == "ok" and out["replacements"] == 2
    assert local_sandbox.read("m.py") == "x = 2\nx = 2\n"


def test_run_command_truncates_output(local_sandbox):
    ts = Toolset(local_sandbox, Limits(out_chars=100))
    out = call(ts, "run_command", command="python3 -c \"print('a' * 1000)\"")
    assert out["exit_code"] == 0
    assert len(out["output"]) <= 100 + 60
    assert out["truncated"] is True


def test_unknown_tool_returns_error_json(local_sandbox):
    out = json.loads(Toolset(local_sandbox, Limits()).call("delete_everything", "{}"))
    assert out["status"] == "error" and "unknown tool" in out["error"]


def test_bad_json_arguments_return_error_json(local_sandbox):
    out = json.loads(Toolset(local_sandbox, Limits()).call("read_file", "{not json"))
    assert out["status"] == "error"
    out = json.loads(Toolset(local_sandbox, Limits()).call("read_file", '{"wrong": 1}'))
    assert out["status"] == "error"


def test_submit_patch_strips_protected(local_sandbox):
    ts = Toolset(local_sandbox, Limits())
    call(ts, "edit_file", filepath="pkg/core.py", old_string="a - b", new_string="a + b")
    call(ts, "write_file", filepath="tests/test_core.py", content="def test_add():\n    pass\n")
    out = call(ts, "submit_patch")
    assert out["status"] == "submitted"
    assert ts.submitted is True
    assert "pkg/core.py" in ts.patch
    assert "tests/test_core.py" not in ts.patch


def test_tool_call_counting_and_status(local_sandbox):
    ts = Toolset(local_sandbox, Limits(tool_calls=5))
    call(ts, "run_command", command="true")
    call(ts, "get_status")
    call(ts, "submit_patch")
    assert ts.tool_calls_used == 1
    assert call(ts, "get_status")["tool_calls_remaining"] == 4
