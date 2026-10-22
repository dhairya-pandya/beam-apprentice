from apprentice.patches import is_protected, strip_protected

TEST_HUNK = """diff --git a/tests/test_a.py b/tests/test_a.py
--- a/tests/test_a.py
+++ b/tests/test_a.py
@@ -1 +1 @@
-assert False
+assert True
"""
CONFIG_HUNK = """diff --git a/pyproject.toml b/pyproject.toml
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -1 +1 @@
-x = 1
+x = 2
"""
SRC_HUNK = """diff --git a/src/pkg/a.py b/src/pkg/a.py
--- a/src/pkg/a.py
+++ b/src/pkg/a.py
@@ -1 +1 @@
-y = 1
+y = 2
"""
NEW_FILE = """diff --git a/pkg/new_helper.py b/pkg/new_helper.py
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/pkg/new_helper.py
@@ -0,0 +1 @@
+z = 3
"""


def test_strip_removes_test_and_config_hunks():
    out = strip_protected(TEST_HUNK + CONFIG_HUNK + SRC_HUNK)
    assert out == SRC_HUNK


def test_strip_keeps_new_non_test_file():
    assert strip_protected(NEW_FILE) == NEW_FILE


def test_strip_empty_patch():
    assert strip_protected("") == ""


def test_is_protected_rules():
    for p in [
        "tests/test_x.py", "pkg/test_x.py", "pkg/x_test.py", "test/helpers.py",
        "a/testing/util.py", "conftest.py", "sub/conftest.py", "pytest.ini", "pyproject.toml",
        "tox.ini", "setup.cfg", ".pytest.ini", "sitecustomize.py", "usercustomize.py", "x.pth",
    ]:
        assert is_protected(p), p
    for p in ["pkg/core.py", "docs/tests.md", "tests/data.json", "src/contest.py"]:
        assert not is_protected(p), p
