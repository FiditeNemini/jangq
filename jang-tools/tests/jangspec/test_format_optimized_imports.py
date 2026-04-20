"""M122 (iter 46): verify format.py struct-size checks survive `python -O`.

Background: `assert` statements are stripped by `python -O` (and by
`python -OO`). The binary-layout size checks in `jang_tools.jangspec.format`
gate the on-disk format. If someone changes `BLOB_HEADER_FORMAT` but
forgets to update the `== 32` constant, a plain `assert` would silently
skip the check under -O and readers would misalign tensors at runtime.
Iter 46 converted the asserts to `if ... != ...: raise ImportError(...)`
so the check runs unconditionally. This test pins that behavior.
"""
import subprocess
import sys


def test_format_module_imports_under_regular_python():
    """Sanity: import works at all under normal python."""
    r = subprocess.run(
        [sys.executable, "-c", "import jang_tools.jangspec.format as fmt; "
                               "print(fmt.BLOB_HEADER_SIZE, fmt.TENSOR_HEADER_SIZE, "
                               "fmt.INDEX_ENTRY_SIZE, fmt.INDEX_HEADER_SIZE)"],
        capture_output=True, text=True, check=True,
    )
    assert r.stdout.strip() == "32 36 28 24", f"unexpected sizes: {r.stdout!r}"


def test_format_module_imports_under_python_O():
    """The module must import cleanly under `python -O` — and the size checks
    must STILL run (they use `if ... != ...: raise`, not `assert`)."""
    r = subprocess.run(
        [sys.executable, "-O", "-c",
         "import jang_tools.jangspec.format as fmt; "
         "print(fmt.BLOB_HEADER_SIZE, fmt.TENSOR_HEADER_SIZE, "
         "fmt.INDEX_ENTRY_SIZE, fmt.INDEX_HEADER_SIZE)"],
        capture_output=True, text=True, check=True,
    )
    assert r.stdout.strip() == "32 36 28 24", f"unexpected sizes: {r.stdout!r}"


def test_format_module_imports_under_python_OO():
    """-OO additionally strips docstrings. Size-check behavior must be
    unchanged (same reasoning as -O)."""
    r = subprocess.run(
        [sys.executable, "-OO", "-c",
         "import jang_tools.jangspec.format as fmt; "
         "print(fmt.BLOB_HEADER_SIZE)"],
        capture_output=True, text=True, check=True,
    )
    assert r.stdout.strip() == "32", f"unexpected: {r.stdout!r}"


def test_format_source_has_no_plain_asserts_on_size_constants():
    """Regression guard: prevent a future edit from re-introducing `assert`
    on the size constants. If someone types `assert BLOB_HEADER_SIZE ==` in
    format.py, this test fails. Unique-substring match scoped to the four
    constants we explicitly migrated (leaves other asserts — none currently
    exist — untouched)."""
    import jang_tools.jangspec.format as fmt
    import inspect

    source = inspect.getsource(fmt)
    # Grep the source for any `assert <SIZE_CONSTANT> == ...` pattern.
    forbidden = [
        "assert BLOB_HEADER_SIZE",
        "assert TENSOR_HEADER_SIZE",
        "assert INDEX_ENTRY_SIZE",
        "assert INDEX_HEADER_SIZE",
    ]
    for pattern in forbidden:
        assert pattern not in source, (
            f"format.py reintroduced `{pattern}` — `assert` is stripped by "
            f"python -O and would silently skip this size gate. Use "
            f"`if X != N: raise ImportError(...)` instead (M122)."
        )
