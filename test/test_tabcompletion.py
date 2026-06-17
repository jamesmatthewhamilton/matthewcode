"""Unit tests for the REPL slash-command completer (res/tabcompletion.py).

Drives the completer with constructed prompt_toolkit Documents — no TTY needed.
"""

from prompt_toolkit.document import Document

from res.tabcompletion import build_slash_completer


def _complete(completer, text):
    doc = Document(text, cursor_position=len(text))
    return [c.text for c in completer.get_completions(doc, None)]


def test_completes_command_prefix():
    c = build_slash_completer(["/help", "/history", "/provider"])
    assert set(_complete(c, "/h")) == {"/help", "/history"}


def test_empty_slash_lists_all_commands():
    c = build_slash_completer(["/help", "/exit", "/provider"])
    assert set(_complete(c, "/")) == {"/help", "/exit", "/provider"}


def test_non_slash_text_yields_nothing():
    c = build_slash_completer(["/help", "/provider"])
    assert _complete(c, "hello world") == []


def test_arg_completion_lists_values():
    c = build_slash_completer(
        ["/provider"], arg_value_funcs={"/provider": lambda: ["alpha", "beta"]}
    )
    assert set(_complete(c, "/provider ")) == {"alpha", "beta"}


def test_arg_completion_filters_by_prefix():
    c = build_slash_completer(
        ["/provider"], arg_value_funcs={"/provider": lambda: ["alpha", "beta"]}
    )
    assert _complete(c, "/provider al") == ["alpha"]


def test_arg_values_resolved_lazily():
    box = {"vals": ["one"]}
    c = build_slash_completer(["/session"],
                             arg_value_funcs={"/session": lambda: box["vals"]})
    assert set(_complete(c, "/session ")) == {"one"}
    box["vals"] = ["one", "two"]                       # mutate after construction
    assert set(_complete(c, "/session ")) == {"one", "two"}  # reflects live state


def test_command_without_arg_completer_offers_no_args():
    c = build_slash_completer(["/help"])
    assert _complete(c, "/help ") == []
