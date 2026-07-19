"""Shared pytest fixtures for the matthewcode test suite.

matthewcode is imported as a module (main() is __main__-guarded); tests drive
handlers with a hand-built ReplContext — no REPL, no provider, no network.

Ground rule for this suite: tests may reference functions (m.cmd_clear,
m.expand_snippets — renaming a function is a code change that should touch
tests), but must NOT hardcode command tokens, alias names, snippet names, or
prompt text as string literals. Those live in the registry/config: iterate
them, or inject synthetic entries with monkeypatch.
"""

import pytest

import matthewcode as m


@pytest.fixture
def make_ctx():
    """Factory for a minimal ReplContext."""
    def _make(**kw):
        defaults = dict(args=None, console=None, client=None, messages=[],
                        session_name=None, session_file="x.json", ctx_tokens=0)
        defaults.update(kw)
        return m.ReplContext(**defaults)
    return _make


@pytest.fixture
def make_command(monkeypatch):
    """Factory for a synthetic Command registered under a token no real command
    uses, so dispatch tests exercise the machinery without naming any real
    command. Returns (command, calls) — calls records each (ctx, arg) run."""
    def _make(token="zz-test-cmd"):
        assert token not in m.COMMAND_BY_TOKEN     # never shadow a real command
        calls = []
        cmd = m.Command(flag_command=(token,), help="synthetic test command",
                        run=lambda ctx, arg: calls.append((ctx, arg)))
        monkeypatch.setitem(m.COMMAND_BY_TOKEN, token, cmd)
        return cmd, calls
    return _make
