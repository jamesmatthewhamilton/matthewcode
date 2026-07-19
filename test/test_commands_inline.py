"""Inline command injection: !`command` in a prompt runs and inlines its output."""

import matthewcode as m


def test_inline_command_expansion():
    for text, want in [
        ("branch: !`echo hi`", "branch: hi"),                   # single
        ("a !`echo one` b !`echo two` c", "a one b two c"),     # multiple
        ("no commands here", "no commands here"),               # untouched
        ("bang! and `backticks` apart", "bang! and `backticks` apart"),
        ("!`unclosed", "!`unclosed"),
        ("staged: !`true`", "staged: (empty)"),                 # silent success marker
    ]:
        assert m.expand_inline_commands(text, interactive=False) == want


def test_inline_command_failure_keeps_exit_code():
    assert "[exit code: 3]" in m.expand_inline_commands("!`exit 3`", interactive=False)


def test_inline_command_restricted_skipped_when_not_interactive(monkeypatch):
    # 'rm' is a restricted word — non-interactive prompts must not run it
    monkeypatch.setitem(m.CONFIG, "restricted_bash_commands", ["rm"])
    out = m.expand_inline_commands("do !`rm -rf /tmp/nope`", interactive=False)
    assert out == "do [skipped restricted command: rm -rf /tmp/nope]"
