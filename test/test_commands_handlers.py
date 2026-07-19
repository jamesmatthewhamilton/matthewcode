"""Behavior of individual /command handlers (cmd_clear, cmd_history, cmd_help)."""

import matthewcode as m


def test_cmd_clear_resets_state(make_ctx, tmp_path):
    sf = str(tmp_path / "s.json")
    ctx = make_ctx(messages=[{"role": "user", "content": "hi"}],
                   session_name="foo", session_file=sf, ctx_tokens=99)
    m.cmd_clear(ctx, "")
    assert ctx.ctx_tokens == 0
    assert [x["role"] for x in ctx.messages] == ["system"]
    assert ctx.session_name == "foo"                  # cleared in place, name kept
    assert ctx.session_file == sf
    assert [x["role"] for x in m.load_history(sf)] == ["system"]   # persisted


def test_cmd_history_empty_is_safe(make_ctx, capsys):
    m.cmd_history(make_ctx(messages=[]), "")
    assert "No history" in capsys.readouterr().out


def test_help_lists_every_command_token(make_ctx, capsys):
    m.cmd_help(make_ctx(), "")
    out = capsys.readouterr().out
    for c in m.COMMANDS:
        if not c.is_command:              # flag-only entries don't appear in /help
            continue
        for t in c.flag_command:          # every token shown, whatever it's named
            assert f"/{t}" in out
