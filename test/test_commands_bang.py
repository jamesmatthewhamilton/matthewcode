"""Bang commands: a !command line runs immediately, before /commands and the model."""

import matthewcode as m


def test_non_bang_lines_fall_through(make_ctx):
    ctx = make_ctx()
    for text in ["plain prompt text", "/help", "!`echo hi` inline form", "!", "!   "]:
        assert m.try_run_bang_command(text, ctx, interactive=False) is False
    assert ctx.messages == []


def test_bang_runs_and_records_context(make_ctx, tmp_path, capsys):
    ctx = make_ctx(session_file=str(tmp_path / "s.json"))
    assert m.try_run_bang_command("!echo hi", ctx, interactive=False) is True
    assert "hi" in capsys.readouterr().out
    assert len(ctx.messages) == 1
    assert ctx.messages[0]["role"] == "user"
    assert ctx.messages[0]["content"] == m.get_prompt(
        "pipeline_bang_command", "user_prompt", command="echo hi", output="hi")


def test_bang_restricted_skipped_when_not_interactive(make_ctx, tmp_path, monkeypatch):
    monkeypatch.setitem(m.CONFIG, "restricted_bash_commands", ["rm"])
    ctx = make_ctx(session_file=str(tmp_path / "s.json"))
    assert m.try_run_bang_command("!rm -rf /tmp/nope", ctx, interactive=False) is True
    assert "[skipped restricted command: rm -rf /tmp/nope]" in ctx.messages[0]["content"]


def test_handle_input_runs_bang_before_agent_turn(make_ctx, tmp_path, monkeypatch):
    turns = []
    monkeypatch.setattr(m, "run_agent_loop", lambda ctx, *, interactive: turns.append(ctx))
    ctx = make_ctx(session_file=str(tmp_path / "s.json"))
    m.handle_input("!echo hi", ctx, interactive=False)
    assert turns == []                              # no model turn for a bang line
    assert len(ctx.messages) == 1                   # only the recorded command context
