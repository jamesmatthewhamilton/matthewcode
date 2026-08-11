"""Plan mode: /plan and /next drive a per-session TODO_<session>.md checklist."""

import os

import matthewcode as m


def test_plan_file_follows_session(make_ctx, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()
    unnamed = m.plan_file(ctx)
    assert os.path.dirname(unnamed) == str(tmp_path)        # lives in the cwd
    assert m.UNNAMED_SESSION_LABEL in os.path.basename(unnamed)
    ctx.session_name = "zz-test-session"
    named = m.plan_file(ctx)
    assert named != unnamed                                 # recomputed per call
    assert "zz-test-session" in os.path.basename(named)


def test_expand_snippets_dynamic_vars(make_ctx, monkeypatch):
    ctx = make_ctx(session_name="zz-test-session")
    for name, value in m.session_vars(ctx).items():
        token = "{{" + name + "}}"
        assert m.expand_snippets(token) == token            # no ctx: untouched
        assert m.expand_snippets(token, ctx) == value       # with ctx: resolved
        monkeypatch.setitem(m.CONFIG.setdefault("snippets", {}), name, "shadowed")
        assert m.expand_snippets(token, ctx) == value       # built-ins win


def test_plan_and_next_registered():
    for fn in (m.cmd_plan, m.cmd_next):
        (cmd,) = [c for c in m.COMMANDS if c.run is fn]
        assert cmd.is_command and cmd.is_flag
        assert cmd.one_shot and cmd.needs_client
        for tok in cmd.flag_command:
            assert m.COMMAND_BY_TOKEN[tok] is cmd


def test_cmd_plan_no_arg_reads_file_without_model(make_ctx, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()                     # client=None: a model call would explode
    m.cmd_plan(ctx, "")
    assert m.plan_file(ctx) in capsys.readouterr().out      # usage hint, no file
    with open(m.plan_file(ctx), "w") as f:
        f.write("- [ ] first step\n")
    m.cmd_plan(ctx, "")
    assert "first step" in capsys.readouterr().out
    assert ctx.messages == []                               # no conversation turn


def test_cmd_plan_with_arg_composes_and_dispatches(make_ctx, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = []
    monkeypatch.setattr(m, "handle_input",
                        lambda text, ctx, *, interactive: captured.append(text))
    monkeypatch.setitem(m.CONFIG, m.PIPELINE_PLAN, {"user_prompt": "plan for {plan_file}"})
    ctx = make_ctx()
    m.cmd_plan(ctx, "do the thing")
    assert len(captured) == 1
    assert m.plan_file(ctx) in captured[0]                  # template var filled
    assert captured[0].endswith("do the thing")             # typed arg at bottom


def test_cmd_next_short_circuits_locally(make_ctx, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = []
    monkeypatch.setattr(m, "handle_input",
                        lambda text, ctx, *, interactive: captured.append(text))
    monkeypatch.setitem(m.CONFIG, m.PIPELINE_NEXT, {"user_prompt": "next in {plan_file}"})
    ctx = make_ctx()
    m.cmd_next(ctx, "")                                     # no plan file
    assert captured == []
    with open(m.plan_file(ctx), "w") as f:
        f.write("- [x] done step\n")
    m.cmd_next(ctx, "")                                     # everything checked
    assert captured == []
    with open(m.plan_file(ctx), "a") as f:
        f.write("- [ ] open step\n")
    m.cmd_next(ctx, "")                                     # one open step: dispatch
    assert len(captured) == 1 and m.plan_file(ctx) in captured[0]


def test_dispatch_prompt_appends_arg(make_ctx, monkeypatch):
    captured = []
    monkeypatch.setattr(m, "handle_input",
                        lambda text, ctx, *, interactive: captured.append(text))
    ctx = make_ctx()
    m._dispatch_prompt(ctx, "base prompt")
    m._dispatch_prompt(ctx, "base prompt", "extra")
    assert captured == ["base prompt", "base prompt\n\nextra"]


def test_prompt_command_routes_through_dispatch(make_ctx, monkeypatch):
    # regression: aliases share the same dispatch tail as the plan commands
    captured = []
    monkeypatch.setattr(m, "handle_input",
                        lambda text, ctx, *, interactive: captured.append(text))
    monkeypatch.setitem(m.CONFIG, "aliases", {"zz-test-alias": "alias prompt"})
    ctx = make_ctx()
    m._prompt_command("zz-test-alias")(ctx, "tail")
    assert captured == ["alias prompt\n\ntail"]


def test_live_config_plan_pipelines_format_clean(make_ctx):
    # success doubles as the no-stray-braces check: .format would raise
    ctx = make_ctx(session_name="zz-live")
    for pipeline in (m.PIPELINE_PLAN, m.PIPELINE_NEXT):
        text = m.get_prompt(pipeline, "user_prompt", **m.session_vars(ctx))
        assert m.plan_file(ctx) in text
