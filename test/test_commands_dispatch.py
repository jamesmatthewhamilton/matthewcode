"""Slash-command dispatch (try_dispatch_command).

Tested against a synthetic command injected into the registry — no real
command is named, so dispatch behavior is pinned independently of what
commands exist.
"""

import matthewcode as m


def test_dispatch_runs_command_and_splits_arg(make_ctx, make_command):
    cmd, calls = make_command()
    tok = cmd.flag_command[0]
    ctx = make_ctx()
    assert m.try_dispatch_command(f"/{tok} some arg", ctx) is True
    assert calls == [(ctx, "some arg")]             # arg split off and stripped
    assert m.try_dispatch_command(f"/{tok}", ctx) is True
    assert calls[-1] == (ctx, "")                   # bare token: empty arg


def test_dispatch_rejects_unknown_and_non_slash(make_ctx):
    tok = "zz-no-such-cmd"
    assert tok not in m.COMMAND_BY_TOKEN
    ctx = make_ctx()
    assert m.try_dispatch_command(f"/{tok}", ctx) is False
    assert m.try_dispatch_command("plain prompt text", ctx) is False
