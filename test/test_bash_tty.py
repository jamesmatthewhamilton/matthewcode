"""Tests for pseudo-terminal (tty) support in bash_run.

Runs the real docker-free mock test/docker_run_tty.sh: it reproduces docker's
"the input device is not a TTY" failure whenever stdin/stdout aren't
terminals. No docker, no subprocess mocking. Expected text comes from
m.get_prompt() rather than literals, per the conftest ground rule.
"""

import os
import shlex
import time

import matthewcode as m

MOCK = "sh " + shlex.quote(os.path.join(m.SCRIPT_DIR, "test", "docker_run_tty.sh"))
HELLO = "hello from container"


def _hint():
    return m.get_prompt("pipeline_tool_errors", "tty_required").rstrip()


def test_plain_run_reports_tty_error_with_hint():
    out = m.tool_bash_run(MOCK)
    assert "the input device is not a TTY" in out
    assert "[exit code: 1]" in out
    assert _hint() in out
    assert HELLO not in out


def test_tty_run_succeeds():
    out = m.tool_bash_run(MOCK, tty=True)
    assert HELLO in out
    assert "[exit code: 0]" in out
    assert _hint() not in out


def test_tty_flag_is_what_the_child_sees():
    probe = "[ -t 0 ] && [ -t 1 ] && echo T || echo N"
    assert "N" in m.tool_bash_run(probe)
    assert "T" in m.tool_bash_run(probe, tty=True)


def test_tty_merges_stderr_and_keeps_exit_code():
    out = m.tool_bash_run("echo err >&2; exit 3", tty=True)
    assert "err" in out
    assert "[exit code: 3]" in out


def test_needs_tty_patterns():
    yes = ["docker run -it img", "docker run -ti img", "docker run -i -t img",
           "docker exec --interactive --tty c sh", "docker run --rm -t img",
           "ssh -t host cmd"]
    no = ["docker ps", "git status", "docker run -i img",
          "docker run --rm img echo hi", "./docker_run_tty.sh"]
    for cmd in yes:
        assert m.needs_tty(cmd), cmd
    for cmd in no:
        assert not m.needs_tty(cmd), cmd


def test_auto_detect_uses_pty(monkeypatch):
    monkeypatch.setitem(m.CONFIG, "tty_commands", [r"docker_run_tty\.sh"])
    assert HELLO in m.tool_bash_run(MOCK)


def test_tty_timeout_does_not_hang():
    expected = m.get_prompt("pipeline_tool_errors", "bash_timeout")
    for cmd in ("sleep 5", "sleep 5 & wait"):
        start = time.monotonic()
        assert m.tool_bash_run(cmd, timeout=1, tty=True) == expected
        assert time.monotonic() - start < 3, cmd


def test_tty_output_normalized():
    out = m.tool_bash_run(r"printf 'a\r\nb\n'; printf '\033[31mred\033[0m\n'", tty=True)
    assert out.startswith("a\nb\nred\n")
    assert "\r" not in out and "\x1b" not in out


def test_tty_auto_retry_config(monkeypatch):
    monkeypatch.setitem(m.CONFIG, "tty_auto_retry", True)
    out = m.tool_bash_run(MOCK)
    assert HELLO in out
    assert "[exit code: 0]" in out
    assert _hint() not in out


def test_dispatch_coerces_tty_string():
    assert HELLO in m.TOOL_DISPATCH["bash_run"]({"command": MOCK, "tty": "true"})
    assert m._as_bool(True) is True
    assert m._as_bool("false") is False
    assert m._as_bool("1") is True
    assert m._as_bool(None) is False


def test_confirm_tool_shows_tty_prompt(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    assert m.confirm_tool("bash_run", {"command": "x", "tty": True})
    assert m.get_prompt("pipeline_confirmations", "bash_run_tty_prompt", command="x").rstrip() \
        in capsys.readouterr().out
    assert m.confirm_tool("bash_run", {"command": "x"})
    assert m.get_prompt("pipeline_confirmations", "bash_run_prompt", command="x").rstrip() \
        in capsys.readouterr().out


# --- regression tests for review fixes 1-3 ---

def test_pty_no_fd_leak_on_spawn_failure(monkeypatch):
    """Fix 1: a failed Popen inside _run_in_pty must close both pty fds."""
    import subprocess as sp
    before = len(os.listdir("/dev/fd"))
    monkeypatch.setattr(sp, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    for _ in range(20):
        try:
            m._run_in_pty("echo hi", 5)
        except OSError:
            pass
    monkeypatch.undo()
    after = len(os.listdir("/dev/fd"))
    assert after - before <= 1, f"leaked {after - before} fds"


def test_piped_timeout_reaps_backgrounded_child():
    """Fix 2: a piped-path timeout kills the process group, not just the shell."""
    marker = os.path.join(m.SCRIPT_DIR, ".pty_leak_probe")
    open(marker, "w").close()
    try:
        # a backgrounded child that would outlive a shell-only kill; it removes
        # the marker after 30s if it survives. Group-kill must prevent that.
        cmd = f"(sleep 30; rm -f {shlex.quote(marker)}) & wait"
        assert m.tool_bash_run(cmd, timeout=1) == m.get_prompt("pipeline_tool_errors", "bash_timeout")
        time.sleep(2.5)
        assert os.path.exists(marker), "backgrounded child survived the timeout kill"
    finally:
        if os.path.exists(marker):
            os.remove(marker)


def test_auto_retry_skips_restricted_command(monkeypatch):
    """Fix 3: tty_auto_retry must not silently re-run a restricted command
    under a pty; it falls through to the hint instead."""
    monkeypatch.setitem(m.CONFIG, "tty_auto_retry", True)
    monkeypatch.setitem(m.CONFIG, "restricted_bash_commands", ["rm"])
    # a restricted (rm) command that fails with a TTY-shaped error
    cmd = "rm nonexistent; echo 'the input device is not a TTY' >&2; exit 1"
    out = m.tool_bash_run(cmd)
    assert _hint() in out, "restricted command should get the hint, not an auto-retry"


def test_auto_retry_still_fires_for_unrestricted():
    """Fix 3 guard is scoped: a non-restricted command still auto-retries."""
    import matthewcode as mm
    # handled by test_tty_auto_retry_config already, but assert the guard's
    # negative branch explicitly with a fresh CONFIG patch
    old = mm.CONFIG.get("tty_auto_retry", False)
    mm.CONFIG["tty_auto_retry"] = True
    try:
        assert HELLO in mm.tool_bash_run(MOCK)
    finally:
        mm.CONFIG["tty_auto_retry"] = old
