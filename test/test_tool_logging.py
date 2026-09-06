"""Tests for tool-call logging — especially that bash_run's FULL command is
printed on its own line so it can be selected and copied verbatim from the log.
"""

import matthewcode as m


def test_bash_full_command_on_own_line(capsys):
    cmd = "grep -rn 'needle' . --include=*.py | sort -u | head -100  # long-ish command"
    m._log_tool_call("bash_run", {"command": cmd}, interactive=True, verbose=False)
    lines = capsys.readouterr().out.splitlines()
    assert cmd in lines                     # exact full command is its own standalone line
    assert not any("..." in ln and "grep" in ln for ln in lines)  # never truncated


def test_bash_tty_marker_and_command(capsys):
    m._log_tool_call("bash_run", {"command": "docker run -it img sh", "tty": True},
                     interactive=True, verbose=False)
    out = capsys.readouterr().out
    assert "[bash_run (tty)]" in out
    assert "docker run -it img sh" in out.splitlines()


def test_noninteractive_bash_goes_to_stderr_not_stdout(capsys):
    cmd = "echo hello world"
    m._log_tool_call("bash_run", {"command": cmd}, interactive=False, verbose=False)
    captured = capsys.readouterr()
    assert cmd in captured.err.splitlines()     # visible in the log (stderr)
    assert cmd not in captured.out              # must not pollute --prompt's stdout result


def test_non_bash_uses_compact_summary(capsys):
    m._log_tool_call("file_grep", {"pattern": "foo"}, interactive=True, verbose=False)
    assert "[file_grep: foo]" in capsys.readouterr().out


def test_verbose_dumps_full_args(capsys):
    m._log_tool_call("bash_run", {"command": "ls", "tty": True}, interactive=True, verbose=True)
    out = capsys.readouterr().out
    assert '"command": "ls"' in out and '"tty": true' in out
