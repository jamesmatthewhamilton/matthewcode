"""Tests for exit-status-aware trimming of bash_run output and the full-output
log file. A long *failed* command must keep the error at its tail (the old
head-only truncation dropped it); a long *successful* one keeps only a short
tail. Expected marker text comes from m.get_prompt, per the conftest rule.
"""

import os
import re

import pytest

import matthewcode as m

BIG_FAIL = 'seq 1 100000; echo "error: undefined reference to foo" >&2; exit 2'
BIG_OK = "seq 1 100000"


@pytest.fixture(autouse=True)
def _log_in_tmp(monkeypatch, tmp_path):
    # keep the real ~/.matthewcode/logs untouched; also proves the dir is created
    monkeypatch.setattr(m, "BASH_LOG_FILE", str(tmp_path / "logs" / "last_bash.log"))


def _marker_prefix():
    # the omitted count varies; match on the stable part before it
    return m.get_prompt("pipeline_tool_errors", "bash_output_omitted",
                        omitted=0, log_path=m.BASH_LOG_FILE).split("0")[0]


@pytest.mark.parametrize("tty", [False, True])
def test_failed_build_keeps_error_at_tail(tty):
    out = m.tool_bash_run(BIG_FAIL, tty=tty)
    assert "error: undefined reference to foo" in out       # the whole point
    assert "[exit code: 2]" in out
    assert out.startswith("1\n2\n3\n")                     # head slice kept
    assert _marker_prefix() in out and m.BASH_LOG_FILE in out
    assert len(out) <= m.MAX_BASH_OUTPUT + 400              # marker + exit line only


def test_successful_build_keeps_only_tail():
    out = m.tool_bash_run(BIG_OK)
    assert out.startswith(_marker_prefix())                 # no head on success
    assert "100000\n[exit code: 0]" in out
    assert len(out) <= m.BASH_SUCCESS_TAIL + 400


def test_cuts_snap_to_line_boundaries():
    out = m.tool_bash_run(BIG_FAIL)
    lines = out.splitlines()
    marker_idx = next(i for i, ln in enumerate(lines) if ln.startswith(_marker_prefix()))
    assert lines[marker_idx - 1].isdigit() and lines[marker_idx + 1].isdigit()


def test_small_output_untouched():
    out = m.tool_bash_run("echo hi")
    assert out == "hi\n[exit code: 0]"
    assert _marker_prefix() not in out


def test_full_output_saved_to_log():
    m.tool_bash_run(BIG_FAIL)
    assert os.path.exists(m.BASH_LOG_FILE)
    log = open(m.BASH_LOG_FILE).read()
    assert log.startswith("$ " + BIG_FAIL + "\n")
    assert "\n50000\n" in log                                # the omitted middle is here
    assert "error: undefined reference to foo" in log
    assert log.rstrip().endswith("[exit code: 2]")


def test_log_write_failure_does_not_break_tool(monkeypatch, tmp_path):
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("x")
    monkeypatch.setattr(m, "BASH_LOG_FILE", str(blocker / "last_bash.log"))
    assert m.tool_bash_run("echo ok") == "ok\n[exit code: 0]"


# ---- omitted-region scan: surfaces early errors buried behind a long tail ----

# an early error + warning at lines ~3000, then a huge cascade at the tail
SCAN_CMD = ('seq 1 3000; echo "src/foo.c:88:5: error: bar undeclared"; '
            'echo "src/foo.c:90:1: warning: unused x"; seq 3001 100000; '
            'for i in $(seq 30); do echo "ld: undefined reference to foo"; done; exit 2')


def _scan_header(shown, total):
    return m.get_prompt("pipeline_tool_errors", "bash_omitted_matches_begin",
                        shown=shown, total=total, log_path=m.BASH_LOG_FILE).rstrip()


def _scan_footer():
    return m.get_prompt("pipeline_tool_errors", "bash_omitted_matches_end").rstrip()


@pytest.mark.parametrize("tty", [False, True])
def test_scan_surfaces_early_error_with_log_line_numbers(tty):
    out = m.tool_bash_run(SCAN_CMD, tty=tty)
    assert _scan_header(2, 2) in out
    assert _scan_footer() in out
    assert out.index(_scan_header(2, 2)) < out.index("bar undeclared") < out.index(_scan_footer())
    assert "error: bar undeclared" in out                   # was only in the omitted middle
    assert "warning: unused x" in out
    # every reported L<n> must point at exactly that line in the log file
    log = open(m.BASH_LOG_FILE).read().splitlines()
    hits = re.findall(r"^  L(\d+): (.*)$", out, re.M)
    assert hits
    for n, text in hits:
        assert log[int(n) - 1] == text


def test_scan_never_runs_on_success():
    out = m.tool_bash_run("seq 1 50000; echo error here; seq 1 50000")
    assert "error here" not in out
    assert "BEGIN picked lines" not in out


def test_scan_never_runs_when_nothing_omitted():
    out = m.tool_bash_run("echo error: small; exit 1")
    assert "BEGIN picked lines" not in out
    assert "error: small" in out                            # still there, untrimmed


def test_scan_respects_max_lines(monkeypatch):
    monkeypatch.setitem(m.CONFIG, "bash_omitted_scan",
                        {"patterns": [r"\berror\b"], "max_lines": 3})
    # errors must sit past the head slice so they land in the omitted region
    cmd = 'seq 1 3000; for i in $(seq 10); do echo "error $i"; done; seq 1 100000; exit 1'
    out = m.tool_bash_run(cmd)
    assert _scan_header(3, 10) in out
    assert "  L" in out and out.count("\n  L") == 3


def test_scan_disabled_by_empty_patterns(monkeypatch):
    monkeypatch.setitem(m.CONFIG, "bash_omitted_scan", {"patterns": [], "max_lines": 40})
    out = m.tool_bash_run(SCAN_CMD)
    assert "bar undeclared" not in out
    assert "BEGIN picked lines" not in out


# ---- end-to-end: a long, always-failing build script with the cause buried ----

BUILD = "sh " + os.path.join(m.SCRIPT_DIR, "test", "build_fail_long.sh")


@pytest.mark.parametrize("tty", [False, True])
def test_long_failing_build_surfaces_buried_error(tty):
    out = m.tool_bash_run(BUILD, tty=tty)
    assert "[exit code: 1]" in out
    assert "Error: The solution is simple!" in out          # buried at line ~30000
    assert _scan_header(1, 1) in out                         # ...and surfaced by the scan, not luck
    assert "ld: undefined reference" in out                  # the tail cascade is still there
    assert len(out) <= m.MAX_BASH_OUTPUT + 2000              # scan block + markers only
