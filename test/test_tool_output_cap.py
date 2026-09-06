"""Tests for the central tool-output backstop cap (_cap_tool_output / _safe).

Every dispatched tool result passes through _cap_tool_output, so an unbounded
search cannot overflow the context. Expected hint text comes from m.get_prompt,
not literals, per the conftest ground rule.
"""

import matthewcode as m


def _hint(limit):
    return m.get_prompt("pipeline_tool_errors", "output_truncated", max_limit=limit).rstrip()


def test_small_output_unchanged():
    assert m._cap_tool_output("hello") == "hello"


def test_non_string_passthrough():
    # defensive: a tool that returned a non-str must not blow up the cap
    assert m._cap_tool_output(1234) == 1234


def test_large_output_capped_with_hint(monkeypatch):
    monkeypatch.setattr(m, "MAX_TOOL_OUTPUT", 100)
    out = m._cap_tool_output("x" * 5000)
    assert out.startswith("x" * 100)
    assert "x" * 101 not in out            # truncated at the cap
    assert _hint(100) in out               # hint appended


def test_cap_applies_through_dispatch(monkeypatch):
    # the cap lives in _safe, so ANY tool is covered — prove it with a fake one
    monkeypatch.setattr(m, "MAX_TOOL_OUTPUT", 50)
    wrapped = m._safe("faux", lambda a: "y" * 9000)
    out = wrapped({})
    assert out.startswith("y" * 50)
    assert _hint(50) in out


def test_safe_still_reports_missing_arg():
    # regression: wrapping the result must not swallow the KeyError guidance
    wrapped = m._safe("faux", lambda a: a["required"])
    assert "missing required argument" in wrapped({})


def test_backstop_not_below_bash_cap():
    # invariant: the generic backstop must sit at or above bash_run's own limit,
    # else bash output would get the generic hint instead of its own marker
    assert m.MAX_TOOL_OUTPUT >= m.MAX_BASH_OUTPUT
