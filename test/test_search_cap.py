"""file_find / file_grep stop at max_search_results and tell the model to
narrow the search (text from config, per the conftest rule)."""

import matthewcode as m


def _hint():
    return m.get_prompt("pipeline_tool_errors", "search_truncated",
                        max_results=m.MAX_SEARCH_RESULTS).rstrip()


def _populate(tmp_path, n):
    for i in range(n):
        (tmp_path / f"f{i:03}.txt").write_text("needle\n")


def test_file_find_caps_and_hints(tmp_path):
    _populate(tmp_path, m.MAX_SEARCH_RESULTS + 10)
    out = m.tool_file_find("*.txt", str(tmp_path))
    assert out.endswith(_hint())
    assert len(out.splitlines()) == m.MAX_SEARCH_RESULTS + 1   # matches + hint line


def test_file_grep_caps_and_hints(tmp_path):
    _populate(tmp_path, m.MAX_SEARCH_RESULTS + 10)
    out = m.tool_file_grep("needle", str(tmp_path))
    assert out.endswith(_hint())
    assert out.count("needle") == m.MAX_SEARCH_RESULTS


def test_under_cap_has_no_hint(tmp_path):
    _populate(tmp_path, 3)
    assert _hint() not in m.tool_file_find("*.txt", str(tmp_path))
    assert _hint() not in m.tool_file_grep("needle", str(tmp_path))
