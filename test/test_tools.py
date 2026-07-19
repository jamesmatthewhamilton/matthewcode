"""Editor-on-save normalization applied by the file tools (file_write, file_edit)."""

import matthewcode as m


def test_normalize_trailing_whitespace():
    # per-line trailing ws stripped, indent and interior blanks kept, trailing
    # blank lines collapsed, final newline ensured, empty stays empty
    for raw, want in [
        ("a   \n\tb\t\n\nc\n", "a\n\tb\n\nc\n"),
        ("a\nb\n\n\n\n", "a\nb\n"),
        ("a\nb", "a\nb\n"),
        ("", ""),
        ("   \n\n", ""),
    ]:
        assert m._normalize_trailing_whitespace(raw) == want


def test_file_write_and_edit_apply_normalization(tmp_path):
    p = str(tmp_path / "f.txt")
    m.tool_file_write(p, "a   \n\tb\t\n\n\n")
    with open(p) as f:
        assert f.read() == "a\n\tb\n"             # write path normalizes
    m.tool_file_edit(p, "a", "z   ")
    with open(p) as f:
        assert f.read() == "z\n\tb\n"             # edit path normalizes too


def test_trim_respects_config_flag(monkeypatch):
    # when disabled, content is returned verbatim
    monkeypatch.setattr(m, "TRIM_TRAILING_WHITESPACE", False)
    assert m._normalize_trailing_whitespace("a   \n\n\n") == "a   \n\n\n"
