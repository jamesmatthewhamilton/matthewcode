"""Shared prompt snippets: {{name}} placeholders resolved from config.yaml.

The mechanism is tested with synthetic snippets; the single-source invariants
are checked over whatever snippets/aliases the live config defines — no
snippet or alias is named.
"""

import matthewcode as m


def test_snippet_mechanism(monkeypatch):
    monkeypatch.setitem(m.CONFIG, "snippets", {"zz_rules": "- rule one\n- rule two\n"})
    out = m.expand_snippets("head:\n{{zz_rules}}\n- tail")
    assert out == "head:\n- rule one\n- rule two\n- tail"   # no stray blank line
    assert m.expand_snippets("keep {{zz_unknown}} as-is") == "keep {{zz_unknown}} as-is"


def test_snippets_expand_in_every_alias_and_live_once_in_config():
    snippets = m.CONFIG.get("snippets") or {}
    aliases = m.CONFIG.get("aliases") or {}
    assert snippets and aliases
    with open(m.CONFIG_FILE) as f:
        source = f.read()
    referenced = set()
    for tok in aliases:
        prompt = m._alias_spec(tok)[1]
        expanded = m.expand_snippets(prompt)
        for name in m.SNIPPET_RE.findall(prompt):
            if name not in snippets:
                continue
            referenced.add(name)
            assert "{{" + name + "}}" not in expanded, (tok, name)   # fully expanded
            first_line = str(snippets[name]).strip().splitlines()[0]
            assert first_line in expanded, (tok, name)               # content arrived
    assert referenced                               # at least one alias uses a snippet
    for name in referenced:
        first_line = str(snippets[name]).strip().splitlines()[0]
        assert source.count(first_line) == 1        # written once, referenced many
