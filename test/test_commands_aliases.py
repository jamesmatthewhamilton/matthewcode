"""Alias commands (config.yaml `aliases`).

Registration is checked for whatever aliases the live config defines — no
alias is named. The _alias_spec value forms are exercised with synthetic
config entries.
"""

import matthewcode as m


def test_every_configured_alias_is_registered():
    aliases = m.CONFIG.get("aliases") or {}
    assert aliases                                  # config defines at least one
    canned = "_prompt_command."
    builtin = {t for c in m.COMMANDS for t in c.flag_command
               if not getattr(c.run, "__qualname__", "").startswith(canned)}
    for tok in aliases:
        if tok in builtin:                          # collision: built-in wins, skipped
            continue
        c = m.COMMAND_BY_TOKEN[tok]
        assert c.is_command and c.is_flag and c.one_shot and c.needs_client
        assert getattr(c.run, "__qualname__", "").startswith(canned)


def test_alias_spec_value_forms(monkeypatch):
    monkeypatch.setitem(m.CONFIG, "aliases", {
        "zz-str": "just a prompt",
        "zz-map": {"help": "a help line", "prompt": "mapped prompt"},
    })
    assert m._alias_spec("zz-str") == ("", "just a prompt")     # string form
    assert m._alias_spec("zz-map") == ("a help line", "mapped prompt")   # mapping form
    assert m._alias_spec("zz-absent") == ("", "")               # unknown token
