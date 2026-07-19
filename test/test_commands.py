"""Generic invariants of the command registry (COMMANDS).

Everything here is derived from the registry itself — no command token appears
as a string literal, so renaming or adding commands never touches this file.
Design intent that cannot be derived (e.g. exactly which built-in handlers
need a live client) is deliberately not pinned here.
"""

import matthewcode as m


def test_registry_invariants():
    # One pass over COMMANDS: every entry is well-formed and both lookup
    # surfaces (slash dispatch, argparse flags) can be built from it.
    for c in m.COMMANDS:
        assert c.flag_command                               # at least one token
        assert c.is_command or c.is_flag                    # on at least one surface
        assert all(not t.startswith(("/", "-")) for t in c.flag_command)
        if c.one_shot:
            assert c.is_flag and c.run is not None          # a one-shot IS a flag
        if c.is_flag:
            assert m._flag_dest(c).isidentifier()           # valid argparse dest
        for tok in c.flag_command:
            if c.is_command:
                assert m.COMMAND_BY_TOKEN[tok] is c         # dispatchable
            else:
                assert tok not in m.COMMAND_BY_TOKEN        # flag-only: not dispatchable


def test_multi_token_commands_share_one_entry():
    # multi-name commands (e.g. exit-style synonyms) resolve to a single entry
    multi = [c for c in m.COMMANDS if c.is_command and len(c.flag_command) > 1]
    assert multi                                    # property stays exercised
    for c in multi:
        assert all(m.COMMAND_BY_TOKEN[t] is c for t in c.flag_command)


def test_canned_prompt_commands_need_client():
    # every alias built by _prompt_command chats with the model: one-shot + client
    canned = [c for c in m.COMMANDS
              if getattr(c.run, "__qualname__", "").startswith("_prompt_command.")]
    assert canned                                   # config defines at least one alias
    for c in canned:
        assert c.one_shot and c.needs_client, c.flag_command[0]
