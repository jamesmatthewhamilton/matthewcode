"""TAB completion for the matthewcode REPL.

Built on prompt_toolkit's `Completer`/`Completion` API. We deliberately do NOT use
the higher-level `NestedCompleter`: it builds an internal `WordCompleter` that uses
prompt_toolkit's default word-boundary regex, where `/` is a non-word character —
so it fails to complete `/`-prefixed command words (e.g. `/he` → nothing). This
small subclass handles the leading slash correctly and resolves argument values
lazily so live state (providers/sessions) is always current.

Hardcodes **no** command names: the caller injects the command list and per-command
argument resolvers, so the single source of truth for "what commands exist" lives
in one place (matthewcode.py) and is never mirrored here.
"""

from prompt_toolkit.completion import Completer, Completion


class SlashCommandCompleter(Completer):
    """Completes ``/commands`` and, where provided, their first argument.

    Completion fires only for input starting with ``/`` — normal prose returns
    nothing, so typing a message never pops a completion menu.
    """

    def __init__(self, commands, arg_completers=None):
        self.commands = sorted(commands)
        self.arg_completers = dict(arg_completers or {})

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        if " " not in text:
            # Still typing the command word — complete it.
            for cmd in self.commands:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))
            return

        # Past the first space — complete the command's argument, if it has one.
        cmd, _, arg = text.partition(" ")
        resolver = self.arg_completers.get(cmd)
        if resolver is None:
            return
        try:
            candidates = resolver()                 # lazy: reflects live state
        except Exception:
            return                                  # never crash the prompt
        for cand in candidates:
            if cand.startswith(arg):
                yield Completion(cand, start_position=-len(arg))


def build_slash_completer(commands, arg_value_funcs=None):
    """Construct the REPL completer.

    Args:
        commands: iterable of command tokens, e.g. ``["/help", "/provider", …]``.
        arg_value_funcs: maps a command token to a zero-arg callable returning the
            candidate argument values (resolved lazily on each TAB press).
    """
    return SlashCommandCompleter(commands, arg_completers=arg_value_funcs)
