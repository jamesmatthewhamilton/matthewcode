#!/usr/bin/env python3
"""MatthewCode - Local LLM CLI coding assistant powered by llm_connections."""

import argparse
import difflib
import json
import os
import subprocess
import sys
import time

# Add common/python to path for llm_connections import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "common", "python"))

from llm_connections import LLMConnection
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme

console = Console(theme=Theme({
    "info": "dim",
    "warning": "yellow",
    "error": "bold red",
    "success": "green",
    "prompt": "green",
}))

# ANSI colors (kept for simple inline use)
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def render_markdown(text: str):
    """Render markdown text (tables, code blocks, bold, etc.) to terminal."""
    if not text.strip():
        return
    try:
        console.print(Markdown(text))
    except Exception:
        # Fallback to plain text if rich fails
        print(text)

HISTORY_DIR = os.path.expanduser("~/.matthewcode")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config", "config.yaml")

def load_config() -> dict:
    """Load app config from config/config.yaml and LLM providers from global config."""
    import yaml

    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f) or {}

    # Load global LLM providers (~/.llm-connections/config.yaml)
    LLMConnection.load()

    # Also load from app config if it has llm-providers (overrides/extends)
    if "llm-providers" in config:
        LLMConnection.load(CONFIG_FILE)

    return config


CONFIG = load_config()
MAX_BASH_OUTPUT = CONFIG.get("max_bash_output", 30_000)

SYSTEM_PROMPT = f"""You are MatthewCode, a coding assistant with access to tools. \
You MUST use your tools to accomplish tasks. NEVER describe what you would do — DO IT.

Environment:
- Home directory: {os.path.expanduser("~")}
- Working directory: {os.getcwd()}

You have these tools: file_read, file_write, file_edit, bash_run, dir_list, file_find, file_grep.
When the user asks you to create, read, edit, or run something, CALL THE TOOL IMMEDIATELY. \
Do not print the tool call as text. Do not ask the user to run commands. Execute them yourself.

Workflow:
1. Think briefly about what to do (1-2 sentences max)
2. Call the appropriate tool(s)
3. If a tool returns an error, try to fix it (max 3 attempts, then explain the issue)
4. Report the result concisely

Rules:
- ALWAYS read a file before editing it
- Use file_edit for small changes, file_write for new files or full rewrites
- For file_edit: old_text must match EXACTLY what is in the file
- If bash_run fails, read the error, fix the issue, and retry
- Never delete files unless explicitly asked
- Do not repeat yourself. If you already tried something and it failed, try a DIFFERENT approach
- Be concise. No unnecessary explanations."""

# --- Tool definitions ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read the contents of a file. Use this to examine code or "
            "understand existing file structure before making changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path to read"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Create a new file or completely overwrite an existing file. "
            "Use file_edit instead for surgical changes to existing files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path to write"},
                    "content": {"type": "string", "description": "The full content to write to the file"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_edit",
            "description": "Edit an existing file by replacing a specific text block with new text. "
            "The old_text must match exactly as it appears in the file. "
            "Always read the file first to get the exact text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path to edit"},
                    "old_text": {"type": "string", "description": "The exact text to find and replace"},
                    "new_text": {"type": "string", "description": "The replacement text"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash_run",
            "description": "Execute a shell command and return its output. "
            "Use for running tests, builds, git commands, or inspecting system state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dir_list",
            "description": "List the contents of a directory. Shows files and subdirectories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list (default: current directory)"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_find",
            "description": "Recursively search for files matching a pattern (like 'main.cpp' or '*.py'). "
            "Use this to locate files in a project before reading or building them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Filename or glob pattern to search for"},
                    "path": {"type": "string", "description": "Directory to search in (default: current directory)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_grep",
            "description": "Search for a text pattern inside files. Returns matching lines with file paths. "
            "Use this to find where functions, classes, or variables are defined.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text or regex pattern to search for"},
                    "path": {"type": "string", "description": "File or directory to search in (default: current directory)"},
                    "glob": {"type": "string", "description": "Only search files matching this glob (e.g. '*.cpp', '*.py')"},
                },
                "required": ["pattern"],
            },
        },
    },
]

SAFE_TOOLS = {"file_read", "dir_list", "file_find", "file_grep"}


# --- Tool implementations ---


def tool_file_read(path):
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return f"Error: File not found: {path}"
    try:
        with open(path, "r") as f:
            content = f.read()
        if len(content) > 100_000:
            return content[:100_000] + f"\n\n[Truncated: file is {len(content)} chars]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def tool_file_write(path, content):
    path = os.path.expanduser(path)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def tool_file_edit(path, old_text, new_text):
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return f"Error: File not found: {path}"
    try:
        with open(path, "r") as f:
            content = f.read()
        if old_text in content:
            count = content.count(old_text)
            if count > 1:
                return f"Error: old_text matches {count} locations. Provide more context."
            new_content = content.replace(old_text, new_text, 1)
            with open(path, "w") as f:
                f.write(new_content)
            return f"Edited {path} (replaced 1 occurrence)"
        lines = content.split("\n")
        old_lines = old_text.split("\n")
        matcher = difflib.SequenceMatcher(None, lines, old_lines)
        best = matcher.find_longest_match(0, len(lines), 0, len(old_lines))
        if best.size > 0 and best.size >= len(old_lines) * 0.6:
            return (f"Error: Exact match not found. Closest match at lines "
                    f"{best.a + 1}-{best.a + best.size}. Read the file and use exact text.")
        return "Error: old_text not found in file. Read the file first to get exact text."
    except Exception as e:
        return f"Error editing file: {e}"


def tool_bash_run(command, timeout=120):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if not output:
            output = "(no output)"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        if len(output) > MAX_BASH_OUTPUT:
            output = output[:MAX_BASH_OUTPUT] + f"\n[truncated at {MAX_BASH_OUTPUT} chars]"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error running command: {e}"


def tool_dir_list(path):
    path = os.path.expanduser(path or ".")
    if not os.path.isdir(path):
        return f"Error: Directory not found: {path}"
    try:
        entries = sorted(os.listdir(path))
        result = []
        for entry in entries:
            full = os.path.join(path, entry)
            if os.path.isdir(full):
                result.append(f"  {entry}/")
            else:
                result.append(f"  {entry} ({os.path.getsize(full)} bytes)")
        return f"{path}/\n" + "\n".join(result) if result else f"{path}/ (empty)"
    except Exception as e:
        return f"Error listing directory: {e}"


def tool_file_find(pattern, path="."):
    import fnmatch
    path = os.path.expanduser(path or ".")
    if not os.path.isdir(path):
        return f"Error: Directory not found: {path}"
    matches = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in
                   ("node_modules", "__pycache__", ".git", "venv", "env")]
        for f in files:
            if fnmatch.fnmatch(f, pattern):
                matches.append(os.path.join(root, f))
                if len(matches) >= 50:
                    return "\n".join(matches) + "\n[truncated at 50 results]"
    return "\n".join(matches) if matches else f"No files matching '{pattern}' found in {path}"


def tool_file_grep(pattern, path=".", file_glob=None):
    import fnmatch, re
    path = os.path.expanduser(path or ".")
    try:
        regex = re.compile(pattern)
    except re.error:
        regex = re.compile(re.escape(pattern))
    matches = []
    files_to_search = [path] if os.path.isfile(path) else []
    if not files_to_search:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in
                       ("node_modules", "__pycache__", ".git", "venv", "env")]
            for f in files:
                if file_glob and not fnmatch.fnmatch(f, file_glob):
                    continue
                files_to_search.append(os.path.join(root, f))
    for fpath in files_to_search[:200]:
        try:
            with open(fpath, "r", errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    if regex.search(line):
                        matches.append(f"{fpath}:{i}: {line.rstrip()}")
                        if len(matches) >= 50:
                            return "\n".join(matches) + "\n[truncated at 50 results]"
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(matches) if matches else f"No matches for '{pattern}' in {path}"


TOOL_DISPATCH = {
    "file_read": lambda a: tool_file_read(a["path"]),
    "file_write": lambda a: tool_file_write(a["path"], a["content"]),
    "file_edit": lambda a: tool_file_edit(a["path"], a["old_text"], a["new_text"]),
    "bash_run": lambda a: tool_bash_run(a["command"], a.get("timeout", 120)),
    "dir_list": lambda a: tool_dir_list(a.get("path", ".")),
    "file_find": lambda a: tool_file_find(a["pattern"], a.get("path", ".")),
    "file_grep": lambda a: tool_file_grep(a["pattern"], a.get("path", "."), a.get("glob")),
}


# --- Confirmation ---


def confirm_tool(name, args):
    if name == "file_write":
        path = args.get("path", "?")
        print(f"\n{YELLOW}Write to {path} ({len(args.get('content', ''))} chars)?{RESET}")
        if os.path.isfile(os.path.expanduser(path)):
            print(f"{DIM}(file exists, will be overwritten){RESET}")
    elif name == "file_edit":
        print(f"\n{YELLOW}Edit {args.get('path', '?')}:{RESET}")
        for line in difflib.unified_diff(
            args.get("old_text", "").splitlines(keepends=True),
            args.get("new_text", "").splitlines(keepends=True),
            fromfile="before", tofile="after",
        ):
            if line.startswith("+") and not line.startswith("+++"):
                print(f"  {GREEN}{line.rstrip()}{RESET}")
            elif line.startswith("-") and not line.startswith("---"):
                print(f"  {RED}{line.rstrip()}{RESET}")
            else:
                print(f"  {line.rstrip()}")
    elif name == "bash_run":
        print(f"\n{YELLOW}Run: {args.get('command', '?')}{RESET}")
    try:
        return input(f"{BOLD}Allow? [Y/n] {RESET}").strip().lower() in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


# --- Fallback tool call parser ---


def parse_tool_calls_from_text(text):
    import re
    calls = []
    clean = re.sub(r'```(?:json)?\s*', '', text)
    i = 0
    while i < len(clean):
        if clean[i] == '{' and '"name"' in clean[i:i+200]:
            depth = 0
            start = i
            for j in range(i, len(clean)):
                if clean[j] == '{':
                    depth += 1
                elif clean[j] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(clean[start:j+1])
                            name = obj.get("name")
                            tc_args = obj.get("arguments") or obj.get("parameters") or {}
                            if name and name in TOOL_DISPATCH:
                                calls.append((name, tc_args))
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1
    return calls


# --- Conversation ---


def save_history(messages, session_file):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    with open(session_file, "w") as f:
        json.dump(messages, f, indent=2, default=str)


def load_history(session_file):
    if os.path.isfile(session_file):
        with open(session_file, "r") as f:
            return json.load(f)
    return []


def _tool_summary(name, args):
    if name == "file_read": return args.get("path", "?")
    if name == "file_write": return f"{args.get('path', '?')} ({len(args.get('content', ''))} chars)"
    if name == "file_edit": return args.get("path", "?")
    if name == "bash_run":
        cmd = args.get("command", "?")
        return cmd if len(cmd) < 60 else cmd[:57] + "..."
    if name == "dir_list": return args.get("path", ".")
    if name == "file_find": return args.get("pattern", "?")
    if name == "file_grep": return args.get("pattern", "?")
    return str(args)


# --- Main ---


def main():
    parser = argparse.ArgumentParser(description="MatthewCode - Local LLM coding assistant")
    parser.add_argument("--provider", default="default", help="LLM provider name from config")
    parser.add_argument("--yes", "-y", action="store_true",
                        default=CONFIG.get("auto_approve", False), help="Auto-approve all tool calls")
    parser.add_argument("--continue", "-c", dest="resume", action="store_true",
                        help="Resume last conversation")
    parser.add_argument("--resume", dest="resume_name", default=None,
                        help="Resume a named session (e.g. --resume myproject)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        default=CONFIG.get("verbose", False), help="Show tool call details")
    args = parser.parse_args()

    # Get LLM connection from registry
    try:
        client = LLMConnection.get(args.provider)
    except KeyError as e:
        print(f"{RED}{e}{RESET}")
        sys.exit(1)

    session_name = None
    if args.resume_name:
        session_name = args.resume_name
        session_file = os.path.join(HISTORY_DIR, f"{session_name}.json")
        if not os.path.isfile(session_file):
            print(f"{RED}Session '{session_name}' not found.{RESET}")
            # List available sessions
            sessions = [f[:-5] for f in os.listdir(HISTORY_DIR) if f.endswith(".json")]
            if sessions:
                print(f"{DIM}Available sessions: {', '.join(sessions)}{RESET}")
            sys.exit(1)
    else:
        session_file = os.path.join(HISTORY_DIR, "last_session.json")

    messages = []
    if args.resume or args.resume_name:
        messages = load_history(session_file)
        if messages:
            label = session_name or "last session"
            print(f"{DIM}Resumed '{label}' ({len(messages)} messages){RESET}")

    if not messages:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"{BOLD}{CYAN}MatthewCode{RESET} {DIM}({client}){RESET}")
    print(f"{DIM}Type /help for commands{RESET}")
    print()

    max_iters = CONFIG.get("max_iterations", 10)
    ctx_tokens = 0  # tracks prompt tokens from last LLM call

    while True:
        try:
            if ctx_tokens > 0:
                # Show token count in prompt
                num_ctx = getattr(client, '_provider', None) and client._provider.config.get("num_ctx", 0) or 0
                if num_ctx:
                    pct = int(ctx_tokens / num_ctx * 100)
                    ctx_color = RED if pct >= 90 else YELLOW if pct >= 70 else DIM
                    ctx_info = f"{ctx_color}[{ctx_tokens}/{num_ctx} tokens]{RESET} "
                else:
                    ctx_info = f"{DIM}[{ctx_tokens} tokens]{RESET} "
            else:
                ctx_info = ""
            user_input = input(f"{ctx_info}◗ ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Bye!{RESET}")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input == "/help":
            print(f"{DIM}Commands:{RESET}")
            print(f"  /help                 Show this help")
            print(f"  /exit, /quit          Exit MatthewCode")
            print(f"  /clear                Reset conversation")
            print(f"  /name <name>          Save/name this session")
            print(f"  /sessions             List saved sessions")
            print(f"  /provider             List available LLM providers")
            print(f"  /provider <name>      Switch to a different provider")
            print(f"  /verbose              Toggle verbose mode")
            continue
        elif user_input == "/verbose":
            args.verbose = not args.verbose
            print(f"{DIM}Verbose: {'on' if args.verbose else 'off'}{RESET}")
            continue
        elif user_input in ("/exit", "/quit"):
            print(f"{DIM}Bye!{RESET}")
            break
        elif user_input == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            session_name = None
            session_file = os.path.join(HISTORY_DIR, "last_session.json")
            print(f"{DIM}Conversation cleared.{RESET}")
            continue
        elif user_input.startswith("/name ") or user_input.startswith("/rename "):
            new_name = user_input.split(" ", 1)[1].strip()
            if not new_name:
                print(f"{RED}Usage: /name <session_name>{RESET}")
                continue
            new_file = os.path.join(HISTORY_DIR, f"{new_name}.json")
            if os.path.isfile(new_file) and new_name != session_name:
                try:
                    answer = input(f"{YELLOW}Session '{new_name}' exists. Overwrite? [y/N] {RESET}").strip().lower()
                    if answer not in ("y", "yes"):
                        print(f"{DIM}Cancelled.{RESET}")
                        continue
                except (EOFError, KeyboardInterrupt):
                    print()
                    continue
            session_name = new_name
            session_file = new_file
            save_history(messages, session_file)
            print(f"{DIM}Session saved as '{session_name}'{RESET}")
            continue
        elif user_input in ("/name", "/rename"):
            if session_name:
                print(f"{DIM}Current session: '{session_name}'{RESET}")
            else:
                print(f"{DIM}Session unnamed. Usage: /name <session_name>{RESET}")
            continue
        elif user_input == "/sessions":
            os.makedirs(HISTORY_DIR, exist_ok=True)
            sessions = sorted(f[:-5] for f in os.listdir(HISTORY_DIR) if f.endswith(".json"))
            if sessions:
                print(f"{DIM}Saved sessions:{RESET}")
                for s in sessions:
                    marker = f" {CYAN}<- active{RESET}" if s == session_name else ""
                    fpath = os.path.join(HISTORY_DIR, f"{s}.json")
                    msgs = load_history(fpath)
                    print(f"  {s} ({len(msgs)} messages){marker}")
            else:
                print(f"{DIM}No saved sessions.{RESET}")
            continue
        elif user_input == "/provider":
            print(f"{DIM}Available providers:{RESET}")
            for name in LLMConnection.list_providers():
                p = LLMConnection.get(name)
                marker = f" {CYAN}<- active{RESET}" if name == args.provider else ""
                print(f"  {name} ({p.model}){marker}")
            continue
        elif user_input.startswith("/provider "):
            name = user_input.split(" ", 1)[1].strip()
            try:
                client = LLMConnection.get(name)
                args.provider = name
                print(f"{DIM}Switched to {client}{RESET}")
            except KeyError as e:
                print(f"{RED}{e}{RESET}")
            continue

        messages.append({"role": "user", "content": user_input})

        # Agent loop
        for _iter in range(max_iters):
            try:
                # Use llm_connections for the chat call
                response = client.chat(messages, tools=TOOLS, stream=True)

                full_content = ""
                tool_calls = []
                llama_frames = ["🦙      ", "  🦙    ", "    🦙  "]
                frame_idx = 0
                chunk_count = 0
                for chunk in response:
                    if chunk.text:
                        full_content += chunk.text
                        chunk_count += 1
                        if chunk_count % 3 == 0:  # animate every 3 chunks
                            frame = llama_frames[frame_idx % len(llama_frames)]
                            sys.stdout.write(f"\r{frame}")
                            sys.stdout.flush()
                            frame_idx += 1
                    if chunk.tool_calls:
                        tool_calls.extend(chunk.tool_calls)

                # After stream: grab accumulated values
                if not full_content:
                    full_content = response.text
                if not tool_calls:
                    tool_calls = response.tool_calls

                # Track context window usage (prompt_tokens = full input context)
                if response.prompt_tokens:
                    ctx_tokens = response.prompt_tokens

                # Clear the llama animation and render the response
                if full_content:
                    sys.stdout.write("\r\033[K")  # clear the animation line
                    render_markdown(full_content)

                # No native tool calls — try fallback text parsing
                if not tool_calls and full_content:
                    fallback = parse_tool_calls_from_text(full_content)
                    if fallback:
                        print(f"{DIM}(parsed tool call from text){RESET}")
                        messages.append({"role": "assistant", "content": full_content})
                        for name, tc_args in fallback:
                            print(f"{DIM}[{name}: {_tool_summary(name, tc_args)}]{RESET}")
                            if name not in SAFE_TOOLS and not args.yes:
                                if not confirm_tool(name, tc_args):
                                    messages.append({"role": "tool", "content": "User rejected this action. Do NOT retry the same action. Ask the user what they want or try a different approach."})
                                    continue
                            result = TOOL_DISPATCH[name](tc_args)
                            messages.append({"role": "tool", "content": result})
                        continue

                # No tool calls at all — done
                if not tool_calls:
                    if full_content:
                        messages.append({"role": "assistant", "content": full_content})
                    save_history(messages, session_file)
                    break

                # Add assistant message with tool calls
                assistant_msg = {"role": "assistant", "content": full_content or ""}
                assistant_msg["tool_calls"] = [
                    {"id": f"call_{i}", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for i, tc in enumerate(tool_calls)
                ]
                messages.append(assistant_msg)

                # Execute tool calls
                for tc in tool_calls:
                    name = tc["name"]
                    tc_args = tc["arguments"]
                    if isinstance(tc_args, str):
                        tc_args = json.loads(tc_args)

                    if args.verbose:
                        print(f"{DIM}[tool: {name}({json.dumps(tc_args, indent=2)})]{RESET}")
                    else:
                        print(f"{DIM}[{name}: {_tool_summary(name, tc_args)}]{RESET}")

                    if name not in SAFE_TOOLS and not args.yes:
                        if not confirm_tool(name, tc_args):
                            messages.append({"role": "tool", "content": "User denied this action."})
                            continue

                    if name in TOOL_DISPATCH:
                        result = TOOL_DISPATCH[name](tc_args)
                    else:
                        result = f"Error: Unknown tool '{name}'"
                    messages.append({"role": "tool", "content": result})

                    if args.verbose:
                        preview = result[:500] + ("..." if len(result) > 500 else "")
                        print(f"{DIM}{preview}{RESET}")

            except KeyboardInterrupt:
                print(f"\n{DIM}(interrupted){RESET}")
                break
            except Exception as e:
                print(f"\n{RED}Error: {e}{RESET}")
                break
        else:
            print(f"\n{YELLOW}(stopped after {max_iters} iterations){RESET}")

    save_history(messages, session_file)


if __name__ == "__main__":
    main()
