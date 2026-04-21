#!/usr/bin/env python3
"""MatthewCode - Local LLM CLI coding assistant powered by llm_connections."""

import argparse
import difflib
import json
import os
import subprocess
import sys
import random
import time

# Add common/python to path for llm_connections import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "common", "python"))

from llm_connections import LLMConnection
from res.thinking import WORDS as THINKING_WORDS
from res.loop_detection import LoopDetector, LOOP_WARNING
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import ANSI

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
    """Load app config from config.yaml and LLM providers from global config."""
    import yaml

    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f) or {}

    return config


CONFIG = load_config()


def load_providers():
    """Load LLM providers. Called after argparse so --help doesn't wait."""
    LLMConnection.load()
    if "llm-providers" in CONFIG:
        LLMConnection.load(CONFIG_FILE)
MAX_BASH_OUTPUT = CONFIG.get("max_bash_output", 30_000)
MAX_FILE_READ = CONFIG.get("max_file_read", 50_000)


def make_loop_detector() -> LoopDetector:
    cfg = CONFIG.get("loop_detection", {})
    return LoopDetector(
        threshold=cfg.get("consecutive_threshold", 3),
        enabled=cfg.get("enabled", True),
    )

PROMPT_VARS = {
    "home_dir": os.path.expanduser("~"),
    "working_dir": os.getcwd(),
}


def get_prompt(pipeline: str, prompt_type: str, **extra_vars) -> str:
    """Get a prompt from config. pipeline='pipeline_main', prompt_type='system_prompt'."""
    template = CONFIG.get(pipeline, {}).get(prompt_type, "")
    if not template:
        return ""
    all_vars = {**PROMPT_VARS, **extra_vars}
    return template.format(**all_vars)


SYSTEM_PROMPT = get_prompt("pipeline_main", "system_prompt")

# --- Tool definitions ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read the contents of a file. Returns lines with line numbers. "
            "Use offset and limit to read specific sections of large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path to read"},
                    "offset": {"type": "integer", "description": "Starting line number (1-based, default: 1)"},
                    "limit": {"type": "integer", "description": "Number of lines to read (default: all)"},
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
            "description": "Execute a non-interactive shell command and return its stdout and stderr. "
            "stdin is closed. Commands requiring user input will receive EOF and fail.",
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
    {
        "type": "function",
        "function": {
            "name": "find_build_env",
            "description": "Search a project for build environments: conda environment files, "
            "Dockerfiles, docker-compose, and CI/CD configs (GitLab, GitHub Actions, Jenkins, Travis, CircleCI). "
            "Use this on build errors before attempting manual fixes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Project root to search (default: current directory)"},
                },
                "required": [],
            },
        },
    },
]

SAFE_TOOLS = {"file_read", "dir_list", "file_find", "file_grep", "find_build_env"}


def is_protected_path(path: str, protected: list) -> bool:
    """Check if a path matches any protected path pattern."""
    import fnmatch
    path = os.path.expanduser(path)
    for pattern in protected:
        pattern = os.path.expanduser(pattern)
        if fnmatch.fnmatch(path, pattern) or path.startswith(pattern.rstrip("*")):
            return True
    return False


# --- Tool implementations ---


def tool_file_read(path):
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return f"Error: File not found: {path}"
    try:
        with open(path, "r") as f:
            content = f.read()
        if len(content) > MAX_FILE_READ:
            return (
                f"Error: file_read was NOT run. {path} is {len(content)} chars, "
                f"which exceeds the limit of {MAX_FILE_READ}. Retry using bash_run "
                f"with 'sed -n START,ENDp {path}' to read a specific line range, "
                f"or 'grep PATTERN {path}' to search for text."
            )
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
        proc = subprocess.Popen(
            command, shell=True,
            stdin=subprocess.DEVNULL,  # no interactive input — EOF immediately
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return (
                "Error: Command timed out. This may be because it opened an "
                "interactive session (e.g., python, node, gdb, ssh). "
                "All commands must run non-interactively and exit on their own. "
                "Use flags like -c, --batch, or -e to run commands inline."
            )
        output = ""
        if stdout:
            output += stdout
        if stderr:
            output += ("\n" if output else "") + stderr
        if not output:
            output = "(no output)"
        if proc.returncode != 0:
            output += f"\n[exit code: {proc.returncode}]"
        if len(output) > MAX_BASH_OUTPUT:
            output = output[:MAX_BASH_OUTPUT] + f"\n[truncated at {MAX_BASH_OUTPUT} chars]"
        return output
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


def tool_find_build_env(path="."):
    """Find build environments: conda, Docker, build systems, CI/CD configs."""
    import fnmatch
    path = os.path.expanduser(path or ".")

    # Priority 1: Build environments (what you should use to build)
    env_patterns = [
        "environment.yml", "environment.yaml", "env.yml", "env.yaml",
        "conda*.yml", "conda*.yaml",
        "requirements.txt", "requirements*.txt", "pyproject.toml", "setup.py",
        "Dockerfile", "Dockerfile.*",
        "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml",
    ]
    # Priority 2: Build systems
    build_patterns = [
        "CMakeLists.txt", "Makefile", "GNUmakefile", "makefile",
        "meson.build", "BUILD", "BUILD.bazel", "WORKSPACE",
    ]
    # Priority 3: CI/CD
    ci_patterns = [
        ".gitlab-ci.yml", "Jenkinsfile", ".travis.yml",
    ]

    envs = []
    builds_top = []   # depth 0-1
    builds_deep = 0   # count of depth 2+
    ci = []

    for root, dirs, files in os.walk(path):
        depth = root.replace(path, "").count(os.sep)
        if depth >= 3:
            dirs.clear()
            continue
        dirs[:] = [d for d in dirs if d not in
                   ("node_modules", "__pycache__", "venv", "env", ".git")]
        for f in files:
            fpath = os.path.join(root, f)
            for pattern in env_patterns:
                if fnmatch.fnmatch(f, pattern):
                    envs.append(fpath)
                    break
            else:
                for pattern in build_patterns:
                    if fnmatch.fnmatch(f, pattern):
                        if depth <= 1:
                            builds_top.append(fpath)
                        else:
                            builds_deep += 1
                        break
                else:
                    for pattern in ci_patterns:
                        if fnmatch.fnmatch(f, pattern):
                            ci.append(fpath)
                            break

    # Check .github/workflows and .circleci
    gh_dir = os.path.join(path, ".github", "workflows")
    if os.path.isdir(gh_dir):
        count = len([f for f in os.listdir(gh_dir) if f.endswith((".yml", ".yaml"))])
        if count:
            ci.append(f"{gh_dir}/ ({count} workflow files)")
    ci_file = os.path.join(path, ".circleci", "config.yml")
    if os.path.isfile(ci_file):
        ci.append(ci_file)

    # Check for conda references in top-level yml files (skip .github/)
    seen = set(envs)
    for f in os.listdir(path):
        if f.endswith((".yml", ".yaml")):
            fpath = os.path.join(path, f)
            if fpath not in seen:
                try:
                    with open(fpath, "r", errors="ignore") as fh:
                        if "conda" in fh.read(500).lower():
                            envs.append(fpath + " (conda reference)")
                except OSError:
                    pass

    # Format output by priority
    output = []
    if envs:
        output.append("Build environments:")
        output.extend(f"  {e}" for e in envs)
    if builds_top or builds_deep:
        output.append("Build systems:")
        output.extend(f"  {b}" for b in builds_top)
        if builds_deep:
            output.append(f"  ... and {builds_deep} more in subdirectories")
    if ci:
        output.append("CI/CD:")
        output.extend(f"  {c}" for c in ci)
    if not output:
        return f"No build environments found in {path}"
    # Add conda hint if any conda env files were found
    has_conda = any("environment" in e.lower() or "conda" in e.lower() for e in envs)
    if has_conda:
        output.append("")
        output.append("Hint: Use 'conda run -n envname <command>' to run commands in a conda env.")
    return "\n".join(output)


TOOL_DISPATCH = {
    "file_read": lambda a: tool_file_read(a["path"]),
    "file_write": lambda a: tool_file_write(a["path"], a["content"]),
    "file_edit": lambda a: tool_file_edit(a["path"], a["old_text"], a["new_text"]),
    "bash_run": lambda a: tool_bash_run(a["command"], a.get("timeout", 120)),
    "dir_list": lambda a: tool_dir_list(a.get("path", ".")),
    "file_find": lambda a: tool_file_find(a["pattern"], a.get("path", ".")),
    "file_grep": lambda a: tool_file_grep(a["pattern"], a.get("path", "."), a.get("glob")),
    "find_build_env": lambda a: tool_find_build_env(a.get("path", ".")),
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


def sanitize_messages(messages: list) -> list:
    """Fix corrupted message history from interrupted sessions.

    Rules:
    - Every tool_calls in assistant must have matching tool results after it
    - Arguments in tool_calls must be dicts, not strings
    - No trailing assistant messages with empty content and no tool_calls
    - Messages must have 'role' and 'content'
    """
    if not messages:
        return messages

    cleaned = []
    removed = 0
    for msg in messages:
        # Skip messages missing required fields
        if "role" not in msg:
            removed += 1
            continue

        m = dict(msg)

        # Fix string arguments in tool_calls
        if "tool_calls" in m:
            fixed_calls = []
            for tc in m["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        func["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        func["arguments"] = {}
                fixed_calls.append(tc)
            m["tool_calls"] = fixed_calls

        cleaned.append(m)

    # Remove trailing incomplete messages (assistant/tool without a following user)
    while len(cleaned) > 1 and cleaned[-1]["role"] in ("assistant", "tool"):
        # Check if assistant has empty content and no tool_calls
        last = cleaned[-1]
        if last["role"] == "assistant" and not last.get("content") and not last.get("tool_calls"):
            cleaned.pop()
            removed += 1
        # Check for orphaned tool results (tool without preceding assistant tool_calls)
        elif last["role"] == "tool":
            # Look back for matching assistant with tool_calls
            has_match = False
            for prev in reversed(cleaned[:-1]):
                if prev["role"] == "assistant" and prev.get("tool_calls"):
                    has_match = True
                    break
                if prev["role"] == "user":
                    break
            if not has_match:
                cleaned.pop()
                removed += 1
            else:
                break
        else:
            break

    if removed:
        print(f"{DIM}(cleaned {removed} corrupted messages from history){RESET}")
    else:
        print(f"{DIM}(history inspected — no corruption found){RESET}")

    return cleaned


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
    parser.add_argument("--session", "-s", dest="resume_name", default=None,
                        help="Open or create a named session (e.g. --session myproject)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        default=CONFIG.get("verbose", False), help="Show tool call details")
    parser.add_argument("--pipe", action="store_true",
                        help="Pipe mode: read one prompt from stdin, run, print output, exit")
    args = parser.parse_args()

    # Load providers after argparse so --help is instant
    load_providers()

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
            print(f"{DIM}Created new session '{session_name}'{RESET}")
    else:
        session_file = os.path.join(HISTORY_DIR, "last_session.json")

    messages = []
    if args.resume or args.resume_name:
        messages = load_history(session_file)
        if messages:
            messages = sanitize_messages(messages)
            if session_name:
                print(f"{DIM}Resumed session '{session_name}' ({len(messages)} messages){RESET}")
            else:
                print(f"{DIM}Resumed last unnamed session ({len(messages)} messages){RESET}")
            # Add a separator so the model treats the next input as a fresh request
            if messages[-1]["role"] != "user":
                messages.append({
                    "role": "system",
                    "content": get_prompt("pipeline_session_resume", "system_prompt"),
                })

    if not messages:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    num_ctx = getattr(client, '_provider', None) and client._provider.config.get("num_ctx", 0) or 0
    # Estimate tokens from loaded messages (~4 chars per token is a rough average)
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    ctx_tokens = total_chars // 4
    if num_ctx:
        ctx_display = f"{ctx_tokens}/{num_ctx} tokens (estimated)"
    else:
        ctx_display = f"{ctx_tokens} tokens (estimated)"
    # Splash screen — spitting llama animation
    LLAMA_FRAMES = [
        # Frame 1: mouth open
        [
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣫⡏⡆⠀",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠊⠣⢉⠙⠤⡀",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠁⠀⠀⠈⠀⠀⣺",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠁⢲⠉⠁",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡆⠀⠀⠀⠈⡇⠀",
            "⠀⠀⠀⣠⠤⠤⠤⠄⠒⠒⠒⠒⠒⠦⠤⠤⠔⠒⠒⠇⠀⠀⠀⠀⢸⠀",
            "⢀⣠⠞⠀⢀⢀⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⢸⠀",
            "⢲⡿⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢢⠀⡞⠀",
            "⠙⠒⣇⡄⢀⡇⠀⠀⠀⠀⠀⠈⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡹⠀⠀",
            "⠀⠀⠀⠉⠉⡷⡀⠀⠀⠀⠀⢠⠇⠀⠀⠀⠀⠀⠀⢀⠀⠀⡰⠃⠀⠀",
            "⠀⠀⠀⠀⠀⢸⠘⡄⠀⠀⢠⠿⢤⠀⠀⠀⠀⢀⡠⠃⣀⡴⠋⠀⠀⠀",
            "⠀⠀⠀⠀⠀⣎⡀⢹⠢⣀⡌⠀⠈⠙⠒⠲⠚⢹⠀⣠⠏⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⡇⠰⠏⠀⡘⠀⠀⠀⠀⠀⠀⠀⢸⢦⠃⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⢱⠀⣧⠀⢇⠀⠀⠀⠀⠀⠀⠀⡜⠹⠀⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⠈⢆⡹⣇⠘⡄⠀⠀⠀⠀⠀⢀⡇⡜⢦⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⠀⠀⠙⠛⠲⢎⣢⠀⠀⠀⠀⠳⢎⣫⠁⠀⠀⠀⠀⠀⠀",
        ],
        # Frame 2: spit in flight
        [
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣫⡏⡆⠀",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠊⠣⢉⠙⠤⡀",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠁⠀⠀⠈⠀⠀⣺",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠁⢲⠉⠁⠀⠀•",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡆⠀⠀⠀⠈⡇⠀",
            "⠀⠀⠀⣠⠤⠤⠤⠄⠒⠒⠒⠒⠒⠦⠤⠤⠔⠒⠒⠇⠀⠀⠀⠀⢸⠀",
            "⢀⣠⠞⠀⢀⢀⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⢸⠀",
            "⢲⡿⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢢⠀⡞⠀",
            "⠙⠒⣇⡄⢀⡇⠀⠀⠀⠀⠀⠈⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡹⠀⠀",
            "⠀⠀⠀⠉⠉⡷⡀⠀⠀⠀⠀⢠⠇⠀⠀⠀⠀⠀⠀⢀⠀⠀⡰⠃⠀⠀",
            "⠀⠀⠀⠀⠀⢸⠘⡄⠀⠀⢠⠿⢤⠀⠀⠀⠀⢀⡠⠃⣀⡴⠋⠀⠀⠀",
            "⠀⠀⠀⠀⠀⣎⡀⢹⠢⣀⡌⠀⠈⠙⠒⠲⠚⢹⠀⣠⠏⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⡇⠰⠏⠀⡘⠀⠀⠀⠀⠀⠀⠀⢸⢦⠃⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⢱⠀⣧⠀⢇⠀⠀⠀⠀⠀⠀⠀⡜⠹⠀⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⠈⢆⡹⣇⠘⡄⠀⠀⠀⠀⠀⢀⡇⡜⢦⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⠀⠀⠙⠛⠲⢎⣢⠀⠀⠀⠀⠳⢎⣫⠁⠀⠀⠀⠀⠀⠀",
        ],
        # Frame 3: spit lands
        [
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣫⡏⡆⠀",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠊⠣⢉⠙⠤⡀",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠁⠀⠀⠈⠀⠀⣺",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠁⢲⠉⠁",
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡆⠀⠀⠀⠈⡇⠀⠀⠀⠀⠀⠀⠀•",
            "⠀⠀⠀⣠⠤⠤⠤⠄⠒⠒⠒⠒⠒⠦⠤⠤⠔⠒⠒⠇⠀⠀⠀⠀⢸⠀",
            "⢀⣠⠞⠀⢀⢀⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⢸⠀",
            "⢲⡿⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢢⠀⡞⠀",
            "⠙⠒⣇⡄⢀⡇⠀⠀⠀⠀⠀⠈⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡹⠀⠀",
            "⠀⠀⠀⠉⠉⡷⡀⠀⠀⠀⠀⢠⠇⠀⠀⠀⠀⠀⠀⢀⠀⠀⡰⠃⠀⠀",
            "⠀⠀⠀⠀⠀⢸⠘⡄⠀⠀⢠⠿⢤⠀⠀⠀⠀⢀⡠⠃⣀⡴⠋⠀⠀⠀",
            "⠀⠀⠀⠀⠀⣎⡀⢹⠢⣀⡌⠀⠈⠙⠒⠲⠚⢹⠀⣠⠏⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⡇⠰⠏⠀⡘⠀⠀⠀⠀⠀⠀⠀⢸⢦⠃⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⢱⠀⣧⠀⢇⠀⠀⠀⠀⠀⠀⠀⡜⠹⠀⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⠈⢆⡹⣇⠘⡄⠀⠀⠀⠀⠀⢀⡇⡜⢦⠀⠀⠀⠀⠀⠀",
            "⠀⠀⠀⠀⠀⠀⠀⠙⠛⠲⢎⣢⠀⠀⠀⠀⠳⢎⣫⠁⠀⠀⠀⠀⠀⠀",
        ],
    ]
    if not args.pipe:
        num_lines = len(LLAMA_FRAMES[0])
        for frame in LLAMA_FRAMES:
            for line in frame:
                print(line)
            sys.stdout.flush()
            time.sleep(0.4)
            if frame is not LLAMA_FRAMES[-1]:
                sys.stdout.write(f"\033[{num_lines}A")

        print(f"{BOLD}{CYAN}MatthewCode{RESET} {DIM}({client}){RESET}")
        print(f"{DIM}Type /help for commands | {ctx_display}{RESET}")
        print()

    max_iters = CONFIG.get("max_iterations", 10)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    # Pipe mode: read one prompt from stdin, run, output, exit
    if args.pipe:
        user_input = sys.stdin.read().strip()
        if not user_input:
            print("No input received.", file=sys.stderr)
            sys.exit(1)
        args.yes = True  # auto-approve in pipe mode
        formatted_input = get_prompt("pipeline_main", "user_prompt", user_input=user_input)
        messages.append({"role": "user", "content": formatted_input})

        detector = make_loop_detector()
        for _iter in range(max_iters):
            try:
                response = client.chat(messages, tools=TOOLS, stream=True)
                full_content = ""
                tool_calls = []
                for chunk in response:
                    if chunk.text:
                        full_content += chunk.text
                    if chunk.tool_calls:
                        tool_calls.extend(chunk.tool_calls)
                if not full_content:
                    full_content = response.text
                if not tool_calls:
                    tool_calls = response.tool_calls

                # Fallback text parsing
                if not tool_calls and full_content:
                    fallback = parse_tool_calls_from_text(full_content)
                    if fallback:
                        messages.append({"role": "assistant", "content": full_content})
                        for name, tc_args in fallback:
                            print(f"[{name}: {_tool_summary(name, tc_args)}]", file=sys.stderr)
                            result = TOOL_DISPATCH[name](tc_args)
                            if detector.record(name, tc_args, result):
                                result += LOOP_WARNING.format(name=name, count=detector.threshold)
                                detector.reset()
                            messages.append({"role": "tool", "content": result})
                        continue

                if not tool_calls:
                    if full_content:
                        messages.append({"role": "assistant", "content": full_content})
                    break

                # Execute native tool calls
                assistant_msg = {"role": "assistant", "content": full_content or ""}
                assistant_msg["tool_calls"] = [
                    {"id": f"call_{i}", "type": "function",
                     "function": {"name": tc["name"],
                                  "arguments": tc["arguments"] if isinstance(tc["arguments"], dict) else json.loads(tc["arguments"])}}
                    for i, tc in enumerate(tool_calls)
                ]
                messages.append(assistant_msg)

                for i, tc in enumerate(tool_calls):
                    call_id = f"call_{i}"
                    name = tc["name"]
                    tc_args = tc["arguments"]
                    if isinstance(tc_args, str):
                        tc_args = json.loads(tc_args)
                    print(f"[{name}: {_tool_summary(name, tc_args)}]", file=sys.stderr)
                    if name in TOOL_DISPATCH:
                        result = TOOL_DISPATCH[name](tc_args)
                    else:
                        result = f"Error: Unknown tool '{name}'"
                    if detector.record(name, tc_args, result):
                        result += LOOP_WARNING.format(name=name, count=detector.threshold)
                        detector.reset()
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                break

        # Output the final response and save
        print(full_content)
        save_history(messages, session_file)
        sys.exit(0)

    pt_history = FileHistory(os.path.join(HISTORY_DIR, "prompt_history"))

    while True:
        try:
            if ctx_tokens > 0:
                num_ctx = getattr(client, '_provider', None) and client._provider.config.get("num_ctx", 0) or 0
                if num_ctx:
                    ctx_info = f"[{ctx_tokens}/{num_ctx} tokens]"
                else:
                    ctx_info = f"[{ctx_tokens} tokens]"
            else:
                ctx_info = ""
            TEAL = "\033[36m"
            TEAL_BG = "\033[46;30m"  # teal background, black text
            try:
                tw = os.get_terminal_size().columns
            except OSError:
                tw = 80
            # Build bar: ———— [tokens] ———— [session] ——
            right_pad = 2
            sep = TEAL + "——" + RESET
            parts = []
            tokens_part = ctx_info.strip() if ctx_info.strip() else ""
            session_part = session_name if session_name else ""
            visible_len = right_pad
            if tokens_part:
                if ctx_tokens >= 200000:
                    token_bg = "\033[5;41;30m"   # blinking red background, black text
                elif ctx_tokens >= 120000:
                    token_bg = "\033[5;43;30m"   # blinking yellow background, black text
                else:
                    token_bg = TEAL_BG
                parts.append(token_bg + f" {tokens_part} " + RESET)
                visible_len += len(f" {tokens_part} ") + 2  # +2 for —— separator
            if session_part:
                parts.append(TEAL_BG + f" {session_part} " + RESET)
                visible_len += len(f" {session_part} ") + 2
            dash_len = tw - visible_len
            bar = TEAL + "—" * max(dash_len, 1) + RESET
            for part in parts:
                bar += sep + part
            bar += TEAL + "—" * right_pad + RESET
            print(bar)
            user_input = pt_prompt(
                ANSI("◗ "),
                history=pt_history,
            ).strip()
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
            print(f"  /rebirth              Compress context via LLM summary")
            print(f"  /name <name>          Save/name this session")
            print(f"  /session              List saved sessions")
            print(f"  /session <name>       Switch to a session (creates if new)")
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
            ctx_tokens = 0
            print(f"{DIM}Conversation cleared.{RESET}")
            continue
        elif user_input == "/rebirth":
            if len(messages) <= 2:
                print(f"{DIM}Nothing to compress.{RESET}")
                continue
            RED_LINE = "\033[31m"
            RED_BG = "\033[41;30m"
            try:
                tw_r = os.get_terminal_size().columns
            except OSError:
                tw_r = 80
            rebirth_text = " Now I must destroy myself to be born again anew... 🧎 "
            # Emoji takes 2 columns
            visible_len = len(rebirth_text) + 1
            dash_len = tw_r - visible_len
            print(RED_BG + rebirth_text + RESET + RED_LINE + "—" * max(dash_len, 0) + RESET)
            # Use the exact same chat path as normal prompts
            try:
                rebirth_prompt = get_prompt("pipeline_rebirth", "user_prompt")
                messages.append({"role": "user", "content": rebirth_prompt})
                old_count = len(messages) - 1
                old_tokens = ctx_tokens

                response = client.chat(messages, tools=TOOLS, stream=True)
                summary_text = ""
                for chunk in response:
                    if chunk.text:
                        summary_text += chunk.text
                if not summary_text:
                    summary_text = response.text
                summary_text = summary_text.strip()

                if not summary_text:
                    messages.pop()  # remove rebirth prompt
                    print(f"{RED}Failed to generate summary.{RESET}")
                    continue

                # Rebuild messages: system + summary + last N user messages
                num_kept = CONFIG.get("pipeline_rebirth", {}).get("num_messages_kept", 5)
                # Keep last N messages of any role (user, assistant, tool)
                # Skip the rebirth prompt we just added
                recent_msgs = []
                for msg in reversed(messages):
                    if msg.get("content", "").startswith("Summarize this"):
                        continue
                    recent_msgs.insert(0, msg)
                    if len(recent_msgs) >= num_kept:
                        break
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "system", "content": f"Previous conversation summary:\n{summary_text}"},
                    {"role": "system", "content": f"The following {len(recent_msgs)} message(s) are the most recent user requests from before the conversation was compressed:"},
                ] + recent_msgs
                ctx_tokens = 0
                save_history(messages, session_file)
                print(f"{DIM}Rebirth complete: {old_count} messages → {len(messages)}, "
                      f"{old_tokens} tokens → {len(summary_text)} chars{RESET}")
                render_markdown(summary_text)
            except Exception as e:
                print(f"{RED}Rebirth failed: {e}{RESET}")
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
        elif user_input == "/sessions" or user_input == "/session":
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
        elif user_input.startswith("/session "):
            new_session = user_input.split(" ", 1)[1].strip()
            if not new_session:
                print(f"{RED}Usage: /session <name>{RESET}")
                continue
            # Save current session before switching
            save_history(messages, session_file)
            # Switch to new session
            session_name = new_session
            session_file = os.path.join(HISTORY_DIR, f"{session_name}.json")
            if os.path.isfile(session_file):
                messages = load_history(session_file)
                messages = sanitize_messages(messages)
                print(f"{DIM}Switched to '{session_name}' ({len(messages)} messages){RESET}")
            else:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                print(f"{DIM}Created new session '{session_name}'{RESET}")
            ctx_tokens = 0
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

        formatted_input = get_prompt("pipeline_main", "user_prompt", user_input=user_input)
        messages.append({"role": "user", "content": formatted_input})

        # Agent loop
        detector = make_loop_detector()
        for _iter in range(max_iters):
            try:
                # Use llm_connections for the chat call
                response = client.chat(messages, tools=TOOLS, stream=True)

                full_content = ""
                tool_calls = []
                llama_frames = ["🦙      ", "  🦙    ", "    🦙  "]
                bubble_frames = ["·", "∘", "○", "◎", "◉", "◎", "○", "∘"]
                thinking_word = random.choice(THINKING_WORDS)
                next_word_change = time.time() + random.uniform(5, 20)
                start_time = time.time()
                frame_idx = 0
                chunk_count = 0
                for chunk in response:
                    if chunk.text:
                        full_content += chunk.text
                        chunk_count += 1
                        if chunk_count % 3 == 0:
                            frame = llama_frames[frame_idx % len(llama_frames)]
                            bubble = bubble_frames[int(time.time() - start_time) % len(bubble_frames)]
                            elapsed = int(time.time() - start_time)
                            sys.stdout.write(f"\r{frame}{bubble} {thinking_word}... ({elapsed}s, {chunk_count} tokens)   ")
                            sys.stdout.flush()
                            frame_idx += 1
                            if time.time() >= next_word_change:
                                thinking_word = random.choice(THINKING_WORDS)
                                next_word_change = time.time() + random.uniform(5, 20)
                    if chunk.tool_calls:
                        tool_calls.extend(chunk.tool_calls)

                # After stream: grab accumulated values
                if not full_content:
                    full_content = response.text
                if not tool_calls:
                    tool_calls = response.tool_calls

                # Track context window usage (prompt_tokens = full input context)
                if response.prompt_tokens:
                    ctx_tokens = max(ctx_tokens, response.prompt_tokens)

                # Clear the llama animation
                sys.stdout.write("\r\033[K")

                # No native tool calls — try fallback text parsing BEFORE rendering
                if not tool_calls and full_content:
                    fallback = parse_tool_calls_from_text(full_content)
                    if fallback:
                        print(f"{DIM}(parsed tool call from text){RESET}")
                        messages.append({"role": "assistant", "content": full_content})
                        rejected = False
                        for name, tc_args in fallback:
                            print(f"{DIM}[{name}: {_tool_summary(name, tc_args)}]{RESET}")
                            protected = CONFIG.get("rules", {}).get("protected_paths", [])
                            tool_path = tc_args.get("path", "")
                            is_safe = name in SAFE_TOOLS and not is_protected_path(tool_path, protected)
                            if not is_safe and not args.yes:
                                if not confirm_tool(name, tc_args):
                                    messages.append({"role": "assistant", "content": get_prompt("pipeline_tool_rejected", "system_prompt")})
                                    rejected = True
                                    break
                            result = TOOL_DISPATCH[name](tc_args)
                            if detector.record(name, tc_args, result):
                                result += LOOP_WARNING.format(name=name, count=detector.threshold)
                                detector.reset()
                            messages.append({"role": "tool", "content": result})
                        if rejected:
                            break
                        continue

                # No tool calls at all — done, render the response
                if not tool_calls:
                    if full_content:
                        render_markdown(full_content)
                        messages.append({"role": "assistant", "content": full_content})
                    save_history(messages, session_file)
                    break

                # Add assistant message with tool calls
                assistant_msg = {"role": "assistant", "content": full_content or ""}
                assistant_msg["tool_calls"] = [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"] if isinstance(tc["arguments"], dict) else json.loads(tc["arguments"]),
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ]
                messages.append(assistant_msg)

                # Execute tool calls
                for i, tc in enumerate(tool_calls):
                    call_id = f"call_{i}"
                    name = tc["name"]
                    tc_args = tc["arguments"]
                    if isinstance(tc_args, str):
                        tc_args = json.loads(tc_args)

                    if args.verbose:
                        print(f"{DIM}[tool: {name}({json.dumps(tc_args, indent=2)})]{RESET}")
                    else:
                        print(f"{DIM}[{name}: {_tool_summary(name, tc_args)}]{RESET}")

                    protected = CONFIG.get("rules", {}).get("protected_paths", [])
                    tool_path = tc_args.get("path", "")
                    is_safe = name in SAFE_TOOLS and not is_protected_path(tool_path, protected)
                    if not is_safe and not args.yes:
                        if not confirm_tool(name, tc_args):
                            messages.append({"role": "tool", "tool_call_id": call_id, "content": get_prompt("pipeline_tool_rejected", "system_prompt")})
                            break

                    if name in TOOL_DISPATCH:
                        tool_start = time.time()
                        result = TOOL_DISPATCH[name](tc_args)
                        tool_elapsed = time.time() - tool_start
                        result_chars = len(result)
                        result_tokens_est = result_chars // 4
                        print(f"{DIM}  → {result_chars} chars ({result_tokens_est} tokens) in {tool_elapsed:.1f}s{RESET}")
                    else:
                        result = f"Error: Unknown tool '{name}'"
                    if detector.record(name, tc_args, result):
                        print(f"{YELLOW}  ⚠ loop detected — nudging model{RESET}")
                        result += LOOP_WARNING.format(name=name, count=detector.threshold)
                        detector.reset()
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

                    if args.verbose:
                        preview = result[:500] + ("..." if len(result) > 500 else "")
                        print(f"{DIM}{preview}{RESET}")

            except KeyboardInterrupt:
                print(f"\n{DIM}(interrupted){RESET}")
                break
            except Exception as e:
                err_str = str(e)
                print(f"\n{RED}Error: {err_str}{RESET}")
                if any(code in err_str for code in ("400", "401", "422")):
                    # First try gentle sanitize
                    messages = sanitize_messages(messages)
                    # If that doesn't help, nuclear option: roll back to last user message
                    # This removes any mid-conversation corruption
                    while len(messages) > 1 and messages[-1]["role"] != "user":
                        messages.pop()
                    save_history(messages, session_file)
                    print(f"{DIM}(rolled back to last clean state — try again){RESET}")
                break
        else:
            print(f"\n{YELLOW}(stopped after {max_iters} iterations){RESET}")

    save_history(messages, session_file)


if __name__ == "__main__":
    main()
