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

# Add llm-connections/python to path. The optional slurm-manipulator nested
# submodule is added if present (under llm-connections/slurm-manipulator/) or
# from a sibling working tree at ~/Repos/slurm-manipulator/ (dev fallback).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "llm-connections", "python"))
for _slurm_path in (
    os.path.join(SCRIPT_DIR, "llm-connections", "slurm-manipulator", "python"),
    os.path.expanduser("~/Repos/slurm-manipulator/python"),
):
    if os.path.isdir(_slurm_path):
        sys.path.insert(0, _slurm_path)
        break

from llm_connections import LLMConnection, ProviderCatalog
from res.thinking import WORDS as THINKING_WORDS
from res.loop_detection import LoopDetector
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.shortcuts import CompleteStyle
from res.tabcompletion import build_slash_completer

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

SESSION_DIR_KEY = os.getcwd()  # directory matthewcode was launched in


def _encode_path(path):
    # emacs ~/.emacs.d/backup convention: escape '!' then '/' -> '!'
    return os.path.abspath(path).replace("!", "!!").replace("/", "!")


def session_path(name="last_session"):
    return os.path.join(HISTORY_DIR, f"{_encode_path(SESSION_DIR_KEY)}!{name}.json")


def _list_session_names():
    """Names of saved sessions for the current directory.

    Single source for both the `/session` listing and TAB completion.
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    prefix = f"{_encode_path(SESSION_DIR_KEY)}!"
    return sorted(f[len(prefix):-5] for f in os.listdir(HISTORY_DIR)
                  if f.startswith(prefix) and f.endswith(".json"))


# Single source of truth for slash commands: drives the `/help` listing AND TAB
# completion. Add a command here once; both update. (label, help) — `label` is
# the display form including any `<arg>` and comma-listed aliases.
SLASH_COMMANDS = [
    ("/help", "Show this help"),
    ("/exit, /quit", "Exit MatthewCode"),
    ("/clear", "Reset conversation"),
    ("/history", "View raw session JSON in less (from the bottom)"),
    ("/rebirth", "Compress context via LLM summary"),
    ("/name <name>", "Save/name this session"),
    ("/session", "List saved sessions"),
    ("/session <name>", "Switch to a session (creates if new)"),
    ("/provider", "List available LLM providers"),
    ("/provider <name>", "Switch to a different provider"),
    ("/verbose", "Toggle verbose mode"),
    ("/kill-model", "Terminate the active AI (local ollama serve OR Slurm job)"),
    ("/diagnose", "Probe the active AI: tunnel, /api/tags, /api/ps, Slurm job"),
]

# Hidden aliases the dispatch accepts but `/help` doesn't advertise as own rows.
_SLASH_ALIASES = ["/rename", "/sessions"]


def _slash_command_tokens():
    """Bare command tokens (deduped) derived from SLASH_COMMANDS, for completion."""
    tokens = list(_SLASH_ALIASES)
    for label, _ in SLASH_COMMANDS:
        for piece in label.split(","):              # split "/exit, /quit" aliases
            tokens.append(piece.strip().split(" ", 1)[0])  # drop any "<arg>"
    return sorted(set(tokens))

def load_config() -> dict:
    """Load app config from config.yaml and LLM providers from global config."""
    import yaml

    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f) or {}

    return config


CONFIG = load_config()


def _resolve_slurm_sessions(yaml_path: str, provider_filter: str = None) -> str:
    """If any provider in yaml_path has 'slurm_session: <name>', spin up the
    Slurm job, replace the field with 'base_url: <local_tunnel_url>', and
    return a path to a temp YAML with the rewrite. Otherwise return yaml_path
    unchanged. Skips silently if the file doesn't exist.
    """
    import tempfile
    import yaml as _yaml

    yaml_path = os.path.expanduser(yaml_path)
    if not os.path.isfile(yaml_path):
        return yaml_path
    with open(yaml_path, "r") as f:
        cfg = _yaml.safe_load(f) or {}

    providers = cfg.get("llm-providers") or {}
    # If provider_filter is specified, only process that provider
    if provider_filter:
        if provider_filter not in providers:
            return yaml_path
        provider_config = providers[provider_filter]
        if not (isinstance(provider_config, dict) and provider_config.get("slurm_session")):
            return yaml_path
        refs = {provider_filter: provider_config["slurm_session"]}
    else:
        refs = {n: p["slurm_session"] for n, p in providers.items()
                if isinstance(p, dict) and p.get("slurm_session")}

    if not refs:
        return yaml_path

    from llm_connections import SlurmSession, connect_ssh
    from llm_connections.ssh import can_reach
    SlurmSession.load()  # reads ~/.llm-connections/config.yaml's slurm-sessions: key

    started: dict = {}  # session_name -> SessionHandle (cache to dedupe)
    for prov_name, sess_name in refs.items():
        if sess_name not in started:
            sess = SlurmSession.get(sess_name)
            ssh = sess._cluster_cfg["ssh"]
            # Fast pre-flight: 3 TCP attempts to host:22, ~6s max. Bails early
            # when VPN is down rather than sitting in connect_ssh's retry loop.
            status_text = get_prompt("pipeline_status", "pinging_host", host=ssh['host'])
            console.print(status_text)
            if not can_reach(ssh["host"], port=22):
                raise ConnectionError(
                    f"Cannot reach {ssh['host']}:22 after 3 attempts (~6s). "
                    f"Are you on the VPN?"
                )
            connect_ssh(ssh["user"], ssh["host"], ssh.get("password", ""))
            existing = sess.peek_existing()
            if existing is not None:
                state, job_id = existing
                status_text = get_prompt("pipeline_status", "found_existing_job",
                                        job_id=job_id, state=state)
                console.print(status_text)
            else:
                status_text = get_prompt("pipeline_status", "no_existing_job", name=sess_name)
                console.print(status_text)
            status_text = get_prompt("pipeline_status", "starting_slurm", name=sess_name)
            console.print(status_text)
            started[sess_name] = sess.start()
        handle = started[sess_name]
        provider_cfg = providers[prov_name]
        provider_cfg["base_url"] = handle.local_url
        provider_cfg.pop("slurm_session", None)
        provider_cfg.pop("ssh_tunnel", None)  # remove any stale tunnel block
        status_text = get_prompt("pipeline_status", "provider_ready",
                                provider=prov_name, url=handle.local_url,
                                job_id=handle.job_id, gpu=handle.gpu_info)
        console.print(status_text)

    tf = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    _yaml.safe_dump(cfg, tf)
    tf.close()
    return tf.name


def load_providers(provider_name=None):
    """Load LLM providers. Called after argparse so --help doesn't wait."""
    home_yaml = os.path.expanduser("~/.llm-connections/config.yaml")
    LLMConnection.load(_resolve_slurm_sessions(home_yaml, provider_name))
    if "llm-providers" in CONFIG:
        LLMConnection.load(_resolve_slurm_sessions(CONFIG_FILE, provider_name))


def _providers_help_text():
    """Formatted list of configured providers for --help.

    Builds a ProviderCatalog from the home + project YAMLs (project wins on name
    clashes — it's merged last) and renders it. The catalog reads config only —
    no Slurm jobs, SSH tunnels, or Ollama probes — so --help stays instant.
    """
    paths = (os.path.expanduser("~/.llm-connections/config.yaml"), CONFIG_FILE)
    catalog = ProviderCatalog.from_paths(paths)
    return catalog.format(title="Configured LLM providers (select with --provider NAME)")
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
    "project_root": os.getcwd(),
}



def get_prompt(pipeline: str, prompt_type: str, **extra_vars) -> str:
    """Get a prompt from config. pipeline='pipeline_main', prompt_type='system_prompt'."""
    template = CONFIG.get(pipeline, {}).get(prompt_type, "")
    if not template:
        raise ValueError(f"Missing prompt: {pipeline}.{prompt_type} not found in config.yaml")
    all_vars = {**PROMPT_VARS, **extra_vars}
    return template.format(**all_vars)


SYSTEM_PROMPT = get_prompt("pipeline_main", "system_prompt")

# --- Tool definitions ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read the contents of a file.",
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
            "The old_text must match exactly as it appears in the file.",
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
            "Dockerfiles, docker-compose, and CI/CD configs (GitLab, GitHub Actions, Jenkins, Travis, CircleCI).",
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


def is_restricted_bash(name: str, args: dict) -> bool:
    """True if a bash_run command contains a restricted word. These force a
    y/N prompt even when --yes / auto_approve is set."""
    if name != "bash_run":
        return False
    import re
    cmd = args.get("command", "")
    for word in CONFIG.get("restricted_bash_commands", []):
        if re.search(rf"\b{re.escape(word)}\b", cmd):
            return True
    return False


# --- Tool implementations ---


def tool_file_read(path):
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return get_prompt("pipeline_tool_errors", "file_not_found", path=path)
    try:
        with open(path, "r") as f:
            content = f.read()
        if len(content) > MAX_FILE_READ:
            return get_prompt("pipeline_tool_errors", "file_read_too_large",
                            path=path, len_content=len(content), max_limit=MAX_FILE_READ)
        return content
    except Exception as e:
        return get_prompt("pipeline_tool_errors", "file_read_error", error=e)


def tool_file_write(path, content):
    path = os.path.expanduser(path)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return get_prompt("pipeline_tool_success", "file_write", path=path, len_content=len(content))
    except Exception as e:
        return get_prompt("pipeline_tool_errors", "file_write_error", error=e)


def tool_file_edit(path, old_text, new_text):
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return get_prompt("pipeline_tool_errors", "file_not_found", path=path)
    try:
        with open(path, "r") as f:
            content = f.read()
        if old_text in content:
            count = content.count(old_text)
            if count > 1:
                return get_prompt("pipeline_tool_errors", "file_edit_multiple_matches", count=count)
            new_content = content.replace(old_text, new_text, 1)
            with open(path, "w") as f:
                f.write(new_content)
            return get_prompt("pipeline_tool_success", "file_edit", path=path)
        lines = content.split("\n")
        old_lines = old_text.split("\n")
        matcher = difflib.SequenceMatcher(None, lines, old_lines)
        best = matcher.find_longest_match(0, len(lines), 0, len(old_lines))
        if best.size > 0 and best.size >= len(old_lines) * 0.6:
            return get_prompt("pipeline_tool_errors", "file_edit_fuzzy_match",
                            start_line=best.a + 1, end_line=best.a + best.size)
        return get_prompt("pipeline_tool_errors", "file_edit_not_found")
    except Exception as e:
        return get_prompt("pipeline_tool_errors", "file_edit_error", error=e)


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
            return get_prompt("pipeline_tool_errors", "bash_timeout")
        output = ""
        if stdout:
            output += stdout
        if stderr:
            output += ("\n" if output else "") + stderr
        rc = proc.returncode
        # Truncate the command's own output first so the exit-code marker below
        # is never cut off.
        if len(output) > MAX_BASH_OUTPUT:
            output = output[:MAX_BASH_OUTPUT] + f"\n[truncated at {MAX_BASH_OUTPUT} chars]"
        if not output:
            # Give silent success an explicit signal so the model doesn't
            # misread it as failure and re-probe in a loop.
            output = ("(no output, exit 0 — command succeeded)" if rc == 0
                      else f"(no output)\n[exit code: {rc}]")
        else:
            output += f"\n[exit code: {rc}]"
        return output
    except Exception as e:
        return get_prompt("pipeline_tool_errors", "bash_error", error=e)


def tool_dir_list(path):
    path = os.path.expanduser(path or ".")
    if not os.path.isdir(path):
        return get_prompt("pipeline_tool_errors", "directory_not_found", path=path)
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
        return get_prompt("pipeline_tool_errors", "directory_not_found", path=path)
    matches = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in
                   ("node_modules", "__pycache__", ".git", "venv", "env")]
        for f in files:
            if fnmatch.fnmatch(f, pattern):
                matches.append(os.path.join(root, f))
                if len(matches) >= 50:
                    return "\n".join(matches) + "\n[truncated at 50 results]"
    return "\n".join(matches) if matches else get_prompt("pipeline_tool_success", "find_no_matches", pattern=pattern, path=path)


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
    return "\n".join(matches) if matches else get_prompt("pipeline_tool_success", "grep_no_matches", pattern=pattern, path=path)


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


def _safe(name, fn):
    def call(a):
        try:
            return fn(a)
        except KeyError as e:
            return f"Error: Tool call {name} is missing required argument {e}. Retry with the argument included."
    return call


TOOL_DISPATCH = {
    "file_read": _safe("file_read", lambda a: tool_file_read(a["path"])),
    "file_write": _safe("file_write", lambda a: tool_file_write(a["path"], a["content"])),
    "file_edit": _safe("file_edit", lambda a: tool_file_edit(a["path"], a["old_text"], a["new_text"])),
    "bash_run": _safe("bash_run", lambda a: tool_bash_run(a["command"], a.get("timeout", 120))),
    "dir_list": _safe("dir_list", lambda a: tool_dir_list(a.get("path", "."))),
    "file_find": _safe("file_find", lambda a: tool_file_find(a["pattern"], a.get("path", "."))),
    "file_grep": _safe("file_grep", lambda a: tool_file_grep(a["pattern"], a.get("path", "."), a.get("glob"))),
    "find_build_env": _safe("find_build_env", lambda a: tool_find_build_env(a.get("path", "."))),
}


# --- Confirmation ---


def confirm_tool(name, args, restricted=False):
    if name == "file_write":
        path = args.get("path", "?")
        prompt_text = get_prompt("pipeline_confirmations", "file_write_prompt",
                                path=path, len_content=len(args.get('content', '')))
        print(f"\n{YELLOW}{prompt_text}{RESET}")
        if os.path.isfile(os.path.expanduser(path)):
            warning_text = get_prompt("pipeline_confirmations", "file_exists_warning")
            print(f"{DIM}{warning_text}{RESET}")
    elif name == "file_edit":
        prompt_text = get_prompt("pipeline_confirmations", "file_edit_prompt", path=args.get('path', '?'))
        print(f"\n{YELLOW}{prompt_text}{RESET}")
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
    if restricted:
        print(f"{RED}Restricted command — approval required even with --yes.{RESET}")
        try:
            return input(f"{BOLD}Run anyway? [y/N] {RESET}").strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            print()
            return False
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


def sanitize_messages(messages: list, quiet: bool = False) -> list:
    """Fix corrupted history so it satisfies strict OpenAI/litellm tool-call
    pairing (Ollama is lenient about this; litellm is not).

    - Drop messages with no 'role'.
    - Coerce tool_calls arguments from JSON strings to dicts.
    - Forward: an assistant 'tool_calls' must be answered by one 'tool' message
      per call id, immediately following. If incomplete, drop the tool_calls
      (keeping the assistant's text if any) and discard the partial results.
    - Backward: a 'tool' message with no matching preceding assistant tool_calls
      is an orphan and is dropped — anywhere in the list, not just the tail.
    - Drop a trailing assistant with empty content and no tool_calls.
    """
    if not messages:
        return messages

    original_len = len(messages)

    # Pass 1: drop role-less messages; coerce string tool-call arguments.
    normalized = []
    for msg in messages:
        if "role" not in msg:
            continue
        m = dict(msg)
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        func["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        func["arguments"] = {}
        normalized.append(m)

    # Pass 2: enforce tool-call pairing in both directions.
    paired = []
    repairs = 0  # in-place fixes (stripped tool_calls) — not captured by len delta
    i = 0
    n = len(normalized)
    while i < n:
        m = normalized[i]
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            ids = [tc.get("id") for tc in m["tool_calls"]]
            # gather the contiguous 'tool' results immediately following
            j = i + 1
            answered = {}
            while j < n and normalized[j].get("role") == "tool":
                answered[normalized[j].get("tool_call_id")] = normalized[j]
                j += 1
            if all(tid in answered for tid in ids):
                paired.append(m)
                for tid in ids:  # canonical order; any extra/dup tools dropped
                    paired.append(answered[tid])
            elif m.get("content"):
                # incomplete tool_calls: keep the text, drop the call + partials
                m2 = dict(m)
                m2.pop("tool_calls", None)
                paired.append(m2)
                repairs += 1
            # else: drop the dangling assistant entirely
            i = j
        elif role == "tool":
            i += 1  # orphan tool result — no assistant block consumed it
        else:
            paired.append(m)
            i += 1

    # Pass 3: drop a trailing assistant with empty content and no tool_calls.
    while len(paired) > 1 and paired[-1].get("role") == "assistant" \
            and not paired[-1].get("content") and not paired[-1].get("tool_calls"):
        paired.pop()

    fixed = (original_len - len(paired)) + repairs
    if fixed:
        print(f"{DIM}(cleaned {fixed} corrupted messages from history){RESET}")
    elif not quiet:
        print(f"{DIM}(history inspected — no corruption found){RESET}")

    return paired


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
    parser = argparse.ArgumentParser(
        description="MatthewCode - Local LLM coding assistant",
        epilog=_providers_help_text(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    load_providers(args.provider)

    # Get LLM connection from registry
    try:
        client = LLMConnection.get(args.provider)
    except KeyError as e:
        print(f"{RED}{e}{RESET}")
        sys.exit(1)

    session_name = None
    resume_existing = False
    if args.resume_name:
        session_name = args.resume_name
        session_file = session_path(session_name)
        if not os.path.isfile(session_file):
            print(f"{DIM}Created new session '{session_name}'{RESET}")
    else:
        # one-time migration of the legacy flat last_session.json into the
        # current directory's per-dir slot.
        legacy = os.path.join(HISTORY_DIR, "last_session.json")
        if os.path.isfile(legacy) and not os.path.isfile(session_path()):
            os.makedirs(HISTORY_DIR, exist_ok=True)
            os.rename(legacy, session_path())
        session_file = session_path()
        # Starting fresh while a last session exists for this directory would
        # silently overwrite it on the first save. Guard against that. Gated on
        # an interactive TTY so pipe mode doesn't consume the piped prompt.
        if not args.resume and sys.stdin.isatty() and os.path.isfile(session_file):
            try:
                answer = input(f"{YELLOW}Last session exists for this directory. Overwrite with new session? [y/N] {RESET}").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer not in ("y", "yes"):
                resume_existing = True
                print(f"{DIM}Keeping existing session.{RESET}")

    messages = []
    if args.resume or args.resume_name or resume_existing:
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
    # Splash screen — spitting llama animation. One static llama body; the
    # spit glob is animated on top of it (see SPIT_SEQUENCE below).
    LLAMA_FRAMES = [
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
    ]
    # The static llama body; the spit glob is overlaid/animated on top of it.
    BASE_LLAMA = LLAMA_FRAMES[0]
    SPIT_ROW = 3  # line at the llama's mouth height where the glob launches

    # Each step: (text drawn to the right of SPIT_ROW, shake = leading-space
    # jitter applied to the whole llama). The glob swells ·→•→●→◉ as it flies,
    # trails behind itself, then freezes mid-air on the final frame (which stays
    # on screen, since the title prints below without erasing it).
    SPIT_SEQUENCE = [
        (" ·",            0),
        ("  ·",           0),
        ("   ·•",         0),
        ("    ·•",        0),
        ("     ·•●",      0),
        ("      ·•●",     0),
        ("       ·•◉",    0),
        ("        ·•◉",   0),
        ("         ·•◉",  0),
        ("          ·•◉", 0),  # spit frozen in mid-air — final frame stays on screen
    ]
    if not args.pipe:
        num_lines = len(BASE_LLAMA)
        for step, (spit, shake) in enumerate(SPIT_SEQUENCE):
            if step:
                sys.stdout.write(f"\033[{num_lines}A")  # rewind to redraw in place
            pad = " " * shake
            for i, line in enumerate(BASE_LLAMA):
                row = pad + line + (spit if i == SPIT_ROW else "")
                sys.stdout.write("\r" + row + "\033[K\n")  # \033[K wipes stale tail
            sys.stdout.flush()
            time.sleep(0.11)

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
                # safety net: repair any malformed tool-call pairing before it
                # reaches a strict provider. quiet so clean iterations stay
                # silent; real cleanups still print
                messages = sanitize_messages(messages, quiet=True)
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

                # fallback text parsing; synthesize a proper assistant tool_calls
                # block + matching tool_call_id results so the payload stays valid
                # on strict providers and survives sanitize/reload
                if not tool_calls and full_content:
                    fallback = parse_tool_calls_from_text(full_content)
                    if fallback:
                        assistant_msg = {"role": "assistant", "content": full_content}
                        assistant_msg["tool_calls"] = [
                            {"id": f"call_{i}", "type": "function",
                             "function": {"name": name,
                                          "arguments": tc_args if isinstance(tc_args, dict) else json.loads(tc_args)}}
                            for i, (name, tc_args) in enumerate(fallback)
                        ]
                        messages.append(assistant_msg)
                        for i, (name, tc_args) in enumerate(fallback):
                            call_id = f"call_{i}"
                            print(f"[{name}: {_tool_summary(name, tc_args)}]", file=sys.stderr)
                            result = TOOL_DISPATCH[name](tc_args)
                            if detector.record(name, tc_args, result):
                                result += get_prompt("pipeline_loop_detected", "message", name=name, count=detector.threshold)
                                detector.reset()
                            messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
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
                        result = get_prompt("pipeline_tool_errors", "unknown_tool", name=name)
                    if detector.record(name, tc_args, result):
                        result += get_prompt("pipeline_loop_detected", "message", name=name, count=detector.threshold)
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
    slash_completer = build_slash_completer(
        _slash_command_tokens(),
        arg_value_funcs={
            "/provider": LLMConnection.list_providers,
            "/session": _list_session_names,
            "/sessions": _list_session_names,
        },
    )

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
                completer=slash_completer,
                complete_while_typing=False,                # TAB-triggered, no live menu
                complete_style=CompleteStyle.READLINE_LIKE,  # bash-style completion
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Bye!{RESET}")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input == "/help":
            print(f"{DIM}Commands:{RESET}")
            for label, helptext in SLASH_COMMANDS:
                print(f"  {label:<22}{helptext}")
            continue
        elif user_input == "/verbose":
            args.verbose = not args.verbose
            print(f"{DIM}Verbose: {'on' if args.verbose else 'off'}{RESET}")
            continue
        elif user_input == "/diagnose":
            import time as _time
            import urllib.error as _urlerr
            import urllib.request as _urlreq

            cur = LLMConnection._registry.get(args.provider)
            base_url = getattr(getattr(cur, "_provider", None), "base_url", None)
            print(f"{DIM}── /diagnose {args.provider} ──{RESET}")
            print(f"  base_url: {base_url or '(none)'}")

            def _probe(path: str, timeout: float = 5.0):
                if not base_url:
                    return ("?", 0.0, "no base_url")
                url = base_url.rstrip("/") + path
                t0 = _time.time()
                try:
                    with _urlreq.urlopen(url, timeout=timeout) as r:
                        body = r.read().decode("utf-8", errors="replace")
                        return (r.status, _time.time() - t0, body[:500])
                except _urlerr.HTTPError as e:
                    return (e.code, _time.time() - t0, str(e))
                except Exception as e:
                    return ("ERR", _time.time() - t0, f"{type(e).__name__}: {e}")

            for path in ("/api/tags", "/api/ps"):
                code, dt, body = _probe(path)
                color = GREEN if code == 200 else RED
                print(f"  {color}{path:<10} {code} ({dt*1000:.0f}ms){RESET}")
                if body and code != 200:
                    print(f"    {DIM}{body}{RESET}")
                elif body and code == 200 and path == "/api/ps":
                    # Show loaded models — most useful diagnostic for "why is chat hanging"
                    print(f"    {DIM}{body}{RESET}")

            from llm_connections import SlurmSession
            handle = SlurmSession.active_by_local_url(base_url) if base_url else None
            if handle is not None:
                print(f"  {DIM}Slurm: cluster={handle.cluster}, job={handle.job_id}, "
                      f"remote={handle.remote_host}:{handle.remote_port}, gpu={handle.gpu_info}{RESET}")
                print(f"  {DIM}Check the job manually:{RESET}")
                print(f"    ssh {handle.ssh_target} 'squeue -j {handle.job_id}; "
                      f"tail -50 ~/log/{handle.name}-{handle.job_id}.log'")
            elif base_url and ("localhost" in base_url or "127.0.0.1" in base_url):
                print(f"  {DIM}Local Ollama; if /api/tags hangs run 'pkill ollama' and restart.{RESET}")
            continue
        elif user_input == "/kill-model":
            cur = LLMConnection._registry.get(args.provider)
            base_url = getattr(getattr(cur, "_provider", None), "base_url", None)
            killed_what = None

            from llm_connections import SlurmSession
            handle = SlurmSession.active_by_local_url(base_url) if base_url else None
            if handle is not None:
                handle.kill()
                killed_what = f"Slurm job {handle.job_id} on cluster {handle.cluster}"
            elif base_url and ("localhost" in base_url or "127.0.0.1" in base_url):
                from llm_connections.llm_providers.ollama import kill_local_server
                if kill_local_server():
                    killed_what = "local ollama serve"
                else:
                    print(f"{YELLOW}No local 'ollama serve' process was running.{RESET}")
            else:
                print(
                    f"{YELLOW}Provider '{args.provider}' is at {base_url or '?'}; "
                    f"not managed by matthewcode (not local, no Slurm session).{RESET}"
                )

            if killed_what:
                print(f"{GREEN}/kill-model: terminated {killed_what}.{RESET}")
                # Don't mark provider as permanently failed - check connectivity on next use
            continue
        elif user_input in ("/exit", "/quit"):
            print(f"{DIM}Bye!{RESET}")
            break
        elif user_input == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            session_name = None
            session_file = session_path()
            ctx_tokens = 0
            print(f"{DIM}Conversation cleared.{RESET}")
            continue
        elif user_input == "/history":
            import tempfile
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                json.dump(messages, tf, indent=2, default=str)
                tmp = tf.name
            try:
                subprocess.run(["less", "+G", tmp])
            except FileNotFoundError:
                print(f"{RED}'less' not found on PATH.{RESET}")
            finally:
                os.unlink(tmp)
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

                # summarize with a dedicated system prompt and no tools so the
                # summary comes back as text, not file_write calls
                rebirth_system = get_prompt("pipeline_rebirth", "system_prompt")
                summarize_messages = [
                    {"role": "system", "content": rebirth_system}
                ] + messages[1:]
                response = client.chat(summarize_messages, tools=None, stream=True)
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
                # The count-based slice can start mid tool-call exchange. A leading
                # 'tool' result whose parent assistant+tool_calls was sliced off is
                # an orphan that litellm/OpenAI reject. Drop leading orphans so the
                # window starts on a clean boundary (a user or assistant message).
                while recent_msgs and recent_msgs[0].get("role") == "tool":
                    recent_msgs.pop(0)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "system", "content": f"Previous conversation summary:\n{summary_text}"},
                    {"role": "system", "content": f"The following {len(recent_msgs)} message(s) are the most recent user requests from before the conversation was compressed:"},
                ] + recent_msgs
                # Sanitize the rebuilt list before persisting — rebirth previously
                # saved directly, so malformed tool sequences were never checked.
                messages = sanitize_messages(messages)
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
            new_file = session_path(new_name)
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
            sessions = _list_session_names()
            if sessions:
                print(f"{DIM}Saved sessions:{RESET}")
                for s in sessions:
                    marker = f" {CYAN}<- active{RESET}" if s == (session_name or "last_session") else ""
                    msgs = load_history(session_path(s))
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
            session_file = session_path(session_name)
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
                # safety net: repair any malformed tool-call pairing before it
                # reaches a strict provider. quiet so clean iterations stay
                # silent; real cleanups still print
                messages = sanitize_messages(messages, quiet=True)
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
                        # synthesize a proper assistant tool_calls block so every
                        # call has an id that a tool result can answer
                        assistant_msg = {"role": "assistant", "content": full_content}
                        assistant_msg["tool_calls"] = [
                            {"id": f"call_{i}", "type": "function",
                             "function": {"name": name,
                                          "arguments": tc_args if isinstance(tc_args, dict) else json.loads(tc_args)}}
                            for i, (name, tc_args) in enumerate(fallback)
                        ]
                        messages.append(assistant_msg)
                        rejected = False
                        for i, (name, tc_args) in enumerate(fallback):
                            call_id = f"call_{i}"
                            print(f"{DIM}[{name}: {_tool_summary(name, tc_args)}]{RESET}")
                            protected = CONFIG.get("rules", {}).get("protected_paths", [])
                            tool_path = tc_args.get("path", "")
                            is_safe = name in SAFE_TOOLS and not is_protected_path(tool_path, protected)
                            restricted = is_restricted_bash(name, tc_args)
                            if restricted or (not is_safe and not args.yes):
                                if not confirm_tool(name, tc_args, restricted=restricted):
                                    messages.append({"role": "tool", "tool_call_id": call_id, "content": get_prompt("pipeline_tool_rejected", "message")})
                                    # answer the remaining declared call ids so the
                                    # assistant tool_calls block stays fully paired
                                    for j in range(i + 1, len(fallback)):
                                        messages.append({"role": "tool", "tool_call_id": f"call_{j}", "content": get_prompt("pipeline_tool_skipped", "message")})
                                    rejected = True
                                    break
                            result = TOOL_DISPATCH[name](tc_args)
                            if detector.record(name, tc_args, result):
                                result += get_prompt("pipeline_loop_detected", "message", name=name, count=detector.threshold)
                                detector.reset()
                            messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
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
                    restricted = is_restricted_bash(name, tc_args)
                    if restricted or (not is_safe and not args.yes):
                        if not confirm_tool(name, tc_args, restricted=restricted):
                            messages.append({"role": "tool", "tool_call_id": call_id, "content": get_prompt("pipeline_tool_rejected", "message")})
                            # answer the remaining declared call ids so the
                            # assistant tool_calls block stays fully paired
                            for j in range(i + 1, len(tool_calls)):
                                messages.append({"role": "tool", "tool_call_id": f"call_{j}", "content": get_prompt("pipeline_tool_skipped", "message")})
                            break

                    if name in TOOL_DISPATCH:
                        tool_start = time.time()
                        result = TOOL_DISPATCH[name](tc_args)
                        tool_elapsed = time.time() - tool_start
                        result_chars = len(result)
                        result_tokens_est = result_chars // 4
                        print(f"{DIM}  → {result_chars} chars ({result_tokens_est} tokens) in {tool_elapsed:.1f}s{RESET}")
                    else:
                        result = get_prompt("pipeline_tool_errors", "unknown_tool", name=name)
                    if detector.record(name, tc_args, result):
                        print(f"{YELLOW}  ⚠ loop detected — nudging model{RESET}")
                        result += get_prompt("pipeline_loop_detected", "message", name=name, count=detector.threshold)
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
