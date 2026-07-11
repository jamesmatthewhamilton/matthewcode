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
from res.thinking import QUOTES as THINKING_QUOTES
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
    return catalog.format(title="providers")
MAX_BASH_OUTPUT = CONFIG.get("max_bash_output", 30_000)
MAX_FILE_READ = CONFIG.get("max_file_read", 50_000)
TRIM_TRAILING_WHITESPACE = CONFIG.get("trim_trailing_whitespace", True)


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


def _normalize_trailing_whitespace(text):
    """Editor-on-save normalization: strip trailing spaces/tabs from every
    line, drop trailing blank lines, and guarantee a single final newline.
    Empty content stays empty (don't create a lone-newline file). No-op when
    the trim_trailing_whitespace config flag is disabled."""
    if not TRIM_TRAILING_WHITESPACE:
        return text
    stripped = "\n".join(line.rstrip() for line in text.split("\n"))
    stripped = stripped.rstrip("\n")
    return stripped + "\n" if stripped else ""


def tool_file_write(path, content):
    path = os.path.expanduser(path)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        content = _normalize_trailing_whitespace(content)
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
            new_content = _normalize_trailing_whitespace(new_content)
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


# --- Command registry ---------------------------------------------------------
# One entry per command is the single source of truth: it drives slash dispatch,
# /help, TAB completion, AND one-shot `--flag` invocation. A `--flag` runs the
# same handler as its `/command` with an empty arg, once, then exits.

from dataclasses import dataclass, field
from typing import Callable, Optional


class ReplContext:
    """Mutable REPL state a command handler reads/writes.

    The same handler runs inside the interactive loop (state copied back into the
    loop locals afterwards) or from a one-shot flag (state discarded on exit).
    """

    def __init__(self, *, args, console, client, messages,
                 session_name, session_file, ctx_tokens, interactive=True):
        self.args = args
        self.console = console
        self.client = client
        self.messages = messages
        self.session_name = session_name
        self.session_file = session_file
        self.ctx_tokens = ctx_tokens
        self.interactive = interactive
        self.should_exit = False


@dataclass(frozen=True)
class Command:
    """One user-facing thing, declared ONCE. `flag_command` holds the bare token
    name(s) — NO `/` or `--`. The surface prefix is added at use-time: `/<token>`
    for the command form, `--<token>` for the flag form. The two booleans decide
    which surfaces this appears on (in help) and is usable from. Both default to
    True — a thing is a command AND a flag unless an entry opts out below."""
    help: str                              # the single description for both surfaces
    flag_command: tuple = ()               # bare token name(s), e.g. ("exit","quit"); no / or --
    run: Optional[Callable] = None         # handler for the command / one-shot flag
    arghint: str = ""                      # "[<name>]" — /help display
    arg_values: Optional[Callable] = None  # tab-completion source for the arg
    flag_kwargs: dict = field(default_factory=dict)  # extra argparse kwargs for the flag
    is_command: bool = True                # shows in /help as /<token> and usable as a command
    is_flag: bool = True                   # shows in --help as --<token> and usable as a flag
    needs_client: bool = False             # one-shot flag runs AFTER provider/client setup
    one_shot: bool = False                 # flag runs run(ctx,"") once, then exits

    def __post_init__(self):
        # Footgun guard: `("session")` is a *string*, not a tuple (missing comma) —
        # iterating it would yield characters ('s','e',...). Coerce a bare string
        # to a 1-tuple so a missing comma can't silently register --s, --e, ...
        if isinstance(self.flag_command, str):
            object.__setattr__(self, "flag_command", (self.flag_command,))


def cmd_help(ctx, arg):
    print(f"{DIM}Commands:{RESET}")
    rows = []
    for c in COMMANDS:
        if not c.is_command:            # flag-only (e.g. --prompt) — see --help
            continue
        label = ", ".join(f"/{t}" for t in c.flag_command)   # add / per surface
        if c.arghint:
            label += f" {c.arghint}"
        rows.append((label, c.help))
    width = max(len(label) for label, _ in rows)
    for label, helptext in rows:
        print(f"  {label:<{width}}  {helptext}")


def cmd_verbose(ctx, arg):
    ctx.args.verbose = not ctx.args.verbose
    print(f"{DIM}Verbose: {'on' if ctx.args.verbose else 'off'}{RESET}")


def cmd_yes(ctx, arg):
    ctx.args.yes = not ctx.args.yes
    print(f"{DIM}Auto-approve: {'on' if ctx.args.yes else 'off'}{RESET}")


def cmd_exit(ctx, arg):
    print(f"{DIM}Bye!{RESET}")
    ctx.should_exit = True


def cmd_clear(ctx, arg):
    # Empty the CURRENT session in place (keep its name/file) and persist — so
    # `--session X --clear` actually wipes X, and /clear wipes the open session.
    ctx.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    ctx.ctx_tokens = 0
    save_history(ctx.messages, ctx.session_file)
    print(f"{DIM}Conversation cleared.{RESET}")


def cmd_history(ctx, arg):
    """Render the session as a chronological transcript — one rich Panel per turn,
    colour-coded by role, with prose as Markdown, tool results as pretty JSON, and
    tool calls as concise `→ name summary` lines (no raw JSON blobs)."""
    from rich.console import Group
    from rich.json import JSON
    from rich.panel import Panel

    if not ctx.messages:
        print(f"{DIM}No history.{RESET}")
        return

    ROLE_STYLE = {"user": "cyan", "assistant": "green", "system": "dim", "tool": "yellow"}

    def _panel(i, msg):
        role = msg.get("role", "?")
        parts = []
        content = msg.get("content")
        if content:
            if role in ("user", "assistant", "system"):
                parts.append(Markdown(content if isinstance(content, str) else str(content)))
            else:                                   # tool result: pretty JSON, else plain text
                try:
                    parts.append(JSON(content))
                except (ValueError, TypeError):
                    parts.append(str(content))
        for tc in msg.get("tool_calls") or []:      # de-mangle tool calls → one tidy line
            fn = tc.get("function", tc)
            name = fn.get("name", "?")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            parts.append(f"[bold]→ {name}[/]  [dim]{_tool_summary(name, args)}[/]")
        body = Group(*parts) if parts else "[dim](no content)[/dim]"
        return Panel(body, title=f"#{i} · {role}", title_align="left",
                     border_style=ROLE_STYLE.get(role, "white"), padding=(0, 1))

    # Print straight to the terminal — rich emits colour, the terminal interprets it,
    # and rich auto-strips colour when piped to a file. (Paging via `less` was the
    # bug: it showed the colour codes as literal `ESC[..m` text unless run with -R.
    # Want paging? `matthewcode --history | less -R`.)
    for i, msg in enumerate(ctx.messages):
        ctx.console.print(_panel(i, msg))


def cmd_diagnose(ctx, arg):
    import time as _time
    import urllib.error as _urlerr
    import urllib.request as _urlreq

    cur = LLMConnection._registry.get(ctx.args.provider)
    base_url = getattr(getattr(cur, "_provider", None), "base_url", None)
    print(f"{DIM}── /diagnose {ctx.args.provider} ──{RESET}")
    print(f"  base_url: {base_url or '(none)'}")

    def _probe(path, timeout=5.0):
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


def cmd_kill_model(ctx, arg):
    cur = LLMConnection._registry.get(ctx.args.provider)
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
            f"{YELLOW}Provider '{ctx.args.provider}' is at {base_url or '?'}; "
            f"not managed by matthewcode (not local, no Slurm session).{RESET}"
        )

    if killed_what:
        print(f"{GREEN}/kill-model: terminated {killed_what}.{RESET}")


def cmd_name(ctx, arg):
    if not arg:
        if ctx.session_name:
            print(f"{DIM}Current session: '{ctx.session_name}'{RESET}")
        else:
            print(f"{DIM}Session unnamed. Usage: /name <session_name>{RESET}")
        return
    new_name = arg.strip()
    new_file = session_path(new_name)
    if os.path.isfile(new_file) and new_name != ctx.session_name:
        try:
            answer = input(f"{YELLOW}Session '{new_name}' exists. Overwrite? [y/N] {RESET}").strip().lower()
            if answer not in ("y", "yes"):
                print(f"{DIM}Cancelled.{RESET}")
                return
        except (EOFError, KeyboardInterrupt):
            print()
            return
    ctx.session_name = new_name
    ctx.session_file = new_file
    save_history(ctx.messages, ctx.session_file)
    print(f"{DIM}Session saved as '{ctx.session_name}'{RESET}")


def cmd_session(ctx, arg):
    if not arg:
        sessions = _list_session_names()
        if sessions:
            print(f"{DIM}Saved sessions:{RESET}")
            for s in sessions:
                marker = f" {CYAN}<--- [Active]{RESET}" if s == (ctx.session_name or "last_session") else ""
                msgs = load_history(session_path(s))
                print(f"  {s} ({len(msgs)} messages){marker}")
        else:
            print(f"{DIM}No saved sessions.{RESET}")
        return
    new_session = arg.strip()
    save_history(ctx.messages, ctx.session_file)            # save current before switching
    ctx.session_name = new_session
    ctx.session_file = session_path(new_session)
    if os.path.isfile(ctx.session_file):
        ctx.messages = sanitize_messages(load_history(ctx.session_file))
        print(f"{DIM}Switched to '{ctx.session_name}' ({len(ctx.messages)} messages){RESET}")
    else:
        ctx.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        print(f"{DIM}Created new session '{ctx.session_name}'{RESET}")
    ctx.ctx_tokens = 0


def cmd_provider(ctx, arg):
    if not arg:
        # offline listing from config (no connection / Slurm spin-up)
        paths = (os.path.expanduser("~/.llm-connections/config.yaml"), CONFIG_FILE)
        catalog = ProviderCatalog.from_paths(paths)
        print(f"{DIM}Providers:{RESET}")
        for p in catalog:
            marker = f" {CYAN}<--- [Active]{RESET}" if p.name == ctx.args.provider else ""
            print(f"  {p.name} ({p.model}){marker}")
        return
    name = arg.strip()
    try:
        ctx.client = LLMConnection.get(name)
        ctx.args.provider = name
        print(f"{DIM}Switched to {ctx.client}{RESET}")
    except KeyError as e:
        print(f"{RED}{e}{RESET}")


def cmd_rebirth(ctx, arg):
    if len(ctx.messages) <= 2:
        print(f"{DIM}Nothing to compress.{RESET}")
        return
    RED_LINE = "\033[31m"
    RED_BG = "\033[41;30m"
    try:
        tw_r = os.get_terminal_size().columns
    except OSError:
        tw_r = 80
    rebirth_text = " Now I must destroy myself to be born again anew... 🧎 "
    visible_len = len(rebirth_text) + 1   # emoji takes 2 columns
    dash_len = tw_r - visible_len
    print(RED_BG + rebirth_text + RESET + RED_LINE + "—" * max(dash_len, 0) + RESET)
    try:
        rebirth_prompt = get_prompt("pipeline_rebirth", "user_prompt")
        ctx.messages.append({"role": "user", "content": rebirth_prompt})
        old_count = len(ctx.messages) - 1
        old_tokens = ctx.ctx_tokens

        # summarize with a dedicated system prompt and no tools so the summary
        # comes back as text, not file_write calls
        rebirth_system = get_prompt("pipeline_rebirth", "system_prompt")
        summarize_messages = [
            {"role": "system", "content": rebirth_system}
        ] + ctx.messages[1:]
        response = ctx.client.chat(summarize_messages, tools=None, stream=True)
        summary_text = ""
        for chunk in response:
            if chunk.text:
                summary_text += chunk.text
        if not summary_text:
            summary_text = response.text
        summary_text = summary_text.strip()

        if not summary_text:
            ctx.messages.pop()  # remove rebirth prompt
            print(f"{RED}Failed to generate summary.{RESET}")
            return

        num_kept = CONFIG.get("pipeline_rebirth", {}).get("num_messages_kept", 5)
        recent_msgs = []
        for msg in reversed(ctx.messages):
            if msg.get("content", "").startswith("Summarize this"):
                continue
            recent_msgs.insert(0, msg)
            if len(recent_msgs) >= num_kept:
                break
        # Drop leading orphan tool results so the window starts on a clean boundary.
        while recent_msgs and recent_msgs[0].get("role") == "tool":
            recent_msgs.pop(0)
        ctx.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Previous conversation summary:\n{summary_text}"},
            {"role": "system", "content": f"The following {len(recent_msgs)} message(s) are the most recent user requests from before the conversation was compressed:"},
        ] + recent_msgs
        ctx.messages = sanitize_messages(ctx.messages)
        ctx.ctx_tokens = 0
        save_history(ctx.messages, ctx.session_file)
        print(f"{DIM}Rebirth complete: {old_count} messages → {len(ctx.messages)}, "
              f"{old_tokens} tokens → {len(summary_text)} chars{RESET}")
        render_markdown(summary_text)
    except Exception as e:
        print(f"{RED}Rebirth failed: {e}{RESET}")


def _prompt_command(token):
    """Build a run() for a canned-prompt command/flag: a commonly-used prompt kept
    in config.yaml under prompt_commands.<token> so users can edit it without code.
    Any arg typed after the token is appended to the configured prompt. The same
    handler serves /<token> (runs in the REPL) and --<token> (one-shot, then exits);
    ctx.interactive decides which."""
    def run(ctx, arg):
        prompt = CONFIG.get("prompt_commands", {}).get(token, "")
        if not prompt:
            print(f"{RED}No prompt configured for '{token}' (prompt_commands.{token} in config.yaml).{RESET}")
            return
        if arg:
            prompt = f"{prompt}\n\n{arg}"
        handle_input(prompt, ctx, interactive=ctx.interactive)
    return run


# THE single list. Each entry has bare `flag_command` token(s); `is_command` and
# `is_flag` decide which surfaces it appears on. /<token> and --<token> are built
# at use-time. /help shows is_command entries; --help shows is_flag entries.
COMMANDS = [
    Command(flag_command=("help",),
            help="",
            run=cmd_help,
            is_flag=False),                 # --help is argparse's builtin
    Command(flag_command=("exit", "quit"),
            help="Exit MatthewCode",
            run=cmd_exit,
            is_flag=False),                 # --exit would be a no-op
    Command(flag_command=("clear",),
            help="Delete all conversation history.",
            run=cmd_clear,
            one_shot=True),
    Command(flag_command=("history",),
            help="Print session history.",
            run=cmd_history,
            one_shot=True),
    Command(flag_command=("rebirth",),
            help="Ask the LLM to summarize the session. Delete all conversation history and replace with the summary.",
            run=cmd_rebirth,
            is_flag=False),
    Command(flag_command=("session",),
            help="Switch to session <arg>. If none exists, create new session named <arg>.",
            run=cmd_session,
            arghint="[<arg>]",
            arg_values=_list_session_names,
            flag_kwargs={"dest": "resume_name", "default": None, "metavar": "<arg>"}),
    Command(flag_command=("session-rename",),
            help="Set this current session's name to <arg>.",
            run=cmd_name,
            arghint="[<arg>]",
            is_flag=False),
    Command(flag_command=("sessions-list",),
            help="List all sessions.",
            run=cmd_session,
            one_shot=True),
    Command(flag_command=("provider",),
            help="Switch to provider <arg>.",
            run=cmd_provider,
            arghint="[<arg>]",
            arg_values=LLMConnection.list_providers,
            flag_kwargs={"default": "default", "metavar": "<arg>"}),
    Command(flag_command=("providers-list",),
            help="List all providers.",
            run=cmd_provider,
            one_shot=True),
    Command(flag_command=("verbose",),
            help="Toogle displaying of detailed tool calls.",
            run=cmd_verbose,
            flag_kwargs={"action": "store_true", "default": CONFIG.get("verbose", False)}),
    Command(flag_command=("kill",),
            help="Terminate the active LLM. Works for local ollama serve or slurm job.",
            run=cmd_kill_model,
            needs_client=True,
            one_shot=True),
    Command(flag_command=("diagnose",),
            help="Provider connection diagnostics.",
            run=cmd_diagnose,
            needs_client=True,
            one_shot=True),
    Command(flag_command=("yes",),
            help="Toggle approval of tool calls.",
            run=cmd_yes,
            flag_kwargs={"action": "store_true", "default": CONFIG.get("auto_approve", True)}),
    Command(flag_command=("prompt",),
            help="Run a single prompt non-interactively then exit.",
            flag_kwargs={"nargs": "?", "const": "-", "default": None, "metavar": "<prompt>"},
            is_command=False),
    # Canned prompts (config.yaml: prompt_commands.<token>). As a flag they run
    # once against the client and exit; as a command they run inside the REPL.
    Command(flag_command=("qcm",),
            help="Quick commit message generated from the current git diff.",
            run=_prompt_command("qcm"),
            needs_client=True,
            one_shot=True),
]
COMMAND_BY_TOKEN = {tok: c for c in COMMANDS if c.is_command for tok in c.flag_command}


def _flag_dest(c):
    """argparse dest for an is_flag entry: explicit flag_kwargs['dest'], else the
    first bare token with dashes normalized (e.g. 'kill-model' → 'kill_model')."""
    return c.flag_kwargs.get("dest") or c.flag_command[0].replace("-", "_")


def try_dispatch_command(user_input, ctx):
    """If `user_input` is a registered /command, run it against `ctx` and return
    True; otherwise return False. This is the ONE dispatch path, shared by the
    interactive REPL and the one-shot --prompt mode — so every command is reachable
    from both, with no per-mode or per-command special-casing to maintain."""
    if not user_input.startswith("/"):
        return False
    head, _, raw_arg = user_input[1:].partition(" ")
    cmd = COMMAND_BY_TOKEN.get(head)
    if cmd is None:
        return False
    cmd.run(ctx, raw_arg.strip())
    return True


def _completer_from_commands():
    """Build the REPL TAB completer from the command registry."""
    return build_slash_completer(
        [f"/{tok}" for c in COMMANDS if c.is_command for tok in c.flag_command],
        arg_value_funcs={
            f"/{tok}": c.arg_values
            for c in COMMANDS if c.is_command and c.arg_values
            for tok in c.flag_command
        },
    )


# --- Shared turn handling (REPL and --prompt run the SAME path) ----------------
# A line of input is processed identically in both modes; `interactive` gates only
# presentation/safety (animation, stdout-vs-stderr, tool confirmation, rendering).


def _execute_tool_calls(calls, ctx, detector, *, interactive, assistant_content):
    """Append the assistant tool_calls message + paired tool results for `calls`
    (a list of (name, args_dict)). Returns True if the turn should stop (a tool was
    rejected). The single tool-execution path, shared by the fallback + native paths
    and both modes — keeps the strict-provider id pairing in exactly one place."""
    ctx.messages.append({
        "role": "assistant",
        "content": assistant_content,
        "tool_calls": [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": name, "arguments": args}}
            for i, (name, args) in enumerate(calls)
        ],
    })
    for i, (name, args) in enumerate(calls):
        call_id = f"call_{i}"
        if interactive:
            if ctx.args.verbose:
                print(f"{DIM}[tool: {name}({json.dumps(args, indent=2)})]{RESET}")
            else:
                print(f"{DIM}[{name}: {_tool_summary(name, args)}]{RESET}")
        else:
            print(f"[{name}: {_tool_summary(name, args)}]", file=sys.stderr)

        # Tool-safety confirmation is interactive-only — non-interactive (--prompt)
        # has no TTY to confirm at, so it auto-executes (caller opted in via --prompt).
        if interactive:
            protected = CONFIG.get("rules", {}).get("protected_paths", [])
            tool_path = args.get("path", "")
            is_safe = name in SAFE_TOOLS and not is_protected_path(tool_path, protected)
            restricted = is_restricted_bash(name, args)
            if restricted or (not is_safe and not ctx.args.yes):
                if not confirm_tool(name, args, restricted=restricted):
                    ctx.messages.append({"role": "tool", "tool_call_id": call_id,
                                         "content": get_prompt("pipeline_tool_rejected", "message")})
                    # answer the remaining declared call ids so the block stays paired
                    for j in range(i + 1, len(calls)):
                        ctx.messages.append({"role": "tool", "tool_call_id": f"call_{j}",
                                             "content": get_prompt("pipeline_tool_skipped", "message")})
                    return True

        if name in TOOL_DISPATCH:
            t0 = time.time()
            result = TOOL_DISPATCH[name](args)
            if interactive:
                rc = len(result)
                print(f"{DIM}  → {rc} chars ({rc // 4} tokens) in {time.time() - t0:.1f}s{RESET}")
        else:
            result = get_prompt("pipeline_tool_errors", "unknown_tool", name=name)

        if detector.record(name, args, result):
            if interactive:
                print(f"{YELLOW}  ⚠ loop detected — nudging model{RESET}")
            result += get_prompt("pipeline_loop_detected", "message", name=name, count=detector.threshold)
            detector.reset()
        ctx.messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

        if interactive and ctx.args.verbose:
            preview = result[:500] + ("..." if len(result) > 500 else "")
            print(f"{DIM}{preview}{RESET}")
    return False


def run_agent_loop(ctx, *, interactive):
    """Run the chat→tools loop for the latest user message in ctx.messages until the
    model produces a final response or max_iters. `interactive` gates only the
    animation, output target, tool confirmation, and rendering."""
    max_iters = CONFIG.get("max_iterations", 10)
    # Occasionally (not every turn) drop a Stoic quote right after the prompt — only
    # interactive, so it never pollutes --prompt's stdout. Gold, 20% of the time.
    if interactive and THINKING_QUOTES and random.random() < 0.20:
        print(f"\n{YELLOW}{random.choice(THINKING_QUOTES)}{RESET}\n")
    detector = make_loop_detector()
    for _iter in range(max_iters):
        try:
            # safety net: repair malformed tool-call pairing before a strict provider
            ctx.messages = sanitize_messages(ctx.messages, quiet=True)
            response = ctx.client.chat(ctx.messages, tools=TOOLS, stream=True)

            full_content = ""
            tool_calls = []
            if interactive:
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
                    if interactive:
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

            if not full_content:
                full_content = response.text
            if not tool_calls:
                tool_calls = response.tool_calls
            if response.prompt_tokens:
                ctx.ctx_tokens = max(ctx.ctx_tokens, response.prompt_tokens)
            if interactive:
                sys.stdout.write("\r\033[K")  # clear the llama animation

            # Fallback text parsing → synthesize tool calls
            if not tool_calls and full_content:
                fallback = parse_tool_calls_from_text(full_content)
                if fallback:
                    if interactive:
                        print(f"{DIM}(parsed tool call from text){RESET}")
                    calls = [(name, a if isinstance(a, dict) else json.loads(a))
                             for name, a in fallback]
                    if _execute_tool_calls(calls, ctx, detector, interactive=interactive,
                                           assistant_content=full_content):
                        break
                    continue

            # No tool calls — final response, done
            if not tool_calls:
                if full_content:
                    if interactive:
                        render_markdown(full_content)
                    else:
                        print(full_content)   # clean stdout for autonomous callers
                    ctx.messages.append({"role": "assistant", "content": full_content})
                save_history(ctx.messages, ctx.session_file)
                break

            # Native tool calls
            calls = [(tc["name"], tc["arguments"] if isinstance(tc["arguments"], dict)
                      else json.loads(tc["arguments"]))
                     for tc in tool_calls]
            if _execute_tool_calls(calls, ctx, detector, interactive=interactive,
                                   assistant_content=(full_content or "")):
                break

        except KeyboardInterrupt:
            if interactive:
                print(f"\n{DIM}(interrupted){RESET}")
            break
        except Exception as e:
            err_str = str(e)
            if interactive:
                print(f"\n{RED}Error: {err_str}{RESET}")
            else:
                print(f"Error: {err_str}", file=sys.stderr)
            if any(code in err_str for code in ("400", "401", "422")):
                # gentle sanitize, then roll back to the last user message
                ctx.messages = sanitize_messages(ctx.messages)
                while len(ctx.messages) > 1 and ctx.messages[-1]["role"] != "user":
                    ctx.messages.pop()
                save_history(ctx.messages, ctx.session_file)
                if interactive:
                    print(f"{DIM}(rolled back to last clean state — try again){RESET}")
            break
    else:
        if interactive:
            print(f"\n{YELLOW}(stopped after {max_iters} iterations){RESET}")
        else:
            print(f"(stopped after {max_iters} iterations)", file=sys.stderr)


def handle_input(user_input, ctx, *, interactive):
    """Process one line of input: dispatch a /command, or run an agent turn. The
    one path shared by the interactive REPL and the one-shot --prompt mode."""
    if try_dispatch_command(user_input, ctx):
        return
    ctx.messages.append({"role": "user",
                         "content": get_prompt("pipeline_main", "user_prompt", user_input=user_input)})
    run_agent_loop(ctx, interactive=interactive)


# --- Terminal banner bar (configurable) ---------------------------------------
# Up to two notifications per side; the repeated sequence fills the bar and
# separates the notifications. Swap the LEFT/RIGHT lists to change what's shown;
# recolour the fill and the notification chips independently.
_TERMINAL_BAR_REPEATED_CHAR = "—"
_TERMINAL_BAR_REPEATED_CHAR_COLOR = "RAINBOW"
_TERMINAL_BAR_NOTIFICATIONS_COLOR = "RAINBOW"
_TERMINAL_BAR_RAINBOW = ["91", "93", "92", "96", "94", "95"]


def _notif_tokens(ctx):
    """Notification: context-token usage, with a threshold colour override."""
    if ctx.ctx_tokens <= 0:
        return None
    num_ctx = getattr(ctx.client, "_provider", None) and ctx.client._provider.config.get("num_ctx", 0) or 0
    text = f"[{ctx.ctx_tokens}/{num_ctx} tokens]" if num_ctx else f"[{ctx.ctx_tokens} tokens]"
    if ctx.ctx_tokens >= 200000:
        return text, "5;41;30"   # blinking red bg, black text
    if ctx.ctx_tokens >= 120000:
        return text, "5;43;30"   # blinking yellow bg, black text
    return text


def _notif_session(ctx):
    """Notification: the open session's name."""
    return ctx.session_name or None


_TERMINAL_BAR_LEFT = []                                 # ≤2 producers (left-to-right)
_TERMINAL_BAR_RIGHT = [_notif_tokens, _notif_session]   # ≤2 producers


def _terminal_width(default=80):
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default


def _colorize(text, color, counter):
    """Colour `text` with `color` — an ANSI SGR code (e.g. "36"), or one of:
    "RAINBOW"        — each char the next colour in _TERMINAL_BAR_RAINBOW, advancing
                       `counter` (a 1-element list) so the cycle flows continuously
                       across the whole bar (fill AND chips);
    "RANDOM_RAINBOW" — (secret) each char a random colour from _TERMINAL_BAR_RAINBOW,
                       duplicates allowed (no cycle, ignores `counter`)."""
    if not text:
        return ""
    if color not in ("RAINBOW", "RANDOM_RAINBOW"):
        return f"\033[{color}m{text}{RESET}"
    out = []
    for ch in text:
        if color == "RANDOM_RAINBOW":
            code = random.choice(_TERMINAL_BAR_RAINBOW)
        else:
            code = _TERMINAL_BAR_RAINBOW[counter[0] % len(_TERMINAL_BAR_RAINBOW)]
            counter[0] += 1
        out.append(f"\033[{code}m{ch}")
    return "".join(out) + RESET


def _bar_fill(n, counter):
    """`n` chars of the repeated sequence, coloured per _TERMINAL_BAR_REPEATED_CHAR_COLOR."""
    if n <= 0:
        return ""
    seq = _TERMINAL_BAR_REPEATED_CHAR
    return _colorize((seq * (n // len(seq) + 1))[:n], _TERMINAL_BAR_REPEATED_CHAR_COLOR, counter)


def render_terminal_bar(ctx, width):
    """The full-width banner bar: up to two notifications per side, each separated
    from the fill by two chars of the repeated sequence; the middle expands to fill.
    One shared counter → RAINBOW (fill and/or chips) flows continuously across the bar."""
    def resolve(notifs):                     # → [(chip_text, colour_override), ...]
        chips = []
        for n in notifs[:2]:
            r = n(ctx)
            if not r:
                continue
            text, override = r if isinstance(r, tuple) else (r, None)
            if text:
                chips.append((f" {text} ", override))
        return chips

    left, right = resolve(_TERMINAL_BAR_LEFT), resolve(_TERMINAL_BAR_RIGHT)
    left_vis = sum(2 + len(t) for t, _ in left)
    right_vis = sum(len(t) + 2 for t, _ in right)
    middle = max(width - left_vis - right_vis, 1)

    counter = [0]
    chip = lambda t, o: _colorize(t, o or _TERMINAL_BAR_NOTIFICATIONS_COLOR, counter)
    out = []
    for text, override in left:              # [fill2 + Lchip] ...
        out.append(_bar_fill(2, counter))
        out.append(chip(text, override))
    out.append(_bar_fill(middle, counter))
    for text, override in right:             # ... [Rchip + fill2]
        out.append(chip(text, override))
        out.append(_bar_fill(2, counter))
    return "".join(out)


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        description="MatthewCode - Matthew Hamilton's personal agentic coding assistant.",
        epilog=_providers_help_text(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,   # we add our own -h/--help so its text is configurable
    )
    # -h/--help: argparse's help action, but the description comes from the registry's
    # `help` command (so it's configurable like every other flag, not hardcoded).
    _help_cmd = COMMAND_BY_TOKEN.get("help")
    parser.add_argument("-h", "--help", action="help",
                        help=_help_cmd.help if _help_cmd else "Show this help and exit")
    # argparse is generated entirely from the one COMMANDS list — every is_flag entry,
    # with `--` added to its bare token(s), using the single `help` /help also shows.
    for c in COMMANDS:
        if c.is_flag:
            kwargs = dict(c.flag_kwargs)
            if c.one_shot:
                # A one-shot flag takes an optional arg, mirroring `/token [arg]`:
                # absent -> None, bare `--token` -> "", `--token foo` -> "foo".
                kwargs.setdefault("nargs", "?")
                kwargs.setdefault("const", "")
                kwargs.setdefault("default", None)
                kwargs.setdefault("metavar", "<arg>")
            parser.add_argument(*[f"--{t}" for t in c.flag_command], help=c.help, **kwargs)
    args = parser.parse_args()

    # Every one-shot `--flag` and --prompt is a "run once and exit" mode — they are
    # mutually exclusive. Collect all that are set and reject if more than one.
    active = [c for c in COMMANDS if c.one_shot and getattr(args, _flag_dest(c)) is not None]
    modes = [f"--{c.flag_command[0]}" for c in active]
    if args.prompt is not None:
        modes.append("--prompt")
    if len(modes) > 1:
        print(f"{RED}[Error] Pick one. These cannot be logically combined: "
              f"{', '.join(modes)}{RESET}", file=sys.stderr)
        sys.exit(2)
    oneshot = active[0] if active else None

    # A one-shot `--flag` runs its matching command once and exits. Offline commands
    # run here, before any provider/Slurm spin-up; client commands run after setup.
    if oneshot is not None and not oneshot.needs_client:
        if args.resume_name:
            _s_name, _s_file = args.resume_name, session_path(args.resume_name)
        else:
            _s_name, _s_file = None, session_path()
        _msgs = load_history(_s_file) if os.path.isfile(_s_file) else []
        _ctx = ReplContext(args=args, console=console, client=None, messages=_msgs,
                           session_name=_s_name, session_file=_s_file, ctx_tokens=0,
                           interactive=False)
        oneshot.run(_ctx, getattr(args, _flag_dest(oneshot)).strip())
        # Persist whatever the command did to the session, so ANY state-mutating
        # one-shot (clear, and future ones) saves automatically — no per-command
        # save needed. Read-only one-shots just re-write the same content (harmless).
        save_history(_ctx.messages, _ctx.session_file)
        sys.exit(0)

    # Load providers after argparse so --help is instant
    load_providers(args.provider)

    # Get LLM connection from registry
    try:
        client = LLMConnection.get(args.provider)
    except KeyError as e:
        print(f"{RED}{e}{RESET}")
        sys.exit(1)

    session_name = None
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

    # Always resume the session for this directory — continuing is the default.
    # (Run /clear for a fresh start.)
    messages = load_history(session_file)
    if messages:
        messages = sanitize_messages(messages)
        if oneshot is not None or args.prompt is not None:
            pass  # one-shot / --prompt: keep stdout clean for the actual output
        elif session_name:
            print(f"{DIM}Resumed session '{session_name}' ({len(messages)} messages){RESET}")
        else:
            print(f"{DIM}Resumed last session ({len(messages)} messages){RESET}")
        # Add a separator so the model treats the next input as a fresh request
        if messages[-1]["role"] != "user":
            messages.append({
                "role": "system",
                "content": get_prompt("pipeline_session_resume", "system_prompt"),
            })
    else:
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
    if args.prompt is None and oneshot is None:
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

    os.makedirs(HISTORY_DIR, exist_ok=True)

    # Pipe mode: read one prompt from stdin, run, output, exit
    if args.prompt is not None:
        args.yes = True  # auto-approve: no TTY to confirm at in --prompt mode
        user_input = sys.stdin.read().strip() if args.prompt == "-" else args.prompt
        if not user_input:
            print("No input received.", file=sys.stderr)
            sys.exit(1)
        # Same path as the REPL: a /command dispatches, anything else runs the agent
        # loop — non-interactively (clean stdout, no animation), then we exit.
        ctx = ReplContext(args=args, console=console, client=client, messages=messages,
                          session_name=session_name, session_file=session_file,
                          ctx_tokens=ctx_tokens, interactive=False)
        handle_input(user_input, ctx, interactive=False)
        save_history(ctx.messages, ctx.session_file)
        sys.exit(0)

    # Client one-shot flags (--diagnose / --kill-model): the model is now set up,
    # so run the command against it and exit without entering the REPL.
    if oneshot is not None:
        _ctx = ReplContext(args=args, console=console, client=client,
                           messages=messages, session_name=session_name,
                           session_file=session_file, ctx_tokens=ctx_tokens,
                           interactive=False)
        oneshot.run(_ctx, getattr(args, _flag_dest(oneshot)).strip())
        save_history(_ctx.messages, _ctx.session_file)
        sys.exit(0)

    pt_history = FileHistory(os.path.join(HISTORY_DIR, "prompt_history"))
    slash_completer = _completer_from_commands()

    ctx = ReplContext(args=args, console=console, client=client, messages=messages,
                      session_name=session_name, session_file=session_file,
                      ctx_tokens=ctx_tokens)
    while True:
        try:
            print(render_terminal_bar(ctx, _terminal_width()))
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

        # The one shared path: dispatch a /command, or run an agent turn — exactly
        # what --prompt does, just interactive=True (animation, confirmation, render).
        handle_input(user_input, ctx, interactive=True)
        if ctx.should_exit:
            break

    save_history(ctx.messages, ctx.session_file)


if __name__ == "__main__":
    main()
