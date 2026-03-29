#!/usr/bin/env python3
"""Test runner for MatthewCode - runs prompts and verifies output files."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_FILE = os.path.join(SCRIPT_DIR, "config", "prompts.json")
MATTHEWCODE = os.path.join(SCRIPT_DIR, "..", "matthewcode.py")

# ANSI colors
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_prompts() -> list:
    with open(PROMPTS_FILE, "r") as f:
        return json.load(f)


def run_matthewcode(prompt: str, model: str, timeout: int = 180) -> tuple:
    """Run MatthewCode with a prompt and return (stdout, stderr, returncode)."""
    result = subprocess.run(
        [sys.executable, MATTHEWCODE, "--model", model, "--yes"],
        input=prompt + "\n",
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout, result.stderr, result.returncode


def check_file_exists(path: str) -> bool:
    return os.path.isfile(path)


def check_file_contents(path: str, expected: dict) -> list:
    """Check that a file contains expected strings. Returns list of failures."""
    failures = []
    try:
        with open(path, "r") as f:
            content = f.read()
    except Exception as e:
        return [f"Could not read {path}: {e}"]

    for s in expected.get("contains", []):
        if s not in content:
            failures.append(f"Missing '{s}' in {os.path.basename(path)}")

    return failures


def run_test(test: dict, model: str, base_tmpdir: str) -> dict:
    """Run a single test case. Returns result dict."""
    test_id = test["id"]
    workdir = os.path.join(base_tmpdir, test_id)
    os.makedirs(workdir, exist_ok=True)

    # Expand {workdir} in prompt and expected paths
    prompt = test["prompt"].replace("{workdir}", workdir)

    print(f"\n{BOLD}{CYAN}[{test_id}]{RESET} {prompt[:80]}...")

    # Run MatthewCode
    start = time.time()
    try:
        stdout, stderr, rc = run_matthewcode(prompt, model)
    except subprocess.TimeoutExpired:
        print(f"  {RED}TIMEOUT{RESET}")
        return {"id": test_id, "status": "timeout", "failures": ["Timed out after 180s"]}
    except Exception as e:
        print(f"  {RED}ERROR: {e}{RESET}")
        return {"id": test_id, "status": "error", "failures": [str(e)]}

    elapsed = time.time() - start
    print(f"  {DIM}Completed in {elapsed:.1f}s{RESET}")

    # Check expected files
    failures = []
    for fpath_template in test.get("expected_files", []):
        fpath = fpath_template.replace("{workdir}", workdir)
        if not check_file_exists(fpath):
            failures.append(f"File missing: {os.path.basename(fpath)}")
            print(f"  {RED}MISSING: {fpath}{RESET}")
        else:
            print(f"  {GREEN}EXISTS:  {os.path.basename(fpath)}{RESET}")

    # Check file contents
    for fpath_template, expected in test.get("expected_contents", {}).items():
        fpath = fpath_template.replace("{workdir}", workdir)
        if not os.path.isfile(fpath):
            continue  # already reported as missing
        content_failures = check_file_contents(fpath, expected)
        for fail in content_failures:
            print(f"  {RED}FAIL: {fail}{RESET}")
        failures.extend(content_failures)

    # Summary
    if not failures:
        print(f"  {GREEN}{BOLD}PASS{RESET}")
        return {"id": test_id, "status": "pass", "failures": [], "time": elapsed}
    else:
        print(f"  {RED}{BOLD}FAIL ({len(failures)} issues){RESET}")
        return {"id": test_id, "status": "fail", "failures": failures, "time": elapsed}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MatthewCode test runner")
    parser.add_argument("--model", default=None, help="Model to test with")
    parser.add_argument("--test", default=None, help="Run specific test by ID")
    parser.add_argument("--keep", action="store_true", help="Keep temp files after test")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout per test (seconds)")
    args = parser.parse_args()

    # Load model from config if not specified
    if args.model is None:
        try:
            import yaml
            config_path = os.path.expanduser("~/.matthewcode/config.yaml")
            with open(config_path) as f:
                config = yaml.safe_load(f)
            args.model = config.get("model", "firefunction-v2")
        except Exception:
            args.model = "firefunction-v2"

    prompts = load_prompts()

    if args.test:
        prompts = [p for p in prompts if p["id"] == args.test]
        if not prompts:
            print(f"{RED}Test '{args.test}' not found{RESET}")
            sys.exit(1)

    # Check Ollama is running
    try:
        import ollama
        ollama.list()
    except Exception:
        print(f"{RED}Ollama is not running. Start with: ollama serve{RESET}")
        sys.exit(1)

    print(f"{BOLD}MatthewCode Test Runner{RESET}")
    print(f"{DIM}Model: {args.model}{RESET}")
    print(f"{DIM}Tests: {len(prompts)}{RESET}")

    # Create temp directory for test outputs
    tmpdir = tempfile.mkdtemp(prefix="matthewcode_test_")
    print(f"{DIM}Workdir: {tmpdir}{RESET}")

    results = []
    for test in prompts:
        result = run_test(test, args.model, tmpdir)
        results.append(result)

    # Final summary
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    errors = sum(1 for r in results if r["status"] in ("timeout", "error"))
    total_time = sum(r.get("time", 0) for r in results)

    print(f"\n{'=' * 50}")
    print(f"{BOLD}Results: {GREEN}{passed} passed{RESET}, ", end="")
    if failed:
        print(f"{RED}{failed} failed{RESET}, ", end="")
    if errors:
        print(f"{YELLOW}{errors} errors{RESET}, ", end="")
    print(f"{DIM}{total_time:.1f}s total{RESET}")

    if not args.keep:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"{DIM}Cleaned up {tmpdir}{RESET}")
    else:
        print(f"{DIM}Kept test files at {tmpdir}{RESET}")

    # Write results to JSON
    results_file = os.path.join(SCRIPT_DIR, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"{DIM}Results saved to {results_file}{RESET}")

    sys.exit(0 if failed == 0 and errors == 0 else 1)


if __name__ == "__main__":
    main()
