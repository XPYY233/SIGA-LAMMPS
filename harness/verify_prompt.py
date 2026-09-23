#!/usr/bin/env python3
"""Verify that M actually reaches the model request.

Unit tests prove the primer's *content* is correct. They cannot prove it is
*injected*: that depends on the harness composing the plugin, registering the
prompt section, and interpolating the task variable. The only honest check is to
run a real session and read the assembled system prompt out of the session log.

This is the check that caught a live bug — a patch setting `activeTask: ''`
rendered ``Active task:`` with nothing after it, because `??` does not fall back
on an empty string. No unit test could have seen that.

Costs one small model call. Requires a DeepSeek credential.

Usage::

    python harness/verify_prompt.py
    python harness/verify_prompt.py --task lj_melt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_ROOT = Path("/Users/fanjunran/deepseek-harness")
DEV_HOME = REPO_ROOT / ".dsh-dev"
PATCH = REPO_ROOT / "harness" / "siga-patch.yml"
CREDENTIALS = Path.home() / ".dsh" / ".credentials.yaml"

PROBE = "Reply with exactly the word OK and nothing else."

#: Distinctive phrases from the primer. Each must survive into the request.
PRIMER_MARKERS = {
    "heading": "LAMMPS input-script primer",
    "ordering rule": "must precede every command",
    "units table": "kcal/mol",
    "pitfall: LJ density": "reduced number density",
    "pitfall: SLLOD": "sllod",
    "MSD syntax": "com yes",
    "task patterns": "nanoindentation",
    "retrieval hint": "search_lammps",
}


def normalise(text: str) -> str:
    """Lowercase and strip Markdown emphasis, for marker matching.

    Markers must survive formatting. The primer writes ``reduced *number
    density*`` and ``**Nanoindentation**``; matching raw bytes would report
    missing content that is plainly present, and a check that cries wolf gets
    ignored — which is worse than not having it.
    """
    return re.sub(r"[*`_]", "", text).lower()


def load_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    if CREDENTIALS.is_file():
        import yaml

        data = yaml.safe_load(CREDENTIALS.read_text(encoding="utf-8")) or {}
        key = str(data.get("DEEPSEEK_API_KEY", "")).strip()
    if not key:
        sys.exit(
            "error: no DeepSeek credential. Set DEEPSEEK_API_KEY, or provide "
            f"{CREDENTIALS}."
        )
    return key


def regenerate_patch(active_task: str | None) -> None:
    cmd = [sys.executable, str(REPO_ROOT / "harness" / "make_patch.py")]
    if active_task:
        cmd += ["--active-task", active_task]
    subprocess.run(cmd, check=True, capture_output=True)


def run_probe() -> None:
    if not HARNESS_ROOT.is_dir():
        sys.exit(f"error: harness checkout not found at {HARNESS_ROOT}")
    env = {**os.environ, "DSH_HOME": str(DEV_HOME), "DEEPSEEK_API_KEY": load_api_key()}
    shutil.rmtree(DEV_HOME / "sessions", ignore_errors=True)
    result = subprocess.run(
        ["pnpm", "dsh", "--profile", "headless", "--patch", str(PATCH), PROBE],
        cwd=HARNESS_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-2000:], file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        sys.exit(f"error: harness run failed (exit {result.returncode})")


def newest_session_log() -> Path:
    logs = sorted((DEV_HOME / "sessions").rglob("session.jsonl.zstd"), key=lambda p: p.stat().st_mtime)
    if not logs:
        sys.exit(f"error: no session log under {DEV_HOME / 'sessions'}")
    return logs[-1]


def assemble_system_prompt(log: Path) -> str:
    """Return the `header.system` of the first model request in *log*."""
    raw = subprocess.run(["unzstd", "-c", str(log)], capture_output=True, check=True).stdout
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "request/header":
            system = event.get("data", {}).get("header", {}).get("system")
            if isinstance(system, str):
                return system
    sys.exit("error: no request/header event in the session log")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default=None, help="bind an active benchmark task id")
    parser.add_argument("--skip-run", action="store_true", help="reuse the newest existing session log")
    args = parser.parse_args(argv)

    if not args.skip_run:
        regenerate_patch(args.task)
        run_probe()

    system = assemble_system_prompt(newest_session_log())
    print(f"assembled system prompt: {len(system)} chars (~{len(system) // 4} tokens)\n")

    failures: list[str] = []
    haystack = normalise(system)
    for label, needle in PRIMER_MARKERS.items():
        ok = normalise(needle) in haystack
        print(f"  {'OK  ' if ok else 'FAIL'} {label}")
        if not ok:
            failures.append(label)

    match = re.search(r"Active task:\*\*\s*(.*)", system)
    value = match.group(1).strip() if match else ""
    print(f"\n  Active task interpolated: {value!r}")
    if not value:
        failures.append("active task rendered empty")

    unresolved = re.findall(r"\{\{[^}]*\}\}", system)
    print(f"  Unresolved '{{{{...}}}}'    : {unresolved or 'none'}")
    if unresolved:
        failures.append(f"unresolved variables: {unresolved}")

    if args.task and args.task not in value:
        failures.append(f"bound task {args.task!r} did not reach the prompt")

    print()
    if failures:
        print(f"M VERIFICATION FAILED: {failures}")
        return 1
    print("M VERIFICATION PASSED — the primer is in the request and interpolated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
