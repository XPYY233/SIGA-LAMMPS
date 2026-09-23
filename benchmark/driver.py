"""The ablation driver: task × configuration cells, run and recorded.

Each cell is one (task, adapter configuration) pair executed in its own workspace
by its own headless harness process. The controlled-comparison requirement is
that the adapter configuration is the only variable, so everything else is held
fixed here rather than left to the caller:

* the same frozen task specification text, read from the task file;
* the same model and harness;
* the same resource ceilings and the same evaluation pipeline;
* a fresh workspace and an isolated harness home per cell, so no cell can see
  another's files or session.

What varies is exactly one thing: which adapter components the overlay mounts.

A cell is expensive — a real agent run takes minutes — so the driver records
everything needed to interpret a result without re-running it: the workspace, the
session log, token usage, the four evaluation levels, and a failure category.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from adapter.tasks import TaskSpec, load_task, load_tasks
from benchmark.evaluator import Evaluation, evaluate_run

__all__ = ["CellResult", "harness_available", "run_cell", "run_matrix", "CONFIGURATIONS"]

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = Path("/Users/fanjunran/deepseek-harness")

#: The four ablation configurations, weakest grounding first.
CONFIGURATIONS = ("vanilla", "m", "mr", "mrsx")


def harness_available() -> bool:
    """Whether a headless run is possible here at all."""
    return (
        (HARNESS_ROOT / "apps" / "cli" / "src" / "bin.ts").is_file()
        and (HARNESS_ROOT / "node_modules" / ".bin" / "tsx").is_file()
        and bool(os.environ.get("DEEPSEEK_API_KEY") or _credential_from_store())
    )


def _credential_from_store() -> str | None:
    """Read the harness's own credential store, so the driver needs no setup."""
    path = Path.home() / ".dsh" / ".credentials.yaml"
    if not path.is_file():
        return None
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return None
    key = str(data.get("DEEPSEEK_API_KEY", "")).strip()
    return key or None


@dataclass
class CellResult:
    """One (task, configuration) cell and its evidence."""

    task_id: str
    configuration: str
    run_id: str
    workspace: Path
    seconds: float = 0.0
    exit_code: int | None = None
    session_log: Path | None = None
    error: str | None = None
    tool_calls: list[str] = field(default_factory=list)
    gate_blocks: int = 0
    tokens: dict[str, int] = field(default_factory=dict)
    evaluation: Evaluation | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "configuration": self.configuration,
            "run_id": self.run_id,
            "workspace": str(self.workspace),
            "seconds": round(self.seconds, 1),
            "exit_code": self.exit_code,
            "error": self.error,
            "tool_calls": self.tool_calls,
            "tool_call_count": len(self.tool_calls),
            "gate_blocks": self.gate_blocks,
            "tokens": self.tokens,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "session_log": str(self.session_log) if self.session_log else None,
        }


def _read_session_log(home: Path) -> tuple[Path | None, list[dict[str, Any]]]:
    """Decompress and parse the newest session log under an isolated home."""
    import subprocess as sp

    logs = sorted(home.rglob("session.jsonl*"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None, []
    path = logs[-1]
    if path.suffix == ".zstd":
        raw = sp.run(["unzstd", "-c", str(path)], capture_output=True).stdout
        text = raw.decode("utf-8", "replace")
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return path, events


def _summarise_events(events: list[dict[str, Any]]) -> tuple[list[str], int, dict[str, int]]:
    """Tool calls, stop-gate blocks, and token usage from a session log."""
    calls: list[str] = []
    blocks = 0
    tokens: dict[str, int] = {}
    for event in events:
        kind = event.get("type")
        data = event.get("data") or {}
        if kind == "tool/call":
            calls.append(str(data.get("name")))
        elif kind == "user/message":
            source = data.get("source") or {}
            # A plugin-sourced user message is S steering the agent back to work.
            if source.get("plugin"):
                blocks += 1
        elif kind == "assistant/message":
            usage = data.get("usage") or {}
            for key in ("inputTokens", "outputTokens", "totalTokens"):
                if isinstance(usage.get(key), int):
                    tokens[key] = tokens.get(key, 0) + usage[key]
    return calls, blocks, tokens


def run_cell(
    task: TaskSpec | str,
    configuration: str,
    *,
    tasks_dir: Path | str | None = None,
    timeout: int = 1800,
    keep_home: bool = False,
) -> CellResult:
    """Run one cell: specification in, evaluated workspace out.

    Raises:
        ValueError: an unknown configuration was requested, or no harness is
            available to run it.
    """
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"unknown configuration {configuration!r}; expected {CONFIGURATIONS}")
    if not harness_available():
        raise RuntimeError(
            "no headless harness available (needs the deepseek-harness checkout and a "
            "DeepSeek credential)"
        )

    spec = task if isinstance(task, TaskSpec) else load_task(task, tasks_dir or REPO_ROOT / "benchmark" / "tasks")

    stamp = time.strftime("%m%d-%H%M%S")
    run_id = f"{spec.id}-{configuration}-{stamp}"
    root = REPO_ROOT / "benchmark" / "runs" / run_id
    workspace = root / "workspace"
    home = root / "dsh-home"
    workspace.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)

    # One overlay per cell: this is the only thing that differs between cells.
    patch = root / "siga-patch.yml"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "harness" / "make_patch.py"),
         "--preset", configuration, "--mode", "headless",
         "--active-task", spec.id, "--task-id", spec.id,
         "--patch-path", str(patch)],
        check=True, capture_output=True,
    )

    env = {
        **os.environ,
        "DSH_HOME": str(home),
        # The workspace-write sandbox cannot start inside the environment this
        # driver itself runs in, and an escalation request would wait for an
        # approval nothing answers. Set once for the whole matrix so every cell
        # has identical permissions.
        "DSH_PERMISSION_MODE": "danger-full-access",
    }
    if not os.environ.get("DEEPSEEK_API_KEY"):
        key = _credential_from_store()
        if key:
            env["DEEPSEEK_API_KEY"] = key

    result = CellResult(
        task_id=spec.id, configuration=configuration, run_id=run_id, workspace=workspace
    )

    started = time.perf_counter()
    try:
        process = subprocess.run(
            [
                str(HARNESS_ROOT / "node_modules" / ".bin" / "tsx"),
                "--tsconfig", str(HARNESS_ROOT / "tsconfig.json"),
                str(HARNESS_ROOT / "apps" / "cli" / "src" / "bin.ts"),
                "--profile", "headless",
                "--patch", str(patch),
                spec.specification,
            ],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result.exit_code = process.returncode
    except subprocess.TimeoutExpired:
        result.error = f"timed out after {timeout}s"
    except OSError as exc:
        result.error = str(exc)
    result.seconds = time.perf_counter() - started

    log_path, events = _read_session_log(home)
    result.session_log = log_path
    result.tool_calls, result.gate_blocks, result.tokens = _summarise_events(events)

    try:
        result.evaluation = evaluate_run(workspace, spec, timeout=timeout)
    except FileNotFoundError as exc:
        # No input script at all is itself the result, and the most damning one.
        result.error = result.error or f"no workspace produced: {exc}"

    (root / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    if not keep_home:
        shutil.rmtree(home, ignore_errors=True)
    return result


def run_matrix(
    *,
    tasks: tuple[str, ...] | None = None,
    configurations: tuple[str, ...] = CONFIGURATIONS,
    timeout: int = 1800,
) -> list[CellResult]:
    """Every requested cell, in a stable order.

    Sequential on purpose: cells share one API quota and one cluster account, and
    running them concurrently would make token usage and wall-clock time
    unattributable — both of which the brief requires per run.
    """
    specs = tuple(t.id for t in load_tasks(REPO_ROOT / "benchmark" / "tasks"))
    chosen = tasks or specs
    results: list[CellResult] = []
    for task_id in chosen:
        for configuration in configurations:
            results.append(run_cell(task_id, configuration, timeout=timeout))
    return results


def summarise(results: list[CellResult]) -> dict[str, Any]:
    """Aggregate a matrix by configuration.

    Reported as capability (level 1-3 success) beside reliability (how often a
    configuration fails at all), because the brief's question is whether the
    harness already authors LAMMPS capably while failing unreliably — a single
    mean would hide exactly that distinction.
    """
    by_config: dict[str, dict[str, Any]] = {}
    for result in results:
        bucket = by_config.setdefault(
            result.configuration,
            {"cells": 0, "valid": 0, "ran": 0, "compliant": 0, "gate_blocks": 0,
             "tool_calls": 0, "seconds": 0.0, "tokens_in": 0, "tokens_out": 0, "failures": {}},
        )
        bucket["cells"] += 1
        bucket["gate_blocks"] += result.gate_blocks
        bucket["tool_calls"] += len(result.tool_calls)
        bucket["seconds"] += result.seconds
        bucket["tokens_in"] += result.tokens.get("inputTokens", 0)
        bucket["tokens_out"] += result.tokens.get("outputTokens", 0)
        evaluation = result.evaluation
        if evaluation is None:
            bucket["failures"]["no_evaluation"] = bucket["failures"].get("no_evaluation", 0) + 1
            continue
        if evaluation.level1_valid:
            bucket["valid"] += 1
        if evaluation.level2_ran:
            bucket["ran"] += 1
        if evaluation.compliance == 1.0:
            bucket["compliant"] += 1
        category = evaluation.failure_category
        if category not in {"none"}:
            bucket["failures"][category] = bucket["failures"].get(category, 0) + 1

    for bucket in by_config.values():
        cells = max(bucket["cells"], 1)
        bucket["valid_rate"] = round(bucket["valid"] / cells, 3)
        bucket["runtime_rate"] = round(bucket["ran"] / cells, 3)
        bucket["compliance_rate"] = round(bucket["compliant"] / cells, 3)
        bucket["mean_seconds"] = round(bucket["seconds"] / cells, 1)
        bucket["mean_tool_calls"] = round(bucket["tool_calls"] / cells, 1)
    return {"cells": len(results), "by_configuration": by_config}


def _main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the SIGA-LAMMPS ablation matrix.")
    parser.add_argument("--tasks", nargs="*", default=None, help="task ids (default: all)")
    parser.add_argument("--configurations", nargs="*", default=list(CONFIGURATIONS))
    parser.add_argument("--timeout", type=int, default=1800, help="per-cell seconds")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    results = run_matrix(
        tasks=tuple(args.tasks) if args.tasks else None,
        configurations=tuple(args.configurations),
        timeout=args.timeout,
    )
    report = summarise(results)
    out = REPO_ROOT / "benchmark" / "runs" / "summary.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"{'config':10s} {'cells':>5s} {'valid':>6s} {'ran':>5s} {'compliant':>10s} "
              f"{'gate':>5s} {'calls':>6s} {'sec':>7s}")
        for name, b in sorted(report["by_configuration"].items()):
            print(f"{name:10s} {b['cells']:5d} {b['valid_rate']:6.0%} {b['runtime_rate']:5.0%} "
                  f"{b['compliance_rate']:10.0%} {b['gate_blocks']:5d} "
                  f"{b['mean_tool_calls']:6.1f} {b['mean_seconds']:7.0f}")
        print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
