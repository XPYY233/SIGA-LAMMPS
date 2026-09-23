"""Local LAMMPS execution — Level 2 runtime validation.

Level 1 (the validator) reads structure. This runs the simulator, which is the
only way to learn whether a script actually works. The two are genuinely
different, and the difference is not hypothetical: the nanoindentation reference
passed structural validation cleanly while LAMMPS refused to run it, twice, for
two unrelated physical reasons that no amount of parsing could have found.

Runtime success is deliberately judged on LAMMPS's own error protocol rather than
on searching for phrases, because **LAMMPS echoes the input script into its log**.
A comment that merely mentions a failure mode will appear in the log, so a
substring search reports a failure that did not happen. An earlier version of
this check did exactly that.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["RunOutcome", "run_local", "logs_show_success"]

#: LAMMPS prints these and exits non-zero on a fatal problem.
ERROR_PREFIX = "ERROR"

#: Printed only by a run that reached the end of its `run` command.
COMPLETION_MARKER = "Total wall time"


@dataclass
class RunOutcome:
    """What happened when LAMMPS was asked to run a script."""

    ran: bool
    exit_code: int
    seconds: float
    log_path: Path | None = None
    #: The first ERROR line, when the run failed. LAMMPS states the cause here.
    error: str | None = None
    #: False when LAMMPS never reached the end of its run, whatever the exit code.
    completed: bool = False
    facts: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "ran": self.ran,
            "completed": self.completed,
            "exit_code": self.exit_code,
            "seconds": round(self.seconds, 3),
            "error": self.error,
            "facts": self.facts,
        }


def logs_show_success(text: str, returncode: int) -> tuple[bool, str | None]:
    """Whether a LAMMPS log represents a successful run.

    Returns ``(ok, error)``. Deliberately does **not** search for phrases like
    "Lost atoms": the log contains an echo of the input script, so a comment
    mentioning an error would be read as one.
    """
    if returncode != 0:
        for line in text.splitlines():
            if line.startswith(ERROR_PREFIX):
                return False, line.strip()
        return False, f"LAMMPS exited {returncode} without an ERROR line"
    if COMPLETION_MARKER not in text:
        return False, f"no '{COMPLETION_MARKER}' in the log; the run did not finish"
    for line in text.splitlines():
        if line.startswith(ERROR_PREFIX):
            return False, line.strip()
    return True, None


def run_local(
    workspace: Path | str,
    *,
    script: str | None = None,
    lammps_bin: str = "lmp",
    timeout: int = 900,
    log_name: str = "log.lammps",
) -> RunOutcome:
    """Run LAMMPS on the input script in *workspace*.

    Args:
        workspace: directory holding the script and any referenced files.
        script: script filename. Defaults to the single best candidate.
        lammps_bin: the LAMMPS executable.
        timeout: wall-clock limit in seconds.
        log_name: where LAMMPS writes its log.

    Returns:
        A :class:`RunOutcome`. A failed run is an outcome, not an exception: the
        evaluator needs to record *how* it failed.

    Raises:
        FileNotFoundError: the workspace or script is missing, which means there
            was nothing to run — a different condition from a failing run.
    """
    from adapter.validator.engine import find_input_script

    workspace = Path(workspace)
    if not workspace.is_dir():
        raise FileNotFoundError(f"workspace not found: {workspace}")

    if script is None:
        found = find_input_script(workspace)
        if found is None:
            raise FileNotFoundError(f"no LAMMPS input script in {workspace}")
        script = found.name

    log_path = workspace / log_name
    started = time.perf_counter()
    try:
        process = subprocess.run(
            [lammps_bin, "-in", script, "-log", log_name, "-screen", "none"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"LAMMPS executable {lammps_bin!r} not found. Set SIGA_LAMMPS_LOCAL_BIN in .env."
        ) from exc
    except subprocess.TimeoutExpired:
        return RunOutcome(
            ran=False,
            exit_code=-1,
            seconds=float(timeout),
            log_path=log_path if log_path.is_file() else None,
            error=f"timed out after {timeout}s",
        )

    elapsed = time.perf_counter() - started
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    ok, error = logs_show_success(text, process.returncode)

    return RunOutcome(
        ran=ok,
        exit_code=process.returncode,
        seconds=elapsed,
        log_path=log_path if log_path.is_file() else None,
        error=error,
        completed=COMPLETION_MARKER in text,
        facts=_extract_facts(text),
    )


def _extract_facts(text: str) -> dict[str, object]:
    """Cheap observables from the log, for the evaluator and the UI.

    Only what LAMMPS prints unconditionally: an atom count and the step count.
    Anything richer depends on what the script chose to output.
    """
    facts: dict[str, object] = {}
    for line in text.splitlines():
        if line.startswith("Created ") and "atoms" in line:
            parts = line.split()
            if parts[1].isdigit():
                facts["atoms"] = int(parts[1])
        if line.startswith("Loop time of"):
            parts = line.split()
            for index, token in enumerate(parts):
                if token == "for" and index + 2 < len(parts):
                    facts["steps"] = parts[index + 1]
    return facts
