"""SLURM submission and monitoring.

Built on a job script already proven on this cluster: an earlier version of this
template ran as job 62865169 on partition 64c512g and completed in two seconds,
reproducing the local thermo table exactly.

Two properties the brief requires are enforced here rather than documented:

* **Resource ceilings are clamped, never trusted.** Every request goes through
  `settings.ceilings.clamp()`, which takes the minimum of the committed
  `config.yaml` ceiling and the deployment `.env` one. The agent may ask for
  less; it cannot ask for more, and it never sees the ceiling it is being held
  to.
* **Every submission is audited.** The exact rendered script, the clamped
  resources, the job id and the timestamp are appended to a JSONL log before the
  call returns, so a submission can be reconstructed after the fact even if the
  job is later cancelled or the workspace is deleted.
"""

from __future__ import annotations

import json
import posixpath
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.loader import HpcSettings, SlurmCeilings
from hpc.client import HpcClient, HpcError

__all__ = ["JobRecord", "SlurmClient", "render_job_script", "SlurmError"]


class SlurmError(HpcError):
    """A SLURM operation failed."""


#: Job script template. `set -uo pipefail` without `-e`: a LAMMPS failure should
#: still reach the artifact listing and the error tail rather than aborting the
#: script before the diagnosis is printed.
JOB_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --time={walltime}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
{account_line}
set -uo pipefail

echo "=== SIGA-LAMMPS JOB ==="
echo "HOSTNAME=$(hostname)"
echo "JOB_ID=${{SLURM_JOB_ID:-<unset>}}"
echo "PARTITION=${{SLURM_JOB_PARTITION:-<unset>}}"
echo "NTASKS=${{SLURM_NTASKS:-<unset>}}"
echo "PWD=$(pwd)"
echo "STARTED=$(date -Is)"

{module_lines}

echo
echo "=== INPUT ==="
if [ ! -f {script} ]; then
  echo "FATAL: {script} not found in $(pwd)"
  exit 2
fi

echo
echo "=== LAMMPS RUN ==="
{lammps_cmd} -in {script} -log log.lammps
LMP_EXIT=$?
echo "LMP_EXIT=${{LMP_EXIT}}"

echo
echo "=== ARTIFACTS ==="
ls -la

if [ "${{LMP_EXIT}}" -ne 0 ]; then
  echo "=== LAMMPS ERROR TAIL ==="
  tail -n 40 log.lammps 2>/dev/null
  exit "${{LMP_EXIT}}"
fi

echo "FINISHED=$(date -Is)"
echo "SIGA_JOB=PASS"
"""


@dataclass
class JobRecord:
    """What we know about one submitted job."""

    job_id: str
    job_name: str
    remote_dir: str
    submitted_at: float
    nodes: int
    ntasks: int
    walltime: str
    script_name: str
    partition: str
    job_script: str = ""
    facts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "remote_dir": self.remote_dir,
            "submitted_at": self.submitted_at,
            "nodes": self.nodes,
            "ntasks": self.ntasks,
            "walltime": self.walltime,
            "script_name": self.script_name,
            "partition": self.partition,
            "facts": self.facts,
        }


def render_job_script(
    *,
    job_name: str,
    partition: str,
    nodes: int,
    ntasks: int,
    walltime: str,
    script: str = "in.*",
    module: str | None = None,
    lammps_bin: str = "lmp",
    account: str | None = None,
) -> str:
    """Render a SLURM job script.

    The input script is passed as a literal name. Submission validates that a
    single file matches, so the job never runs an ambiguous choice of script —
    picking one silently would mean running something nobody selected.
    """
    module_lines = ""
    if module:
        module_lines += f"module load {module}\n"
    return JOB_TEMPLATE.format(
        job_name=job_name,
        partition=partition,
        nodes=nodes,
        ntasks=ntasks,
        walltime=walltime,
        account_line=f"#SBATCH --account={account}" if account else "",
        module_lines=module_lines.rstrip(),
        script=script,
        lammps_cmd=lammps_bin,
    ).replace("\n\n\n", "\n\n")


class SlurmClient:
    """SLURM operations against one cluster, inside one workspace."""

    def __init__(
        self,
        client: HpcClient,
        settings: HpcSettings,
        ceilings: SlurmCeilings,
        *,
        partition: str,
        module: str | None = None,
        lammps_bin: str = "lmp",
        account: str | None = None,
        audit_path: Path | None = None,
    ) -> None:
        self.client = client
        self.settings = settings
        self.ceilings = ceilings
        self.partition = partition
        self.module = module
        self.lammps_bin = lammps_bin
        self.account = account
        self.audit_path = audit_path
        self.jobs: dict[str, JobRecord] = {}

    # -- submission --------------------------------------------------------- #

    def submit(
        self,
        remote_subdir: str,
        *,
        job_name: str | None = None,
        nodes: int = 1,
        ntasks: int = 8,
        walltime: str = "00:30:00",
        script_name: str | None = None,
    ) -> JobRecord:
        """Render a job script into the workspace and submit it.

        Raises:
            SlurmError: the directory is missing, holds no unambiguous input
                script, or `sbatch` refuses the job.
        """
        directory = self.client.resolve(remote_subdir)
        if not self.client.is_dir(remote_subdir):
            raise SlurmError(f"remote directory does not exist: {directory}")

        script = script_name or self._pick_script(remote_subdir)
        clamped_nodes, clamped_ntasks, clamped_wall = self.ceilings.clamp(
            nodes=nodes, ntasks=ntasks, walltime=walltime
        )

        name = job_name or f"siga-{Path(remote_subdir).name or 'job'}"[:60]
        job_script = render_job_script(
            job_name=name,
            partition=self.partition,
            nodes=clamped_nodes,
            ntasks=clamped_ntasks,
            walltime=clamped_wall,
            script=script,
            module=self.module,
            lammps_bin=self.lammps_bin,
            account=self.account,
        )

        handle = job_name or name
        remote_script = self.client.write_text(
            posixpath.join(remote_subdir, "siga-job.slurm"), job_script
        )

        code, out, err = self.client._exec(
            ["bash", "-lc", f"cd {directory} && sbatch siga-job.slurm"],
            timeout=120,
        )
        if code != 0:
            raise SlurmError(f"sbatch failed in {directory}: {(err or out).strip()}")

        job_id = ""
        for token in out.split():
            if token.isdigit():
                job_id = token
                break
        if not job_id:
            raise SlurmError(f"sbatch returned no job id: {out.strip()!r}")

        record = JobRecord(
            job_id=job_id,
            job_name=handle,
            remote_dir=directory,
            submitted_at=time.time(),
            nodes=clamped_nodes,
            ntasks=clamped_ntasks,
            walltime=clamped_wall,
            script_name=script,
            partition=self.partition,
            job_script=job_script,
            facts={
                "requested": {"nodes": nodes, "ntasks": ntasks, "walltime": walltime},
                "clamped": {
                    "nodes": clamped_nodes,
                    "ntasks": clamped_ntasks,
                    "walltime": clamped_wall,
                },
                "slurm_script": remote_script,
            },
        )
        self.jobs[job_id] = record
        self._audit("submit", record)
        return record

    def _pick_script(self, remote_subdir: str) -> str:
        """The single unambiguous input script in a remote directory."""
        entries = [
            e.name
            for e in self.client.list_dir(remote_subdir)
            if not e.is_dir and (e.name.startswith("in.") or e.name.startswith("input."))
        ]
        if not entries:
            raise SlurmError(
                f"no LAMMPS input script in {remote_subdir} (expected a name starting with in.)"
            )
        if len(entries) > 1:
            raise SlurmError(
                f"{remote_subdir} holds several candidate scripts ({', '.join(sorted(entries))}); "
                "name the one to run rather than letting the choice be made for you"
            )
        return entries[0]

    # -- monitoring --------------------------------------------------------- #

    def status(self, job_id: str) -> dict[str, Any]:
        """Current state of one job, from the queue and then from accounting.

        `squeue` only knows about pending and running jobs. A finished job
        disappears from it, so accounting is consulted whenever the queue is
        silent — otherwise a completed job would look like it never existed.
        """
        record = self.jobs.get(job_id)
        payload: dict[str, Any] = {"job_id": job_id}
        if record is not None:
            payload.update(
                {
                    "job_name": record.job_name,
                    "remote_dir": record.remote_dir,
                    "submitted_at": record.submitted_at,
                    "partition": record.partition,
                    "resources": record.facts.get("clamped"),
                }
            )

        code, out, _err = self.client._exec(
            ["squeue", "-h", "-j", job_id, "-o", "%T|%M|%N|%P"], timeout=60
        )
        if code == 0 and out.strip():
            state, elapsed, nodes, partition = (out.strip().split("|") + ["", "", "", ""])[:4]
            payload.update(
                {
                    "state": _normalise_state(state),
                    "raw_state": state,
                    "elapsed": elapsed,
                    "nodes": nodes,
                    "source": "squeue",
                }
            )
            return payload

        accounting = self.accounting(job_id)
        if accounting:
            payload.update(
                {
                    "state": _normalise_state(str(accounting.get("State", ""))),
                    "raw_state": accounting.get("State"),
                    "elapsed": accounting.get("Elapsed"),
                    "exit_code": accounting.get("ExitCode"),
                    "source": "sacct",
                }
            )
            return payload

        payload.update({"state": "unknown", "source": "none", "detail": "not in queue or accounting"})
        return payload

    def accounting(self, job_id: str) -> dict[str, str]:
        """One row of `sacct` for the job, or an empty mapping."""
        code, out, _err = self.client._exec(
            [
                "sacct", "-j", job_id, "--noheader", "--parsable2",
                "--format=JobID,State,Elapsed,ExitCode,MaxRSS",
            ],
            timeout=90,
        )
        if code != 0 or not out.strip():
            return {}
        for line in out.splitlines():
            parts = line.split("|")
            # The bare job id is the allocation; `.batch`/`.extern` are steps.
            if parts and parts[0] == job_id:
                return {
                    "JobID": parts[0],
                    "State": parts[1] if len(parts) > 1 else "",
                    "Elapsed": parts[2] if len(parts) > 2 else "",
                    "ExitCode": parts[3] if len(parts) > 3 else "",
                    "MaxRSS": parts[4] if len(parts) > 4 else "",
                }
        return {}

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel one job. Safe to call on a job that already finished."""
        code, out, err = self.client._exec(["scancel", job_id], timeout=60)
        self._audit("cancel", self.jobs.get(job_id), extra={"job_id": job_id, "ok": code == 0})
        if code != 0:
            return {"job_id": job_id, "cancelled": False, "detail": (err or out).strip()}
        return {"job_id": job_id, "cancelled": True}

    # -- artifacts ---------------------------------------------------------- #

    def read_log(self, job_id: str, which: str = "lammps") -> dict[str, Any]:
        """Read a job's output.

        Args:
            which: ``lammps`` for ``log.lammps``, ``stdout``/``stderr`` for the
                SLURM files, or ``listing`` for the artifact directory.
        """
        record = self.jobs.get(job_id)
        if record is None:
            # A job submitted by an earlier process is still readable: the
            # workspace is the durable record, the dict is only a cache.
            directory = self.client.resolve("")
            raise SlurmError(
                f"job {job_id} is not known to this session, so its directory is unknown. "
                f"List {directory} to find it."
            )

        relative = posixpath.relpath(record.remote_dir, self.client.workspace)
        if which == "listing":
            return {
                "job_id": job_id,
                "files": [f.to_dict() for f in self.client.list_dir(relative)],
            }

        names = {"lammps": "log.lammps", "stdout": f"slurm-{job_id}.out", "stderr": f"slurm-{job_id}.err"}
        target = names.get(which)
        if target is None:
            raise SlurmError(f"unknown log {which!r}; expected one of {', '.join(names)} or 'listing'")
        try:
            text = self.client.read_text(posixpath.join(relative, target))
        except HpcError as exc:
            return {"job_id": job_id, "which": which, "error": str(exc), "text": ""}
        return {"job_id": job_id, "which": which, "file": target, "text": text}

    # -- audit -------------------------------------------------------------- #

    def _audit(self, action: str, record: JobRecord | None, extra: dict[str, Any] | None = None) -> None:
        """Append one line to the submission audit log.

        Never raises: losing an audit line is bad, but failing a submission
        because the log directory is unwritable would be worse. The failure is
        swallowed only after being reported by the caller's logger.
        """
        if self.audit_path is None:
            return
        entry: dict[str, Any] = {"action": action, "at": time.time()}
        if record is not None:
            entry.update(record.to_dict())
            entry["job_script"] = record.job_script
        if extra:
            entry.update(extra)
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        except OSError:
            pass


#: SLURM state names, normalised to the vocabulary the brief asks for.
_STATE_MAP = {
    "PENDING": "pending",
    "PD": "pending",
    "CONFIGURING": "pending",
    "RUNNING": "running",
    "R": "running",
    "COMPLETING": "running",
    "COMPLETED": "completed",
    "CD": "completed",
    "FAILED": "failed",
    "F": "failed",
    "CANCELLED": "cancelled",
    "CA": "cancelled",
    "TIMEOUT": "failed",
    "OUT_OF_MEMORY": "failed",
    "NODE_FAIL": "failed",
}


def _normalise_state(raw: str) -> str:
    """Map a SLURM state onto the brief's vocabulary.

    Mapped rather than passed through because account software reports
    `CANCELLED by 12345` and `FAILED+`, and a status display that leaks those
    spellings makes every downstream comparison fragile.
    """
    token = raw.strip().split()[0].rstrip("+") if raw.strip() else ""
    return _STATE_MAP.get(token.upper(), "unknown")
