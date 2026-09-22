"""Layer 5 — HPC execution over SSH/SFTP and SLURM.

`HpcSession` is the entry point the tools use. It gathers everything from
configuration, so no host, account, partition, module or path is hardcoded
anywhere in the codebase, and it refuses to run at all when HPC is not
configured rather than guessing a target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.loader import ConfigError, Settings
from hpc.client import (
    HpcAuthError,
    HpcClient,
    HpcError,
    HpcUnreachableError,
    RemoteFile,
    WorkspaceEscapeError,
)
from hpc.slurm import JobRecord, SlurmClient, SlurmError, render_job_script

__all__ = [
    "HpcAuthError",
    "HpcClient",
    "HpcError",
    "HpcSession",
    "HpcUnreachableError",
    "JobRecord",
    "RemoteFile",
    "SlurmClient",
    "SlurmError",
    "WorkspaceEscapeError",
    "render_job_script",
    "open_session",
]


@dataclass
class HpcSession:
    """A connected HPC session plus the SLURM client bound to it."""

    client: HpcClient
    slurm: SlurmClient

    def __enter__(self) -> HpcSession:
        self.client.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.client.close()

    def preflight(self) -> dict[str, object]:
        """Check the whole path before anything is uploaded.

        Reports authentication failure separately, because on this cluster
        certificate expiry is a monthly routine and a generic connection error
        sends the reader looking for a network fault that is not there.
        """
        host = self.client.check_alive()
        workspace = self.client.ensure_workspace()
        code, out, _err = self.client._exec(["sinfo", "-h", "-p", self.slurm.partition, "-o", "%P|%a|%D"])
        partition_ok = code == 0 and bool(out.strip())
        return {
            "host": self.client.settings.host,
            "landed_on": host,
            "workspace": workspace,
            "workspace_exists": True,
            "partition": self.slurm.partition,
            "partition_available": partition_ok,
            "partition_detail": out.strip()[:120] if partition_ok else "partition not listed by sinfo",
            "ceilings": {
                "max_nodes": self.slurm.ceilings.max_nodes,
                "max_ntasks": self.slurm.ceilings.max_ntasks,
                "max_walltime": self.slurm.ceilings.max_walltime,
            },
            "lammps_module": self.slurm.module,
            "lammps_bin": self.slurm.lammps_bin,
        }


def open_session(settings: Settings, *, audit_path: Path | None = None) -> HpcSession:
    """Build a session from configuration.

    Raises:
        ConfigError: HPC is not configured. Reported rather than degraded,
            because a tool that silently does nothing on an unconfigured
            deployment is indistinguishable from one that is broken.
    """
    hpc = settings.require_hpc()

    # Read the submission parameters from the resolved settings rather than the
    # process environment. An earlier version read os.environ here, which
    # bypassed the loader entirely: a correctly configured .env was loaded into
    # the settings object and then ignored, so the layer reported a configured
    # deployment as unconfigured.
    if not hpc.partition:
        raise ConfigError(
            "SIGA_SLURM_PARTITION is not set. The cluster requires an explicit "
            "partition; guessing one would submit work somewhere nobody chose."
        )

    client = HpcClient(hpc)
    slurm = SlurmClient(
        client,
        hpc,
        settings.ceilings,
        partition=hpc.partition,
        module=hpc.lammps_module,
        lammps_bin=hpc.lammps_bin,
        account=hpc.account,
        audit_path=audit_path or (settings.repo_root / "benchmark" / "runs" / "hpc-audit.jsonl"),
    )
    return HpcSession(client=client, slurm=slurm)
