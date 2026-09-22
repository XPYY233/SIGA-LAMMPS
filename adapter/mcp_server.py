"""The SIGA-LAMMPS tools, exposed over MCP stdio.

This is how the agent reaches the adapter. The harness spawns this process and
registers each tool as `mcp__<server>__<tool>`, so the adapter stays a separate
Python process and the harness never learns anything about LAMMPS.

Two decisions worth stating:

* **The server loads its own configuration.** The harness does not read our
  `.env` — its own `.env` handling is credentials-only — so relying on it would
  silently produce a server with no corpus path and no index. Instead
  `config.loader` resolves everything from the repository root, which it derives
  from this file's own location and therefore works whatever the cwd is.

* **Errors are returned as structured content, not raised.** A tool that throws
  gives the model an opaque failure; a tool that returns
  ``{"error": ..., "hint": ...}`` lets it decide whether to retry, ask, or take a
  different route. The one exception is a genuinely broken invocation, which is
  a real error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from config.loader import ConfigError, Settings, load_settings

__all__ = ["build_server", "main", "SERVER_NAME"]

SERVER_NAME = "siga-lammps"


def _settings() -> Settings:
    """Load settings on each call, so a config edit does not need a restart."""
    return load_settings()


def build_server() -> MCPServer:
    """Construct the MCP server and register every tool."""
    server = MCPServer(
        name=SERVER_NAME,
        instructions=(
            "Tools for authoring LAMMPS simulations. Use search_lammps for command "
            "syntax and worked examples, and validate_lammps_input to check a "
            "workspace before considering it finished. Validation is deterministic "
            "and its findings are authoritative."
        ),
    )

    # ----------------------------------------------------------------- R --- #
    @server.tool(
        name="search_lammps",
        description=(
            "Search the LAMMPS documentation, example scripts and command reference. "
            "Returns the most relevant passages with their source paths. Call this "
            "before writing a command whose arguments you are not certain of, and "
            "again when validation reports something you do not recognise. Prefer it "
            "over guessing a keyword: LAMMPS does not always error on a wrong "
            "argument, it sometimes silently computes something else."
        ),
    )
    def search_lammps(query: str, k: int = 5, collections: str = "") -> dict[str, Any]:
        """Query the LAMMPS knowledge base.

        Args:
            query: natural language or LAMMPS vocabulary, e.g. "fix deform erate"
                or "mean square displacement".
            k: how many passages to return.
            collections: optional comma-separated subset of examples, docs, syntax.
        """
        from adapter.retrieval.index import IndexNotBuiltError, LammpsIndex

        try:
            settings = _settings()
            index = LammpsIndex(
                settings.retrieval.persist_dir,
                model_cache_dir=settings.retrieval.model_cache_dir,
            )
            wanted = [c.strip() for c in collections.split(",") if c.strip()] or None
            hits = index.search(query, k=k, collections=wanted)
        except IndexNotBuiltError as exc:
            return {
                "error": "index_not_built",
                "detail": str(exc),
                "hint": "The retrieval index has not been built. Report this; do not guess syntax.",
                "results": [],
            }
        except (ValueError, ConfigError) as exc:
            return {"error": "bad_request", "detail": str(exc), "results": []}

        return {
            "query": query,
            "backend": index.backend,
            "count": len(hits),
            "results": [hit.to_result() for hit in hits],
            "note": (
                "Retrieval matches LAMMPS vocabulary well; a paraphrase may return "
                "nothing useful. If these results do not answer the question, say so "
                "rather than inventing syntax."
            ),
        }

    # ----------------------------------------------------------------- X --- #
    @server.tool(
        name="validate_lammps_input",
        description=(
            "Deterministically validate a LAMMPS workspace: required files, command "
            "ordering, units, atom_style, structure initialisation, force field, "
            "referenced files, ensemble, timestep, run, and obvious conflicts. "
            "Call it before considering a task finished. Findings are authoritative "
            "for structure but say nothing about whether the physics is right."
        ),
    )
    def validate_lammps_input(workspace: str = ".", task: str = "") -> dict[str, Any]:
        """Validate the LAMMPS input script in a workspace.

        Args:
            workspace: directory containing the input script. Relative paths
                resolve against the current working directory.
            task: optional benchmark task id to validate against, which adds that
                task's required commands and patterns to the checks.
        """
        from adapter.tasks import TaskSpecError, load_task
        from adapter.validator import validate_workspace

        try:
            settings = _settings()
            path = Path(workspace).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()

            spec = load_task(task, settings.benchmark.tasks_dir) if task else None
            result = validate_workspace(
                path,
                task=spec,
                supported_atom_styles=settings.validator.supported_atom_styles,
                supported_unit_styles=settings.validator.supported_unit_styles,
                max_lines=settings.validator.max_lines,
            )
        except FileNotFoundError as exc:
            # Nothing to validate is not an invalid input.
            return {"error": "no_input_script", "detail": str(exc), "valid": None}
        except TaskSpecError as exc:
            return {"error": "unknown_task", "detail": str(exc), "valid": None}
        except ConfigError as exc:
            return {"error": "config", "detail": str(exc), "valid": None}

        payload = result.to_dict()
        payload["note"] = (
            "Only errors block. Warnings and suggestions are worth reading but do not "
            "make the script invalid."
        )
        return payload

    # --------------------------------------------------------------- HPC --- #
    #
    # There is deliberately no `hpc_run_shell`. The agent cannot express an
    # arbitrary remote command; every operation is a fixed function confined to
    # the configured workspace. That is the confinement guarantee, not a policy
    # the model is asked to respect.

    def _hpc():
        """Open a session, or return a structured error the model can read."""
        from hpc import HpcError, open_session

        try:
            settings = _settings()
            return open_session(settings), None
        except (ConfigError, HpcError) as exc:
            return None, {
                "error": "hpc_unavailable",
                "detail": str(exc),
                "hint": "HPC is not configured or not reachable. Report this; do not retry blindly.",
            }

    @server.tool(
        name="hpc_preflight",
        description=(
            "Check the HPC path before uploading anything: SSH connectivity, the remote "
            "workspace, the SLURM partition, and the resource ceilings in force. Call this "
            "first, and again if a submission fails for a reason that looks environmental."
        ),
    )
    def hpc_preflight() -> dict[str, Any]:
        """Verify connectivity, workspace, partition and ceilings."""
        from hpc import HpcAuthError, HpcError

        session, error = _hpc()
        if error:
            return error
        assert session is not None
        try:
            with session:
                return session.preflight()
        except HpcAuthError as exc:
            return {"error": "hpc_auth", "detail": str(exc), "hint": "Renew the SSH certificate."}
        except HpcError as exc:
            return {"error": "hpc_error", "detail": str(exc)}

    @server.tool(
        name="hpc_upload_workspace",
        description=(
            "Upload a local LAMMPS workspace into the remote scratch area. Files land "
            "under the configured remote workspace only; absolute and escaping paths are "
            "refused. Returns the remote directory to submit."
        ),
    )
    def hpc_upload_workspace(workspace: str, remote_subdir: str = "") -> dict[str, Any]:
        """Upload a workspace directory to the cluster.

        Args:
            workspace: local directory to upload.
            remote_subdir: destination relative to the remote workspace root.
        """
        from hpc import HpcError

        session, error = _hpc()
        if error:
            return error
        assert session is not None
        try:
            with session:
                target = session.client.ensure_workspace()
                if remote_subdir:
                    target = session.client.make_dir(remote_subdir)
                uploaded = session.client.upload_tree(workspace, remote_subdir)
                return {
                    "uploaded": len(uploaded),
                    "remote_dir": target,
                    "files": [Path(p).name for p in uploaded][:50],
                }
        except HpcError as exc:
            return {"error": "hpc_error", "detail": str(exc)}

    @server.tool(
        name="hpc_submit_job",
        description=(
            "Submit the LAMMPS input script in a remote directory through SLURM, and return "
            "a job id immediately. It does not wait for the job. Resources are clamped to "
            "configured ceilings, so asking for more than allowed silently yields the "
            "maximum and the clamped values are reported back."
        ),
    )
    def hpc_submit_job(
        remote_subdir: str,
        nodes: int = 1,
        ntasks: int = 8,
        walltime: str = "00:30:00",
        job_name: str = "",
        script: str = "",
    ) -> dict[str, Any]:
        """Submit a job from an uploaded directory.

        Args:
            remote_subdir: directory relative to the remote workspace root.
            nodes: requested node count, clamped to the configured ceiling.
            ntasks: requested task count, clamped to the configured ceiling.
            walltime: requested wall time as HH:MM:SS, clamped to the ceiling.
            job_name: optional SLURM job name.
            script: optional input script filename when the directory has several.
        """
        from hpc import HpcError

        session, error = _hpc()
        if error:
            return error
        assert session is not None
        try:
            with session:
                record = session.slurm.submit(
                    remote_subdir,
                    job_name=job_name or None,
                    nodes=nodes,
                    ntasks=ntasks,
                    walltime=walltime,
                    script_name=script or None,
                )
                # The registry lives on the session, and the session closes with
                # this block, so it is copied onto the module-level client cache
                # the other tools read.
                _JOB_CACHE[record.job_id] = record.to_dict() | {"job_script": record.job_script}
                payload = record.to_dict()
                payload["note"] = (
                    "Submitted. Poll with hpc_job_status; this call did not wait for the job."
                )
                payload.pop("job_script", None)
                return payload
        except HpcError as exc:
            return {"error": "hpc_error", "detail": str(exc)}

    @server.tool(
        name="hpc_job_status",
        description=(
            "Current state of a submitted job: pending, running, completed, failed or "
            "cancelled, with elapsed time and exit code once it finishes. Cheap to call; "
            "poll it rather than blocking."
        ),
    )
    def hpc_job_status(job_id: str) -> dict[str, Any]:
        """Report a job's state from the queue, falling back to accounting."""
        from hpc import HpcError

        session, error = _hpc()
        if error:
            return error
        assert session is not None
        try:
            with session:
                cached = _JOB_CACHE.get(job_id)
                if cached:
                    session.slurm.jobs[job_id] = _record_from_cache(cached)
                return session.slurm.status(job_id)
        except HpcError as exc:
            return {"error": "hpc_error", "detail": str(exc)}

    @server.tool(
        name="hpc_read_log",
        description=(
            "Read a job's output: the LAMMPS log, the SLURM stdout or stderr, or the "
            "artifact listing. Large files are truncated, and the truncation is stated."
        ),
    )
    def hpc_read_log(job_id: str, which: str = "lammps") -> dict[str, Any]:
        """Read log.lammps, the SLURM output, or the artifact listing.

        Args:
            job_id: the id returned by hpc_submit_job.
            which: 'lammps', 'stdout', 'stderr', or 'listing'.
        """
        from hpc import HpcError

        session, error = _hpc()
        if error:
            return error
        assert session is not None
        try:
            with session:
                cached = _JOB_CACHE.get(job_id)
                if cached:
                    session.slurm.jobs[job_id] = _record_from_cache(cached)
                return session.slurm.read_log(job_id, which)
        except HpcError as exc:
            return {"error": "hpc_error", "detail": str(exc)}

    @server.tool(
        name="hpc_cancel_job",
        description=(
            "Cancel a submitted job. Safe to call on a job that already finished; the "
            "result says whether anything was actually cancelled."
        ),
    )
    def hpc_cancel_job(job_id: str) -> dict[str, Any]:
        """Cancel one job."""
        from hpc import HpcError

        session, error = _hpc()
        if error:
            return error
        assert session is not None
        try:
            with session:
                return session.slurm.cancel(job_id)
        except HpcError as exc:
            return {"error": "hpc_error", "detail": str(exc)}

    return server


#: Jobs submitted by this server process, keyed by SLURM job id.
#:
#: Each tool call opens and closes its own SSH session, so without this the
#: submission record would be lost with the connection and the later status,
#: log and cancel calls would have no idea which directory the job belongs to.
#: The remote workspace remains the durable record; this is a convenience cache.
_JOB_CACHE: dict[str, dict[str, Any]] = {}


def _record_from_cache(cached: dict[str, Any]):
    from hpc.slurm import JobRecord

    return JobRecord(
        job_id=cached["job_id"],
        job_name=cached["job_name"],
        remote_dir=cached["remote_dir"],
        submitted_at=cached["submitted_at"],
        nodes=cached["nodes"],
        ntasks=cached["ntasks"],
        walltime=cached["walltime"],
        script_name=cached["script_name"],
        partition=cached["partition"],
        job_script=cached.get("job_script", ""),
        facts=cached.get("facts", {}),
    )



def main() -> None:
    """Run the server over stdio, as the harness expects."""
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
