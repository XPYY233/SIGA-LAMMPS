"""SIGA-LAMMPS web backend.

Three surfaces, matching the brief's three areas:

* **A — Simulation Request**: create a run from natural language, with optional
  file uploads, then keep talking to it.
* **B — Agent Activity**: a live event relay, showing tool calls, statuses and
  validator feedback. It relays only what the session log already makes durable —
  `tool/call`, `tool/result`, `agent/status`, `user/message` — and never
  synthesises or reveals hidden reasoning.
* **C — Simulation Job**: the generated files, the remote job state, and the
  LAMMPS log, read from the workspace and the cluster.

**This process is the trust boundary.** The harness's `/api` fence is a
DNS-rebinding fence, explicitly not authentication: its own source reasons that
any caller able to start a session can already run commands as this user. Since
this app proxies that surface, it binds loopback only. Exposing it on a LAN
address without adding real authentication would hand out that capability.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from config.loader import ConfigError, Settings, load_settings
from web.backend.harness import HarnessClient, HarnessError, HarnessRpcError

__all__ = ["Run", "create_app"]

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = REPO_ROOT / "web" / "frontend"

#: Files the browser needs for Area B. Everything else in the log is dropped
#: rather than forwarded, so the UI cannot accidentally surface more than the
#: design principles allow.
RELAYED_EVENTS = {
    "tool/call",
    "tool/result",
    "agent/status",
    "user/message",
    "assistant/message",
    "turn/start",
    "turn/end",
    "step/start",
    "step/end",
}


@dataclass
class Run:
    """One researcher request and everything it produced."""

    run_id: str
    task_id: str | None
    configuration: str
    workspace: Path
    session_id: str | None = None
    created_at: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    job: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "configuration": self.configuration,
            "workspace": str(self.workspace),
            "session_id": self.session_id,
            "created_at": self.created_at,
            "job": self.job,
            "error": self.error,
            "event_count": len(self.events),
        }


class CreateRunRequest(BaseModel):
    """Area A: what a researcher submits."""

    request: str
    task_id: str | None = None
    configuration: str = "mrsx"


class MessageRequest(BaseModel):
    """A follow-up instruction on an existing run."""

    message: str


def create_app(settings: Settings | None = None, harness_url: str | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: resolved configuration; loaded from the repository when omitted.
        harness_url: base URL of a running harness. Read from
            ``SIGA_HARNESS_URL`` when omitted.
    """
    import os

    resolved = settings or load_settings()
    base_url = harness_url or os.environ.get("SIGA_HARNESS_URL", "http://127.0.0.1:3081")
    harness = HarnessClient(base_url)
    runs: dict[str, Run] = {}

    app = FastAPI(title="SIGA-LAMMPS", version="0.1.0")
    app.state.runs = runs
    app.state.harness = harness

    # ------------------------------------------------------------- health --- #

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Report both halves: this app, and the harness it depends on."""
        report: dict[str, Any] = {
            "app": "ok",
            "harness_url": base_url,
            "configurations": list(settings_configurations(resolved)),
        }
        try:
            await harness.rpc("session.list", {})
            report["harness"] = "ok"
        except (HarnessError, HarnessRpcError) as exc:
            report["harness"] = f"unreachable: {exc}"
        return report

    # ---------------------------------------------------------- Area A ----- #

    @app.post("/api/runs")
    async def create_run(payload: CreateRunRequest) -> dict[str, Any]:
        """Create a run: allocate a workspace, then start a harness session in it."""
        run_id = f"run-{uuid.uuid4().hex[:10]}"
        workspace = resolved.repo_root / "workspace" / run_id
        workspace.mkdir(parents=True, exist_ok=True)

        run = Run(
            run_id=run_id,
            task_id=payload.task_id,
            configuration=payload.configuration,
            workspace=workspace,
        )
        runs[run_id] = run

        try:
            run.session_id = await harness.create_session(
                agent_preset=payload.configuration, cwd=str(workspace)
            )
            await harness.prompt(run.session_id, _compose_request(payload.request, payload.task_id))
        except (HarnessError, HarnessRpcError) as exc:
            # The run is kept rather than discarded: the workspace and the reason
            # for failure are both useful, and losing them would leave a
            # researcher with an empty directory and no explanation.
            run.error = str(exc)
            return JSONResponse(status_code=502, content={"run": run.to_dict(), "error": str(exc)})

        asyncio.create_task(_pump_events(run, harness))
        return {"run": run.to_dict()}

    @app.get("/api/runs")
    async def list_runs() -> dict[str, Any]:
        return {"runs": [r.to_dict() for r in runs.values()]}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        return {"run": _require(runs, run_id).to_dict()}

    @app.post("/api/runs/{run_id}/message")
    async def send_message(run_id: str, payload: MessageRequest) -> dict[str, Any]:
        """Area A continued: a follow-up instruction on the same run."""
        run = _require(runs, run_id)
        if not run.session_id:
            raise HTTPException(status_code=409, detail="this run has no live session")
        try:
            await harness.prompt(run.session_id, payload.message)
        except (HarnessError, HarnessRpcError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"accepted": True, "run_id": run_id}

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str) -> dict[str, Any]:
        """Cancel the session, and the remote job when one is known."""
        run = _require(runs, run_id)
        result: dict[str, Any] = {"run_id": run_id, "session_cancelled": False}
        if run.session_id:
            try:
                await harness.cancel(run.session_id)
                result["session_cancelled"] = True
            except (HarnessError, HarnessRpcError) as exc:
                result["session_error"] = str(exc)
        job_id = run.job.get("job_id")
        if job_id:
            result["job"] = await asyncio.to_thread(_cancel_remote_job, resolved, job_id)
        return result

    # ---------------------------------------------------------- Area A files - #

    @app.post("/api/runs/{run_id}/files")
    async def upload_files(run_id: str, files: list[UploadFile]) -> dict[str, Any]:
        """Attach structure, data or potential files to a run's workspace."""
        run = _require(runs, run_id)
        written: list[str] = []
        for upload in files:
            name = Path(upload.filename or "upload").name
            if not name or name.startswith("."):
                continue
            target = run.workspace / name
            target.write_bytes(await upload.read())
            written.append(name)
        return {"run_id": run_id, "written": written, "workspace": str(run.workspace)}

    # ---------------------------------------------------------- Area B ----- #

    @app.get("/api/runs/{run_id}/events")
    async def stream_events(run_id: str) -> StreamingResponse:
        """Live agent activity, as Server-Sent Events to the browser.

        SSE here and WebSocket to the harness: the browser half is ours to
        choose and SSE is simpler for a one-way feed, while the harness offers
        only the socket.
        """
        run = _require(runs, run_id)

        async def generate() -> AsyncIterator[bytes]:
            sent = 0
            idle = 0.0
            while True:
                while sent < len(run.events):
                    payload = json.dumps(run.events[sent])
                    sent += 1
                    yield f"data: {payload}\n\n".encode()
                    idle = 0.0
                await asyncio.sleep(0.25)
                idle += 0.25
                if idle > 15:
                    # A comment line keeps an idle connection visibly alive
                    # without inventing an event.
                    yield b": heartbeat\n\n"
                    idle = 0.0

        return StreamingResponse(generate(), media_type="text/event-stream")

    # ---------------------------------------------------------- Area C ----- #

    @app.get("/api/runs/{run_id}/workspace")
    async def workspace_files(run_id: str) -> dict[str, Any]:
        """The generated files, so a researcher can see what was produced."""
        run = _require(runs, run_id)
        if not run.workspace.is_dir():
            return {"run_id": run_id, "files": []}
        files = []
        for path in sorted(run.workspace.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "name": path.relative_to(run.workspace).as_posix(),
                        "size": path.stat().st_size,
                    }
                )
        return {"run_id": run_id, "workspace": str(run.workspace), "files": files}

    @app.get("/api/runs/{run_id}/file")
    async def read_workspace_file(run_id: str, name: str, max_bytes: int = 200_000) -> dict[str, Any]:
        """Read one generated file.

        The path is resolved and checked to be inside the run's workspace: a
        traversal here would turn a file viewer into an arbitrary file reader.
        """
        run = _require(runs, run_id)
        candidate = (run.workspace / name).resolve()
        root = run.workspace.resolve()
        if candidate != root and root not in candidate.parents:
            raise HTTPException(status_code=400, detail="path escapes the run workspace")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"no such file: {name}")
        data = candidate.read_bytes()[:max_bytes]
        return {
            "run_id": run_id,
            "name": name,
            "text": data.decode("utf-8", "replace"),
            "truncated": candidate.stat().st_size > max_bytes,
        }

    @app.get("/api/runs/{run_id}/job")
    async def job_status(run_id: str) -> dict[str, Any]:
        """Remote job state and the LAMMPS log, read from the cluster."""
        run = _require(runs, run_id)
        job_id = run.job.get("job_id")
        if not job_id:
            return {"run_id": run_id, "state": "not_submitted", "job": run.job}
        status = await asyncio.to_thread(_remote_job_status, resolved, job_id, run.job)
        return {"run_id": run_id, **status}

    @app.post("/api/runs/{run_id}/submit")
    async def submit_job(run_id: str, nodes: int = 1, ntasks: int = 8, walltime: str = "00:30:00"):
        """Validate, then upload and submit the run's workspace.

        Validation runs first and a failure refuses the submission. Submitting a
        workspace the deterministic validator rejects would spend cluster quota
        on a job already known to be structurally wrong.
        """
        from adapter.validator import validate_workspace

        run = _require(runs, run_id)
        try:
            validation = validate_workspace(
                run.workspace,
                supported_atom_styles=resolved.validator.supported_atom_styles,
                supported_unit_styles=resolved.validator.supported_unit_styles,
                max_lines=resolved.validator.max_lines,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=f"nothing to submit: {exc}") from exc

        if not validation.valid:
            return JSONResponse(
                status_code=422,
                content={
                    "run_id": run_id,
                    "submitted": False,
                    "reason": "validation failed; the workspace was not uploaded",
                    "validation": validation.to_dict(),
                },
            )

        result = await asyncio.to_thread(
            _submit_remote, resolved, run.workspace, run_id, nodes, ntasks, walltime
        )
        run.job = result
        return {"run_id": run_id, "validation": validation.to_dict(), **result}

    # ---------------------------------------------------------- frontend --- #

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        page = FRONTEND / "index.html"
        if not page.is_file():
            return HTMLResponse("<h1>SIGA-LAMMPS</h1><p>frontend not found</p>", status_code=500)
        return HTMLResponse(page.read_text(encoding="utf-8"))

    return app


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def settings_configurations(settings: Settings) -> tuple[str, ...]:
    return settings.benchmark.configurations


def _require(runs: dict[str, Run], run_id: str) -> Run:
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    return run


def _compose_request(request: str, task_id: str | None) -> str:
    """The text a session receives.

    A benchmark task's frozen specification is used verbatim so every
    configuration receives identical text; the researcher's own wording is used
    unchanged when no task is named.
    """
    if not task_id:
        return request
    from adapter.tasks import load_task

    settings = load_settings()
    task = load_task(task_id, settings.benchmark.tasks_dir)
    return task.specification


async def _pump_events(run: Run, harness: HarnessClient) -> None:
    """Relay this run's durable events into its buffer.

    Filters by session and by event type, so the UI receives the agent's actions
    rather than its reasoning. A dropped stream is not replayed — the socket
    cannot — so the run's session id is retained for a history reconcile.
    """
    try:
        async for frame in harness.events():
            if frame.kind != "session/event":
                continue
            session_id = frame.payload.get("sessionId")
            if run.session_id and session_id and session_id != run.session_id:
                continue
            event = frame.payload.get("event") or {}
            kind = str(event.get("type", ""))
            if kind not in RELAYED_EVENTS:
                continue
            run.events.append(
                {
                    "seq": event.get("seq"),
                    "type": kind,
                    "at": event.get("time"),
                    "data": _summarise(kind, event.get("data") or {}),
                }
            )
    except HarnessError as exc:
        run.events.append({"type": "stream/error", "data": {"detail": str(exc)}})


def _summarise(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    """Reduce a durable event to what Area B is allowed to show.

    Tool calls and results are summarised rather than forwarded whole: a full
    tool result can be enormous, and the brief asks for status and short action
    summaries, not a transcript of everything the model saw.
    """
    if kind == "tool/call":
        return {"name": data.get("name"), "call_id": data.get("callId")}
    if kind == "tool/result":
        content = data.get("content")
        text = ""
        if isinstance(content, list):
            text = " ".join(
                str(block.get("text", "")) for block in content if isinstance(block, dict)
            )
        return {
            "call_id": data.get("callId"),
            "is_error": bool(data.get("isError")),
            "preview": text[:400],
        }
    if kind == "agent/status":
        return {"status": data.get("status")}
    if kind == "user/message":
        source = data.get("source") or {}
        content = data.get("content")
        text = ""
        if isinstance(content, list):
            text = " ".join(
                str(block.get("text", "")) for block in content if isinstance(block, dict)
            )
        return {"source": source.get("plugin") or source.get("kind") or "user", "preview": text[:400]}
    return {k: v for k, v in data.items() if k in {"turn", "step", "reason"}}


def _submit_remote(
    settings: Settings, workspace: Path, run_id: str, nodes: int, ntasks: int, walltime: str
) -> dict[str, Any]:
    """Upload and submit, synchronously, for ``asyncio.to_thread``."""
    from hpc import HpcError, open_session

    try:
        with open_session(settings) as session:
            remote = session.client.upload_tree(workspace, run_id)
            record = session.slurm.submit(
                run_id, nodes=nodes, ntasks=ntasks, walltime=walltime
            )
            return {
                "submitted": True,
                "job_id": record.job_id,
                "remote_dir": record.remote_dir,
                "uploaded": len(remote),
                "resources": record.facts.get("clamped"),
                "requested": record.facts.get("requested"),
            }
    except (HpcError, ConfigError) as exc:
        return {"submitted": False, "error": str(exc)}


def _remote_job_status(settings: Settings, job_id: str, known: dict[str, Any]) -> dict[str, Any]:
    from hpc import HpcError, open_session

    try:
        with open_session(settings) as session:
            status = session.slurm.status(job_id)
            if known.get("remote_dir"):
                session.slurm.jobs[job_id] = _rehydrate(known, job_id)
            if status.get("state") in {"completed", "failed"}:
                status["log"] = session.slurm.read_log(job_id, "lammps")
            return status
    except (HpcError, ConfigError) as exc:
        return {"state": "unknown", "error": str(exc)}


def _cancel_remote_job(settings: Settings, job_id: str) -> dict[str, Any]:
    from hpc import HpcError, open_session

    try:
        with open_session(settings) as session:
            return session.slurm.cancel(job_id)
    except (HpcError, ConfigError) as exc:
        return {"cancelled": False, "error": str(exc)}


def _rehydrate(known: dict[str, Any], job_id: str):
    """Rebuild the minimum job record the log reader needs after a restart."""
    from hpc.slurm import JobRecord

    return JobRecord(
        job_id=job_id,
        job_name=known.get("job_name", job_id),
        remote_dir=known.get("remote_dir", ""),
        submitted_at=known.get("submitted_at", 0.0),
        nodes=int((known.get("resources") or {}).get("nodes", 1)),
        ntasks=int((known.get("resources") or {}).get("ntasks", 1)),
        walltime=str((known.get("resources") or {}).get("walltime", "00:30:00")),
        script_name="in.melt",
        partition=str(known.get("partition", "")),
    )
