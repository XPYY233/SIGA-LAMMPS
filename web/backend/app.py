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
    #: The researcher's own words. Without it a free-form run is indistinguishable
    #: from every other free-form run, which makes the history unusable for
    #: finding anything: the task id is empty and the folder name says nothing.
    request: str = ""
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
            "request": self.request,
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
        """Create a run: allocate a workspace, then start a harness session in it.

        A request with no description and no task is refused rather than filled
        in. An earlier version substituted a generic Lennard-Jones prompt, which
        meant an empty form silently generated a simulation nobody asked for —
        the failure a researcher would least expect and last notice.
        """
        if not payload.request.strip() and not payload.task_id:
            raise HTTPException(
                status_code=422,
                detail="a run needs either a description or a benchmark task; refusing to invent one",
            )
        run_id = f"run-{uuid.uuid4().hex[:10]}"
        workspace = resolved.repo_root / "workspace" / run_id
        workspace.mkdir(parents=True, exist_ok=True)

        run = Run(
            run_id=run_id,
            task_id=payload.task_id,
            configuration=payload.configuration,
            workspace=workspace,
            request=payload.request.strip(),
        )
        runs[run_id] = run

        try:
            run.session_id = await harness.create_session(
                agent_preset=payload.configuration, cwd=str(workspace)
            )
            # Written only once the session id exists. The earlier version wrote
            # before the assignment, so the persisted record always said
            # `session_id: null` and a run recovered from disk had no way to have
            # its activity replayed — which is why history could not be reviewed.
            _write_metadata(run)
            await harness.prompt(run.session_id, _compose_request(payload.request, payload.task_id))
        except (HarnessError, HarnessRpcError) as exc:
            # The run is kept rather than discarded: the workspace and the reason
            # for failure are both useful, and losing them would leave a
            # researcher with an empty directory and no explanation.
            run.error = str(exc)
            _write_metadata(run)
            return JSONResponse(status_code=502, content={"run": run.to_dict(), "error": str(exc)})

        asyncio.create_task(_pump_events(run, harness))
        return {"run": run.to_dict()}

    @app.get("/api/runs")
    async def list_runs() -> dict[str, Any]:
        """Every run this deployment knows about, newest first.

        Live runs carry their session and event count; runs recovered from disk
        carry their files and job state. History is a first-class surface, since
        a researcher returning to a task needs the previous attempt, not only the
        one currently in flight.
        """
        merged = dict(runs)
        for discovered in _discover_runs(resolved, runs):
            merged.setdefault(discovered.run_id, discovered)
        ordered = sorted(merged.values(), key=lambda r: r.created_at, reverse=True)
        return {"runs": [r.to_dict() for r in ordered]}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        return {"run": _require(runs, run_id, resolved).to_dict()}

    @app.post("/api/runs/{run_id}/message")
    async def send_message(run_id: str, payload: MessageRequest) -> dict[str, Any]:
        """Area A continued: a follow-up instruction on the same run."""
        run = _require(runs, run_id, resolved)
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
        run = _require(runs, run_id, resolved)
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
        run = _require(runs, run_id, resolved)
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
        run = _require(runs, run_id, resolved)
        await _backfill_events(run, harness, _dsh_home())

        if not run.events:
            # Say why rather than streaming nothing. A viewer left on "正在加载…"
            # cannot tell a missing session from a slow one.
            if not run.session_id:
                run.events.append({
                    "type": "stream/error",
                    "data": {"detail":
                             "该任务没有记录会话 id，无法回放活动流。"
                             "文件与作业状态仍可在右侧查看；活动流只对此后新建的任务可用。"},
                })
            else:
                run.events.append({
                    "type": "stream/error",
                    "data": {"detail":
                             "harness 中已找不到该会话，活动流无法回放。"
                             "文件与作业状态仍可查看。"},
                })

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
        run = _require(runs, run_id, resolved)
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
        run = _require(runs, run_id, resolved)
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
        run = _require(runs, run_id, resolved)
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

        run = _require(runs, run_id, resolved)
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
        _write_metadata(run)
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


RUN_ID_PREFIX = "run-"

#: Run metadata lives beside the generated files.
#:
#: The directory name carries the id and the files carry the result, but neither
#: says which task was requested or which adapter configuration produced it. That
#: is exactly what a researcher returning to a run needs to know, and what an
#: ablation needs in order to attribute a result.
METADATA_NAME = "run.json"


def _write_metadata(run: "Run") -> None:
    """Persist a run's identity. Never raises: metadata is useful, not vital."""
    try:
        (run.workspace / METADATA_NAME).write_text(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "task_id": run.task_id,
                    "configuration": run.configuration,
                    "request": run.request,
                    "session_id": run.session_id,
                    "created_at": run.created_at,
                    "job": run.job,
                    "error": run.error,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def _read_metadata(workspace: Path) -> dict[str, Any]:
    path = workspace / METADATA_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _discover_runs(settings: Settings, known: dict[str, "Run"]) -> list["Run"]:
    """Runs found on disk that this process does not already know about.

    The in-memory registry is a live-session cache, not the record. A workspace
    directory is the durable artefact: it survives a restart, a crash, and the
    browser being closed, and it holds the generated files and logs a researcher
    comes back for. Without this, reloading the page after a backend restart
    showed an empty history even though every run was still on disk.
    """
    root = settings.repo_root / "workspace"
    if not root.is_dir():
        return []
    discovered: list[Run] = []
    for path in sorted(root.iterdir(), reverse=True):
        if not path.is_dir() or not path.name.startswith(RUN_ID_PREFIX):
            continue
        if path.name in known:
            continue
        meta = _read_metadata(path)
        # A run directory predating run.json still has its files and logs, which
        # is most of what a researcher comes back for. Reporting it as unknown
        # rather than hiding it keeps that value.
        discovered.append(
            Run(
                run_id=path.name,
                task_id=meta.get("task_id"),
                request=str(meta.get("request") or ""),
                configuration=str(meta.get("configuration") or "unknown"),
                workspace=path,
                created_at=float(meta.get("created_at") or path.stat().st_mtime),
                job=dict(meta.get("job") or {}),
                error=meta.get("error"),
            )
        )
    return discovered


def _require(runs: dict[str, Run], run_id: str, settings: Settings | None = None) -> Run:
    run = runs.get(run_id)
    if run is None and settings is not None:
        # Rehydrate from disk, so a past run stays reachable after a restart.
        for candidate in _discover_runs(settings, runs):
            if candidate.run_id == run_id:
                runs[run_id] = candidate
                run = candidate
                break
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
            _relay(run, frame.payload.get("event") or {})
    except HarnessError as exc:
        run.events.append({"type": "stream/error", "data": {"detail": str(exc)}})


#: Which SIGA component a tool belongs to, and what it does.
#:
#: Named explicitly because the whole point of the adapter is that these are
#: distinct mechanisms, and a researcher watching a run should be able to see
#: which one acted. "Called a tool" is not informative; "X validated the script
#: and found two ordering errors" is.
_TOOL_ROLES: dict[str, tuple[str, str, str]] = {
    "mcp__lammps__search_lammps": (
        "R", "检索 LAMMPS 文档",
        "在官方文档、示例脚本与命令参考中做语义检索，返回最相关的片段与出处。",
    ),
    "mcp__lammps__validate_lammps_input": (
        "X", "确定性校验",
        "逐条检查命令顺序、units、atom_style、结构初始化、力场、文件引用、ensemble、"
        "timestep、run 与明显冲突，只报结构性结论，不判断物理对错。",
    ),
    "hpc_preflight": ("HPC", "超算连通性预检", "检查 SSH、远端工作区、SLURM 分区与资源上限。"),
    "hpc_upload_workspace": ("HPC", "上传工作区", "把本地工作区传到远端 workspace 根目录下。"),
    "hpc_submit_job": ("HPC", "提交作业", "渲染作业脚本并 sbatch，资源被夹紧到配置上限。"),
    "hpc_job_status": ("HPC", "查询作业状态", "从 squeue 与 sacct 读取作业状态。"),
    "hpc_read_log": ("HPC", "读取作业日志", "读取 log.lammps、SLURM stdout/stderr 或文件列表。"),
    "hpc_cancel_job": ("HPC", "取消作业", "scancel 指定作业。"),
    "bash": ("执行", "运行命令", "在会话工作区内执行 shell 命令。"),
    "write": ("文件", "写入文件", "创建一个新文件。"),
    "edit": ("文件", "修改文件", "对已有文件做定点替换。"),
    "read": ("查看", "读取文件", "读取文件内容。"),
    "glob": ("查看", "查找文件", "按文件名模式查找。"),
    "grep": ("查看", "搜索内容", "在文件内容中做正则搜索。"),
    "todo_write": ("规划", "更新任务清单", "记录并更新当前任务的待办状态。"),
}


def _role_of(name: str) -> tuple[str, str, str]:
    """The component, label and explanation for a tool name."""
    if name in _TOOL_ROLES:
        return _TOOL_ROLES[name]
    if name.startswith("mcp__lammps__"):
        return ("Adapter", name.replace("mcp__lammps__", ""), "调用适配器工具。")
    if name.startswith("mcp__"):
        return ("MCP", name, "调用外部 MCP 工具。")
    return ("工具", name, "调用通用工具。")


#: The plugin that owns the stop gate, and the one that injects runtime context.
#: Distinct because both arrive as plugin-sourced user messages.
STOP_GATE_PLUGIN = "siga-lammps"
SYSTEM_PROMPT_PLUGIN = "@deepseek-ai/dsh-system-prompt"


def _coerce_arguments(arguments: Any) -> dict[str, Any]:
    """Normalise a tool call's arguments to a mapping.

    They arrive as a JSON *string*, not a mapping, because the event log stores
    the wire form. An isinstance check against dict therefore silently produced
    no summary at all, which is exactly the failure this whole surface exists to
    prevent: the console showed "运行命令" with no command.
    """
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
        except ValueError:
            return {"_raw": arguments}
        return parsed if isinstance(parsed, dict) else {"_raw": parsed}
    return {}


def _argument_summary(name: str, arguments: Any) -> str:
    """A one-line, human-readable summary of what a tool was asked to do.

    The arguments are the difference between "called write()" and "wrote
    in.melt", and between "ran a command" and "ran lmp -in in.melt". Without them
    a viewer cannot tell real work from spinning.
    """
    arguments = _coerce_arguments(arguments)
    if not arguments:
        return ""
    for key in ("command", "file_path", "path", "pattern", "query", "workspace", "message"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            text = " ".join(value.split())
            return text[:220] + ("…" if len(text) > 220 else "")
    for key, value in arguments.items():
        if isinstance(value, str) and value.strip():
            return f"{key}={value[:120]}"
    return ""


#: Plain-language explanation of what a validator finding means.
_FINDING_LABELS: dict[str, str] = {
    "UNITS_MISSING": "缺少 units 命令",
    "UNITS_NOT_FIRST": "units 不在第一行",
    "ATOM_STYLE_MISSING": "缺少 atom_style",
    "STRUCTURE_NOT_INITIALISED": "系统从未被创建",
    "PAIR_STYLE_MISSING": "缺少 pair_style",
    "PAIR_COEFF_MISSING": "缺少 pair_coeff",
    "ORDER_VIOLATION": "命令顺序错误",
    "TIMESTEP_MISSING": "缺少 timestep（默认 0.0，动力学无意义）",
    "TIMESTEP_NONPOSITIVE": "timestep 非正",
    "TIMESTEP_TOO_LARGE": "timestep 对该单位制偏大",
    "RUN_MISSING": "从未请求 run",
    "CONFLICT_MULTIPLE_INTEGRATORS": "同一 group 上有多个积分器",
    "CONFLICT_DEFORM_AND_BAROSTAT_SAME_AXIS": "deform 与恒压器控制同一轴",
    "CONFLICT_READ_DATA_WITH_CONSTRUCTION": "read_data 与建盒命令混用",
    "REFERENCED_FILE_MISSING": "引用的文件不存在",
    "POTENTIAL_FILE_MISSING": "势函数文件不存在",
    "TASK_REQUIRED_COMMAND_MISSING": "缺少任务必需的命令",
    "TASK_REQUIRED_PATTERN_MISSING": "缺少任务要求的写法",
}


def _dsh_home() -> Path:
    """The harness home this console starts. Kept beside the workspace, never ~/.dsh."""
    import os

    return Path(os.environ.get("SIGA_HARNESS_HOME") or (REPO_ROOT / ".dsh-web"))


def _relay(run: "Run", event: dict[str, Any]) -> bool:
    """Append one durable event to a run's buffer, if Area B should show it.

    Shared by the live pump and the history backfill so a restored run looks
    exactly like a live one. Two code paths would drift, and the one that only
    runs after a restart is the one nobody notices breaking.
    """
    kind = str(event.get("type", ""))
    if kind not in RELAYED_EVENTS:
        return False
    run.events.append(
        {
            "seq": event.get("seq"),
            "type": kind,
            "at": event.get("time"),
            "turn": event.get("data", {}).get("turn"),
            "step": event.get("data", {}).get("step"),
            "data": _summarise(kind, event.get("data") or {}),
        }
    )
    return True


def _session_events_from_disk(run: "Run", dsh_home: Path) -> list[dict[str, Any]]:
    """Read a run's events straight from its session log.

    The log is the durable source of truth, and reading it needs neither the
    harness to be running nor a session id to have been recorded — which matters,
    because an earlier version persisted the id before it was assigned and so
    could not replay any run created before that was fixed. Matching is by project
    directory, since the harness names it after the session's cwd.
    """
    import subprocess as sp

    root = dsh_home / "sessions"
    if not root.is_dir():
        return []
    candidates = [d for d in root.iterdir() if d.is_dir() and run.run_id in d.name]
    if not candidates:
        return []
    events: list[dict[str, Any]] = []
    for directory in candidates:
        for log in sorted(directory.rglob("session.jsonl*")):
            try:
                if log.suffix == ".zstd":
                    raw = sp.run(["unzstd", "-c", str(log)], capture_output=True).stdout
                    text = raw.decode("utf-8", "replace")
                else:
                    text = log.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
    return events


async def _backfill_events(run: "Run", harness: HarnessClient, dsh_home: Path | None = None) -> None:
    """Populate a run's activity, from the harness or from the log on disk.

    Runs discovered on disk have no live pump, so without this the console sat on
    "正在加载该任务的活动流…" forever. The events were never lost — the session log is
    durable — they simply had no reader.
    """
    if run.events:
        return

    # Prefer the log: it works for every run, needs no live harness, and does not
    # depend on a session id having been recorded.
    if dsh_home is not None:
        for event in _session_events_from_disk(run, dsh_home):
            _relay(run, event)
        if run.events:
            return

    if not run.session_id:
        return
    try:
        payload = await harness.history(run.session_id, max_messages=400)
    except (HarnessError, HarnessRpcError):
        return
    value = payload if isinstance(payload, dict) else {}
    for entry in value.get("events") or []:
        # `session.history` wraps each event, and the wrapper is the transport's
        # shape rather than the event's, so unwrap before reading it.
        event = entry.get("event") if isinstance(entry, dict) and "event" in entry else entry
        if isinstance(event, dict):
            _relay(run, event)


def _result_text(data: dict[str, Any]) -> str:
    """Flatten a tool result to plain text.

    The text sits at ``message.content[].content[].text`` — a tool result block
    carries its own content list. Reading ``data["content"]`` found nothing and
    rendered every result as "(无输出)", which made a busy run look idle.
    """
    parts: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        if isinstance(node.get("text"), str):
            parts.append(node["text"])
        for key in ("content", "message"):
            if key in node:
                walk(node[key])

    walk(data.get("message") if "message" in data else data)
    return "\n".join(part for part in parts if part).strip()


def _summarise(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    """Reduce a durable event to what Area B is allowed to show.

    Tool results are summarised rather than forwarded whole, and reasoning is
    never included. What is added is enough context to tell real work from
    apparent idling: which adapter component acted, what it was asked to do, and
    what came back.
    """
    if kind == "tool/call":
        name = str(data.get("name", ""))
        component, label, why = _role_of(name)
        return {
            "name": name,
            "call_id": data.get("callId"),
            "component": component,
            "label": label,
            "why": why,
            "args": _argument_summary(name, data.get("arguments")),
        }
    if kind == "tool/result":
        text = _result_text(data)
        payload: dict[str, Any] = {
            "call_id": data.get("callId"),
            "is_error": bool(data.get("isError")),
            "preview": text[:400],
        }
        # A validator result is the most informative thing in the stream, so its
        # findings are surfaced by name rather than left as raw JSON.
        if '"valid"' in text:
            try:
                parsed = json.loads(text[text.index("{"):text.rindex("}") + 1])
                counts = parsed.get("counts") or {}
                payload["validation"] = {
                    "valid": parsed.get("valid"),
                    "errors": counts.get("errors", 0),
                    "warnings": counts.get("warnings", 0),
                    "codes": [
                        _FINDING_LABELS.get(f.get("code"), f.get("code"))
                        for f in (parsed.get("errors") or [])[:4]
                    ],
                }
            except (ValueError, TypeError):
                pass
        return payload
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
        plugin = str(source.get("plugin") or "")
        # Only this adapter's own steer is the stop gate. The harness injects
        # runtime context through the same plugin-sourced channel, and labelling
        # that as an S interception would point a viewer at exactly the wrong
        # moment — the one thing S's display exists to get right.
        is_stop_gate = plugin == STOP_GATE_PLUGIN
        is_environment = plugin == SYSTEM_PROMPT_PLUGIN or source.get("kind") == "system"
        if is_stop_gate:
            return {
                "source": plugin,
                "component": "S",
                "label": "停止门控拦截",
                "why": "agent 请求结束本轮，S 校验失败因此拒绝结束，并把结构化错误交回 agent 继续修复。",
                "preview": text[:500],
            }
        if is_environment:
            return {"source": plugin or "system", "component": "环境",
                    "label": "注入运行时上下文", "preview": text[:220]}
        return {
            "source": plugin or source.get("kind") or "user",
            "component": None,
            "label": None,
            "why": None,
            "preview": text[:500],
        }
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
