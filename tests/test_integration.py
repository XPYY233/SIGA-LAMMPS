"""End-to-end integration: the whole MVP chain, in order.

This is the test that answers the question the project actually cares about —
does a task go from a specification to real results, with every layer's evidence
attached — rather than whether each unit behaves.

The chain, and what proves each link:

1. **Specification** — a frozen task, so every configuration sees one text.
2. **Generation**   — a workspace exists with an input script.
3. **Validation**   — `adapter/cli.py validate` passes, the same implementation
                      the agent's tool and the stop gate call.
4. **Upload**       — the workspace reaches the cluster, inside the workspace root.
5. **Submission**   — SLURM accepts it and reports a job id.
6. **Execution**    — the job reaches a terminal state and LAMMPS produced a log.
7. **Audit**        — the run leaves a durable record of all of the above.

The live legs skip when the cluster or a LAMMPS build is unavailable, so the
suite stays runnable offline; the structural legs always run.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from adapter.runner import run_local
from adapter.tasks import load_task
from adapter.validator import validate_workspace
from config.loader import load_settings
from hpc import HpcError, open_session

REPO_ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH = REPO_ROOT / "benchmark" / "ground_truth"
TASKS_DIR = REPO_ROOT / "benchmark" / "tasks"

#: The representative task for the integration walk. NVT equilibration is chosen
#: because it runs in seconds locally and on the cluster, so the whole chain can
#: be exercised without spending meaningful quota.
TASK_ID = "nvt_equilibration"


def _settings():
    return load_settings()


def _hpc_available() -> bool:
    settings = _settings()
    if settings.hpc is None or not settings.hpc.partition:
        return False
    try:
        with open_session(settings) as session:
            session.client.check_alive()
        return True
    except (HpcError, Exception):  # noqa: BLE001 - reachability is the question
        return False


requires_hpc = pytest.mark.skipif(not _hpc_available(), reason="cluster not reachable")
requires_lammps = pytest.mark.skipif(
    shutil.which(_settings().lammps.local_bin) is None, reason="no local LAMMPS"
)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A complete workspace, built from the task's reference implementation."""
    task = load_task(TASK_ID, TASKS_DIR)
    assert task.reference is not None
    path = tmp_path_factory.mktemp("integration")
    shutil.copy(GROUND_TRUTH / task.reference, path / "in.test")
    return path


# --------------------------------------------------------------------------- #
# 1-3: specification, generation, validation — always runnable
# --------------------------------------------------------------------------- #


def test_1_specification_is_frozen_and_self_contained() -> None:
    task = load_task(TASK_ID, TASKS_DIR)
    assert task.specification.strip()
    assert task.level3 and task.level4
    # The specification must read as an instruction, since it is passed to the
    # agent verbatim. An earlier version of this assertion looked for the word
    # "run", which the NVT task never uses — it says "report ... every 200
    # steps". Checking for imperative phrasing is the property that matters.
    assert any(w in task.specification.lower()
               for w in ("simulate", "equilibrate", "measure", "perform"))


def test_2_a_workspace_holds_an_input_script(workspace: Path) -> None:
    scripts = [p for p in workspace.glob("in.*") if p.is_file()]
    assert len(scripts) == 1, f"expected exactly one input script, found {scripts}"


def test_3_validation_passes_and_is_reproducible(workspace: Path) -> None:
    """Level 1, and identical on a second run — the audit record depends on it."""
    task = load_task(TASK_ID, TASKS_DIR)
    first = validate_workspace(workspace, task=task)
    second = validate_workspace(workspace, task=task)
    assert first.valid, first.render()
    assert first.to_dict() == second.to_dict(), "validation is not deterministic"


def test_3b_validation_refuses_a_broken_workspace(tmp_path: Path) -> None:
    """The gate must be able to fail, or passing it means nothing."""
    (tmp_path / "in.test").write_text("atom_style atomic\nrun 100\n", encoding="utf-8")
    result = validate_workspace(tmp_path, task=load_task(TASK_ID, TASKS_DIR))
    assert not result.valid
    assert "UNITS_MISSING" in set(result.failure_codes)


# --------------------------------------------------------------------------- #
# 6a: execution locally, before spending cluster quota
# --------------------------------------------------------------------------- #


@requires_lammps
def test_6a_the_workspace_runs_locally(workspace: Path) -> None:
    """Level 2 locally.

    Run before any submission: a workspace that cannot execute on this machine
    will not execute on the cluster, and discovering that locally costs seconds
    where discovering it in the queue costs a job.
    """
    outcome = run_local(workspace, timeout=600)
    assert outcome.ran, f"local run failed: {outcome.error}"
    assert outcome.completed
    assert outcome.facts.get("atoms", 0) > 0


# --------------------------------------------------------------------------- #
# 4-7: the cluster legs
# --------------------------------------------------------------------------- #


@requires_hpc
def test_4_to_7_full_chain_on_the_cluster(workspace: Path) -> None:
    """Upload, submit, wait, and read the result — the MVP chain, once.

    Deliberately one test rather than five: the chain is only meaningful as a
    sequence, and splitting it would let a passing upload hide a submission that
    never produced a log.
    """
    settings = _settings()
    remote = f"pytest-integration-{int(time.time())}"

    with open_session(settings) as session:
        # 4 — upload, confined to the configured workspace.
        session.client.ensure_workspace()
        uploaded = session.client.upload_tree(workspace, remote)
        assert uploaded, "nothing was uploaded"
        assert all(p.startswith(session.client.workspace) for p in uploaded), (
            "an uploaded path escaped the workspace root"
        )

        # 5 — submit. Resources are clamped, so asking for the default is safe.
        record = session.slurm.submit(remote, nodes=1, ntasks=2, walltime="00:05:00")
        assert record.job_id.isdigit(), f"no job id: {record.job_id!r}"
        assert record.facts["clamped"]["ntasks"] <= settings.ceilings.max_ntasks

        # 6 — wait for a terminal state. squeue then sacct, since a finished job
        # leaves the queue entirely.
        state = "unknown"
        for _ in range(60):
            time.sleep(5)
            status = session.slurm.status(record.job_id)
            state = str(status.get("state"))
            if state in {"completed", "failed", "cancelled"}:
                break
        assert state != "unknown", "the job never appeared in the queue or accounting"
        assert state == "completed", f"job ended as {state}: {session.slurm.status(record.job_id)}"

        # The job's own marker, distinct from SLURM's view of it.
        stdout = session.slurm.read_log(record.job_id, "stdout")
        assert "SIGA_JOB=PASS" in stdout.get("text", ""), stdout.get("text", "")[-400:]

        # LAMMPS actually ran, rather than the script merely exiting zero.
        log = session.slurm.read_log(record.job_id, "lammps")
        text = log.get("text", "")
        assert "Total wall time" in text, "LAMMPS never reached the end of its run"
        assert not any(line.startswith("ERROR") for line in text.splitlines())

        listing = session.slurm.read_log(record.job_id, "listing")
        names = {f["name"] for f in listing["files"]}
        assert {"log.lammps", "siga-job.slurm"} <= names, names


@requires_hpc
def test_7_the_submission_is_audited(workspace: Path) -> None:
    """The audit log must let a run be reconstructed after the fact."""
    import json

    settings = _settings()
    audit = settings.repo_root / "benchmark" / "runs" / "hpc-audit.jsonl"
    if not audit.is_file():
        pytest.skip("no audit log yet; another test must submit first")

    entries = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
    submits = [e for e in entries if e.get("action") == "submit"]
    assert submits, "a submission left no audit record"

    latest = submits[-1]
    # Everything needed to reconstruct the submission without the workspace.
    for field in ("job_id", "remote_dir", "submitted_at", "partition", "job_script"):
        assert latest.get(field), f"audit record is missing {field!r}"
    assert "#SBATCH" in latest["job_script"], "the exact rendered job script must be recorded"
    assert latest.get("facts", {}).get("clamped"), "the clamped resources must be recorded"


# --------------------------------------------------------------------------- #
# the web surface, when it is running
# --------------------------------------------------------------------------- #


def _web_available() -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:8090/healthz", timeout=4) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


requires_web = pytest.mark.skipif(not _web_available(), reason="SIGA web app not running")


@requires_web
def test_the_web_surface_reports_both_halves() -> None:
    """/healthz must distinguish its own health from the harness's."""
    import json
    import urllib.request

    with urllib.request.urlopen("http://127.0.0.1:8090/healthz", timeout=10) as response:
        payload = json.loads(response.read())
    assert payload["app"] == "ok"
    assert payload["harness"] == "ok", f"harness not reachable: {payload['harness']}"
    assert set(payload["configurations"]) == {"vanilla", "m", "mr", "mrsx"}


@requires_web
def test_the_web_refuses_a_request_with_nothing_in_it() -> None:
    """An empty form must be refused, not filled in.

    A silent default would generate a simulation nobody asked for, which is the
    failure a researcher is least likely to notice.
    """
    import json
    import urllib.error
    import urllib.request

    body = json.dumps({"request": "  ", "task_id": None, "configuration": "mrsx"}).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:8090/api/runs",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=30)
    assert caught.value.code == 422
    assert "refusing to invent one" in json.loads(caught.value.read())["detail"]


@requires_web
def test_the_web_serves_the_three_areas() -> None:
    """The page must actually contain Areas A, B and C, in Chinese."""
    import urllib.request

    with urllib.request.urlopen("http://127.0.0.1:8090/", timeout=10) as response:
        page = response.read().decode("utf-8")
    assert 'lang="zh-CN"' in page
    for marker in ("模拟需求", "Agent 活动", "模拟作业", "生成模拟", "校验并提交到超算"):
        assert marker in page, f"the page is missing {marker!r}"
    # GitHub dark surface, so the styling is verifiable rather than assumed.
    assert "#0d1117" in page and "#2f81f7" in page
