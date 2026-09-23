"""Tests for the HPC layer.

The confinement tests are the important ones and need no network: the guarantee
that the agent cannot reach outside the configured remote workspace is enforced
by path resolution, so it can be tested directly. The live tests are skipped
unless HPC is configured and reachable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.loader import ConfigError, HpcSettings, SlurmCeilings, load_settings
from hpc.client import HpcClient, WorkspaceEscapeError
from hpc.slurm import SlurmError, render_job_script

REPO_ROOT = Path(__file__).resolve().parent.parent


def _settings() -> HpcSettings:
    return HpcSettings(host="probe", user="u", port=22, key_path=None, workspace="/scratch/me/siga")


# --------------------------------------------------------------------------- #
# confinement — the reason this module exists
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "subpath",
    ["a", "a/b", "./a", "a/../b", "", "deep/nested/path"],
)
def test_relative_paths_stay_inside_the_workspace(subpath: str) -> None:
    client = HpcClient(_settings())
    resolved = client.resolve(subpath)
    assert resolved.startswith("/scratch/me/siga")


@pytest.mark.parametrize(
    "subpath",
    [
        "/etc/passwd",
        "../outside",
        "../../etc",
        "a/../../outside",
        "..",
        "./../..",
        "a/b/../../../outside",
    ],
)
def test_escaping_paths_are_refused(subpath: str) -> None:
    """`..` and absolute paths must be rejected, not quietly corrected.

    Clamping to the workspace would hide that the caller asked for something it
    should not have, and a silent correction is indistinguishable from success.
    """
    client = HpcClient(_settings())
    with pytest.raises(WorkspaceEscapeError):
        client.resolve(subpath)


def test_the_workspace_root_itself_resolves() -> None:
    assert HpcClient(_settings()).resolve("") == "/scratch/me/siga"


def test_a_sibling_with_a_shared_prefix_is_not_confused_for_a_child() -> None:
    """`/scratch/me/siga-evil` must not pass as inside `/scratch/me/siga`.

    A plain string prefix check would accept it, which is why the separator is
    part of the comparison.
    """
    client = HpcClient(_settings())
    with pytest.raises(WorkspaceEscapeError):
        client.resolve("../siga-evil/x")


# --------------------------------------------------------------------------- #
# no arbitrary remote shell
# --------------------------------------------------------------------------- #


def test_the_client_exposes_no_arbitrary_command_method() -> None:
    """The confinement story is structural, so it can be asserted.

    Every remote operation is a fixed shape composed inside the client. If a
    pass-through method were ever added, the agent would gain an unrestricted
    remote shell and this test would fail.
    """
    forbidden = {"exec", "execute", "run", "shell", "command", "exec_command", "system", "popen"}
    public = {name for name in dir(HpcClient) if not name.startswith("_")}
    assert not (public & forbidden), f"unexpected pass-through methods: {sorted(public & forbidden)}"
    # The private escape hatch exists for internal use and is named as such.
    assert "_exec" in dir(HpcClient)


# --------------------------------------------------------------------------- #
# job scripts
# --------------------------------------------------------------------------- #


def test_rendered_job_script_carries_the_essentials() -> None:
    script = render_job_script(
        job_name="probe",
        partition="64c512g",
        nodes=1,
        ntasks=4,
        walltime="00:10:00",
        script="in.melt",
        module="lammps/20250722",
        lammps_bin="lmp",
    )
    assert "#SBATCH --partition=64c512g" in script
    assert "#SBATCH --ntasks=4" in script
    assert "#SBATCH --time=00:10:00" in script
    assert "module load lammps/20250722" in script
    assert "lmp -in in.melt" in script
    # A success marker the collector can look for, distinct from LAMMPS' own.
    assert "SIGA_JOB=PASS" in script


def test_job_script_omits_module_line_when_unset() -> None:
    script = render_job_script(
        job_name="p", partition="cpu", nodes=1, ntasks=1, walltime="00:05:00", module=None
    )
    assert "module load" not in script


def test_job_script_guards_against_a_missing_input() -> None:
    """A job that cannot find its script should say so, not fail obscurely."""
    script = render_job_script(
        job_name="p", partition="cpu", nodes=1, ntasks=1, walltime="00:05:00", script="in.x"
    )
    assert "not found in $(pwd)" in script


def test_account_line_only_appears_when_configured() -> None:
    without = render_job_script(
        job_name="p", partition="cpu", nodes=1, ntasks=1, walltime="00:05:00"
    )
    with_account = render_job_script(
        job_name="p", partition="cpu", nodes=1, ntasks=1, walltime="00:05:00", account="acct-1"
    )
    assert "--account" not in without
    assert "#SBATCH --account=acct-1" in with_account


# --------------------------------------------------------------------------- #
# ceilings
# --------------------------------------------------------------------------- #


def test_ceilings_clamp_a_request_above_the_limit() -> None:
    ceilings = SlurmCeilings(max_nodes=1, max_ntasks=64, max_walltime="02:00:00")
    assert ceilings.clamp(nodes=99, ntasks=999, walltime="48:00:00") == (1, 64, "02:00:00")


def test_ceilings_allow_a_smaller_request() -> None:
    ceilings = SlurmCeilings(max_nodes=4, max_ntasks=64, max_walltime="02:00:00")
    assert ceilings.clamp(nodes=2, ntasks=8, walltime="00:10:00") == (2, 8, "00:10:00")


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def test_submission_parameters_come_from_settings_not_the_environment() -> None:
    """The bug this pins: reading os.environ bypassed the loader.

    `.env` values were loaded into the settings object and then ignored, so a
    correctly configured deployment reported itself as unconfigured — and the
    error blamed a missing partition that was in fact present.
    """
    settings = load_settings()
    if settings.hpc is None:
        pytest.skip("HPC not configured in this checkout")
    assert settings.hpc.partition, "partition must be resolved from .env by the loader"
    assert settings.hpc.lammps_bin


def test_hpc_is_optional_until_used() -> None:
    settings = load_settings()
    if settings.hpc is not None:
        pytest.skip("this checkout has HPC configured")
    with pytest.raises(ConfigError, match="HPC is not configured"):
        settings.require_hpc()


def test_open_session_refuses_without_a_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    """No partition means no submission: guessing one sends work somewhere unchosen."""
    from config.loader import Settings
    from hpc import open_session

    settings = load_settings()
    if settings.hpc is None:
        pytest.skip("HPC not configured in this checkout")
    stripped = Settings(
        memory=settings.memory,
        validator=settings.validator,
        retrieval=settings.retrieval,
        lammps=settings.lammps,
        benchmark=settings.benchmark,
        ceilings=settings.ceilings,
        hpc=HpcSettings(
            host=settings.hpc.host,
            user=settings.hpc.user,
            port=settings.hpc.port,
            key_path=settings.hpc.key_path,
            workspace=settings.hpc.workspace,
            partition=None,
        ),
        model=settings.model,
        repo_root=settings.repo_root,
    )
    with pytest.raises(ConfigError, match="SIGA_SLURM_PARTITION"):
        open_session(stripped)


# --------------------------------------------------------------------------- #
# live — skipped unless the cluster is reachable
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_preflight_against_the_real_cluster() -> None:
    from hpc import HpcError, open_session

    settings = load_settings()
    if settings.hpc is None:
        pytest.skip("HPC not configured in this checkout")
    try:
        with open_session(settings) as session:
            report = session.preflight()
    except HpcError as exc:
        pytest.skip(f"cluster not reachable: {exc}")

    assert report["workspace_exists"] is True
    assert report["landed_on"], "preflight must report which login node answered"
    assert report["ceilings"]["max_ntasks"] >= 1
