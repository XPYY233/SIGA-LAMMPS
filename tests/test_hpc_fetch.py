"""The return half of the HPC round trip, and the log parsing it feeds.

Two defects are pinned here.

The HPC layer could upload a workspace and submit a job, but had no way to bring
anything back. A run that completed on the cluster therefore left its trajectory
and its log on the cluster, and the console — which lists the local workspace —
showed only the input script. A finished run looked like it had produced nothing.
Worse, the numbers were unreachable, because the thermo table lives in the remote
log and nothing read it.

The log parser stopped at the first thermo table. LAMMPS reprints the header for
every `run` command, so a staged script leaves several tables in one log, and
`temperature_reached` samples the second half of the rows it is given. On a
two-stage melting run it therefore measured the cold equilibration stage and
failed a script that had done exactly what was asked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from benchmark.evaluator import parse_thermo, run_level4
from hpc.client import DEFAULT_MAX_FILE_BYTES, HpcClient, HpcError, RemoteFile

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = REPO_ROOT / "benchmark" / "tasks"


# --------------------------------------------------------------------------- #
# log parsing across every run in a script
# --------------------------------------------------------------------------- #

TWO_STAGE_LOG = """\
LAMMPS (22 Jul 2025 - Update 4)

   Step          Temp          PotEng         KinEng         TotEng         Press
         0   0.1           -6.7733681      0.1497        -6.6236681     -6.1510661
       500   0.10005743    -6.6441244      0.14978597    -6.4943385     -5.3350161

   Step          Temp          PotEng         KinEng         TotEng         Press
       500   1.5           -6.6441244      2.2455        -4.3986244     -4.1555483
      5500   1.4923271     -5.3844414      2.2333333     -3.1511081      1.2345678
"""


def test_parse_thermo_reads_every_run_not_just_the_first() -> None:
    """A staged script's second stage is the experiment; it must not be dropped.

    Reading only the first table made `temperature_reached` — which samples the
    later half of the rows — measure the equilibration stage and fail a correct
    run with "Temp left [0.6, 2] with 0.087".
    """
    columns, rows = parse_thermo(TWO_STAGE_LOG)
    assert columns[0] == "Step"
    assert len(rows) == 4, "both thermo tables should contribute rows"
    # The final row is the end of the second stage, not the end of the first.
    assert rows[-1][1] == pytest.approx(1.4923271)
    assert [r[0] for r in rows] == [0.0, 500.0, 500.0, 5500.0]


def test_parse_thermo_keeps_tables_in_file_order() -> None:
    _columns, rows = parse_thermo(TWO_STAGE_LOG)
    steps = [r[0] for r in rows]
    assert steps == sorted(steps), "rows must stay in the order LAMMPS wrote them"


def test_parse_thermo_ignores_a_table_with_a_different_header() -> None:
    """Merging rows of differing width would corrupt every index-based reader."""
    text = TWO_STAGE_LOG + """
   Step          c_myRDF[1]    c_myRDF[2]
       100   1.0           2.0
"""
    columns, rows = parse_thermo(text)
    assert columns[1] == "Temp"
    assert all(len(r) == len(columns) for r in rows)
    assert len(rows) == 4, "the three-column table must not be merged in"


def test_melting_run_reaches_its_requested_temperature() -> None:
    """The end-to-end consequence, through the real level-4 checks."""
    spec = tuple(yaml.safe_load((TASKS / "lj_melt.yaml").read_text())["level4"])
    outcomes = {o.id: o for o in run_level4(spec, TWO_STAGE_LOG)}
    assert outcomes["temperature_reached"].passed is True
    assert outcomes["no_atom_loss"].passed is True


def test_a_cold_run_is_still_failed() -> None:
    """The fix must not turn the check into one that always passes."""
    cold = TWO_STAGE_LOG.replace("       500   1.5  ", "       500   0.1  ").replace(
        "      5500   1.4923271", "      5500   0.1012345"
    )
    spec = tuple(yaml.safe_load((TASKS / "lj_melt.yaml").read_text())["level4"])
    outcomes = {o.id: o for o in run_level4(spec, cold)}
    assert outcomes["temperature_reached"].passed is False


# --------------------------------------------------------------------------- #
# every level-4 column must be one LAMMPS actually prints
# --------------------------------------------------------------------------- #

#: Columns any standard `thermo_style` prints. `E_pair` is deliberately absent:
#: it is a real keyword, but nothing prints it unless asked, so a check naming it
#: was undecidable on every run rather than merely on bad ones.
STANDARD_THERMO_COLUMNS = {
    "Step", "Temp", "PotEng", "KinEng", "TotEng", "Press", "E_vdwl", "E_coul",
    "Density", "Volume", "Atoms", "E_long", "Pxx", "Pyy", "Pzz", "Lx", "Ly", "Lz",
}


@pytest.mark.parametrize("path", sorted(TASKS.glob("*.yaml")), ids=lambda p: p.stem)
def test_level4_checks_name_columns_a_script_can_be_expected_to_print(path: Path) -> None:
    """A check that can never be decided is a hole in the evaluation, not rigour.

    `energy_finite` named `E_pair` in three of five tasks. No default
    `thermo_style` prints it, so the check returned "needs human review" on every
    run forever — including runs that were entirely correct.
    """
    level4 = yaml.safe_load(path.read_text()).get("level4") or []
    for check in level4:
        for column in check.get("columns") or []:
            assert column in STANDARD_THERMO_COLUMNS, (
                f"{path.stem}:{check['id']} names column {column!r}, which a standard "
                f"thermo_style does not print"
            )


# --------------------------------------------------------------------------- #
# fetching results back
# --------------------------------------------------------------------------- #


class _FakeClient(HpcClient):
    """An HpcClient with the transport replaced, so no cluster is involved."""

    def __init__(self, files: list[tuple[str, int, bool]]) -> None:
        self.workspace = "/remote/ws"  # resolve() confines against this
        self._files = files
        self.downloaded: list[str] = []

    def list_dir(self, subpath: str = "") -> list[RemoteFile]:
        self.resolve(subpath)
        return [RemoteFile(name=n, size=s, is_dir=d) for n, s, d in self._files]

    def download_file(self, subpath: str, local: Path | str, *, max_bytes: int = 0) -> int:
        size = next(s for n, s, _d in self._files if n == subpath.split("/")[-1])
        if size > (max_bytes or DEFAULT_MAX_FILE_BYTES):
            raise HpcError(f"{subpath} is {size} bytes, over the limit")
        self.downloaded.append(subpath)
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_bytes(b"x" * size)
        return size


def test_fetch_results_brings_the_trajectory_home(tmp_path: Path) -> None:
    client = _FakeClient([("in.melt", 100, False), ("liquid.dump", 5000, False), ("log.lammps", 200, False)])
    report = client.fetch_results("run-1", tmp_path)
    names = [f["name"] for f in report["fetched"]]
    assert names == ["liquid.dump", "log.lammps"]
    assert (tmp_path / "liquid.dump").stat().st_size == 5000
    assert report["total_bytes"] == 5200


def test_fetching_one_named_file_leaves_the_rest_on_the_cluster(tmp_path: Path) -> None:
    """The usual case: list the directory, pull back only what was asked for."""
    client = _FakeClient([("liquid.dump", 5000, False), ("solid.dump", 4000, False), ("log.lammps", 200, False)])
    report = client.fetch_results("run-1", tmp_path, only=("solid.dump",))
    assert [f["name"] for f in report["fetched"]] == ["solid.dump"]
    assert not (tmp_path / "liquid.dump").exists()
    assert not (tmp_path / "log.lammps").exists()


def test_an_explicit_request_overrides_the_suffix_filter(tmp_path: Path) -> None:
    """A caller who named the file has already decided it matters."""
    client = _FakeClient([("traj.weird", 300, False)])
    report = client.fetch_results("run-1", tmp_path, only=("traj.weird",))
    assert [f["name"] for f in report["fetched"]] == ["traj.weird"]


def test_a_missing_name_is_reported_rather_than_silently_skipped(tmp_path: Path) -> None:
    """Silence is the failure mode: a file that did not come back unnoticed."""
    client = _FakeClient([("log.lammps", 200, False)])
    report = client.fetch_results("run-1", tmp_path, only=("ghost.dump",))
    assert report["fetched"] == []
    reasons = {s["name"]: s["reason"] for s in report["skipped"]}
    assert "not present" in reasons["ghost.dump"]


def test_an_oversized_file_is_skipped_with_its_size_stated(tmp_path: Path) -> None:
    """A truncated dump is worse than no dump: it looks complete to every reader."""
    client = _FakeClient([("huge.dump", 10_000_000, False)])
    report = client.fetch_results("run-1", tmp_path, max_file_bytes=1000)
    assert report["fetched"] == []
    reason = report["skipped"][0]["reason"]
    assert "10000000" in reason and "1000" in reason
    assert not (tmp_path / "huge.dump").exists()


def test_the_total_budget_stops_a_large_run_filling_the_disk(tmp_path: Path) -> None:
    client = _FakeClient([("a.dump", 600, False), ("b.dump", 600, False)])
    report = client.fetch_results("run-1", tmp_path, max_file_bytes=1000, max_total_bytes=700)
    assert [f["name"] for f in report["fetched"]] == ["a.dump"]
    assert "total budget" in report["skipped"][0]["reason"]


def test_input_and_job_files_are_not_pulled_back(tmp_path: Path) -> None:
    """The input script is already local; fetching it back is pure noise."""
    client = _FakeClient([("in.melt", 100, False), ("siga-job.slurm", 50, False), ("sub.err", 0, False)])
    report = client.fetch_results("run-1", tmp_path)
    assert report["fetched"] == []
    assert report["total_bytes"] == 0


# --------------------------------------------------------------------------- #
# the console's remote listing
# --------------------------------------------------------------------------- #


def test_remote_leaf_takes_the_directory_name_from_an_absolute_path() -> None:
    """`resolve` refuses absolute paths, so the console passes only the leaf."""
    from web.backend.app import _remote_leaf

    class Run:
        job = {"remote_dir": "/dssg/home/acct-X/user/siga-lammps/run-abc123"}

    assert _remote_leaf(Run(), "run-abc123") == "run-abc123"


def test_remote_leaf_falls_back_to_the_run_id() -> None:
    from web.backend.app import _remote_leaf

    class Run:
        job = {}

    assert _remote_leaf(Run(), "run-xyz") == "run-xyz"


@pytest.mark.parametrize("name", ["", ".", "..", "../../etc/passwd", "sub/dir/file.dump", "a\\b.dump"])
def test_remote_preview_refuses_anything_that_is_not_a_plain_filename(name: str) -> None:
    """The name arrives from a query string and must not walk out of the directory."""
    from web.backend.app import _remote_preview

    result = _remote_preview(None, "run-1", name, 100)  # settings unused: rejected first
    assert result["ok"] is False
    assert "plain filename" in result["error"]
