"""Tests for X — the deterministic validator.

Two properties matter most and are tested hardest:

1. **No false positives on valid scripts.** A validator that rejects a
   correct-but-unusual script is worse than one that misses a defect, because it
   blocks the agent on work it actually completed.
2. **Unknown is not failure.** With no workspace, or with a variable in a
   filename, the validator reports that it could not check — a suggestion, never
   an error. Manufacturing a failure out of ignorance would corrupt both the
   gate and the benchmark.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adapter.tasks import load_task
from adapter.validator import validate_script, validate_workspace
from adapter.validator.engine import find_input_script
from adapter.validator.findings import ValidationResult
from adapter.validator.script import parse_script

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "benchmark" / "tasks"

#: A minimal script that is genuinely correct. Every rule must leave it alone.
VALID_SCRIPT = """\
units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
velocity all create 1.2 87287 loop geom
fix 1 all nve
timestep 0.005
thermo 100
run 2000
"""


def codes(result: ValidationResult) -> set[str]:
    return {f.code for f in result.findings}


def errors(result: ValidationResult) -> set[str]:
    return set(result.failure_codes)


# --------------------------------------------------------------------------- #
# the parser
# --------------------------------------------------------------------------- #


def test_parser_skips_blank_lines_and_comments() -> None:
    script = parse_script("# a comment\n\n   \nunits lj  # trailing\n")
    assert script.keywords() == ["units"]
    assert script.commands[0].line == 4


def test_parser_joins_continuation_lines() -> None:
    script = parse_script("fix 1 all nvt temp &\n  1.0 1.0 0.5\n")
    assert len(script.commands) == 1
    assert script.commands[0].args == ("1", "all", "nvt", "temp", "1.0", "1.0", "0.5")


def test_parser_records_the_first_line_of_a_continued_command() -> None:
    script = parse_script("units lj\nfix 1 all nvt &\n temp 1 1 1\n")
    assert script.commands[1].line == 2


def test_parser_lowercases_keywords_but_keeps_the_raw_form() -> None:
    script = parse_script("Units lj\n")
    assert script.commands[0].keyword == "units"
    assert script.commands[0].raw_keyword == "Units"


def test_parser_detects_control_flow() -> None:
    script = parse_script("variable x equal 1\nif ${x} == 1 then 'jump SELF'\n")
    assert script.has_control_flow
    assert "if" in script.control_flow


def test_parser_detects_variable_references_in_arguments() -> None:
    script = parse_script("read_data ${datafile}\n")
    assert script.commands[0].has_variable


def test_parser_records_variable_definitions() -> None:
    script = parse_script("variable temp equal 1.2\n")
    assert script.variables == {"temp": "equal"}


def test_parser_skips_a_bare_variable_expansion_command() -> None:
    """Its meaning is unknown, so recording it as a keyword would be a lie."""
    script = parse_script("${command}\nunits lj\n")
    assert script.keywords() == ["units"]


# --------------------------------------------------------------------------- #
# no false positives
# --------------------------------------------------------------------------- #


def test_a_correct_script_passes_clean() -> None:
    result = validate_script(VALID_SCRIPT)
    assert result.valid, f"false positives: {result.render()}"
    assert not result.errors


def test_a_correct_script_passes_with_a_workspace(tmp_path: Path) -> None:
    (tmp_path / "in.test").write_text(VALID_SCRIPT, encoding="utf-8")
    result = validate_workspace(tmp_path)
    assert result.valid, f"false positives: {result.render()}"


def test_read_data_script_passes_clean(tmp_path: Path) -> None:
    """The file-based construction path must be as acceptable as the lattice one."""
    (tmp_path / "system.data").write_text("LAMMPS data\n", encoding="utf-8")
    script = """\
units metal
atom_style atomic
read_data system.data
pair_style eam/alloy
pair_coeff * * Cu.eam.alloy Cu
timestep 0.001
fix 1 all nve
thermo 100
run 1000
"""
    (tmp_path / "Cu.eam.alloy").write_text("potential placeholder\n", encoding="utf-8")
    (tmp_path / "in.test").write_text(script, encoding="utf-8")
    result = validate_workspace(tmp_path)
    assert result.valid, f"false positives: {result.render()}"


# --------------------------------------------------------------------------- #
# units, style, structure
# --------------------------------------------------------------------------- #


def test_missing_units_is_an_error() -> None:
    assert "UNITS_MISSING" in errors(validate_script("atom_style atomic\nrun 10\n"))


def test_units_not_first_warns_but_does_not_fail() -> None:
    """It is very likely wrong, but a script can legitimately put a comment first."""
    result = validate_script("atom_style atomic\nunits lj\nrun 10\n")
    assert "UNITS_NOT_FIRST" in codes(result)
    assert "UNITS_NOT_FIRST" not in errors(result)


def test_unsupported_units_style_warns_not_fails() -> None:
    """An unusual-but-valid style must not be rejected."""
    result = validate_script("units electron\natom_style atomic\nrun 10\n", supported_unit_styles=("lj", "metal"))
    assert "UNITS_UNSUPPORTED" in codes(result)
    assert "UNITS_UNSUPPORTED" not in errors(result)


def test_missing_atom_style_is_an_error_only_when_atoms_are_made() -> None:
    assert "ATOM_STYLE_MISSING" in errors(validate_script("units lj\nregion b block 0 1 0 1 0 1\ncreate_box 1 b\nrun 0\n"))
    assert "ATOM_STYLE_MISSING" not in codes(validate_script("units lj\nrun 0\n"))


def test_structure_not_initialised_is_an_error() -> None:
    script = "units lj\natom_style atomic\npair_style lj/cut 2.5\npair_coeff 1 1 1 1\nfix 1 all nve\ntimestep 0.005\nrun 100\n"
    assert "STRUCTURE_NOT_INITIALISED" in errors(validate_script(script))


def test_ordering_violation_is_reported_with_boundaries() -> None:
    script = "units lj\natom_style atomic\nlattice fcc 0.8\ncreate_box 1 box\nregion box block 0 4 0 4 0 4\nrun 0\n"
    result = validate_script(script)
    assert "ORDER_VIOLATION" in errors(result)


def test_read_data_alongside_create_box_is_a_conflict() -> None:
    script = "units lj\natom_style atomic\nread_data x.data\nregion b block 0 1 0 1 0 1\ncreate_box 1 b\nrun 0\n"
    assert "CONFLICT_READ_DATA_WITH_CONSTRUCTION" in errors(validate_script(script))


# --------------------------------------------------------------------------- #
# force field and files
# --------------------------------------------------------------------------- #


def test_missing_pair_style_is_an_error() -> None:
    assert "PAIR_STYLE_MISSING" in errors(validate_script(VALID_SCRIPT.replace("pair_style lj/cut 2.5\n", "")))


def test_missing_pair_coeff_is_an_error() -> None:
    assert "PAIR_COEFF_MISSING" in errors(validate_script(VALID_SCRIPT.replace("pair_coeff 1 1 1.0 1.0 2.5\n", "")))


def test_missing_potential_file_is_an_error(tmp_path: Path) -> None:
    script = "units metal\natom_style atomic\nlattice fcc 3.6\nregion b block 0 2 0 2 0 2\ncreate_box 1 b\ncreate_atoms 1 b\npair_style eam/alloy\npair_coeff * * absent.eam Cu\nfix 1 all nve\ntimestep 0.001\nrun 100\n"
    (tmp_path / "in.test").write_text(script, encoding="utf-8")
    assert "POTENTIAL_FILE_MISSING" in errors(validate_workspace(tmp_path))


def test_existing_potential_file_is_accepted(tmp_path: Path) -> None:
    script = "units metal\natom_style atomic\nlattice fcc 3.6\nregion b block 0 2 0 2 0 2\ncreate_box 1 b\ncreate_atoms 1 b\npair_style eam/alloy\npair_coeff * * present.eam Cu\nfix 1 all nve\ntimestep 0.001\nrun 100\n"
    (tmp_path / "in.test").write_text(script, encoding="utf-8")
    (tmp_path / "present.eam").write_text("x\n", encoding="utf-8")
    assert validate_workspace(tmp_path).valid


def test_missing_referenced_data_file_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "in.test").write_text("units lj\natom_style atomic\nread_data absent.data\nrun 0\n", encoding="utf-8")
    assert "REFERENCED_FILE_MISSING" in errors(validate_workspace(tmp_path))


# --------------------------------------------------------------------------- #
# unknown is not failure
# --------------------------------------------------------------------------- #


def test_file_references_are_unverified_without_a_workspace() -> None:
    result = validate_script("units lj\natom_style atomic\nread_data absent.data\nrun 0\n")
    assert "REFERENCED_FILE_MISSING" not in errors(result)
    assert "REFERENCED_FILE_MISSING" not in codes(result)


def test_variable_filenames_are_unverified_not_missing(tmp_path: Path) -> None:
    """A `${var}` filename cannot be resolved statically, so it is not a failure.

    The script is otherwise complete on purpose: an earlier version of this test
    used a minimal stub and failed for unrelated reasons, which would have hidden
    whether the variable handling was right.
    """
    script = (
        "units lj\n"
        "atom_style atomic\n"
        "read_data ${file}\n"
        "pair_style lj/cut 2.5\n"
        "pair_coeff 1 1 1.0 1.0 2.5\n"
        "fix 1 all nve\n"
        "timestep 0.005\n"
        "run 100\n"
    )
    (tmp_path / "in.test").write_text(script, encoding="utf-8")
    result = validate_workspace(tmp_path)
    assert "REFERENCED_FILE_MISSING" not in errors(result)
    assert result.valid, f"a variable filename must not fail validation: {result.render()}"


def test_variable_timestep_is_a_suggestion_not_an_error() -> None:
    script = "units lj\natom_style atomic\nlattice fcc 0.8\nregion b block 0 2 0 2 0 2\ncreate_box 1 b\ncreate_atoms 1 b\npair_style lj/cut 2.5\npair_coeff 1 1 1 1\ntimestep ${dt}\nrun 100\n"
    result = validate_script(script)
    assert "TIMESTEP_UNVERIFIED" in codes(result)
    assert "TIMESTEP_UNVERIFIED" not in errors(result)


def test_control_flow_downgrades_ordering_to_a_warning() -> None:
    """Written order is not executed order, so the validator must not assert."""
    script = "variable x equal 0\nif ${x} == 0 then 'units lj'\natom_style atomic\nlattice fcc 0.8\ncreate_box 1 box\nregion box block 0 4 0 4 0 4\nrun 0\n"
    result = validate_script(script)
    assert "ORDER_VIOLATION_CONTROL_FLOW" in codes(result)
    assert "ORDER_VIOLATION_CONTROL_FLOW" not in errors(result)


# --------------------------------------------------------------------------- #
# ensemble, timestep, run
# --------------------------------------------------------------------------- #


def test_multiple_integrators_on_one_group_is_an_error() -> None:
    script = VALID_SCRIPT.replace("fix 1 all nve\n", "fix 1 all nve\nfix 2 all nvt temp 1.0 1.0 0.5\n")
    assert "CONFLICT_MULTIPLE_INTEGRATORS" in errors(validate_script(script))


def test_two_integrators_on_different_groups_are_fine() -> None:
    script = VALID_SCRIPT.replace(
        "fix 1 all nve\n",
        "group mobile id 1:100\nfix 1 all nve\nfix 2 mobile nvt temp 1.0 1.0 0.5\n",
    )
    assert "CONFLICT_MULTIPLE_INTEGRATORS" not in codes(validate_script(script))


def test_missing_integrator_warns_but_does_not_fail() -> None:
    result = validate_script(VALID_SCRIPT.replace("fix 1 all nve\n", ""))
    assert "ENSEMBLE_MISSING" in codes(result)
    assert "ENSEMBLE_MISSING" not in errors(result)


@pytest.mark.parametrize("value", ["0.0", "-0.005"])
def test_nonpositive_timestep_is_an_error(value: str) -> None:
    assert "TIMESTEP_NONPOSITIVE" in errors(validate_script(VALID_SCRIPT.replace("timestep 0.005", f"timestep {value}")))


def test_missing_timestep_is_an_error() -> None:
    """The 0.0 default silently makes dynamics meaningless — the classic trap."""
    assert "TIMESTEP_MISSING" in errors(validate_script(VALID_SCRIPT.replace("timestep 0.005\n", "")))


def test_oversized_timestep_warns_for_the_unit_system() -> None:
    result = validate_script(VALID_SCRIPT.replace("timestep 0.005", "timestep 0.05"))
    assert "TIMESTEP_TOO_LARGE" in codes(result)
    assert "TIMESTEP_TOO_LARGE" not in errors(result)


def test_missing_run_is_an_error() -> None:
    assert "RUN_MISSING" in errors(validate_script(VALID_SCRIPT.replace("run 2000\n", "")))


def test_run_zero_is_a_suggestion_not_an_error() -> None:
    result = validate_script(VALID_SCRIPT.replace("run 2000", "run 0"))
    assert "RUN_ZERO_STEPS" in codes(result)
    assert "RUN_ZERO_STEPS" not in errors(result)


# --------------------------------------------------------------------------- #
# conflicts
# --------------------------------------------------------------------------- #


def test_deform_and_barostat_on_the_same_axis_is_an_error() -> None:
    """`npt` is itself the integrator here, so there is exactly one.

    An earlier version added a separate `nvt`, which tripped the
    multiple-integrator rule instead and masked the conflict being tested.
    """
    script = VALID_SCRIPT.replace(
        "fix 1 all nve\n",
        "fix 1 all npt temp 0.5 0.5 0.5 z 0 0 1.0\nfix 2 all deform 1 z erate 0.001\n",
    )
    assert "CONFLICT_MULTIPLE_INTEGRATORS" not in codes(validate_script(script))
    assert "CONFLICT_DEFORM_AND_BAROSTAT_SAME_AXIS" in errors(validate_script(script))


def test_barostat_on_a_different_axis_is_accepted() -> None:
    """The correct pattern: deform z, couple the barostat to x and y only."""
    script = VALID_SCRIPT.replace(
        "fix 1 all nve\n",
        "fix 1 all npt temp 0.5 0.5 0.5 x 0 0 1.0 y 0 0 1.0\nfix 2 all deform 1 z erate 0.001\n",
    )
    assert "CONFLICT_DEFORM_AND_BAROSTAT_SAME_AXIS" not in codes(validate_script(script))


def test_compute_used_before_defined_is_an_error() -> None:
    script = VALID_SCRIPT.replace("run 2000\n", "thermo_style custom step c_msd[4]\ncompute msd all msd com yes\nrun 2000\n")
    assert "COMPUTE_USED_BEFORE_DEFINED" in errors(validate_script(script))


def test_fix_id_redefined_without_unfix_warns() -> None:
    script = VALID_SCRIPT.replace("fix 1 all nve\n", "fix 1 all nve\nfix 1 all nvt temp 1 1 1\n")
    assert "FIX_ID_REDEFINED_WITHOUT_UNFIX" in codes(validate_script(script))


# --------------------------------------------------------------------------- #
# task requirements
# --------------------------------------------------------------------------- #


def test_task_required_command_is_reported_including_which_task() -> None:
    task = load_task("lj_melt", TASKS_DIR)
    result = validate_script("units lj\natom_style atomic\nrun 0\n", task=task)
    missing = [f for f in result.errors if f.code == "TASK_REQUIRED_COMMAND_MISSING"]
    assert missing
    assert missing[0].evidence["task"] == "lj_melt"


def test_task_pattern_failure_explains_itself() -> None:
    """A bare pattern name would not tell the agent what to write instead."""
    task = load_task("lj_melt", TASKS_DIR)
    result = validate_script("units metal\natom_style atomic\nrun 0\n", task=task)
    pattern_errors = [f for f in result.errors if f.code == "TASK_REQUIRED_PATTERN_MISSING"]
    assert pattern_errors
    assert pattern_errors[0].message.strip()
    assert "expected a line matching" in pattern_errors[0].message


def test_the_reference_solution_for_every_task_passes(tmp_path: Path) -> None:
    """The strongest available check: the tasks must be satisfiable.

    A task whose own reference cannot pass is unsatisfiable, and would fail every
    agent for a reason no adapter could fix.
    """
    for task_id in ("lj_melt", "nvt_equilibration", "msd_diffusion", "uniaxial_tension", "nanoindentation"):
        task = load_task(task_id, TASKS_DIR)
        if task.reference is None:
            continue
        reference = REPO_ROOT / "benchmark" / "ground_truth" / task.reference
        if not reference.is_file():
            pytest.skip(f"{task_id}: reference not yet authored")
        workspace = tmp_path / task_id
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "in.test").write_text(reference.read_text(encoding="utf-8"), encoding="utf-8")
        result = validate_workspace(workspace, task=task)
        assert result.valid, f"{task_id} reference does not pass its own task: {result.render()}"


# --------------------------------------------------------------------------- #
# the result contract
# --------------------------------------------------------------------------- #


def test_result_dict_matches_the_agreed_shape() -> None:
    payload = validate_script(VALID_SCRIPT).to_dict()
    assert set(payload) >= {"valid", "errors", "warnings", "suggestions"}
    assert payload["valid"] is True
    assert isinstance(payload["errors"], list)


def test_only_errors_affect_validity() -> None:
    """Warnings must never block; blocking on one would reject valid work."""
    warned = validate_script(VALID_SCRIPT.replace("timestep 0.005", "timestep 0.05"))
    assert warned.warnings
    assert warned.valid


def test_output_is_deterministic() -> None:
    """Two runs must agree exactly; the benchmark compares them."""
    first = validate_script(VALID_SCRIPT).to_dict()
    second = validate_script(VALID_SCRIPT).to_dict()
    assert first == second


# --------------------------------------------------------------------------- #
# finding the input script
# --------------------------------------------------------------------------- #


def test_find_input_script_prefers_the_canonical_name(tmp_path: Path) -> None:
    (tmp_path / "in.melt").write_text("units lj\n", encoding="utf-8")
    (tmp_path / "in.melt.gpu").write_text("units lj\n", encoding="utf-8")
    assert find_input_script(tmp_path).name == "in.melt"


def test_find_input_script_returns_none_for_an_empty_workspace(tmp_path: Path) -> None:
    assert find_input_script(tmp_path) is None


def test_missing_workspace_raises_rather_than_reporting_invalid(tmp_path: Path) -> None:
    """Nothing to validate is a run failure, not an invalid input."""
    with pytest.raises(FileNotFoundError):
        validate_workspace(tmp_path / "absent")


def test_workspace_without_a_script_raises_rather_than_reporting_invalid(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no LAMMPS input script"):
        validate_workspace(tmp_path)
