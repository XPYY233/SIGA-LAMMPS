"""Tests for the configuration layer.

The important properties here are the *invariants*, not the plumbing: ceilings
that cannot be raised from either source, secrets that cannot reach a log, and
HPC being optional so that M/R/X work with no cluster access.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from config.loader import (
    ConfigError,
    format_walltime,
    load_settings,
    parse_env_text,
    parse_walltime,
)

# --------------------------------------------------------------------------- #
# .env parsing
# --------------------------------------------------------------------------- #


def test_parse_env_handles_comments_blanks_and_export() -> None:
    text = textwrap.dedent(
        """
        # a comment
          # an indented comment

        SIGA_HPC_HOST=sy_hl_login
        export SIGA_HPC_PORT=22
        SIGA_EMPTY=
        """
    )
    assert parse_env_text(text) == {
        "SIGA_HPC_HOST": "sy_hl_login",
        "SIGA_HPC_PORT": "22",
        "SIGA_EMPTY": "",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('K="quoted value"', "quoted value"),
        ("K='single quoted'", "single quoted"),
        ("K=  spaced  ", "spaced"),
        ('K="with=equals"', "with=equals"),
    ],
)
def test_parse_env_quoting(raw: str, expected: str) -> None:
    assert parse_env_text(raw)["K"] == expected


@pytest.mark.parametrize("raw", ["JUSTAKEY", "=novalue", "K='unterminated"])
def test_parse_env_rejects_malformed_lines(raw: str) -> None:
    with pytest.raises(ConfigError):
        parse_env_text(raw)


# --------------------------------------------------------------------------- #
# walltime
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("30", 1800),
        ("05:00", 300),
        ("02:00:00", 7200),
        ("1-00:00:00", 86400),
        ("00:30:00", 1800),
    ],
)
def test_parse_walltime(text: str, seconds: int) -> None:
    assert parse_walltime(text) == seconds


@pytest.mark.parametrize("text", ["", "abc", "1:2:3:4", "00:99:00"])
def test_parse_walltime_rejects_garbage(text: str) -> None:
    with pytest.raises(ConfigError):
        parse_walltime(text)


def test_walltime_roundtrip_and_day_format() -> None:
    assert format_walltime(7200) == "02:00:00"
    assert format_walltime(86400 + 3600) == "1-01:00:00"
    for text in ("00:30:00", "2-12:15:30"):
        assert format_walltime(parse_walltime(text)) == text


# --------------------------------------------------------------------------- #
# the real repository configuration
# --------------------------------------------------------------------------- #


@pytest.fixture()
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_loads_the_real_repository_config(repo_root: Path) -> None:
    """The committed config.yaml plus the developer's .env must both load."""
    settings = load_settings(repo_root=repo_root)
    assert settings.memory.path.name == "lammps_memory.md"
    assert settings.memory.max_chars > 0
    assert settings.retrieval.collections == ("examples", "docs", "syntax")
    assert settings.benchmark.configurations == ("vanilla", "m", "mr", "mrsx")
    assert settings.ceilings.max_nodes >= 1
    assert settings.ceilings.max_ntasks >= 1


def test_universal_required_commands_are_lowercase_keywords(repo_root: Path) -> None:
    """LAMMPS keywords are case-sensitive; the validator compares lowercased."""
    settings = load_settings(repo_root=repo_root)
    for command in settings.validator.universal_required:
        assert command == command.lower(), f"{command!r} must be lowercased"


# --------------------------------------------------------------------------- #
# the ceiling invariant — the core security property of this module
# --------------------------------------------------------------------------- #


def _write_config(tmp_path: Path, max_nodes: int, max_ntasks: int, walltime: str = "02:00:00") -> Path:
    """A minimal config.yaml with a chosen ceiling triple."""
    config = tmp_path / "config.yaml"
    config.write_text(
        textwrap.dedent(
            f"""
            memory:
              path: adapter/memory/lammps_memory.md
              max_chars: 4200
            validator:
              universal_required: [units, atom_style, run]
              supported_atom_styles: [atomic]
              supported_unit_styles: [lj]
              max_lines: 400
              block_on_warnings: false
            retrieval:
              persist_dir: data/index
              corpus_dir: data/raw
              collections: [examples, docs, syntax]
              top_k: 5
            slurm:
              ceilings:
                max_nodes: {max_nodes}
                max_ntasks: {max_ntasks}
                max_walltime: "{walltime}"
              defaults:
                nodes: 1
                ntasks: 8
                walltime: "00:30:00"
            lammps:
              local_bin: lmp
              init_success_marker: "Setting up Verlet run"
              run_success_marker: "Total wall time"
            benchmark:
              tasks_dir: benchmark/tasks
              ground_truth_dir: benchmark/ground_truth
              runs_dir: benchmark/runs
              configurations: [vanilla, m, mr, mrsx]
            """
        ),
        encoding="utf-8",
    )
    return config


def test_env_cannot_raise_a_ceiling_beyond_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.env` asking for MORE must be clamped down to config.yaml."""
    monkeypatch.setenv("SIGA_SLURM_MAX_NODES", "8")
    monkeypatch.setenv("SIGA_SLURM_MAX_NTASKS", "512")
    config = _write_config(tmp_path, max_nodes=1, max_ntasks=64)
    settings = load_settings(env_file=None, config_file=config, repo_root=tmp_path)
    assert settings.ceilings.max_nodes == 1
    assert settings.ceilings.max_ntasks == 64


def test_env_may_tighten_a_ceiling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`.env` asking for LESS must win — tightening is always allowed."""
    monkeypatch.setenv("SIGA_SLURM_MAX_NODES", "1")
    monkeypatch.setenv("SIGA_SLURM_MAX_NTASKS", "4")
    config = _write_config(tmp_path, max_nodes=4, max_ntasks=64)
    settings = load_settings(env_file=None, config_file=config, repo_root=tmp_path)
    assert settings.ceilings.max_nodes == 1
    assert settings.ceilings.max_ntasks == 4


def test_walltime_ceiling_takes_the_shorter_of_the_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGA_SLURM_MAX_WALLTIME", "00:10:00")
    config = _write_config(tmp_path, max_nodes=1, max_ntasks=8, walltime="02:00:00")
    settings = load_settings(env_file=None, config_file=config, repo_root=tmp_path)
    assert settings.ceilings.max_walltime == "00:10:00"


def test_clamp_never_exceeds_ceilings(tmp_path: Path) -> None:
    config = _write_config(tmp_path, max_nodes=2, max_ntasks=16, walltime="01:00:00")
    settings = load_settings(env_file=None, config_file=config, repo_root=tmp_path)
    nodes, ntasks, walltime = settings.ceilings.clamp(nodes=99, ntasks=999, walltime="10:00:00")
    assert (nodes, ntasks, walltime) == (2, 16, "01:00:00")


def test_clamp_rejects_nonsense_requests(tmp_path: Path) -> None:
    config = _write_config(tmp_path, max_nodes=2, max_ntasks=16)
    settings = load_settings(env_file=None, config_file=config, repo_root=tmp_path)
    with pytest.raises(ConfigError):
        settings.ceilings.clamp(nodes=0, ntasks=1, walltime="00:10:00")


# --------------------------------------------------------------------------- #
# HPC is optional; secrets never reach a log
# --------------------------------------------------------------------------- #


def test_hpc_is_absent_when_unconfigured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SIGA_HPC_HOST", "SIGA_HPC_WORKSPACE"):
        monkeypatch.delenv(name, raising=False)
    config = _write_config(tmp_path, max_nodes=1, max_ntasks=8)
    settings = load_settings(env_file=None, config_file=config, repo_root=tmp_path)
    assert settings.hpc is None
    with pytest.raises(ConfigError, match="HPC is not configured"):
        settings.require_hpc()


def test_half_configured_hpc_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo'd pair must not silently degrade into 'no HPC'."""
    monkeypatch.setenv("SIGA_HPC_HOST", "sy_hl_login")
    monkeypatch.delenv("SIGA_HPC_WORKSPACE", raising=False)
    config = _write_config(tmp_path, max_nodes=1, max_ntasks=8)
    with pytest.raises(ConfigError, match="SIGA_HPC_WORKSPACE"):
        load_settings(env_file=None, config_file=config, repo_root=tmp_path)


def test_relative_remote_workspace_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A remote workspace must be absolute — a relative one would resolve on the
    wrong machine and defeat the confinement guarantee."""
    monkeypatch.setenv("SIGA_HPC_HOST", "sy_hl_login")
    monkeypatch.setenv("SIGA_HPC_WORKSPACE", "siga-lammps")
    config = _write_config(tmp_path, max_nodes=1, max_ntasks=8)
    with pytest.raises(ConfigError, match="absolute remote path"):
        load_settings(env_file=None, config_file=config, repo_root=tmp_path)


def test_log_dict_redacts_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGA_HPC_HOST", "sy_hl_login")
    monkeypatch.setenv("SIGA_HPC_WORKSPACE", "/scratch/x/siga")
    monkeypatch.setenv("SIGA_HPC_KEY_PASSPHRASE", "hunter2-do-not-log")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-also-secret")
    config = _write_config(tmp_path, max_nodes=1, max_ntasks=8)
    settings = load_settings(env_file=None, config_file=config, repo_root=tmp_path)
    rendered = repr(settings.to_log_dict())
    assert "hunter2-do-not-log" not in rendered
    assert "sk-also-secret" not in rendered
    # ...while still being useful for diagnosis.
    assert "sy_hl_login" in rendered


def test_process_environment_wins_over_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`SIGA_X=y pytest` must behave as an operator expects."""
    env = tmp_path / ".env"
    env.write_text("SIGA_SLURM_MAX_NTASKS=2\n", encoding="utf-8")
    monkeypatch.setenv("SIGA_SLURM_MAX_NTASKS", "7")
    config = _write_config(tmp_path, max_nodes=1, max_ntasks=64)
    settings = load_settings(env_file=env, config_file=config, repo_root=tmp_path)
    assert settings.ceilings.max_ntasks == 7


def test_missing_env_file_is_not_an_error(tmp_path: Path) -> None:
    """M/R/X must work with no .env at all."""
    config = _write_config(tmp_path, max_nodes=1, max_ntasks=8)
    settings = load_settings(env_file=tmp_path / "does-not-exist", config_file=config, repo_root=tmp_path)
    assert settings.hpc is None


def test_loading_does_not_mutate_the_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading settings must be side-effect free.

    An earlier implementation populated ``os.environ`` via ``setdefault``, which
    leaked the developer's real ``.env`` into every later read and made otherwise
    unrelated tests order-dependent. Catching that required this assertion: the
    loader returns an effective mapping instead of writing global state.
    """
    for name in ("SIGA_HPC_HOST", "SIGA_HPC_WORKSPACE", "SIGA_SLURM_MAX_NODES"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "SIGA_HPC_HOST=sy_hl_login\nSIGA_HPC_WORKSPACE=/scratch/x\nSIGA_SLURM_MAX_NODES=3\n",
        encoding="utf-8",
    )
    config = _write_config(tmp_path, max_nodes=8, max_ntasks=8)
    settings = load_settings(env_file=env, config_file=config, repo_root=tmp_path)

    # The values were honoured...
    assert settings.hpc is not None
    assert settings.hpc.host == "sy_hl_login"
    assert settings.ceilings.max_nodes == 3
    # ...without escaping into the process environment.
    import os

    assert "SIGA_HPC_HOST" not in os.environ
    assert "SIGA_SLURM_MAX_NODES" not in os.environ


def test_read_env_is_process_env_first(tmp_path: Path) -> None:
    """The precedence rule, asserted directly on the helper."""
    from config.loader import read_env

    env = tmp_path / ".env"
    env.write_text("A=from_file\nB=from_file\n", encoding="utf-8")
    resolved = read_env(env)
    assert resolved["A"] == "from_file"
    assert resolved["B"] == "from_file"
    # PATH is always present in the process environment and must win.
    assert resolved["PATH"] == __import__("os").environ["PATH"]


def test_invalid_yaml_fails_loudly(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("memory: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(env_file=None, config_file=config, repo_root=tmp_path)


def test_missing_required_key_names_itself(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("memory:\n  max_chars: 100\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="memory.path"):
        load_settings(env_file=None, config_file=config, repo_root=tmp_path)
