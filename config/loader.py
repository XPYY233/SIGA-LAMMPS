"""Configuration loading for SIGA-LAMMPS.

Two sources, deliberately separated:

* ``config/config.yaml`` — versioned policy, thresholds, and hard ceilings.
  Contains no secrets and no machine-specific paths.
* ``.env`` — deployment specifics and secrets. Git-ignored, never committed.

Precedence follows the convention the harness itself uses for credentials: an
already-set process environment variable **wins** over the same key in the
``.env`` file. That makes ``SIGA_FOO=bar pytest`` behave the way an operator
expects without editing files.

Resource ceilings are the exception to plain precedence: they resolve to the
**minimum** of the two sources, so neither file can unilaterally raise a limit
the other tightened. The agent never participates in this decision at all.

Nothing here is on the model's tool surface.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

__all__ = [
    "ConfigError",
    "SlurmCeilings",
    "HpcSettings",
    "MemorySettings",
    "ValidatorSettings",
    "RetrievalSettings",
    "LammpsSettings",
    "BenchmarkSettings",
    "Settings",
    "load_settings",
    "parse_env_text",
    "read_env",
    "parse_walltime",
    "format_walltime",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_CONFIG_FILE = REPO_ROOT / "config" / "config.yaml"

#: Keys whose values must never appear in a log line.
_SECRET_KEY_PATTERN = re.compile(r"(passphrase|password|secret|token|api_key|key$)", re.IGNORECASE)

#: SLURM time formats: "MM", "MM:SS", "HH:MM:SS", "D-HH:MM:SS".
_WALLTIME_PATTERN = re.compile(r"^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$|^(\d+)$")


class ConfigError(RuntimeError):
    """A configuration value is missing, malformed, or contradictory."""


# --------------------------------------------------------------------------- #
# .env parsing
# --------------------------------------------------------------------------- #


def parse_env_text(text: str) -> dict[str, str]:
    """Parse ``.env`` text into a mapping.

    Handles the subset this project actually uses, and is strict about it:
    blank lines, ``#`` comments, an optional ``export`` prefix, unquoted values,
    and single- or double-quoted values. No variable expansion is performed —
    predictable beats clever for a file that holds credentials.

    Raises:
        ConfigError: a non-comment line is not ``KEY=VALUE``.
    """
    values: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            raise ConfigError(f".env line {lineno}: expected KEY=VALUE, got {raw!r}")
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            raise ConfigError(f".env line {lineno}: empty key")
        value = value.strip()
        # A value that OPENS a quote must close it. Checking only for matching
        # ends would silently pass `K='unterminated` through as a literal.
        if value[:1] in ("'", '"'):
            quote = value[0]
            if len(value) < 2 or value[-1] != quote:
                raise ConfigError(f".env line {lineno}: unterminated quote in {raw!r}")
            try:
                parts = shlex.split(value)
            except ValueError as exc:
                raise ConfigError(f".env line {lineno}: malformed quoting in {raw!r}") from exc
            if len(parts) != 1:
                raise ConfigError(f".env line {lineno}: expected one quoted value, got {raw!r}")
            value = parts[0]
        values[key] = value
    return values


def read_env(env_file: Path | None) -> dict[str, str]:
    """Resolve the effective environment without mutating the process.

    Values from *env_file* are the base; an already-set process environment
    variable overrides its file counterpart. Returning a mapping rather than
    writing into ``os.environ`` keeps loading side-effect free, so a caller that
    loads settings cannot silently change what any later code observes.

    A missing file is not an error: local-only work (M/R/X) needs no HPC config.
    """
    merged: dict[str, str] = {}
    if env_file is not None and Path(env_file).is_file():
        merged.update(parse_env_text(Path(env_file).read_text(encoding="utf-8")))
    merged.update({k: v for k, v in os.environ.items() if v is not None})
    return merged


# --------------------------------------------------------------------------- #
# value helpers
# --------------------------------------------------------------------------- #


def parse_walltime(value: str) -> int:
    """Parse a SLURM time limit into seconds.

    Accepts ``MM``, ``MM:SS``, ``HH:MM:SS``, and ``D-HH:MM:SS``.

    Raises:
        ConfigError: the value is not a recognised SLURM duration.
    """
    text = str(value).strip()
    if not text:
        raise ConfigError("walltime is empty")
    match = _WALLTIME_PATTERN.match(text)
    if match is None:
        raise ConfigError(f"invalid SLURM walltime {value!r} (expected HH:MM:SS or D-HH:MM:SS)")
    if match.group(5) is not None:  # bare minutes
        return int(match.group(5)) * 60
    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)
    if minutes > 59 or seconds > 59:
        raise ConfigError(f"invalid SLURM walltime {value!r} (minutes and seconds must be < 60)")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def format_walltime(seconds: int) -> str:
    """Render seconds as ``HH:MM:SS`` (or ``D-HH:MM:SS`` past a day)."""
    if seconds < 0:
        raise ConfigError("walltime cannot be negative")
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}-{clock}" if days else clock


def _env_int(env: Mapping[str, str], name: str, default: int | None = None) -> int | None:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_str(env: Mapping[str, str], name: str, default: str | None = None) -> str | None:
    raw = env.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw or default


def _expand(path: str) -> str:
    """Expand ``~`` and ``$VARS`` in a user-supplied path."""
    return os.path.expanduser(os.path.expandvars(path))


# --------------------------------------------------------------------------- #
# typed settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SlurmCeilings:
    """Hard upper bounds on requested SLURM resources.

    The agent may request less than these. It can never exceed them, and it
    never sees this object.
    """

    max_nodes: int
    max_ntasks: int
    max_walltime: str

    @property
    def max_walltime_seconds(self) -> int:
        return parse_walltime(self.max_walltime)

    def clamp(self, *, nodes: int, ntasks: int, walltime: str) -> tuple[int, int, str]:
        """Clamp a request to the ceilings, returning the effective triple."""
        if nodes < 1 or ntasks < 1:
            raise ConfigError(f"nodes and ntasks must be positive, got nodes={nodes} ntasks={ntasks}")
        return (
            min(nodes, self.max_nodes),
            min(ntasks, self.max_ntasks),
            format_walltime(min(parse_walltime(walltime), self.max_walltime_seconds)),
        )


@dataclass(frozen=True)
class HpcSettings:
    """Connection details for the cluster. Every value is config, not code."""

    host: str
    user: str | None
    port: int
    key_path: str | None
    workspace: str

    def __post_init__(self) -> None:
        if not self.host:
            raise ConfigError("SIGA_HPC_HOST is required for HPC operations")
        if not self.workspace.startswith("/"):
            raise ConfigError(
                f"SIGA_HPC_WORKSPACE must be an absolute remote path, got {self.workspace!r}"
            )
        if self.workspace.rstrip("/") == "":
            raise ConfigError("SIGA_HPC_WORKSPACE must not be the filesystem root")


@dataclass(frozen=True)
class MemorySettings:
    """M — the always-on procedural-memory primer."""

    path: Path
    max_chars: int

    def text(self) -> str:
        """Read the primer verbatim."""
        if not self.path.is_file():
            raise ConfigError(f"memory primer not found: {self.path}")
        return self.path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class ValidatorSettings:
    """X — deterministic validation policy."""

    universal_required: tuple[str, ...]
    supported_atom_styles: tuple[str, ...]
    supported_unit_styles: tuple[str, ...]
    max_lines: int
    block_on_warnings: bool


@dataclass(frozen=True)
class RetrievalSettings:
    """R — the LAMMPS knowledge base."""

    persist_dir: Path
    corpus_dir: Path
    model_cache_dir: Path
    collections: tuple[str, ...]
    top_k: int


@dataclass(frozen=True)
class LammpsSettings:
    """Layer 4 — the simulator."""

    local_bin: str
    init_success_marker: str
    run_success_marker: str


@dataclass(frozen=True)
class BenchmarkSettings:
    """The frozen task set and the ablation configurations."""

    tasks_dir: Path
    ground_truth_dir: Path
    runs_dir: Path
    configurations: tuple[str, ...]


@dataclass(frozen=True)
class Settings:
    """The complete resolved configuration."""

    memory: MemorySettings
    validator: ValidatorSettings
    retrieval: RetrievalSettings
    lammps: LammpsSettings
    benchmark: BenchmarkSettings
    ceilings: SlurmCeilings
    hpc: HpcSettings | None
    model: str
    repo_root: Path = field(default=REPO_ROOT)

    def require_hpc(self) -> HpcSettings:
        """Return HPC settings, or explain precisely what is missing.

        M, R, and X need no cluster access, so HPC config is optional until an
        HPC operation is actually attempted.
        """
        if self.hpc is None:
            raise ConfigError(
                "HPC is not configured. Set SIGA_HPC_HOST and SIGA_HPC_WORKSPACE "
                "in .env (see .env.example). Local validation and LAMMPS smoke "
                "tests work without it."
            )
        return self.hpc

    def to_log_dict(self) -> dict[str, Any]:
        """A redacted view safe to write to a log.

        Every secret-bearing value is replaced by its presence, never its
        content, so a stray ``logger.info(settings)`` cannot leak a credential.
        """
        hpc: dict[str, Any] | None = None
        if self.hpc is not None:
            hpc = {
                "host": self.hpc.host,
                "user": self.hpc.user,
                "port": self.hpc.port,
                "key_path": self.hpc.key_path,
                "workspace": self.hpc.workspace,
            }
        return {
            "model": self.model,
            "hpc": hpc,
            "ceilings": {
                "max_nodes": self.ceilings.max_nodes,
                "max_ntasks": self.ceilings.max_ntasks,
                "max_walltime": self.ceilings.max_walltime,
            },
            "memory": {"path": str(self.memory.path), "max_chars": self.memory.max_chars},
            "retrieval": {
                "persist_dir": str(self.retrieval.persist_dir),
                "collections": list(self.retrieval.collections),
                "top_k": self.retrieval.top_k,
            },
            "benchmark_configurations": list(self.benchmark.configurations),
        }


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def _require(mapping: dict[str, Any], path: str, where: str) -> Any:
    node: Any = mapping
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"{where}: missing required key {path!r}")
        node = node[part]
    return node


def _as_str_tuple(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{where} must be a non-empty list")
    out = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{where} must contain non-empty strings, got {item!r}")
        out.append(item.strip())
    return tuple(out)


def _resolve(root: Path, value: str) -> Path:
    """Resolve a possibly-relative path against the repository root."""
    path = Path(_expand(value))
    return path if path.is_absolute() else (root / path).resolve()


def load_settings(
    env_file: Path | str | None = DEFAULT_ENV_FILE,
    config_file: Path | str | None = DEFAULT_CONFIG_FILE,
    *,
    repo_root: Path | str | None = None,
) -> Settings:
    """Load and validate the complete configuration.

    Args:
        env_file: ``.env`` path, or ``None`` to skip and read only the process
            environment.
        config_file: ``config.yaml`` path.
        repo_root: Override for resolving relative paths (used by tests).

    Returns:
        A frozen :class:`Settings`.

    Raises:
        ConfigError: any value is missing, malformed, or contradictory.
    """
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT

    env = read_env(Path(env_file) if env_file is not None else None)

    if config_file is None:
        raise ConfigError("a config.yaml path is required")
    config_path = Path(config_file)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: top level must be a mapping")

    # ----- M -----
    memory_path = _resolve(root, str(_require(raw, "memory.path", "config.yaml")))
    memory_max = _require(raw, "memory.max_chars", "config.yaml")
    if not isinstance(memory_max, int) or memory_max <= 0:
        raise ConfigError(f"memory.max_chars must be a positive integer, got {memory_max!r}")

    # ----- X -----
    validator = ValidatorSettings(
        universal_required=_as_str_tuple(
            _require(raw, "validator.universal_required", "config.yaml"),
            "validator.universal_required",
        ),
        supported_atom_styles=_as_str_tuple(
            _require(raw, "validator.supported_atom_styles", "config.yaml"),
            "validator.supported_atom_styles",
        ),
        supported_unit_styles=_as_str_tuple(
            _require(raw, "validator.supported_unit_styles", "config.yaml"),
            "validator.supported_unit_styles",
        ),
        max_lines=int(_require(raw, "validator.max_lines", "config.yaml")),
        block_on_warnings=bool(_require(raw, "validator.block_on_warnings", "config.yaml")),
    )

    # ----- R -----
    # The corpus is a LAMMPS checkout, which lives somewhere specific on this
    # machine, so .env may override the committed default path.
    corpus_override = _env_str(env, "SIGA_LAMMPS_CORPUS")
    retrieval = RetrievalSettings(
        persist_dir=_resolve(root, str(_require(raw, "retrieval.persist_dir", "config.yaml"))),
        corpus_dir=(
            Path(_expand(corpus_override))
            if corpus_override
            else _resolve(root, str(_require(raw, "retrieval.corpus_dir", "config.yaml")))
        ),
        model_cache_dir=_resolve(
            root, str(_require(raw, "retrieval.model_cache_dir", "config.yaml"))
        ),
        collections=_as_str_tuple(
            _require(raw, "retrieval.collections", "config.yaml"), "retrieval.collections"
        ),
        top_k=int(_require(raw, "retrieval.top_k", "config.yaml")),
    )

    # ----- LAMMPS -----
    lammps = LammpsSettings(
        local_bin=_env_str(env, "SIGA_LAMMPS_LOCAL_BIN")
        or str(_require(raw, "lammps.local_bin", "config.yaml")),
        init_success_marker=str(_require(raw, "lammps.init_success_marker", "config.yaml")),
        run_success_marker=str(_require(raw, "lammps.run_success_marker", "config.yaml")),
    )

    # ----- benchmark -----
    benchmark = BenchmarkSettings(
        tasks_dir=_resolve(root, str(_require(raw, "benchmark.tasks_dir", "config.yaml"))),
        ground_truth_dir=_resolve(
            root, str(_require(raw, "benchmark.ground_truth_dir", "config.yaml"))
        ),
        runs_dir=_resolve(root, str(_require(raw, "benchmark.runs_dir", "config.yaml"))),
        configurations=_as_str_tuple(
            _require(raw, "benchmark.configurations", "config.yaml"), "benchmark.configurations"
        ),
    )

    # ----- SLURM ceilings: the MINIMUM of config.yaml and .env -----
    # Neither source may raise a limit the other tightened.
    def _ceiling(key: str, env_name: str, default: int) -> int:
        from_config = int(_require(raw, f"slurm.ceilings.{key}", "config.yaml"))
        from_env = _env_int(env, env_name)
        values = [v for v in (from_config, from_env) if v is not None]
        effective = min(values) if values else default
        if effective < 1:
            raise ConfigError(f"slurm.ceilings.{key} resolves to {effective}; must be >= 1")
        return effective

    walltimes = [str(_require(raw, "slurm.ceilings.max_walltime", "config.yaml"))]
    if (env_wall := _env_str(env, "SIGA_SLURM_MAX_WALLTIME")) is not None:
        walltimes.append(env_wall)
    ceilings = SlurmCeilings(
        max_nodes=_ceiling("max_nodes", "SIGA_SLURM_MAX_NODES", 1),
        max_ntasks=_ceiling("max_ntasks", "SIGA_SLURM_MAX_NTASKS", 64),
        max_walltime=format_walltime(min(parse_walltime(w) for w in walltimes)),
    )

    # ----- HPC: optional -----
    hpc: HpcSettings | None = None
    host = _env_str(env, "SIGA_HPC_HOST")
    workspace = _env_str(env, "SIGA_HPC_WORKSPACE")
    if host and workspace:
        hpc = HpcSettings(
            host=host,
            user=_env_str(env, "SIGA_HPC_USER"),
            port=_env_int(env, "SIGA_HPC_PORT", 22) or 22,
            key_path=_expand(key) if (key := _env_str(env, "SIGA_HPC_KEY_PATH")) else None,
            workspace=_expand(workspace),
        )
    elif host or workspace:
        missing = "SIGA_HPC_WORKSPACE" if host else "SIGA_HPC_HOST"
        raise ConfigError(
            f"{missing} is missing while the other HPC variable is set; "
            "configure both or neither."
        )

    model = _env_str(env, "DSH_MODEL") or "deepseek-v4-flash"

    return Settings(
        memory=MemorySettings(path=memory_path, max_chars=memory_max),
        validator=validator,
        retrieval=retrieval,
        lammps=lammps,
        benchmark=benchmark,
        ceilings=ceilings,
        hpc=hpc,
        model=model,
        repo_root=root,
    )
