"""SIGA-LAMMPS configuration.

Public entry point: :func:`load_settings`, which merges the git-ignored ``.env``
with the versioned ``config/config.yaml`` and validates the result.
"""

from config.loader import (
    BenchmarkSettings,
    ConfigError,
    HpcSettings,
    LammpsSettings,
    MemorySettings,
    RetrievalSettings,
    Settings,
    SlurmCeilings,
    ValidatorSettings,
    format_walltime,
    load_settings,
    parse_env_text,
    parse_walltime,
)

__all__ = [
    "BenchmarkSettings",
    "ConfigError",
    "HpcSettings",
    "LammpsSettings",
    "MemorySettings",
    "RetrievalSettings",
    "Settings",
    "SlurmCeilings",
    "ValidatorSettings",
    "format_walltime",
    "load_settings",
    "parse_env_text",
    "parse_walltime",
]
