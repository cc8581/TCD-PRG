"""Portable project-path and executable resolution helpers."""

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository root, independent of CWD."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_executable(value: str | Path, *, must_exist: bool = True) -> Path:
    """Resolve either a path-like executable or a command found on ``PATH``."""

    raw = str(value)
    path = Path(raw).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        resolved = project_path(path)
        if must_exist and not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved
    located = shutil.which(raw)
    if located is None and must_exist:
        raise FileNotFoundError(f"Executable is not on PATH: {raw}")
    return Path(located).resolve() if located is not None else project_path(path)
