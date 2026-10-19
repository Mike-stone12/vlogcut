"""Path conversion at workspace JSON cache boundaries."""

from pathlib import Path


def to_workspace_relative(path: str | Path, workspace: str | Path) -> str:
    """Store workspace paths as POSIX relatives; keep external paths absolute."""
    path = Path(path)
    absolute = path.resolve()
    try:
        return absolute.relative_to(Path(workspace).resolve()).as_posix()
    except ValueError:
        return str(path if path.is_absolute() else absolute)


def from_workspace(value: str | Path, workspace: str | Path) -> Path:
    """Resolve a cached relative path; preserve legacy absolute paths as-is."""
    path = Path(value)
    return path if path.is_absolute() else (Path(workspace) / path).resolve()
