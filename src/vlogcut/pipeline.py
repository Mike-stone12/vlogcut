"""Footage folder -> proxies -> shots -> quality -> rough cut -> preview.

The steps every front end (CLI now, UI later) shares. Each stage caches in the
workspace, so re-running with another target, strategy or style profile only re-plans
and re-renders.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from vlogcut.ingest import ProbeError, make_proxy, probe_asset
from vlogcut.schemas import EDL, MediaAsset, MusicCue, Shot, StyleProfile
from vlogcut.validate import Violation, validate_edl

log = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v", ".mkv", ".avi", ".webm", ".mts", ".3gp"}

Progress = Callable[[str], None]


class CutResult(BaseModel):
    assets: list[MediaAsset]
    skipped: list[tuple[str, str]]  # (file, reason)
    shots: list[Shot]
    edl: EDL
    edl_path: Path
    preview_path: Path | None
    profile: StyleProfile | None = None  # the style the cut was paced with, if any
    violations: list[Violation] = Field(default_factory=list)


def find_footage(folder: str | Path) -> list[Path]:
    """Video files directly in `folder` (not recursive), hidden files skipped, sorted."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder")
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in VIDEO_SUFFIXES
    )


def ingest(
    files: list[Path], workspace: Path, progress: Progress = log.info
) -> tuple[list[MediaAsset], list[tuple[str, str]]]:
    assets, skipped = [], []
    for i, path in enumerate(files, 1):
        progress(f"[{i}/{len(files)}] ingest {path.name}")
        try:
            assets.append(make_proxy(probe_asset(path), workspace))
        except ProbeError as exc:
            skipped.append((path.name, str(exc)))
            log.warning("skipping %s: %s", path.name, exc)
    return assets, skipped


def analyze(assets: list[MediaAsset], workspace: Path, progress: Progress = log.info) -> list[Shot]:
    from vlogcut.analyze import assess_shots, detect_shots  # heavy optional deps
    from vlogcut.analyze.pauses import analyze_pauses, cut_points_for_span

    shots: list[Shot] = []
    for i, asset in enumerate(assets, 1):
        progress(f"[{i}/{len(assets)}] analyze {Path(asset.path).name}")
        asset_shots = assess_shots(asset, detect_shots(asset, workspace), workspace)
        progress(f"[{i}/{len(assets)}] pauses {Path(asset.path).name}")
        midpoints = [pause.mid_s for pause in analyze_pauses(asset, workspace)]
        for shot in asset_shots:
            # Refresh even on quality-cache hits: old records have no cut points.
            shot.cut_points = cut_points_for_span(midpoints, shot.start_s, shot.end_s)
        shots += asset_shots
    return shots


def validate_and_write(
    edl: EDL,
    shots: Sequence[Shot],
    edl_path: Path,
    *,
    target_s: float,
    profile: StyleProfile | None = None,
) -> list[Violation]:
    """Check an EDL and replace its JSON report, including when no issues remain."""
    violations = validate_edl(edl, shots, profile, target_s=target_s)
    path = edl_path.with_suffix(".violations.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.part")
    temp.write_text(
        json.dumps([v.model_dump(mode="json") for v in violations], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    temp.replace(path)
    for violation in violations:
        log.warning(
            "validation %s (clip %s): %s",
            violation.code,
            violation.clip_index,
            violation.message,
        )
    if not violations:
        log.info("validation: OK")
    return violations


def cut(
    folder: str | Path,
    workspace: str | Path,
    *,
    target_s: float = 180.0,
    strategy: str = "coverage",
    style: Path | None = None,
    music: Path | None = None,
    music_gain_db: float = -14.0,
    out: Path | None = None,
    render: bool = True,
    preview_params: dict | None = None,
    progress: Progress = log.info,
) -> CutResult:
    from vlogcut.plan import RoughCutParams, rough_cut
    from vlogcut.render import PreviewParams, render_preview
    from vlogcut.style import load_profile

    workspace = Path(workspace)
    # read the profile first: a bad file should fail before minutes of analysis
    profile = load_profile(style) if style is not None else None
    files = find_footage(folder)
    if not files:
        raise FileNotFoundError(f"no video files in {folder}")
    assets, skipped = ingest(files, workspace, progress)
    if not assets:
        raise FileNotFoundError(f"none of the {len(files)} files in {folder} is usable footage")
    shots = analyze(assets, workspace, progress)

    progress(
        f"plan {strategy}, target {target_s:.0f} s" + (f", style {style.name}" if style else "")
    )
    edl = rough_cut(shots, assets, RoughCutParams(target_s=target_s, strategy=strategy), profile)
    if style is not None and profile is not None:
        edl.notes += f"; style {style.name}: {profile.notes}"  # an EDL says what paced it
    if music is not None:
        edl.music = [MusicCue(track_id="music", gain_db=music_gain_db)]
    # outputs with and without a style sit side by side (the run manifest replaces this)
    name = f"{strategy}_{target_s:.0f}s" + (f"_{style.stem}" if style else "")
    edl_path = workspace / "edl" / f"{name}.json"
    violations = validate_and_write(edl, shots, edl_path, target_s=target_s, profile=profile)
    edl_path.parent.mkdir(parents=True, exist_ok=True)
    edl_path.write_text(edl.model_dump_json(indent=1), encoding="utf-8")
    shots_path = workspace / "shots.json"
    shots_path.write_text(
        json.dumps([s.model_dump(mode="json") for s in shots], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    preview = None
    if render:
        preview = out or workspace / f"preview_{name}.mp4"
        progress(f"render {preview}")
        render_preview(
            edl, shots, assets, workspace, preview,
            music={"music": music} if music else None,
            params=PreviewParams(**(preview_params or {})),
        )  # fmt: skip
    return CutResult(
        assets=assets, skipped=skipped, shots=shots, edl=edl, edl_path=edl_path,
        preview_path=preview, profile=profile, violations=violations,
    )  # fmt: skip
