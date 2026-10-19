"""Footage folder -> proxies -> shots -> quality -> rough cut -> preview.

The steps every front end (CLI now, UI later) shares. Each stage caches in the
workspace, so re-running with another target, strategy or style profile only re-plans
and re-renders.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Literal, TypeVar

from pydantic import BaseModel, Field

from vlogcut.ingest import ProbeError, make_proxy, probe_asset
from vlogcut.learn.ranker import RankerModel, score
from vlogcut.llm.router import ModelRouter
from vlogcut.paths import to_workspace_relative
from vlogcut.schemas import EDL, MediaAsset, MusicCue, Shot, StyleProfile
from vlogcut.validate import Violation, repair_edl, validate_edl

log = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v", ".mkv", ".avi", ".webm", ".mts", ".3gp"}
REPAIRABLE_CODES = frozenset(
    {"unknown_shot", "rejected_shot", "transition_not_allowed", "dissolve_too_long"}
)

Progress = Callable[[str], None]
_T = TypeVar("_T")
_R = TypeVar("_R")


class CutResult(BaseModel):
    assets: list[MediaAsset]
    skipped: list[tuple[str, str]]  # (file, reason)
    shots: list[Shot]
    edl: EDL
    edl_path: Path
    preview_path: Path | None
    profile: StyleProfile | None = None  # the style the cut was paced with, if any
    violations: list[Violation] = Field(default_factory=list)
    repaired: bool = False
    planner: Literal["rules", "llm"] = "rules"
    fallback: bool = False
    repair_rounds: int = 0


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


def _run_ordered(
    step: Callable[[int, _T], _R],
    items: Sequence[_T],
    workers: int,
    collect: Callable[[_R], None],
) -> None:
    """Collect in input order, including errors; cancel pending work on failure."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers == 1:
        for i, item in enumerate(items, 1):
            collect(step(i, item))
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        try:
            for i, item in enumerate(items, 1):
                futures.append(executor.submit(step, i, item))
            for future in futures:
                collect(future.result())
        finally:
            for future in futures:
                future.cancel()


def ingest(
    files: list[Path], workspace: Path, progress: Progress = log.info, *, workers: int = 1
) -> tuple[list[MediaAsset], list[tuple[str, str]]]:
    """Probe and proxy in input order; progress may run on worker threads."""
    assets, skipped = [], []
    locks: dict[str, Lock] = {}
    registry_lock = Lock()

    def ingest_asset(i: int, path: Path) -> MediaAsset | tuple[str, str]:
        progress(f"[{i}/{len(files)}] ingest {path.name}")
        try:
            asset = probe_asset(path)
            if workers == 1:
                return make_proxy(asset, workspace)
            # Identical source content shares an ID and a proxy temp filename.
            with registry_lock:
                lock = locks.setdefault(asset.asset_id, Lock())
            with lock:
                return make_proxy(asset, workspace)
        except ProbeError as exc:
            return path.name, str(exc)

    def collect(result: MediaAsset | tuple[str, str]) -> None:
        if isinstance(result, MediaAsset):
            assets.append(result)
        else:
            skipped.append(result)
            log.warning("skipping %s: %s", *result)

    _run_ordered(ingest_asset, files, workers, collect)
    return assets, skipped


def analyze(
    assets: list[MediaAsset],
    workspace: Path,
    progress: Progress = log.info,
    *,
    workers: int = 1,
    caption: bool = False,
    router: ModelRouter | None = None,
) -> list[Shot]:
    """Analyze in input order, then optionally caption sequentially via the router."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if caption and router is None:
        raise ValueError("caption=True requires a ModelRouter")
    from vlogcut.analyze import assess_shots, detect_shots  # heavy optional deps
    from vlogcut.analyze.pauses import analyze_pauses, cut_points_for_span

    shots: list[Shot] = []
    locks = {asset.asset_id: Lock() for asset in assets} if workers > 1 else {}

    def analyze_asset(i: int, asset: MediaAsset) -> list[Shot]:
        progress(f"[{i}/{len(assets)}] analyze {Path(asset.path).name}")
        asset_shots = assess_shots(asset, detect_shots(asset, workspace), workspace)
        progress(f"[{i}/{len(assets)}] pauses {Path(asset.path).name}")
        midpoints = [pause.mid_s for pause in analyze_pauses(asset, workspace)]
        for shot in asset_shots:
            # Refresh even on quality-cache hits: old records have no cut points.
            shot.cut_points = cut_points_for_span(midpoints, shot.start_s, shot.end_s)
        return asset_shots

    def step(i: int, asset: MediaAsset) -> list[Shot]:
        if workers == 1:
            return analyze_asset(i, asset)
        # Protect shot/keyframe, quality and pause caches for duplicate IDs.
        with locks[asset.asset_id]:
            return analyze_asset(i, asset)

    _run_ordered(step, assets, workers, shots.extend)
    if caption:
        from vlogcut.analyze.caption import caption_shots

        captions = caption_shots(
            shots, router, workspace, model=router.roles["captioner"].model, progress=progress
        )
        for shot in shots:
            if result := captions.get(shot.shot_id):
                shot.caption = result.description
                shot.tags = result.tags
                shot.interest = result.interest
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


def repair_if_needed(
    edl: EDL,
    shots: Sequence[Shot],
    violations: Sequence[Violation],
    profile: StyleProfile | None = None,
) -> tuple[EDL, bool]:
    """Apply the deterministic fallback only for repairable validation codes.

    Keep the original EDL if all clips would be dropped. The caller must validate
    a repaired EDL again, since remaining violations and clip indices can change.
    """
    if not any(v.code in REPAIRABLE_CODES for v in violations):
        return edl, False
    try:
        return repair_edl(edl, shots, profile), True
    except ValueError as exc:
        log.warning("EDL repair failed; keeping original EDL and violations: %s", exc)
        return edl, False


def cut(
    folder: str | Path,
    workspace: str | Path,
    *,
    target_s: float = 180.0,
    strategy: str = "coverage",
    style: Path | None = None,
    cold_open: bool = False,
    planner: Literal["rules", "llm"] = "rules",
    router: ModelRouter | None = None,
    caption: bool = False,
    brief: str | None = None,
    constraints: Sequence[str] | None = None,
    ranker: RankerModel | None = None,
    max_repair_rounds: int = 2,
    music: Path | None = None,
    music_gain_db: float = -14.0,
    out: Path | None = None,
    render: bool = True,
    preview_params: dict | None = None,
    progress: Progress = log.info,
    workers: int = 1,
) -> CutResult:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    from vlogcut.plan import RoughCutParams, rough_cut
    from vlogcut.plan.llm import PlannerError, plan_with_repair
    from vlogcut.render import PreviewParams, render_preview
    from vlogcut.style import load_profile

    if planner == "llm" and router is None:
        raise ValueError('planner="llm" requires a ModelRouter')
    if caption and router is None:
        raise ValueError("caption=True requires a ModelRouter")
    workspace = Path(workspace).resolve()
    if music is not None and not music.is_file():
        raise FileNotFoundError(f"music file does not exist or is not a file: {music}")
    # read the profile first: a bad file should fail before minutes of analysis
    profile = load_profile(style) if style is not None else None
    if cold_open:
        if profile is None or profile.cold_open_clips == 0:
            log.warning("Cold-open allowance is 0; no cold open is produced")
        if planner == "llm":
            log.warning("LLM planner ignores cold_open; the flag still applies to a rules fallback")
    files = find_footage(folder)
    if not files:
        raise FileNotFoundError(f"no video files in {folder}")
    worker_options = {"workers": workers} if workers != 1 else {}
    assets, skipped = ingest(files, workspace, progress, **worker_options)
    if not assets:
        raise FileNotFoundError(f"none of the {len(files)} files in {folder} is usable footage")
    caption_options = {"caption": True, "router": router} if caption else {}
    shots = analyze(assets, workspace, progress, **worker_options, **caption_options)
    ranker_scores = {shot.shot_id: score(ranker, shot) for shot in shots} if ranker else None

    progress(
        f"plan {strategy}, target {target_s:.0f} s" + (f", style {style.name}" if style else "")
    )
    used_planner, fallback, repair_rounds = planner, False, 0
    if planner == "llm":
        try:
            planned = plan_with_repair(
                router, shots, target_s=target_s, profile=profile, brief=brief,
                max_rounds=max_repair_rounds, constraints=constraints, ranker_scores=ranker_scores,
            )  # fmt: skip
        except PlannerError as exc:
            log.warning("LLM planner failed; falling back to rules: %s", exc)
            used_planner, fallback = "rules", True
        else:
            edl, repair_rounds = planned.edl, planned.rounds
    if used_planner == "rules":
        if any(text.strip() for text in constraints or ()):
            log.warning("Edit constraints are ignored by the rules planner")
        edl = rough_cut(
            shots, assets, RoughCutParams(target_s=target_s, strategy=strategy), profile,
            cold_open=cold_open, ranker_scores=ranker_scores,
        )  # fmt: skip
    if style is not None and profile is not None:
        edl.notes += f"; style {style.name}: {profile.notes}"  # an EDL says what paced it
    if music is not None:
        from vlogcut.plan.snap import snap_to_beats

        try:
            from vlogcut.analyze.beats import BeatParams, detect_beats
            from vlogcut.analyze.pauses import load_mono_audio

            sample_rate = BeatParams().sample_rate
            samples = load_mono_audio(music, sample_rate)
            beats = detect_beats(samples, sample_rate).beats if samples is not None else []
        except ImportError as exc:
            raise ImportError('music beat detection needs: pip install -e ".[analyze]"') from exc
        if beats:
            # User music starts at timeline zero, so no beat-time offset is needed.
            edl = snap_to_beats(edl, shots, beats)
        else:
            log.warning("No beats detected in %s; leaving EDL unsnapped", music)
        edl.music = [MusicCue(track_id="music", gain_db=music_gain_db)]
    # outputs with and without a style sit side by side (the run manifest replaces this)
    name = f"{strategy}_{target_s:.0f}s" + (f"_{style.stem}" if style else "")
    if used_planner == "llm":
        name += "_llm"
    edl_path = workspace / "edl" / f"{name}.json"
    violations = validate_and_write(edl, shots, edl_path, target_s=target_s, profile=profile)
    edl_path.parent.mkdir(parents=True, exist_ok=True)
    edl_path.write_text(edl.model_dump_json(indent=1), encoding="utf-8")
    edl, repaired = repair_if_needed(edl, shots, violations, profile)
    if repaired:
        temp = edl_path.with_suffix(".json.part")
        temp.write_text(edl.model_dump_json(indent=1), encoding="utf-8")
        temp.replace(edl_path)
        violations = validate_and_write(edl, shots, edl_path, target_s=target_s, profile=profile)
    shots_path = workspace / "shots.json"
    stored = [
        {
            **s.model_dump(mode="json"),
            "keyframes": [to_workspace_relative(k, workspace) for k in s.keyframes],
        }
        for s in shots
    ]
    shots_path.write_text(
        json.dumps(stored, ensure_ascii=False, indent=1),
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
        preview_path=preview, profile=profile, violations=violations, repaired=repaired,
        planner=used_planner, fallback=fallback, repair_rounds=repair_rounds,
    )  # fmt: skip
