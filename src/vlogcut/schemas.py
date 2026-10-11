"""Data contracts shared by every pipeline stage.

These models are the source of truth for what flows between modules.
Changing one means updating ARCHITECTURE.md and the tests in the same change.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Ingest / analyze
# ---------------------------------------------------------------------------


class MediaAsset(BaseModel):
    """One source file as found on disk."""

    asset_id: str
    path: str
    proxy_path: str | None = None
    duration_s: float = Field(gt=0)
    fps: float = Field(gt=0)
    width: int
    height: int
    created_at: datetime | None = None
    gps: tuple[float, float] | None = None  # (lat, lon)
    has_audio: bool = True


class QualityFlags(BaseModel):
    """Cheap signal-level quality measurements for one shot."""

    sharpness: float  # variance of Laplacian; higher = sharper
    exposure_clip: float = Field(ge=0, le=1)  # fraction of clipped luma pixels
    shake: float = Field(ge=0)  # optical-flow magnitude variance
    rejected: bool = False
    reject_reasons: list[str] = Field(default_factory=list)


class Shot(BaseModel):
    """A continuous segment of one asset, the unit the planner selects from."""

    shot_id: str
    asset_id: str
    start_s: float = Field(ge=0)  # absolute time within the asset
    end_s: float
    keyframes: list[str] = Field(default_factory=list)  # image paths
    quality: QualityFlags
    caption: str | None = None
    tags: list[str] = Field(default_factory=list)
    interest: int | None = Field(default=None, ge=1, le=5)
    transcript: str | None = None
    dup_group: str | None = None  # shots in the same group are near-duplicates

    @model_validator(mode="after")
    def _check_bounds(self) -> Shot:
        if self.end_s <= self.start_s:
            raise ValueError(f"shot {self.shot_id}: end_s must be > start_s")
        return self

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


class DurationStats(BaseModel):
    p10: float = Field(gt=0)
    median: float = Field(gt=0)
    p90: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> DurationStats:
        if not (self.p10 <= self.median <= self.p90):
            raise ValueError("expected p10 <= median <= p90")
        return self


class SubtitleStyle(BaseModel):
    present: bool = False
    position: str | None = None  # "bottom", "center", ...
    relative_size: float | None = None  # text height / frame height


class StyleProfile(BaseModel):
    """Explicit, editable description of a creator's editing style."""

    version: int = 1
    source_refs: list[str] = Field(default_factory=list)
    shot_duration_s: DurationStats
    cuts_per_min: float = Field(gt=0)
    beat_aligned_ratio: float = Field(ge=0, le=1)
    bpm_range: tuple[float, float] | None = None
    intro_s: float | None = None
    speech_ratio: float = Field(default=0.0, ge=0, le=1)
    look: dict[str, float] = Field(default_factory=dict)  # brightness, saturation, warmth...
    subtitle: SubtitleStyle = Field(default_factory=SubtitleStyle)
    notes: str = ""  # short natural-language summary


# ---------------------------------------------------------------------------
# Plan (edit decision list)
# ---------------------------------------------------------------------------


class Clip(BaseModel):
    """One piece of a shot placed on the timeline, in order."""

    shot_id: str
    in_s: float = Field(ge=0)  # offset from the shot's start
    out_s: float
    speed: float = Field(default=1.0, gt=0.25, le=4.0)
    subtitle: str | None = None
    reason: str = ""  # why the planner chose it; useful for debugging and eval

    @model_validator(mode="after")
    def _check_order(self) -> Clip:
        if self.out_s <= self.in_s:
            raise ValueError(f"clip from {self.shot_id}: out_s must be > in_s")
        return self

    @property
    def timeline_duration_s(self) -> float:
        return (self.out_s - self.in_s) / self.speed


class MusicCue(BaseModel):
    track_id: str  # refers to a user-supplied track, never a path the model invents
    start_s: float = Field(default=0.0, ge=0)  # position on the output timeline
    gain_db: float = 0.0


class EDL(BaseModel):
    """Edit decision list produced by the planner, checked by validate/."""

    variant_id: str
    target_duration_s: float = Field(gt=0)
    clips: list[Clip] = Field(min_length=1)
    music: list[MusicCue] = Field(default_factory=list)
    notes: str = ""

    @property
    def total_duration_s(self) -> float:
        return sum(c.timeline_duration_s for c in self.clips)
