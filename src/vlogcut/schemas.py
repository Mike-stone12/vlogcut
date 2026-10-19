"""Data contracts shared by every pipeline stage.

These models are the source of truth for what flows between modules.
Changing one means updating ARCHITECTURE.md and the tests in the same change.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

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
    """Cheap signal-level quality measurements for one shot.

    Measurements, not taste: blur and shake can be deliberate in a vlog, so only
    clearly unusable footage is `rejected`. Soft issues go to `warnings`; whether they
    matter is decided later from the style profile and the planner.
    """

    sharpness: float  # variance of Laplacian; higher = sharper
    exposure_clip: float = Field(ge=0, le=1)  # fraction of clipped luma pixels
    # high-frequency camera jitter: global motion minus its smoothed path (RMS, px at
    # 270p), so deliberate pans and zooms score low and hand shake scores high
    shake: float = Field(ge=0)
    rejected: bool = False
    reject_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)  # soft issues; never reject alone
    # suggested (in_s, out_s) offsets from the shot start, trimming e.g. the phone being
    # raised or lowered; None = the whole shot is usable
    usable_range: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _check_range(self) -> QualityFlags:
        if self.usable_range is not None:
            lo, hi = self.usable_range
            if not 0 <= lo < hi:
                raise ValueError("usable_range must satisfy 0 <= in_s < out_s")
        return self


class CutPoint(BaseModel):
    """Code-proposed cut at an absolute time on the source asset's timeline."""

    time_s: float
    kind: Literal["pause"]


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
    cut_points: list[CutPoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_bounds(self) -> Shot:
        if self.end_s <= self.start_s:
            raise ValueError(f"shot {self.shot_id}: end_s must be > start_s")
        rng = self.quality.usable_range
        if rng is not None and rng[1] > self.end_s - self.start_s + 1e-6:
            raise ValueError(f"shot {self.shot_id}: usable_range ends after the shot")
        previous = self.start_s
        for point in self.cut_points:
            if not self.start_s < point.time_s < self.end_s:
                raise ValueError(
                    f"shot {self.shot_id}: cut_points must be strictly inside the shot"
                )
            if point.time_s <= previous:
                raise ValueError(
                    f"shot {self.shot_id}: cut_points must be sorted with no duplicates"
                )
            previous = point.time_s
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


class TransitionMix(BaseModel):
    """Shares of measured cut types; fades currently count as dissolves."""

    hard_cut: float = Field(ge=0, le=1)
    dissolve: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _sum_to_one(self) -> TransitionMix:
        if abs(self.hard_cut + self.dissolve - 1) > 1e-6:
            raise ValueError("transition shares must sum to 1 within 1e-6")
        return self


class StructureStats(BaseModel):
    """Measured editing structure, independent of absolute pacing."""

    montage_share: float = Field(ge=0, le=1)


class StyleProfile(BaseModel):
    """Explicit, editable description of a creator's editing style."""

    version: int = 1
    source_refs: list[str] = Field(default_factory=list)
    shot_duration_s: DurationStats
    cuts_per_min: float = Field(gt=0)
    beat_aligned_ratio: float | None = Field(default=None, ge=0, le=1)  # None = not measured
    bpm_range: tuple[float, float] | None = None
    intro_s: float | None = None
    # User-set leading clips exempt from chronology; 0 disables cold opens.
    cold_open_clips: int = Field(default=0, ge=0)
    speech_ratio: float | None = Field(default=None, ge=0, le=1)  # None = not measured
    transitions: TransitionMix | None = None  # None = not measured
    structure: StructureStats | None = None  # None = not measured
    look: dict[str, float] = Field(default_factory=dict)  # brightness, saturation, warmth...
    subtitle: SubtitleStyle = Field(default_factory=SubtitleStyle)
    notes: str = ""  # short natural-language summary


# ---------------------------------------------------------------------------
# Plan (edit decision list)
# ---------------------------------------------------------------------------


class Transition(BaseModel):
    """Entry at a clip boundary; dissolves are centred without changing EDL timing."""

    kind: Literal["cut", "dissolve"] = "cut"
    duration_s: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _check_duration(self) -> Transition:
        if self.kind == "cut" and self.duration_s != 0:
            raise ValueError("a cut must have duration 0")
        if self.kind == "dissolve" and self.duration_s <= 0:
            raise ValueError("a dissolve must have duration > 0")
        return self


AudioMode = Literal["keep", "mute", "duck"]


class Clip(BaseModel):
    """One piece of a shot placed on the timeline, in order."""

    shot_id: str
    in_s: float = Field(ge=0)  # offset from the shot's start
    out_s: float
    speed: float = Field(default=1.0, gt=0.25, le=4.0)
    transition_in: Transition = Field(default_factory=Transition)
    audio: AudioMode = "keep"  # camera audio: normal level, silence, or fixed attenuation
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

    @model_validator(mode="after")
    def _check_first_transition(self) -> EDL:
        if self.clips[0].transition_in.kind != "cut":
            raise ValueError("the first clip's transition_in must be a cut")
        return self

    @property
    def total_duration_s(self) -> float:
        return sum(c.timeline_duration_s for c in self.clips)


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


class Feedback(BaseModel):
    """A verdict or free-text feedback on one variant, identified by its EDL stem."""

    variant_id: str
    verdict: Literal["pick", "reject"] | None = None
    text: str = ""

    @model_validator(mode="after")
    def _check_feedback(self) -> Feedback:
        if not self.variant_id.strip():
            raise ValueError("variant_id must not be blank")
        if self.verdict is None and not self.text.strip():
            raise ValueError("feedback requires a verdict or non-blank text")
        return self


class PreferencePair(BaseModel):
    """A chosen variant and a rejected variant for the same project or brief."""

    context: str
    chosen: str
    rejected: str

    @model_validator(mode="after")
    def _check_variants(self) -> PreferencePair:
        if not self.chosen.strip() or not self.rejected.strip():
            raise ValueError("chosen and rejected must not be blank")
        if self.chosen == self.rejected:
            raise ValueError("chosen and rejected must be different variants")
        return self
