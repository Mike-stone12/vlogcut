# vlogcut — Architecture (v0.1 draft)

> `vlogcut` is a placeholder name. This file is the design source of truth;
> `src/vlogcut/schemas.py` is the source of truth for data contracts.

## What this is

An open-source agent that turns raw daily-vlog footage into **editable rough cuts
that match a creator's own style**, learned from a few reference videos. It
produces several variants, takes feedback, and iterates.

## v1 scope

**In**
- Daily vlog footage shot on phones (tens of clips, 0.5–2 h total)
- 1–3 reference videos that define the target style
- Output: JianYing (剪映) draft + low-res preview mp4, 2–4 variants per run
- User-supplied background music only

**Out (for v1)**
- Story understanding / narrative restructuring (timeline stays chronological)
- Generated footage, voice cloning
- Color grading beyond simple parameter matching
- Web UI (CLI first)

## Design principles

1. **Models plan, code executes.** No model ever emits FFmpeg/shell commands or
   touches media. Models read text (and keyframes, for the captioner) and return
   JSON validated against `schemas.py`. Everything after that is deterministic.
2. **Style is explicit parameters, not an embedding.** Every style dimension is
   measurable, human-readable and editable ("faster pacing" = lower shot-duration
   range).
3. **Analyze once, plan many times.** Footage analysis is cached per project;
   planning and variants are cheap.
4. **Roles, not vendors.** Every model call goes through `ModelRouter` and is
   configured by role in `config.yaml`. Every call is logged.
5. **The core runs without any model.** Deterministic parts are testable offline
   with fixtures and a fake client.

## Pipeline

```
ingest → analyze → plan → validate/repair → render/export → feedback
                ↑                                              │
   references → style profile ←────────── profile deltas ──────┘
```

### 1. Ingest
- `ffprobe` metadata: duration, fps, display resolution (rotation applied), creation time,
  GPS (if present)
- Low-res proxies for all analysis and preview rendering
- Output: `MediaAsset` records

### 2. Analyze (per shot)
- Shot detection: PySceneDetect `AdaptiveDetector` for hard cuts + a blend-fit detector
  for cross-dissolves / fades (`analyze/transitions.py`); TransNetV2 later
- Quality (`analyze/quality.py`): blur (variance of Laplacian), exposure (clipped-luma
  fraction), shake (high-frequency camera jitter: tile-median phase-correlation path
  minus its smoothed path) → `rejected` + reasons only for clearly unusable footage,
  soft `warnings`, and a suggested `usable_range` that trims bad starts / ends.
  Measures, does not judge taste: tolerance for blur / shake comes later from the
  style profile (`look`) and the planner
- Near-duplicates: CLIP/SigLIP keyframe embeddings, clustered into `dup_group`
- Caption: **captioner** role (VLM) on 1–3 keyframes → description, tags, interest 1–5
- Speech: VAD + ASR (Whisper / FunASR) only when speech is present
- Output: `Shot` records, cached in `workspace/<project>/`

### 3. Style profile (from references)
Run the same analysis on reference videos, then aggregate:
- Pacing: shot-duration p10/median/p90, cuts per minute, pacing curve (intro vs body)
- Music: BPM, beat-aligned cut ratio (cut within ±80 ms of a beat), energy curve
- Structure: intro length, speech ratio
- Look: brightness / saturation / warmth stats (for later matching)
- Text: subtitle presence/position/size (OCR; v1.5)
- `notes`: short natural-language summary written by a VLM
- Output: `StyleProfile` JSON — versioned and user-editable

### 4. Plan (**planner** role)
- Input: shot table as text (id, time, duration, caption, tags, quality, interest),
  `StyleProfile`, user brief ("~3 min, cozy, focus on the café"), variant knobs
- Output: `EDL`
- If the shot table is too long: two stages — pick segments, then pick shots per segment

### 5. Validate & repair
- Hard checks (code): shot ids exist; in/out inside shot bounds; no rejected shots;
  total duration within ±10 % of target; clip durations within the profile range
  (with tolerance); no back-to-back clips from the same `dup_group`; chronological order
- On failure: structured error list goes back to the planner (max N rounds), then a
  rule-based fallback fix
- Deterministic post-processing: snap cut points to the nearest beat within
  tolerance, audio fades, subtitle timing from ASR

### 6. Render / export
- Preview mp4: FFmpeg on proxies (fast, low-res)
- JianYing draft: `pyJianYingDraft` (generation only; final export happens in the
  JianYing desktop app)
- FCPXML for Premiere / Resolve / FCP (later)

### 7. Variants & feedback
- Variants = different knob settings: pacing ±, selection strategy
  (highlights vs. coverage), music choice
- Feedback (pick / reject / free text) → **feedback** role turns text into
  validated `StyleProfile` deltas
- Every (context, variant A, variant B, choice) is logged as a preference pair →
  later: reward model / preference optimization

## Data contracts

Defined in `src/vlogcut/schemas.py`:

| Model | Produced by | Consumed by |
|---|---|---|
| `MediaAsset` | ingest | analyze, render |
| `Shot` (+ `QualityFlags`) | analyze | plan, validate, render |
| `StyleProfile` | style | plan, validate, feedback |
| `EDL` (+ `Clip`, `MusicCue`) | plan | validate, render |

Changing a schema means updating this file and the tests in the same change.

## Model roles

| Role | Job | Call volume | Typical choice |
|---|---|---|---|
| `captioner` | keyframes → description, tags, interest | high | local VLM (vLLM / Ollama) or cheap API |
| `planner` | shot table + profile + brief → EDL | low | strongest available model |
| `feedback` | free-text feedback → profile deltas | low | mid-tier or planner model |
| `critic` (optional) | check an EDL / preview against the brief | low–mid | mid-tier |

All roles go through `vlogcut.llm.router.ModelRouter`; any OpenAI-compatible
endpoint works (local servers, OpenAI, DeepSeek, Qwen/DashScope, OpenRouter …).
Privacy mode (later): only text descriptions leave the machine, never footage.

## Directory layout

```
src/vlogcut/
  schemas.py        data contracts
  llm/router.py     the only place that talks to models
  ingest/  analyze/  style/  plan/  validate/  render/  feedback/
  cli.py
tests/              offline tests; tiny fixtures only
eval/               benchmark sets, metrics, results
workspace/          per-project cache (gitignored)
```

## Milestones

| Week | Deliverable |
|---|---|
| 1 | ingest + analyze (no LLM); rule-based chronological rough cut; FFmpeg preview |
| 2 | style profile + beat sync; different references visibly change pacing |
| 3 | captioner + planner + validate/repair loop |
| 4 | JianYing export, variants, README + demo video |

### Progress

- [x] Week 1 · ingest: ffprobe metadata → `MediaAsset` (`ingest/probe.py`)
- [x] Week 1 · ingest: low-res proxies, cached per asset (`ingest/proxy.py`)
- [x] Week 1 · analyze: shot detection, incl. dissolves / fades (`analyze/shots.py`, `analyze/transitions.py`)
- [x] Week 1 · analyze: quality flags + full `Shot` records (`analyze/quality.py`)
- [ ] Week 1 · rule-based chronological rough cut
- [ ] Week 1 · FFmpeg preview render

## Evaluation (start in week 1)

- ~10 footage sets, each with reference videos
- Metrics: style distance (shot-duration distribution distance, beat-alignment gap),
  constraint violations, repair rounds, cost and latency per video, blind human preference
- Every model or prompt change is run against the eval set before merging

## Decisions log

- 2026-10: v1 targets daily vlogs; chronological timeline; user-supplied music only.
- 2026-10: role-based model config; local captioner + API planner as default hybrid.
- 2026-10: ingest — `asset_id` is a content fingerprint (size + first/last 1 MiB, sha1)
  so renames keep the cache; `MediaAsset.width/height` are display (post-rotation)
  dimensions, no `rotation` field; `created_at` prefers Apple's `creationdate`
  (keeps local offset), never falls back to file mtime; still images and audio-only
  files are skipped. Proxies: short side 540 px, CFR at min(source fps, 30), same
  timeline as the source so shot times map 1:1.
- 2026-10: tests that need ffmpeg are marked `ffmpeg` and skip when it is missing;
  fixture clips are generated with lavfi at test time, never committed.
- 2026-10: proxy details — H.264 CRF 28 + AAC 96k stereo, keyframe every ~1 s for
  cheap seeking, metadata stripped, `+faststart`. Dimensions round *down* to even so a
  proxy is never larger than its source; sources already ≤ 540 px keep their size.
  Output rate is the exact fraction (29.97 stays 30000/1001). Rotation is baked in
  (ffmpeg autorotate), so proxies are upright with no rotation tag. Cached at
  `<workspace>/proxies/<asset_id>.mp4`, written via a `.part.mp4` temp file; a cached
  proxy is reused only if its duration is within 0.25 s of the source.
  Known gaps: no HDR→SDR tone mapping yet (HLG/Dolby Vision proxies look flat); a VFR
  source whose average rate is < 30 gets that average, not 30.

- 2026-10: shot detection — runs on proxies. Hard cuts: PySceneDetect `AdaptiveDetector`
  (threshold 3.0), which tolerates pans and handheld shake. Shots < 0.5 s merge into
  their shorter neighbour (flash frames); shots > 10 s split into equal parts so the
  planner can pick from long one-takes. Output is `ShotSpan` (times + 3 keyframes at
  25/50/75 %) cached in `<workspace>/shots/<asset_id>.json` with the params as cache
  key; full `Shot` records are built after quality analysis, so `schemas.py` is
  unchanged. `scenedetect` and `numpy` live in the optional `analyze` extra.
- 2026-10: gradual transitions — no model: frames decoded at 15 fps, 64×64 luma; a
  window is a dissolve if its ends differ (mean |Δ| > 0.05), are structurally different
  (z-scored correlation < 0.5, rejects exposure swings) and the 1/4, 1/2, 3/4 frames fit
  the linear blend (relative residual < 0.2, rejects pans/zooms/hard cuts). Among
  overlapping hits, the longest window fitting within 0.02 of the best wins; the cut goes
  at its midpoint, and hard cuts inside a transition are dropped. Fades through black are
  two transitions. Precision is ±1 sample (67 ms) on short dissolves; on long (2 s)
  dissolves over fast-changing content only part of the span fits, so the cut can land
  ~0.3 s off-centre (still inside the transition). Growing spans outward was tried and
  rejected: it overshoots into static footage and merges fade halves. Wipes / slides
  are not detected.

- 2026-10: quality — **schema change**: `QualityFlags` gains `warnings` (soft issues,
  never reject alone) and `usable_range` (suggested in/out offsets within the shot;
  `Shot` validates it fits); `shake` now means high-frequency jitter (RMS px at 270p)
  instead of optical-flow magnitude variance, so pans and zooms are not "shaky".
  Rationale: blur and shake are often deliberate in vlogs. Rejected only when bad from
  start to end *and* near-black (> 95 % of pixels ≤ 5: covered lens), blown out
  (> 90 % ≥ 250), or violently shaky with no sound (audio-active < 30 %). Blur alone
  never rejects (skies, walls, soft focus). Per-frame "bad" = blur below max(10,
  0.2 × shot p90), clipping, or local (0.5 s) RMS jitter > 25 px, majority-voted over
  0.5 s; `usable_range` spans the first to last good run ≥ 0.5 s; shake is re-measured
  on the usable part. Motion is measured at the proxy frame rate (at 10 fps fast shake
  outgrew phase correlation), median over a 3×3 tile grid (a subject crossing the frame
  moves few tiles), smoothed path = order-1 Savitzky-Golay over 1 s (moving average
  inside, line fit at the ends) so a steady pan scores 0. Defaults calibrated on 18
  generated cases (blur levels, black / blown / night, pan, subject crossing, zoom, mild
  / strong / violent shake, phone raise with and without motion blur, focus hunt, sky).
  Known gaps: absolute thresholds are not calibrated on real phone footage yet; shake
  faster than ~45 px/frame at 270p is under-measured (motion blur then catches it); a
  pan from detail to sky at a shot end may be suggested for trimming.

## Open questions

- Default target length and how many variants per run
- How to present variants for comparison (side-by-side preview page?)
- Privacy mode details
- Quality thresholds: calibrate on a real phone-footage eval set; per-project relative
  thresholds?
