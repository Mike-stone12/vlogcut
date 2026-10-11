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
- `ffprobe` metadata: duration, fps, resolution, rotation, creation time, GPS (if present)
- Low-res proxies for all analysis and preview rendering
- Output: `MediaAsset` records

### 2. Analyze (per shot)
- Shot detection: PySceneDetect (content detector) baseline; TransNetV2 later
- Quality: blur (variance of Laplacian), exposure (clipped-luma fraction),
  shake (optical-flow magnitude variance) → `rejected` + reasons
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

## Evaluation (start in week 1)

- ~10 footage sets, each with reference videos
- Metrics: style distance (shot-duration distribution distance, beat-alignment gap),
  constraint violations, repair rounds, cost and latency per video, blind human preference
- Every model or prompt change is run against the eval set before merging

## Decisions log

- 2026-10: v1 targets daily vlogs; chronological timeline; user-supplied music only.
- 2026-10: role-based model config; local captioner + API planner as default hybrid.

## Open questions

- Default target length and how many variants per run
- How to present variants for comparison (side-by-side preview page?)
- Privacy mode details
