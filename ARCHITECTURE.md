# vlogcut — Architecture (v0.2 draft)

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
- The creator's own past JianYing drafts as selection examples (optional; see
  "Learning from past edits")
- Output: JianYing (剪映) draft + low-res preview mp4, 2–4 variants per run
- User-supplied background music only

**Out (for v1)**
- Story understanding / narrative restructuring (timeline stays chronological, except
  an optional cold open; see Validate)
- Generated footage, voice cloning
- Color grading beyond simple parameter matching
- Web UI (CLI first)

## Design principles

1. **Models plan, code executes.** No model ever emits FFmpeg/shell commands or
   touches media. Models read text (and keyframes, for the captioner) and return
   JSON validated against `schemas.py`. Everything after that is deterministic.
2. **Style is explicit parameters, not an embedding.** Every style dimension is
   measurable, human-readable and editable ("faster pacing" = lower shot-duration
   range). A learned model may *score* shots (see "Learning from past edits"), but it
   is a plug-in scorer; the profile itself stays explicit.
3. **Analyze once, plan many times.** Footage analysis is cached per project;
   planning and variants are cheap.
4. **Roles, not vendors.** Every model call goes through `ModelRouter` and is
   configured by role in `config.yaml`. Every call is logged.
5. **The core runs without any model.** Deterministic parts are testable offline
   with fixtures and a fake client.
6. **Code proposes, models choose.** Cut points, candidate shots and transitions are
   enumerated deterministically from signals; models pick among candidates rather than
   inventing times.
7. **Measure before trusting.** Every style or planning change is checked against the
   eval set; the LLM planner becomes the default only if it beats rules + ranker there.

## Pipeline

```
ingest → analyze → plan → validate/repair → render/export → feedback
                   ↑  ↑                                        │
  references → style profile ←─────────── profile deltas ──────┤
  past drafts → selection examples → ranker ←─ preference pairs ┘
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
- Near-duplicates: CLIP/SigLIP keyframe embeddings, clustered into `dup_group`. Chunks
  split from one long take share a group without any embedding.
- Speech: VAD + ASR (Whisper / FunASR) only when speech is present
- Shot kind (planned): `talking` / `broll` from VAD + caption; feeds the structure
  dimension of the style profile and the planner
- Caption: **captioner** role (VLM) on 1–3 keyframes → description, tags, interest 1–5.
  Only for shots that survive quality + dedup; the cache key includes the model name
- Cut-point candidates (planned): speech pauses, beats of the chosen music, motion
  minima → `Shot.cut_points`; planners choose in/out among these instead of inventing
  times or inheriting the equal 10 s split boundaries
- Output: `Shot` records, cached in `workspace/<project>/`. Per-asset work is
  independent; `pipeline.py` runs it sequentially today, a thread pool is the easy
  speed-up

### 3. Style profile (from references)
Run shot detection, transitions, VAD and (later) captions on reference videos (quality
flags are skipped: references are finished edits), then aggregate:
- Pacing: shot-duration p10/median/p90, cuts per minute, pacing curve (intro vs body)
- Music: BPM, beat-aligned cut ratio (cut within ±80 ms of a beat), energy curve
- Structure ("grammar"): intro length, speech ratio, talking ↔ b-roll alternation
  (transition counts between shot kinds), montage bursts (runs of short shots), cold
  open present or not. Pacing alone reads as a speed slider; structure is what makes a
  cut recognisably someone's.
- Transitions: share of hard cuts / dissolves (from `analyze/transitions.py`); fades
  through black currently count as two dissolves
- Audio: speech-over-music ratio, whether clip audio is kept under music
- Look: brightness / saturation / warmth stats (for later matching)
- Text: subtitle presence/position/size (OCR; v1.5)
- `notes`: short natural-language summary written by a VLM
- Output: `StyleProfile` JSON — versioned and user-editable (`vlogcut style build`);
  `vlogcut style show / diff` make it a first-class artefact users can share and compare

### 3b. Learning from past edits (ranker)
References show only what a creator kept, never what they skipped, so *selection*
cannot be learned from them. Two sources do carry that signal:
- The creator's own past JianYing drafts (`import/jianying.py`, planned): a draft
  names the source files, the chosen in/out points and the order, i.e. a (raw footage
  → edit) pair. Running analyze on those sources yields chosen vs. skipped shots.
- Preference pairs from the variant feedback loop (§7).

A small pairwise ranker (gradient-boosted trees or a tiny MLP over quality metrics,
motion, speech, shot kind and a CLIP embedding; CPU, seconds to train, fully local)
turns these into a `score(shot)` that plugs into the rules planner and is shown to the
LLM planner. No end-to-end "video → edit" model: there is no data for it, it is not
explainable, and it would break principle 2.

### 4. Plan (**planner** role, or rules + ranker)
- Input: shot table as text (id, time, duration, kind, caption, tags, quality, ranker
  score, cut-point candidates), `StyleProfile`, user brief ("~3 min, cozy, focus on
  the café"), variant knobs
- Output: `EDL`; in/out points are chosen among the shot's cut-point candidates
- If the shot table is too long: two stages — pick segments, then pick shots per segment
- `plan/rules.py` (+ ranker) is the no-model baseline and the fallback. The LLM planner
  becomes the default only once the eval shows it beats that baseline on style distance
  and blind preference; its distinct value is the brief and narrative ordering.

### 5. Validate & repair
- Deterministic post-processing first: snap cut points to the nearest beat within
  tolerance (needs the chosen music, so it runs after planning and before the checks),
  audio fades, subtitle timing from ASR
- Hard checks (code): shot ids exist; in/out inside shot bounds; no rejected shots;
  total duration within ±10 % of target; clip durations within the profile range
  (with tolerance); no back-to-back clips from the same `dup_group`; chronological
  order, except an optional cold open of at most `StyleProfile.cold_open_clips` clips
  at the start (a hook from later in the day, then back to the beginning); transition
  types and audio modes allowed by the profile
- On failure: structured error list goes back to the planner (max N rounds), then a
  rule-based fallback fix

### 6. Render / export
- Preview mp4 (`render/preview.py`): FFmpeg on proxies (fast, low-res); cached per-clip
  segments joined losslessly; debug subtitle track with source + reason per clip.
  Planned: dissolves / fades per `Clip.transition_in`, per-clip audio mode (keep / mute
  / duck under music)
- Interchange: an OpenTimelineIO timeline is the canonical export; JianYing and FCPXML
  are exporters from it, so a change in one app's format never reaches the planner
- JianYing draft: `pyJianYingDraft` (generation only; final export happens in the
  JianYing desktop app). Risk: undocumented format, and newer JianYing versions
  reportedly encrypt drafts → verify with a one-day spike (a two-clip draft opened in
  JianYing) before building on it
- FCPXML for Premiere / Resolve / FCP (later)

### 7. Variants & feedback
- Variants = different knob settings: pacing ±, selection strategy
  (highlights vs. coverage), music choice, cold open on / off
- Presentation: the CLI writes a local HTML page with the previews side by side and
  pick / reject buttons that save a feedback JSON (no server)
- Feedback (pick / reject / free text) → **feedback** role first *classifies* it:
  "the café part was too short" is about this edit and becomes a re-plan constraint;
  "too fast overall" is about style and becomes a validated `StyleProfile` delta.
  Mixing the two would drift the profile.
- Every (context, variant A, variant B, choice) is logged as a preference pair and
  feeds the ranker (§3b)

## Data contracts

Defined in `src/vlogcut/schemas.py`:

| Model | Produced by | Consumed by |
|---|---|---|
| `MediaAsset` | ingest | analyze, render |
| `Shot` (+ `QualityFlags`) | analyze | plan, validate, render |
| `StyleProfile` | style | plan, validate, feedback |
| `EDL` (+ `Clip`, `MusicCue`) | plan | validate, render |

Changing a schema means updating this file and the tests in the same change.

`StyleProfile.beat_aligned_ratio` and `speech_ratio` are `None` until the beat / speech
analysis exists: `None` means "not measured", which is not the same as 0.
`StyleProfile.transitions: TransitionMix | None = None` follows the same convention:
`hard_cut` and `dissolve` are shares in [0, 1], summing to 1 within 1e-6. Old profile
JSON without this field still loads. The mix is unmeasured when dissolve detection is off
or no cuts survive merging. `cuts_per_min` permits zero for uncut references.

Planned additions (not yet in `schemas.py`; each lands with the module that needs it):

| Field / model | Why |
|---|---|
| `Clip.transition_in` (cut / dissolve / fade + duration) | references' transitions are measured but cannot be reproduced |
| `Clip.audio` (keep / mute / duck) | how voice sits under music is part of a style |
| `MusicCue.end_s`, ducking params | music mixing needs an end and a level under speech |
| `Shot.kind` (talking / broll), `Shot.cut_points` | structure dimension; code-proposed in/out candidates |
| `StyleProfile.structure`, `.cold_open_clips` | the "grammar" dimensions |
| `Run` manifest (inputs fingerprint → outputs) | EDL file names collide once profile version, brief, variant and model all vary |
| `Feedback`, `PreferencePair` | the flywheel must be logged from the first real use |

Cached JSON should store paths relative to the workspace so a moved workspace stays
valid (today they are absolute).

## Model roles

| Role | Job | Call volume | Typical choice |
|---|---|---|---|
| `captioner` | keyframes → description, tags, interest | high | local VLM (vLLM / Ollama) or cheap API |
| `planner` | shot table + profile + brief → EDL | low | strongest available model |
| `feedback` | free-text feedback → profile deltas | low | mid-tier or planner model |
| `critic` (optional) | check an EDL / preview against the brief | low–mid | mid-tier |

All roles go through `vlogcut.llm.router.ModelRouter`; any OpenAI-compatible
endpoint works (local servers, OpenAI, DeepSeek, Qwen/DashScope, OpenRouter …).
Privacy mode (later): only text descriptions leave the machine, never footage. The
captioner sees keyframes, so privacy mode pins `captioner` to a local provider and the
router refuses a remote one.

## Directory layout

```
src/vlogcut/
  schemas.py        data contracts
  llm/router.py     the only place that talks to models
  ingest/  analyze/  style/  plan/  validate/  render/  feedback/
  import/           past JianYing drafts → selection examples
  learn/            ranker training (optional, tiny, CPU)
  export/           OTIO timeline → JianYing draft / FCPXML
  cli.py
tests/              offline tests; tiny fixtures only
eval/               benchmark sets, metrics, results
workspace/          per-project cache (gitignored)
```

## Milestones

| Week | Deliverable |
|---|---|
| 1 | ingest + analyze (no LLM); rule-based chronological rough cut; FFmpeg preview |
| 2 | **PoC**: style profile from references + `cut --style`; round-trip eval; JianYing draft spike |
| 3 | structure dimension; cut-point candidates; transitions + audio modes in EDL and preview; beat sync |
| 4 | draft import + ranker; captioner; planner + validate/repair loop, measured against rules + ranker |
| 5 | OTIO + JianYing export; variants page + feedback; README + demo video |

Revised 2026-10-14 after the week-1 review: the PoC and the eval move first because
"references visibly change the cut" is the hypothesis everything else depends on; the
JianYing spike moves up because its format is the biggest external risk; the LLM planner
moves after the ranker so it is measured against a real baseline.

### PoC (week 2): cut unrelated footage in creator X's style

Goal: show that a profile extracted from someone's published vlogs changes a cut of
unrelated footage in a way viewers recognise. Protocol:
- References: two published vlogs each from two creators with opposite pacing (one
  fast-cut, one slow / cinematic). Analysed locally only; never committed or
  redistributed.
- Footage: 30–60 min shot on a phone over one day, the realistic input. Public *raw*
  vlog footage is rare; stock clips (Pexels / Pixabay, CC0) are a fallback but are
  already-edited short clips, so they exercise little of ingest / quality / coverage.
- Runs: the same footage cut with the default pacing and with each creator's profile.
- Pass: (a) round-trip style distance (profile of the output vs. the reference profile)
  is well below the distance between the two creators; (b) in a blind test, viewers
  match each cut to its creator at ≥ 80 %.
- Code: `style/profile.py` (shot detection + transitions on references →
  `StyleProfile`), `cut --style <json>`, `style/distance.py` + `vlogcut style diff`. All
  in place as of 2026-10-14; what remains is running it on real references and footage.
- Known limit: with pacing and transitions only, the PoC shows rhythm, not selection or
  structure; those come in weeks 3–4.

### Progress

- [x] Week 1 · ingest: ffprobe metadata → `MediaAsset` (`ingest/probe.py`)
- [x] Week 1 · ingest: low-res proxies, cached per asset (`ingest/proxy.py`)
- [x] Week 1 · analyze: shot detection, incl. dissolves / fades (`analyze/shots.py`, `analyze/transitions.py`)
- [x] Week 1 · analyze: quality flags + full `Shot` records (`analyze/quality.py`)
- [x] Week 1 · rule-based chronological rough cut (`plan/rules.py`)
- [x] Week 1 · FFmpeg preview render (`render/preview.py`)
- [x] Week 1 · CLI: footage folder → preview in one command (`cli.py`, `pipeline.py`)
- [x] Week 2 · style: pacing profile from reference videos (`style/profile.py`,
      `vlogcut style build`)
- [x] Week 2 · `vlogcut cut --style <profile.json>`: the rules planner paces from a
      profile (`pipeline.py`)
- [x] Week 2 · style distance + round-trip eval (`style/distance.py`,
      `vlogcut style show / diff`, `tests/test_style_distance.py`)
- [x] Week 2 · rules planner reproduces the profile's shot-length distribution
- [x] Week 3 · style: transition mix from references (`style/profile.py`)

Week 1 complete (2026-10-14): `vlogcut cut <folder>` runs end to end.

## Evaluation (starts with the PoC; `eval/` does not exist yet)

- Round-trip test (offline, synthetic footage): reference → profile → cut → profile of
  the cut; the distance must be small. This is the first eval and doubles as a unit test:
  `tests/test_style_distance.py` cuts generated footage with a fast and a slow reference
  profile and requires the round trip (via the EDL and via the rendered preview) to land
  within a third of the distance between the two references. The fast-reference EDL
  distance must also be < 0.1 octaves; the rendered-preview distance must be < 0.2.
- Style distance (`style/distance.py`): per dimension |log2(a / b)| for the positive
  pacing quantities (shot-duration p10 / median / p90, cuts per minute), so the unit is
  octaves (1.0 = twice as fast or slow); absolute difference for the ratio dimensions in
  [0, 1], including the new `dissolve` ratio dimension (`transitions.dissolve`);
  a dimension unmeasured on either side is skipped and reported; equal weights
  until the eval set says otherwise. `vlogcut style diff a.json b.json` prints it.
  Zero cuts/min versus zero has distance 0; zero versus a positive rate has infinite
  log distance.
- ~10 footage sets, each with reference videos
- Metrics: style distance (shot-duration distribution distance, beat-alignment gap,
  structure / transition-mix distance), constraint violations, repair rounds, cost and
  latency per video, blind human preference
- Baseline comparison: rules + ranker vs. LLM planner on the same sets
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
- 2026-10: rule-based rough cut — no model; Week-1 baseline and later the fallback when
  a planner EDL cannot be repaired. Default target 3 min (everything if less is usable),
  default strategy "coverage", default pacing without a style profile p10 1.5 /
  median 3 / p90 5 s. Coverage treats each recording as a moment of the day: every
  moment gets a clip when slots allow, the rest by usable footage (largest remainder),
  best shot per chronological bin; with fewer slots than moments, evenly spaced bins of
  moments. (Plain shot-count bins let a long one-take crowd out the last scene.)
  "highlights" = top-k by score, re-sorted by time. Lengths share one offset clamped
  to p10..p90 and each shot's usable part; k grows / shrinks until the total is within
  ±10 %. Base lengths now sample the profile distribution (see 2026-10-15 below).
  Clips sit mid-usable-part; shots `kept_for_audio` start at the usable start
  and run to p90. Score is quality-only and pluggable (highlight features — faces,
  audio peaks, motion peaks, composition — come with the style profile in week 2, with
  weights learned from reference videos, not hard-coded).
  Known gaps: "talking" is only known for shots kept_for_audio until VAD / ASR exists;
  no near-duplicate check until keyframe embeddings exist.
- 2026-10: preview — every clip is rendered from its proxy into a cached segment
  (H.264 + PCM in .mov, key = asset, source span, output frames, canvas, params) and
  segments are joined with the concat demuxer (video copied, audio encoded once to
  AAC). Clip boundaries are snapped to the frame grid on the cumulative timeline, so
  the preview matches the EDL within half a frame and audio cannot drift. Canvas =
  orientation covering most of the timeline (ties → portrait), 540 px short side,
  30 fps; other-orientation clips are letterboxed in black (blur fill is a param, off
  by default). 30 ms audio fades at every cut; clips without audio get silence; speed
  via setpts + chained atempo. Music: `MusicCue.track_id` → a user-supplied file,
  gain + delay, mixed under programme audio (no ducking yet), fade-out at the end.
  Debug info is a soft `mov_text` subtitle track (this FFmpeg build has no
  freetype / libass, so nothing is burned in).
- 2026-10: CLI — `vlogcut cut <folder>` (argparse, no new dependency; console script in
  `pyproject.toml`). Shared steps live in `pipeline.py` so a later UI reuses them.
  Folder scan is not recursive, skips hidden files, matches common video suffixes;
  files ffprobe rejects are listed as skipped, not fatal. Default workspace
  `./workspace/<folder name>`; writes `edl/<strategy>_<target>s.json`, `shots.json` and
  `preview_<strategy>_<target>s.mp4`. Every stage is cached, so changing target or
  strategy re-plans in ~2 s. Errors are one line with exit code 1. Coverage keeps
  pacing first: when target / median gives fewer slots than recordings, it spreads
  over the day instead of shortening every clip.

- 2026-10-14: week-1 review → design changes (schemas unchanged until each module
  lands): EDL gains transitions and per-clip audio modes; chronological order allows an
  optional cold open; the style profile gains structure / transitions / audio
  dimensions; cut points are code-proposed candidates that planners choose among;
  feedback is classified (this edit vs. style) before it touches the profile; OTIO is
  the canonical export with JianYing and FCPXML as exporters; a run manifest replaces
  name-based EDL files; a learned ranker trained on the creator's own drafts and
  preference pairs is the only learned component and plugs in as `score`. Milestones
  re-ordered: PoC + eval + JianYing spike first (week 2), LLM planner after the ranker.
- 2026-10-14: style profile, PoC slice — references are finished edits, so
  `style/profile.py` calls the cut detectors directly (`shots.detect_cuts` +
  `transitions`) and skips `detect_shots`: no `split_long` (a 30 s shot *is* the style)
  and no keyframes. `min_shot_s` is 0.25 s (footage uses 0.5): 0.3 s montage shots are
  real in fast edits. Raw cut times are cached at `<workspace>/style/<asset_id>.cuts.json`
  keyed by asset fingerprint + `ReferenceParams`; aggregation is pure. Several references
  pool their shot durations and `cuts_per_min = total cuts / total minutes`, not a mean
  of per-video rates. **Schema change**: `StyleProfile.beat_aligned_ratio` and
  `speech_ratio` become `float | None = None` (not measured yet). `notes` holds a
  deterministic one-line summary until a VLM writes one. `vlogcut style <refs> -o
  profile.json` caches under `workspace/references` by default so profiles share
  proxies. Known gaps: `transitions.load_frames` decodes the whole proxy at 15 fps,
  64×64 float32 (≈ 440 MB for a 30 min reference), fine for 5–15 min PoC references;
  intro length, structure, transitions and look are not measured yet.
- 2026-10-14: `cut --style <profile.json>` — the profile is read and validated before
  any analysis, so a bad file fails in milliseconds, not after minutes of proxies; only
  the rules planner reads it (pacing p10 / median / p90). Outputs carry the profile's
  file stem as a suffix (`edl/coverage_180s_fast.json`, `preview_coverage_180s_fast.mp4`)
  so cuts with and without a style sit side by side in one workspace until the run
  manifest replaces name-based outputs. The EDL `notes` record the profile summary, so
  an EDL says what paced it.
- 2026-10-14: style distance + round-trip eval — `style/distance.py`, no model, no numpy.
  Pacing dimensions are positive ratio quantities, so each is compared as |log2(a / b)|
  and the unit is octaves (1.0 = twice as fast or slow); the ratio dimensions in [0, 1]
  (`beat_aligned_ratio`, `speech_ratio`) use the absolute difference; a dimension that is
  `None` on either side is skipped and listed in `StyleDistance.unmeasured`; equal
  weights until the eval set says otherwise (cuts per minute is not independent of the
  duration distribution, so pacing counts about twice). `StyleDistance` is an analysis
  result, not a cross-module contract, so it stays out of `schemas.py`. Two ways to
  profile a cut: `profile_from_edl` (clip timeline durations; exact, no render, no
  detection; what the unit test uses) and `style build` on the rendered preview (the
  path users take; also checks the detector's recall on our own output). CLI: `vlogcut
  style` became `style build | show | diff`, since argparse cannot mix positional
  references with subcommands and the interface was unreleased. The initial round-trip
  bar was relative (under a third of the fast-vs-slow reference distance): the rules
  planner gave all clips nearly one length (one shared offset), pushing p10 above and
  p90 below the reference's and leaving even a perfect run at ≈ 0.3 octaves. This known
  gap was resolved by reference-distribution sampling on 2026-10-15, adding absolute
  round-trip limits. No `eval/` directory yet: the first eval doubles as a unit test;
  real-footage protocols and results live in the private notes.
- 2026-10-15: rules planner length sampling — Q(u) is piecewise linear through
  (0.1, p10), (0.5, median), (0.9, p90), with flat tails to stay in p10..p90. Sample
  at j/(k-1), or the median for k=1, to reproduce inclusive duration quantiles.
  Choose k = round(target / expected_length), at least one, using Q's integral
  0.3·p10 + 0.4·median + 0.3·p90; the median over-counted clips for right-skewed
  profiles. Assign descending targets by descending usable cap, then score, then
  earlier chronological order. Ties are deterministic; no interleaving or pacing
  curve yet. Each `kept_for_audio` shot starts with its cap, replacing the nearest
  remaining target, and still starts at the usable segment's beginning. The existing
  shared-offset fit, p10..cap bounds and ±10 % clip-count adjustment remain unchanged;
  selection and placement are unchanged. Offline simulation: fast reference
  0.50 / 0.77 / 1.00 s, 12 roughly 3 s candidates, target 9 s gives 0.017 octaves
  (previously ≈ 0.26); default pacing at 60 s gives ≈ 0.03. Slow-reference distance
  only improves ≈ 0.33 → 0.18 because roughly 3 s footage cannot fit its 4.7 s p90.
  These are simulation figures, not measurements from real footage.

- 2026-10-18: transition mix — **schema change**: optional `StyleProfile.transitions`
  holds `TransitionMix(hard_cut, dissolve)`; absent fields in old profiles mean not
  measured. Shares pool surviving dissolve boundaries / total surviving boundaries
  across references, after short-shot merging, matching the cuts/min denominator.
  Fades through black count as two dissolves (known classification gap; no fade field).
  If any reference has dissolve detection off, or there are no surviving cuts, the
  mix is `None`. To support uncut references, cuts/min now permits zero; its distance
  is zero against zero and infinite against a positive rate. Single-shot reference
  duration quantiles all equal that shot's length; single-clip EDLs still raise the
  existing pacing error. Reference caches
  require `dissolve_cuts` (a subset of `cuts`); validation failure invalidates old
  records and reruns detection instead of assuming zero dissolves. Style distance
  adds the absolute difference of dissolve shares, skipped when either is unmeasured.
  EDL profiles stay unmeasured until clips carry transitions; planning and rendering
  are unchanged.

## Open questions

- How many variants per run
- JianYing draft format: does the installed version encrypt drafts? (spike, week 2)
- PoC inputs: own phone footage vs. stock clips; which two creators as references
- Where music analysis (BPM, beat times) lives: a `MusicTrack` record next to assets?
- Learned ranker: how many drafts / preference pairs before it beats the quality-only
  score
- Privacy mode details
- Music: auto-pick / beat-match from a user's own library (week 3), and whether
  voice cloning for narration (user's own voice, local, explicit consent) belongs in a
  post-v1 plugin
- Quality thresholds: calibrate on a real phone-footage eval set; per-project relative
  thresholds?
