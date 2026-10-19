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
- Shot kind: `talking` / `broll` from the existing `kept_for_audio` signal via
  `analyze/kind.py`, populated by the pipeline on fresh analysis and cache hits.
  This is the rules planner's audio-activity heuristic, not speech-specific VAD;
  rejected shots remain unmeasured because quality analysis clears their warnings.
  Caption-based refinement remains planned; EDL-derived profiles can measure
  talking/b-roll alternation from these kinds
- Caption: **captioner** role (VLM) on 1–3 keyframes → description, tags, interest 1–5.
  The core and cached batch layer exist in `analyze/caption.py` via `ModelRouter`,
  with rejected-shot filtering and caching (model name in the cache key).
  Pipeline/CLI wiring exists as opt-in `--caption`, populating `Shot.caption`,
  `tags` and `interest`, now exposed to the LLM planner in the shot table.
  Within each non-None `dup_group`, only the first eligible shot in input order
  uses the captioner/cache path; other eligible members receive its description,
  tags and interest with independent tags lists. Rejected shots are excluded by
  default; ungrouped shots retain per-shot captioning.
- Cut-point candidates: energy-based audio pauses populate `Shot.cut_points` during
  analysis. The rules planner shifts fixed-length clips onto nearby candidates after
  length fitting and rounding. Beats of the chosen music, motion minima and LLM
  planner use remain planned
- Output: `Shot` records, cached in `workspace/<project>/`. Per-asset work is
  independent; `pipeline.py` can run it in a thread pool via `workers` / `--jobs`
  (default 1, sequential).

### 3. Style profile (from references)
Run shot detection, transitions, VAD and (later) captions on reference videos (quality
flags are skipped: references are finished edits), then aggregate:
- Pacing: shot-duration p10/median/p90, cuts per minute; intro-vs-body median ratio
  is measured under structure; a full pacing curve remains planned
- Music: beat-aligned cut ratio (cut within ±80 ms of a beat in the reference's own
  audio), pooled over references with detected beats via `analyze/beats.py`.
  Chosen-music beat snapping is wired as opt-in via `pipeline.cut(music=...)`.
  Profile BPM and energy curve remain planned
- Structure ("grammar"): montage bursts are now measured as the share of shots in
  runs of at least three shots shorter than half the pooled median, within each
  reference. Intro-vs-body pacing is the pooled median duration of shots starting
  before 20 s divided by the pooled body median (at least two shots in each pool).
  Talking ↔ b-roll alternation is measured for EDL-derived profiles supplied with
  analyzed shots; references skip quality analysis and leave it unmeasured.
  Measured alternation contributes to style distance and appears in style show/diff;
  the LLM prompt describes a measured value, including zero, as guidance. The
  rules planner swaps eligible shots to approach it without reordering clips;
  validation still ignores this dimension.
  Intro length, speech ratio and cold-open measurement remain planned. Pacing alone
  reads as a speed slider; structure is what makes a cut recognisably someone's.
  Cold-open allowance is user-set. The rules planner uses the intro pacing ratio
  to shape clip lengths before fitting the total, uses measured `montage_share`
  to group short clips into consecutive non-talking bursts, and can produce one opt-in cold
  open with `pipeline.cut(cold_open=True)` or `vlogcut cut --cold-open` when a
  `--style` profile has `cold_open_clips > 0`. The LLM planner prompt describes
  measured `montage_share`, `intro_pace_ratio`, `kind_alternation` and the cold-open allowance;
  validation is unchanged.
- Transitions: share of hard cuts / dissolves (from `analyze/transitions.py`); fades
  through black currently count as two dissolves
- Audio: speech-over-music ratio, whether clip audio is kept under music
- Look: brightness / saturation / warmth stats measured from reference proxies into
  `StyleProfile.look`, using the pure helper in `analyze/look.py`. Samples are cached
  per reference and pooled by frame count. Style distance uses all three look
  dimensions; planners do not yet use them.
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

`learn/ranker.py` now provides pure-Python linear pairwise logistic training on
chosen/skipped `Shot` pairs, with deterministic full-batch gradient descent and
standardisation stored alongside weights in a JSON-round-trippable `RankerModel`.
Features use existing shot durations, usable ranges, quality metrics, warning and
cut-point counts, and interest; missing optional values have explicit indicators.
Training and scoring run locally on CPU with no new dependencies. Callers decide
whether to exclude rejected shots. The shot table and rules planner accept optional
precomputed ranker scores keyed by shot ID, without importing the ranker. The rules
planner blends 50% existing shot score with 50% within-table ranker percentile;
ties use average rank, and single or missing scores default to 0.5.
`pipeline.cut(ranker=...)` accepts a trained `RankerModel`; `cut --ranker MODEL_JSON`
loads and validates it before analysis. Every analyzed shot is scored after optional
captioning, so interest is available, and scores reach rules planning (including
LLM-failure fallback) and the LLM prompt.
`learn/feedback_pairs.shot_pairs_from_preferences` derives training tuples from
variant preference pairs, EDLs keyed by file stem, and analyzed shots. It pairs
each chosen-only shot with each rejected-only shot, excluding shared and unknown
shots, preserving first-clip order and deduplicating ordered shot-ID pairs.
Draft import and a CLI to train and save a `RankerModel` from feedback remain planned.
No CLIP embedding is used yet. No end-to-end "video → edit" model: there is no data
for it, it is not explainable, and it would break principle 2.

### 4. Plan (**planner** role, or rules + ranker)
- Input: shot table as text (id, time, duration, kind, caption, tags, quality, ranker
  score, cut-point candidates), `StyleProfile`, user brief ("~3 min, cozy, focus on
  the café"), variant knobs
- `plan/shot_table.py` accepts optional precomputed `ranker_scores` keyed by shot ID.
  When supplied, a `rank` column follows interest: two decimals, `-` for missing
  scores, higher = more likely chosen, relative within this table only. Scores for
  supplied shots must be finite; unknown IDs are ignored. Without scores the table
  is unchanged. `cut --ranker MODEL_JSON` supplies scores through `pipeline.cut`
  and the optional `ranker_scores` argument in `plan/llm.py`.
- `plan/rules.rough_cut` accepts optional precomputed `ranker_scores` with the same
  higher-is-better convention and finite-value validation for every known shot,
  including rejected or too-short shots; unknown IDs are ignored. Among eligible
  candidates with scores, convert zero-based average ranks to percentiles by
  dividing by n - 1. Ties share their average rank; a single scored shot and
  missing scores get 0.5. Blend `0.5 * existing_score + 0.5 * percentile` for
  selection, length assignment and cold-open ranking. The default quality score
  already lies in [0, 1]. None or empty mappings preserve EDL bytes; hard filters,
  chronology and pacing rules are unchanged. No ranker import is needed in the
  planner: `cut --ranker MODEL_JSON` scores shots in the pipeline and forwards
  the mapping to both default rules planning and the fallback after LLM failure.
  Omitting the ranker preserves EDL and prompt bytes.
- After clip-count selection, measured `structure.kind_alternation` enables a
  bounded greedy swap pass before length assignment, montage shaping, snapping
  and cold-open selection. It reuses eligible candidates and blended scores,
  preserving chronology, count, duplicate-group adjacency and duration capacity.
  Missing profile, structure or alternation preserves the existing EDL bytes.
- Output: `EDL`; the rules planner aligns an in- or out-point to a cut-point candidate
  when a fixed-length window fits within the usable part and the maximum shift
- `plan/llm.py` provides deterministic prompt building and EDL parsing via
  `ModelRouter` with the `planner` role, plus a bounded repair loop feeding parse
  errors and hard-check violations back into the same conversation. The LLM planner
  and repair loop are opt-in via `pipeline.cut(planner="llm", router=...)` or
  `vlogcut cut --planner llm`; the rules planner stays the default. The CLI reads
  `--config` (default `config.yaml`) for LLM planning or `--caption`, sharing one
  router when both are requested, and accepts `--brief`. Optional edit constraints
  reach the LLM planner via `pipeline.cut(constraints=...)` or `cut --constraints`.
  Non-blank strings appear in input order, one per line in a labelled section of
  the initial prompt; repair rounds reuse that conversation without repeating the
  section. Missing, empty or blank-only constraints preserve prompt bytes. Rules
  planning (including fallback) ignores constraints with one warning. Constraints
  do not affect validation, deterministic repair, snapping or rendering.
- If the shot table is too long: two stages — pick segments, then pick shots per segment
- `plan/rules.py` (+ ranker) is the no-model baseline and the fallback. The LLM planner
  becomes the default only once the eval shows it beats that baseline on style distance
  and blind preference; its distinct value is the brief and narrative ordering.

### 5. Validate & repair
- Checks now run after planning in `pipeline.cut`, using the analyzed shots, requested
  target duration and style profile (or `None` without `--style`). Every run writes
  a JSON list to `edl/<stem>.violations.json`, replacing any previous report even
  when empty. The CLI reports the results; violations are non-fatal and rendering
  continues. Check coverage and current limits are recorded under 2026-10-26 below.
- Deterministic post-processing: `pipeline.cut(music=...)`, reachable via
  `vlogcut cut --music PATH`, decodes user music with
  the existing mono-audio loader and detects beats via `analyze/beats.detect_beats`.
  After rules or LLM planning, `plan/snap.snap_to_beats` snaps boundaries to the
  nearest beat within its default tolerance, before checks and deterministic repair.
  Music starts at timeline 0; without music there is no snapping and EDL bytes are
  unchanged. No detected beats logs a warning and leaves timing unchanged; missing
  music files fail before planning and missing analysis dependencies raise ImportError.
  Optional music cue ends and linear fades are applied deterministically during
  preview mixing; pipeline-created cues keep defaults. `MusicTrack` and subtitle
  timing from ASR remain planned.
- Hard checks (code): shot ids exist; in/out inside shot bounds; no rejected shots;
  total duration within ±10 % of target; clip durations within the profile range
  (with tolerance); no back-to-back clips from the same `dup_group`; chronological
  order, except an optional cold open of at most `StyleProfile.cold_open_clips` clips
  at the start (a hook from later in the day, then back to the beginning); transition
  types and audio modes allowed by the profile
- A pure rule-based fallback helper, `validate/repair.repair_edl`, drops unknown or
  rejected clips and repairs transitions without changing retained selections or
  timing. It now runs in `pipeline.cut` when checks report a repairable violation.
  Successful repair atomically replaces the EDL JSON and re-runs validation before
  rendering, so the report, `CutResult.violations` and preview reflect the final EDL.
  If all clips would be dropped, a warning is logged and the original EDL and
  violations are kept. `CutResult.repaired` and an extra CLI line report applied
  repairs; clean EDLs are unchanged. The LLM repair loop now exists separately in
  `plan/llm.py`: structured errors are sent back to the planner for at most
  `max_rounds` extra rounds, without calling deterministic repair. It runs before
  this validation/repair/render flow with `--planner llm` (two extra rounds by
  default). If no reply parses, a warning is logged and rules planning takes over;
  every resulting EDL goes through the same checks and deterministic repair.
  With user music, these checks and repairs receive the beat-snapped EDL.

### 6. Render / export
- Preview mp4 (`render/preview.py`): FFmpeg on proxies (fast, low-res); cached clip
  bodies and centred dissolves per `Clip.transition_in`, joined with video copy;
  linear audio crossfades; debug subtitle track with source + reason per clip.
  Per-clip camera audio: keep at normal level, mute to silence, or duck by a fixed
  render gain (default -12 dB), applied to each source before dissolve crossfades.
  Music remains audible independently of clip audio modes. Each cue can end early
  and fade linearly in at its start or out at its effective end, capped to the
  timeline end. Explicit fades are shortened to the available cue span if needed;
  they may overlap after timeline truncation. A positive cue fade-out overrides
  `PreviewParams.music_fade_out_s`; zero keeps that existing fallback. Default cues
  preserve the exact music filter graph and command.
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
  pick / reject buttons that save a feedback JSON (no server). The static page
  generator and JSON loader in `feedback/page.py` are wired through
  `vlogcut feedback page PREVIEW [PREVIEW ...] [-o variants.html]`.
  `vlogcut feedback pairs FEEDBACK_JSON [--context TEXT] [-o pairs.json]` derives
  preference-pair JSON for storage or stdout.
- Feedback (pick / reject / free text) → **feedback** role first *classifies* it:
  "the café part was too short" is about this edit and becomes a re-plan constraint;
  "too fast overall" is about style and becomes a validated `StyleProfile` delta.
  Mixing the two would drift the profile.
  `vlogcut feedback classify FEEDBACK_JSON --profile PROFILE_JSON [-o OUT_PROFILE_JSON]`
  uses `feedback/classify.py` through the router's `feedback` role (`--config`, default
  `config.yaml`). Non-blank text is classified against the input profile; verdict-only
  records are skipped without model calls. Pure profile deltas apply in record order.
  Output defaults to `<profile stem>.updated.json` beside the input; only explicit `-o`
  can replace the input profile. Edit constraints print one per line, or are saved as a
  JSON list with `--constraints-out CONSTRAINTS_JSON`. A stdout summary counts classified
  records, applied deltas and constraints. Re-plan with
  `vlogcut cut FOOTAGE --planner llm --constraints CONSTRAINTS_JSON`; the CLI loads
  this JSON list of strings before analysis and reports missing or invalid files
  as a one-line error with exit 1. Re-planning from constraints is supported only
  by the LLM planner; the rules planner logs that constraints are ignored.
  Classification itself is not run inside `cut`.
- Every (context, variant A, variant B, choice) is logged as a preference pair and
  feeds the ranker (§3b)

## Data contracts

Defined in `src/vlogcut/schemas.py`:

| Model | Produced by | Consumed by |
|---|---|---|
| `MediaAsset` | ingest | analyze, render |
| `Shot` (+ `QualityFlags`) | analyze | plan, validate, render |
| `StyleProfile` | style | plan, validate, feedback |
| `EDL` (+ `Clip`, `Transition`, `MusicCue`) | plan | validate, render |
| `Run`, `RunInputs` | pipeline | run manifest (readers planned) |
| `Feedback`, `PreferencePair` | feedback | ranker / feedback role |

Changing a schema means updating this file and the tests in the same change.

`Feedback` records `variant_id: str`, `verdict: "pick" | "reject" | None = None`
and free-form `text: str = ""`. The ID must be non-empty after stripping whitespace;
at least a verdict or non-blank text is required. `variant_id` identifies an EDL
file stem, such as `coverage_180s_fast`; run manifests do not change that identity.
`PreferencePair` records `context: str` (project/brief description, possibly empty),
`chosen: str` and `rejected: str` (variant IDs). Both IDs must be non-empty after
stripping whitespace, and `chosen != rejected`. Supplied strings are preserved.
The static variants page downloads a JSON array of `Feedback` records, validated by
`feedback.load_feedback` and ready for `feedback.preference_pairs`. Preference-pair
persistence is available through `vlogcut feedback pairs -o`; the pure
`learn.shot_pairs_from_preferences` helper converts these records into ranker
training pairs. A CLI to train and save the ranker from feedback remains planned.

`Shot.cut_points` holds code-proposed `CutPoint(time_s, kind)` candidates at absolute
source-asset times, on the same timeline as `Shot.start_s` and `end_s`. Times must be
sorted, unique, and strictly inside the shot; they need not be inside `usable_range`
(the planner decides that). The field defaults to an empty list, so old shot JSON
still loads. `kind` is only `pause` for now; other kinds will add literals later.

`StyleProfile.beat_aligned_ratio` pools aligned cuts / total cuts over references with
detected beats. It is `None` if beat detection is disabled for any reference, no
reference has beats, or the pooled cut count is zero. `speech_ratio` remains `None`
until its analysis is wired in: `None` means "not measured", which is not the same as 0.
`StyleProfile.transitions: TransitionMix | None = None` follows the same convention:
`hard_cut` and `dissolve` are shares in [0, 1], summing to 1 within 1e-6. Old profile
JSON without this field still loads. The mix is unmeasured when dissolve detection is off
or no cuts survive merging. Profiles still require at least one cut to describe pacing.

`LookStats` holds brightness (mean Rec.709 luma) and saturation (mean HSV saturation)
in [0, 1], warmth (mean R-B) in [-1, 1], and a positive integer `frames` count.
All values are finite. `StyleProfile.look: LookStats | None = None` stores frame-weighted
means across references. `None` means not measured (disabled or no sampled frames),
distinct from measured black. Old profile JSON without `look`, or with the previous
empty `{}` placeholder, loads with `None`; measured values round-trip with frame counts.
EDL-derived profiles leave look unmeasured. The analysis module re-exports the shared
model so existing helper imports continue to work. Style distance uses brightness,
saturation and warmth when both profiles have look; planners still ignore look.

`StyleProfile.structure: StructureStats | None = None` holds measured structure
dimensions. `StructureStats.montage_share` is a share in [0, 1]. A burst is a
maximal run of at least three consecutive shots strictly shorter than half the pooled
profile median. Runs stay within each reference's timeline-ordered, post-merge shot
durations; the share pools shots inside bursts / total shots, not per-reference ratios.
EDLs use speed-adjusted clip timeline durations in clip order as one sequence.
The rules planner uses measured positive `montage_share` to request
`round(montage_share * clip_count)` clips in consecutive non-talking runs of at
least three, preferring later clips. Short lengths stay at or above p10 and below
half the profile median after shared-offset fitting where feasible. Insufficient
room can reduce the burst count; fewer than three requested clips, p10 at or above
half the median, or no feasible run leave planning unchanged. Missing structure,
`montage_share` of `None`, and zero preserve the previous EDL JSON. The LLM planner
prompt describes the measured share, including zero, and the half-median burst
threshold in seconds; validation is unchanged.
`None` means not measured, distinct from a measured zero; old profile JSON without
`structure` still loads with `None`. The pure helper returns `None` for zero shots;
profile construction retains its existing pacing errors for insufficient cuts.
`StructureStats.intro_pace_ratio: float | None = None` is strictly positive when
measured: pooled median intro shot duration / pooled median body shot duration.
Each reference restarts at zero; whole shots starting strictly before 20 s belong
to the intro, all others to the body. Pool durations, not per-reference ratios,
using the same ordered, post-merge durations (speed-adjusted clips for EDLs).
Fewer than two shots in either pool yields `None`, meaning not measured. Old JSON
with structure but without this field loads with `None`. The rules planner uses a
measured ratio to scale intro base lengths before fitting the total, preserving
the cap-based assignment for `kept_for_audio` clips. Unmeasured ratios leave EDL
JSON unchanged. The LLM planner prompt describes the measured ratio and an
approximate intro median in seconds, using the profile median as the body estimate;
validation is unchanged.

`StructureStats.kind_alternation: float | None = None` is the share of adjacent
known-kind pairs whose kinds differ, constrained to [0, 1]. Unknown kinds break
adjacency: skip pairs touching `None`, never bridge across them. Zero eligible
pairs yields `None` (not measured), distinct from measured zero (all eligible
pairs have equal kinds). Old structure JSON without the field loads with `None`.
Unmeasured alternation is omitted from serialized profiles to preserve existing
JSON bytes; measured zero is included. Only `profile_from_edl(..., shots=...)`
measures it, mapping shot IDs through `analyze/kind.classify_kind` in clip order;
unknown IDs map to `None`. Reference videos skip quality analysis and leave
alternation unmeasured. Style distance uses its absolute difference when both
profiles measure it, including zero; otherwise it is skipped and reported as
unmeasured. Style show displays measured values; style diff displays a row when
both sides measure it, otherwise listing it in the unmeasured summary. The LLM
planner's initial prompt describes a measured value, including zero, with two
decimal places as guidance on adjacent talking/b-roll pairs switching kind.
Unmeasured alternation adds no line and preserves prompt bytes; repair rounds
reuse the conversation without repeating the line. The rules planner uses measured
alternation, including zero, for deterministic single-shot swaps after selection
and before length assignment. Each swap strictly reduces the absolute target gap;
unknown kinds break adjacency and zero eligible pairs leave selection unchanged.
Validation remains unchanged and still ignores this dimension.

`StyleProfile.cold_open_clips` is a non-negative, user-set integer defaulting to 0
(no cold open), including when absent from old profile JSON. It exempts the first
`min(cold_open_clips, len(clips))` EDL clips from chronology only: a hook from later
in the day can precede the day from the beginning. Chronology starts afresh among
the remaining clips; all other checks still apply to the opening clips. Reference
analysis does not measure this allowance and keeps the default. It is not a style
distance dimension. The rules planner can move one later hook to the front when
requested via `pipeline.cut(cold_open=True)` or `vlogcut cut --cold-open` and this
allowance is positive; it is off by default. Clip timing is preserved,
the remaining clips stay chronological, and newly joined duplicate groups are avoided.
The LLM planner prompt describes a positive allowance as at most N leading hook
clips before the chronological remainder; validation is unchanged. Missing structure
dimensions add no prompt lines. Without structure and with zero cold-open allowance,
or without a profile, prompt bytes are unchanged.

`Clip.transition_in: Transition` defaults to a hard cut, so old EDL JSON still loads.
`Transition.kind` is `cut` or `dissolve`; `duration_s` is 0 for a cut and strictly
positive for a dissolve. The first EDL clip must enter with a cut. A dissolve is
centred on the boundary: it changes no clip in/out points, timeline durations, or
EDL total. The preview renders overlapping source handles, holding edge frames
when an asset ends or has not started. `profile_from_edl` measures the
dissolve share over the `len(clips) - 1` boundaries (and still rejects single clips).

`Clip.audio` is `keep` (camera audio at normal level), `mute` (camera audio replaced
by silence; music still plays), or `duck` (camera audio attenuated by a fixed gain
to sit under music). It defaults to `keep`, so old EDL JSON without the field still
loads with the same audio behaviour.

`MusicCue.end_s: float | None = None` is an optional absolute timeline stop time;
when set, it must be finite and strictly greater than `start_s`. `None` plays to
the timeline end. `fade_in_s: float = 0.0` and `fade_out_s: float = 0.0` are finite,
non-negative linear fade durations. When `end_s` is set, their sum must not exceed
`end_s - start_s`. Preview mixing caps the end to the timeline duration and clamps
explicit fades to the remaining cue span. A zero cue fade-out retains the preview's
global fade-out fallback. Each new field is omitted from serialization at its
default value, including explicitly supplied defaults: old EDL JSON loads and
default-cue EDL bytes remain identical; non-default values round-trip. Pipeline
and CLI music creation still use defaults, with no new flags or ducking parameters.

`Shot.kind: Literal["talking", "broll"] | None = None` makes the existing planner
kind explicit. `None` means not measured, distinct from either kind; old shot JSON
without the field still loads with `None`. The pure `analyze/kind.classify_kind`
prefers an explicit kind, otherwise maps the `kept_for_audio` quality warning to
`talking` and its absence on eligible shots to `broll`. Rejected shots return `None`
because quality analysis clears their warnings. The flag comes from audio activity
saving an otherwise shaky shot, so this is a coarse heuristic, not confirmed speech
or confirmed absence of speech. The pipeline backfills legacy quality-cache records
after loading without re-analysis or cache invalidation, and writes the field in
`shots.json`. Rules scoring, length assignment and placement share this helper;
legacy eligible shots retain identical EDL bytes. Caption refinement remains planned.

`RunInputs` records sorted `asset_ids`, `target_s`, selection `strategy`, requested
`planner` (`rules` / `llm`), and `planner_model` (configured model for LLM planning,
otherwise `None`, even with a captioner router). LLM fallback retains the requested
planner identity. Optional `profile_sha256` and `ranker_sha256` are full hex SHA-256
digests of the loaded models' compact `model_dump_json()` encoded as UTF-8. This
normalizes source JSON whitespace, key order and omitted defaults through model
validation. `music_sha256` hashes the file bytes, streamed without decoding. Missing
objects have `None` hashes. `brief` defaults to `""`, `constraints` to an independent
empty list, and `cold_open` / `caption` to `False`. Constraints retain their order
and exact strings even when ignored by rules planning.

The pure `run.run_id(inputs)` returns the first 16 hex characters of SHA-256 over
`RunInputs.model_dump_json()` encoded as UTF-8. `Run` stores that `run_id`, `inputs`,
the EDL file stem as `variant_id`, and `edl`, `violations`, and optional `preview`
paths (`None` when rendering is off). Workspace-internal paths are relative POSIX
strings, following the cache convention below; explicitly external previews retain
absolute paths. After final outputs are written, `pipeline.cut` atomically replaces
`runs/<run_id>.json` through a `.json.part` file and returns its path as
`CutResult.manifest`; the CLI prints it. No timestamps are included: identical
inputs and output locations produce identical manifest bytes. Fingerprints do not
rename or snapshot outputs, so distinct manifests can point to the same overwritten
EDL. Rendering options and other fields outside `RunInputs` are not fingerprinted.

Planned additions (remaining fields land with the module that needs them):

| Field / model | Why |
|---|---|
| `MusicCue` ducking params | music mixing needs a level under speech |
| `StyleProfile.structure` | exists with `montage_share`, `intro_pace_ratio` and EDL-derived `kind_alternation`; reference alternation, intro length and cold-open measurement remain planned |
| `Run` manifest (inputs fingerprint → outputs) | exists with `RunInputs`; file names are unchanged and variant IDs remain EDL stems |

Cached JSON stores paths inside the workspace as relative POSIX strings, so a moved
workspace stays valid. External source paths stay absolute. Cache readers resolve
relative paths against the current workspace and preserve legacy absolute paths.
In-memory media and keyframe paths remain absolute strings with unchanged schema
types. Reference analysis preserves the supplied source string for
`ReferenceCuts.source` and `StyleProfile.source_refs` on both cache misses and hits.

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

- [x] Week 3 · style: talking/b-roll alternation measured for EDL-derived profiles (schemas.py, style/)
- [x] Week 3 · analyze: Shot.kind talking / broll from the existing speech signal (schemas.py, analyze/kind.py)
- [x] Week 4 · learn: shot pairs derived from variant preference pairs (learn/feedback_pairs.py)
- [x] Week 4 · plan: opt-in ranker scores in cut via --ranker (pipeline.py, cli.py, plan/llm.py)
- [x] Week 4 · plan: rules planner blends optional ranker scores (plan/rules.py)
- [x] Week 4 · plan: optional ranker score column in the planner shot table (plan/shot_table.py)
- [x] Week 4 · learn: pure-Python pairwise shot ranker (learn/ranker.py)
- [x] Week 5 · feedback: edit constraints fed into the LLM planner via cut --constraints (plan/llm.py, pipeline.py, cli.py)
- [x] Week 3 · style: look stats in style distance (style/distance.py)
- [x] Week 3 · style: look stats from references (schemas.py, style/profile.py)

- [x] Week 4 · plan: LLM planner prompt describes structure dimensions (plan/llm.py)
- [x] Week 4 · cache: workspace-relative paths in cached JSON (paths.py, analyze/shots.py, analyze/quality.py, style/profile.py, pipeline.py)
- [x] Week 5 · feedback: CLI classify subcommand applying profile deltas (cli.py)
- [x] Week 3 · structure: rules planner produces montage bursts from montage_share (plan/rules.py)
- [x] Week 3 · music: --music CLI flag for beat snapping (cli.py)
- [x] Week 3 · music: opt-in beat snapping of planned EDLs to user music (pipeline.py)
- [x] Week 4 · analyze: opt-in captioning wired into pipeline and CLI (pipeline.py, cli.py)
- [x] Week 4 · analyze: captioner core via ModelRouter (analyze/caption.py)
- [x] Week 5 · feedback: feedback-role classification into edit constraints / profile deltas (feedback/classify.py)
- [x] Week 3 · structure: rules planner honours intro_pace_ratio (plan/rules.py)
- [x] Week 3 · structure: opt-in cold open in the rules planner (plan/rules.py)
- [x] Week 3 · structure: opt-in cold open wired into pipeline.cut and CLI (pipeline.py, cli.py)
- [x] Week 5 · feedback: CLI page + pairs subcommands (cli.py)
- [x] Week 5 · feedback: static variants page + feedback JSON loader (feedback/page.py)
- [x] Week 5 · feedback: Feedback / PreferencePair contracts and pair derivation (schemas.py, feedback/pairs.py)
- [x] Week 4 · pipeline: optional thread pool for per-asset analysis (pipeline.py, cli.py)
- [x] Week 4 · plan: opt-in LLM planner wired into pipeline.cut and CLI (pipeline.py, cli.py)
- [x] Week 4 · plan: LLM repair loop feeding violations back to the planner (plan/llm.py)
- [x] Week 4 · plan: LLM planner core, prompt + EDL parsing via ModelRouter (plan/llm.py)
- [x] Week 4 · plan: shot table as planner input text (plan/shot_table.py)
- [x] Week 4 · plan: caption, tags and interest in the planner shot table (plan/shot_table.py)
- [x] Week 4 · plan: rules planner avoids back-to-back dup_group clips (plan/rules.py)
- [x] Week 4 · validate: deterministic repair wired into pipeline.cut after checks (pipeline.py, cli.py)
- [x] Week 4 · validate: deterministic repair helper for unknown/rejected clips and transitions (validate/repair.py)
- [x] Week 4 · validate: checks run on every planned EDL, reported by the CLI (pipeline.py, cli.py)
- [x] Week 4 · validate: deterministic hard checks (validate/checks.py)
- [x] Week 3 · cut-point candidates: audio pauses → Shot.cut_points → rules planner placement (analyze/pauses.py, plan/rules.py)
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
- [x] Week 3 · structure: StyleProfile.cold_open_clips honoured by chronology check (schemas.py, validate/checks.py)
- [x] Week 3 · structure: montage-burst share from references and EDLs (schemas.py, style/profile.py, style/distance.py)
- [x] Week 3 · structure: intro-vs-body pacing ratio from references and EDLs (schemas.py, style/profile.py, style/distance.py)
- [x] Week 3 · style: transition mix from references (`style/profile.py`)
- [x] Week 3 · style: beat-aligned cut ratio from references (style/profile.py)

Week 1 complete (2026-10-14): `vlogcut cut <folder>` runs end to end.

## Evaluation (starts with the PoC; `eval/` does not exist yet)

- Round-trip test (offline, synthetic footage): reference → profile → cut → profile of
  the cut; the distance must be small. This is the first eval and doubles as a unit test:
  `tests/test_style_distance.py` cuts generated footage with a fast and a slow reference
  profile and requires the round trip (via the EDL and via the rendered preview) to land
  within a third of the distance between the two references. The fast-reference EDL
  distance must also be < 0.1 octaves; the rendered-preview distance must be < 0.2.
  Look measurement is disabled for these comparisons: synthetic footage differs in
  look from the references, and planners do not match appearance.
- Style distance (`style/distance.py`): per dimension |log2(a / b)| for the positive
  pacing quantities (shot-duration p10 / median / p90, cuts per minute, and
  `intro_pace` from `structure.intro_pace_ratio`), so the unit is
  octaves (1.0 = twice as fast or slow); absolute difference for the ratio dimensions in
  [0, 1]: `beat_aligned_ratio`, `speech_ratio`, `dissolve` (`transitions.dissolve`),
  `montage_share` (`structure.montage_share`), `kind_alternation`
  (`structure.kind_alternation`), and look `brightness` and `saturation`.
  Look `warmth` uses |a - b| / 2 to map its [-1, 1] range onto [0, 1].
  A dimension unmeasured on either side is skipped and reported; if either look is
  `None`, all three look dimensions are skipped. Total is the mean over measured
  dimensions with equal weights until the eval set says otherwise.
  `vlogcut style diff a.json b.json` prints it.
- ~10 footage sets, each with reference videos
- Metrics: style distance (shot-duration distribution distance, beat-alignment gap,
  structure / transition-mix distance), constraint violations, repair rounds, cost and
  latency per video, blind human preference
- Baseline comparison: rules + ranker vs. LLM planner on the same sets
- Every model or prompt change is run against the eval set before merging

## Decisions log

- 2026-10-19: music cue timing — **schema change**: optional finite `end_s` must
  follow `start_s`; finite non-negative `fade_in_s` and `fade_out_s` must fit their
  sum within an explicitly bounded cue. Omit each new default field via a wrap
  serializer to preserve old EDL bytes. The pure `render/preview.music_cue_filter`
  builds timeline-capped trims and linear fades without FFmpeg or model calls.
  Positive cue fade-out overrides the existing preview-wide fade-out; zero retains
  that fallback, keeping default graphs and commands identical. Explicit fades
  clamp independently to the remaining cue span and can overlap after truncation;
  cues starting at or beyond the timeline end add no audible music. Pipeline/CLI,
  beat snapping, validation checks and prompts are unchanged; ducking stays planned.

- 2026-10-19: rules planner kind alternation — a pure post-selection pass runs
  only for measured `structure.kind_alternation`, including 0.0. Reuse
  `analyze/kind.classify_kind` and the style measurement helper: skip adjacent
  pairs touching `None`, never bridge them; no eligible pairs means no swaps.
  Consider single replacements from the unpicked, hard-filtered candidate pool,
  strictly between the pick's chronological neighbours, with a known kind
  different from the replaced kind. Neither new neighbour may share the
  replacement's non-None `dup_group`. Preserve clip count and capped duration
  capacity for the existing ±10% fit (or do not worsen an existing shortfall).
  Among swaps strictly reducing the absolute target gap, choose the highest
  blended candidate score, including ranker percentiles; ties use earliest source
  start, then shot ID. For the same candidate, prefer the smallest gap, then the
  earliest picked position. Recompute after each swap; stop without improvement
  or after at most the picked count. Clip-count fitting is unchanged; lengths are
  reassigned after the swaps. Final length assignment, intro pacing, montage
  bursts, snapping and optional cold open all consume the swapped set. Cold open
  can change measured adjacency.
  Missing profile, structure or alternation bypasses the pass and preserves EDL
  JSON bytes. Validation remains unchanged and ignores alternation; no schema,
  LLM prompt, repair, reference measurement or style-distance changes.

- 2026-10-19: LLM planner alternation guidance — describe measured
  `structure.kind_alternation`, including zero, with two decimal places alongside
  the existing structure guidance in the initial prompt. Unmeasured values preserve
  prompt bytes; repair messages do not repeat the line. Rules planning, validation,
  deterministic repair, snapping, rendering and schemas are unchanged.

- 2026-10-19: batch caption deduplication — the first member of each non-None
  `dup_group` that passes the existing rejection filter is the representative.
  Only its existing model-aware cache and captioner path are used; peers receive
  independent copies of its result in input order. If it has no keyframes, the
  group remains uncaptioned; if captioning fails, the existing error propagates.
  Never retry with another member. Ungrouped shots and the explicit
  `include_rejected` override retain their existing behavior. Inputs, schemas,
  router, cache keys and pipeline/CLI interfaces are unchanged.

- 2026-10-19: **schema change**: add `RunInputs` and `Run` manifests, with a pure
  16-character SHA-256 input fingerprint. Hash loaded profile/ranker model JSON
  in UTF-8 and stream music file bytes. Write deterministic, timestamp-free manifests
  atomically after final validation/repair and optional rendering; overwrite matching
  run IDs and expose the path through `CutResult` and CLI. Preserve EDL, violations
  and preview names/bytes and feedback variant IDs. Workspace-internal paths use
  relative POSIX strings; external preview paths follow the existing cache convention.
  This adds provenance without changing analysis caches or introducing readers.

- 2026-10-19: alternation distance and display — include measured `kind_alternation`
  as an equally weighted absolute-difference dimension, including measured zero.
  Skip and report it when either side is unmeasured, preserving previous totals.
  Shared style show/diff formatters display it using the existing share convention;
  without alternation, only the diff's unmeasured summary gains its name. Reference
  measurement, schemas, serialization, planners and prompts are unchanged.

- 2026-10-19: talking/b-roll alternation — **schema change**:
  `StructureStats.kind_alternation` is optional, bounded to [0, 1], and measured
  only by `profile_from_edl` when analyzed shots are supplied. Count differing
  known adjacent pairs / all known adjacent pairs in clip order. Unknown IDs and
  unmeasured kinds break adjacency; no eligible pairs means `None`, not zero.
  Reuse `classify_kind`, including explicit kinds and the `kept_for_audio` fallback.
  Reference analysis remains unchanged because it has no quality-derived kinds.
  Omit unmeasured alternation during serialization to preserve existing JSON bytes,
  including EDL profiles with omitted, `None` or empty shots. Use a wrap serializer
  compatible with the existing Pydantic minimum version. Keep `montage_share`
  required and its validation unchanged: successful EDL pacing always measures it,
  so an alternation-only structure cannot arise on this path. Supporting such a
  structure would require a separate change to the existing montage contract.
  Distance and style show/diff now consume alternation as described above;
  the LLM prompt now describes measured values as guidance. Rules planner swaps
  are described above; validation still ignores this dimension.

- 2026-10-19: Shot.kind reuses the rules planner's `kept_for_audio` heuristic;
  no new speech detector or dependency. Explicit kinds take precedence. Rejected
  shots are unmeasured because quality analysis discards their audio warning;
  eligible shots without the warning retain the planner's b-roll classification.
  Backfill in the pipeline after quality-cache loading without changing cache keys.
  The shot table has no kind column, so its prompt bytes remain unchanged.

- 2026-10-19: feedback-derived shot pairs — the pure
  `learn/feedback_pairs.shot_pairs_from_preferences` looks up EDLs by variant ID
  (file stem) and returns `(chosen Shot, skipped Shot)` tuples for `train_pairwise`.
  Use the cross product of chosen-only and rejected-only shot IDs present in the
  supplied shots; shared shots carry no selection signal. Missing EDLs skip the
  preference, with one warning per missing variant ID per call. Unknown shot IDs
  are ignored. Preserve preference order, then first-clip order on each side,
  chosen outermost; repeated clips contribute a shot only once. Deduplicate ordered
  shot-ID pairs across preferences, retaining both orientations when preferences
  conflict because the ranker's loss handles the conflict. Do not filter quality-
  rejected shots; callers decide. Inputs are unchanged. No I/O, model calls,
  dependencies, schemas, pipeline, CLI or ranker algorithm changes.

- 2026-10-19: rules planner accepts precomputed ranker scores without importing
  `vlogcut.learn`. Normalize only scored eligible candidates to [0, 1] using
  zero-based average ranks / (n - 1), averaging ties; a single score and missing
  scores default to 0.5. Blend 50/50 with the existing per-shot score, which
  `score_shot` already clamps to [0, 1], for selection, length assignment ties,
  duplicate replacement and cold-open ranking. Custom scorers should use [0, 1]
  when blending. Validate finite scores for all known shots before filtering;
  ignore unknown IDs. None or empty mappings preserve EDL JSON bytes. Candidate
  eligibility, usable ranges, chronology and pacing rules are unchanged.
  Pipeline/CLI wiring, draft import and feedback-derived shot pairs remain planned.

- 2026-10-19: first shot ranker uses linear pairwise logistic training in pure
  Python with no new dependencies. Population means and scales are stored in
  `RankerModel` with its weights; zero-variance features use scale 1. Features
  are limited to existing `Shot` and `QualityFlags` fields, with neutral defaults
  and missing indicators for optional values. No CLIP embedding yet. This is a
  standalone analysis artefact: no schema, planner, pipeline or CLI changes;
  draft import and feedback-to-shot pair derivation remain planned.

- 2026-10-19: feedback constraints reach the LLM planner through `cut --constraints`
  and `pipeline.cut(constraints=...)`. Load the JSON string list saved by
  `feedback classify --constraints-out`, rejecting missing files, invalid JSON
  and non-string-list shapes before analysis. Preserve input order, omit blank
  strings and flatten whitespace so each constraint occupies one prompt line.
  Add the labelled section only to the initial user message; both public LLM
  planning entry points accept constraints and repair rounds reuse the conversation.
  Absent or empty constraints preserve prompt bytes. Rules planning, including
  fallback, logs one warning for non-blank constraints and preserves EDL output.
  No schema, validation, deterministic repair, snapping or rendering changes.

- 2026-10-19: look stats in style distance — compare `StyleProfile.look` brightness
  and saturation by absolute difference, and warmth by half the absolute difference
  to normalize its [-1, 1] range. All three dimensions have equal weight alongside
  existing measured dimensions and appear in the diff table. If either look is
  `None`, skip all three and list them as unmeasured; existing totals and per-dimension
  values are unchanged. Disable look measurement in the synthetic round-trip test
  to retain its pacing thresholds. No schema, measurement or planner changes.

- 2026-10-19: reference look statistics — **schema change**: move `LookStats` from
  the analysis helper into `schemas.py` and replace the profile's dictionary placeholder
  with optional typed measurements. Sample 64×64 RGB frames from the reference proxy
  at approximately 1 fps, lowering the rate over long references to spread at most 32
  frames across the duration; sub-second references request one sample. An explicit
  ffmpeg output cap enforces the bound. Cache per-reference means and actual frame count,
  then reuse `pool_look` to weight by sampled frames, skipping unmeasured records.
  Look is `None` when disabled globally or for any reference, or no frames were sampled.
  `ReferenceParams.look=True` participates in cache matching; missing look data or a
  missing look flag invalidates old caches when requested. Explicit cached `look: null`
  is a valid empty measurement and avoids re-sampling. EDL look remains `None`.
  Distance, planners, prompts, rendering and colour matching are unchanged.

- 2026-10-19: portable workspace caches — `paths.py` converts paths only at JSON
  read/write boundaries. Shot-span and quality caches store `keyframes` relative
  to the workspace; the pipeline's `shots.json` snapshot uses the same format.
  Reference-cut caches store internal `source` paths relatively and external sources
  absolutely. Readers restore absolute strings before file checks or downstream use;
  old absolute-path caches still qualify as hits without migration or key changes.
  Reference analysis keeps the caller's source string for profile provenance on
  both misses and hits. The pipeline's `shots.json` snapshot has no reader yet.
  Ingest stores proxy media but no JSON; caption and pause caches contain no paths.
  Schema types, EDL output and the planned Run manifest are unchanged. Offline tests
  rename a populated workspace and fail if detection or quality analysis runs again.

- 2026-10-19: feedback classification CLI — `feedback classify` loads feedback and
  style JSON through the existing loaders, then builds `ModelRouter.from_yaml` just
  as `cut` does, requiring the `feedback` role. Classify each non-blank record once
  against the original profile, then apply deltas in input order with the existing
  pure helper. Verdict-only records make no model call; config is still required.
  Write the updated profile to `-o`, defaulting to `<stem>.updated.json` next to the
  input; replacing the input requires explicit `-o`. Constraints go to stdout one
  per line (embedded newlines become spaces), or to `--constraints-out` as a JSON
  list preserving the strings. Constraint output must differ from both profiles.
  Each file is replaced atomically after classification and application succeed.
  Report classified/delta/constraint counts on stdout; file, config and validation
  errors use one-line `_fail` messages and exit 1. No classifier, schema, pipeline,
  prompt, re-planning or ranker changes.

- 2026-10-19: music CLI — the existing `vlogcut cut --music PATH` forwards a
  `Path` to `pipeline.cut(music=...)` for opt-in beat snapping. It combines with
  `--style`, `--planner llm`, `--cold-open` and `--caption`; omission still passes
  `None`, preserving the default call and EDL bytes. Missing files fail before
  running the pipeline; dependency ImportErrors use a one-line error with the
  existing `[analyze]` install hint. No snapping status is exposed by `CutResult`,
  so no success summary is added. Pipeline behavior and schemas are unchanged.

- 2026-10-19: opt-in music beat snapping — `pipeline.cut(music=...)` treats music
  beats as timeline times starting at 0. Snap after either planner and before checks
  and deterministic repair, so reports and `CutResult.violations` reflect the final
  snapped EDL. No music means no snapping and byte-identical EDL output. Existing
  music cues and preview behavior are preserved; no schema or CLI changes.

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
  any analysis, so a bad file fails in milliseconds, not after minutes of proxies;
  the rules planner reads it (pacing p10 / median / p90), and validation now uses it
  after planning (2026-10-27). Outputs carry the profile's
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
  mix is `None` (a defensive guard; zero-cut references still raise the existing
  pacing error). Reference caches
  require `dissolve_cuts` (a subset of `cuts`); validation failure invalidates old
  records and reruns detection instead of assuming zero dissolves. Style distance
  adds the absolute difference of dissolve shares, skipped when either is unmeasured.
  EDL profiles now measure the mix from `Clip.transition_in`, and the rules planner
  assigns it (see the EDL transitions and preview dissolves entries below).
  `dissolve_cuts` are taken from the merged cut list.

- 2026-10-18: EDL transitions — **schema change**: `Transition(kind, duration_s)` and
  `Clip.transition_in` describe entry at each boundary. Old EDLs default to hard cuts;
  cuts require duration 0, dissolves require duration > 0, and clip 0 must be a cut.
  No `fade` kind until the reference detector distinguishes fades through black,
  which currently count as two dissolves. The rules planner assigns
  `n = round(dissolve_share * (k - 1))` dissolves at clip indices
  `floor((j + 0.5) * (k - 1) / n) + 1`, for `j = 0..n-1`: even, deterministic
  placement over the boundaries, never clip 0. Without a measured mix, or with fewer
  than two clips, all entries are cuts. Default dissolve duration is 0.5 s, capped
  at half the shorter neighbouring clip's final timeline duration, after fitting and
  in/out rounding. Selection, length sampling, fitting and placement are unchanged.
  Dissolves are centred on boundaries without changing in/out points, timeline
  durations or the EDL total; the preview now renders the media overlap (2026-10-19).
  `profile_from_edl` measures dissolves / `(len(clips) - 1)` and returns a measured
  `TransitionMix`, including zero dissolves for old EDLs; single clips still raise
  the pacing error. The preview supports these centred dissolves as of 2026-10-19.

- 2026-10-19: preview dissolves — a pure segment plan shortens clip bodies by half
  each adjacent dissolve and inserts a transition over `[b - d/2, b + d/2]`.
  Window edges snap on the cumulative frame grid; segment frame counts preserve
  the original total. Source handles respect speed and may extend beyond shot
  bounds; missing asset handles use cloned edge video frames and silent audio.
  Video uses `xfade=fade`, audio a linear `acrossfade`; 30 ms fades remain only
  at hard cuts and programme endpoints. Overlapping windows or windows outside
  the timeline raise a render error; the rules planner's capped durations fit.
  Bodies and transitions use the same H.264/PCM MOV format, canvas and fps, joined
  by the concat demuxer with video copy. Hard-cut-only cache keys are unchanged
  except for speed-adjusted silent sources: generated silence now uses signed
  samples before `atempo`, avoiding its unsigned-silence DC offset. Dissolve keys
  include both source spans, speeds, padding, duration, canvas and render params.
  Subtitle cues still cover each full resolved clip span.

- 2026-10-20: per-clip audio — **schema change**: `Clip.audio` adds `keep` (default),
  `mute`, and `duck`; old EDLs still load. `PreviewParams.duck_gain_db` defaults to
  -12 dB. Modes apply in clip bodies and separately to each dissolve source before
  the audio crossfade; mute uses signed-sample silence, including at non-unit speeds.
  Existing 30 ms fades, video output, frame snapping and timeline durations stay
  unchanged. Cache keys include non-default audio modes and the duck gain only for
  duck; keep keys remain byte-identical, including dissolve keys. The planner still
  emits `keep`; style assignment and measurement, `MusicCue.end_s` and music ducking
  parameters remain future work.

- 2026-10-21: audio pauses — `analyze/pauses.py` measures energy, with no VAD model.
  Defaults: 16 kHz mono, 20 ms RMS frames (digital silence floored at -100 dBFS),
  quiet below max(-50 dBFS, envelope p95 minus 25 dB), a 60 ms majority window
  (three frames), and a minimum pause of 0.25 s. The relative threshold adapts to
  recording level and admits quiet background noise between louder bursts even
  when it exceeds the absolute floor. Only internal gaps bounded by sound count;
  an entirely quiet recording has no structure to cut on. Pauses are per asset;
  sorted midpoints will be mapped to shots later with a strict interior margin
  (default 0.1 s). No `Shot.cut_points`, pipeline wiring or cache yet, so cut-point
  candidates remain incomplete. Quality shares the existing signed-PCM decoder;
  its 100 ms, 8 kHz audio-activity measurement for shake rejection is unchanged.
  Known gaps: music or background noise can hide pauses, and energy does not yet
  distinguish speech from non-speech.

- 2026-10-22: shot cut points — **schema change**: `Shot.cut_points` defaults to an
  empty list for old JSON and holds `CutPoint(time_s, kind)` with only `pause` as a
  kind today. Times are absolute source-asset times, sorted without duplicates and
  strictly inside the shot's start/end; `usable_range` does not constrain them.
  `analyze/pauses.py` maps pause midpoints to sorted, deduplicated candidates with
  `start_s + margin_s < time_s < end_s - margin_s`; the default margin is 0.1 s,
  zero is allowed, and negative margins raise `ValueError`. Pipeline wiring and
  caching are implemented, as is rules planner placement (2026-10-25); LLM planner
  use remains planned.

- 2026-10-23: pause pipeline wiring — `pipeline.analyze` loads or detects pauses once
  per asset using the existing signed-PCM decoder (16 kHz mono by default), then
  maps their midpoints to every shot with the default strict 0.1 s interior margin.
  Assets without audio yield no candidates. `PauseCache` stays in `analyze/pauses.py`;
  `<workspace>/pauses/<asset_id>.json` is keyed by asset fingerprint and all
  `PauseParams`. Mismatches or invalid cache records trigger detection; writes use
  a `.part` file followed by replace. Cut points are refreshed after loading quality
  caches, including legacy shots without candidates, so returned shots and the
  written `shots.json` use current pause results without repeating quality analysis.
  Rules planner placement completes the audio-pause cut-point path (2026-10-25).
  Beats, motion minima and LLM planner use remain planned.

- 2026-10-24: cut-point placement helper — `plan/rules.place_on_cut_points` moves a
  fixed-length clip within an absolute usable window (1e-9 s feasibility tolerance),
  aligning its in- or out-point to a `Shot.cut_points` candidate. The smallest shift
  from the default wins, subject to an optional maximum shift; ties prefer both ends
  on candidates, then the earlier in-point. If no placement qualifies, the default is
  returned unchanged. Clip length is never changed. The rules planner now uses this
  helper after fitting and rounding (2026-10-25); its tie-breaking is unchanged.

- 2026-10-25: rules planner cut-point placement — after length fitting, default
  placement and the existing millisecond in/out rounding, `rough_cut` shifts each
  fixed-length clip with `place_on_cut_points`. Its keyword-only `max_shift_s`
  defaults to 1.0 s; `None` allows an unbounded shift within the usable window.
  The helper receives absolute bounds (`Shot.start_s` plus `usable_range` offsets)
  and candidates; EDL in/out remain shot-relative offsets. No further rounding is
  applied, so candidate alignment is preserved to floating-point precision (with
  the helper's 1e-9 s feasibility tolerance clamped at usable boundaries).
  `kept_for_audio` clips also participate, starting from their usual usable-start
  default; a clip filling the usable window has no room to move. If legacy rounding
  makes a full-window clip longer than that window, it stays unchanged because no
  fixed-length placement fits. Empty candidate lists retain byte-identical EDL JSON.
  Selection, chronological order, length sampling/fitting and clip audio are unchanged;
  rounded lengths and timeline totals are preserved to floating-point precision.
  Transitions are assigned from the original rounded lengths before placement, so
  their kinds and durations remain identical. Beats, motion minima and LLM planner
  placement remain planned.

- 2026-10-26: validate hard checks — `validate/checks.py` returns ordered `Violation`
  records for unknown shot IDs, shot-relative in/out bounds (1e-6 s tolerance),
  rejected shots, target timeline duration (default ±10%), profile clip timeline
  durations (p10 minus 25% through p90 plus 25%), adjacent non-None `dup_group`
  matches, strict chronology in the supplied shot order (including repeated shots),
  dissolves forbidden by a measured zero dissolve share, and oversized dissolves.
  Each dissolve must be no longer than either neighbouring timeline clip; incoming
  and outgoing half-windows must not overlap within a clip. Speed affects timeline
  durations; dissolves do not shorten the total. Boundary errors belong to the
  incoming clip, overlap errors to the affected clip; EDL-level errors come last.
  Unknown IDs receive only `unknown_shot`; chronology compares successive known
  shots across unknown IDs, while duplicate groups compare actual neighbours only.
  `target_s=None` skips the target check, even when the EDL declares a target;
  `profile=None` skips pacing and transition-mix checks. Unmeasured mixes allow
  dissolves. `usable_range` is a suggestion and is not enforced. `Shot.dup_group`
  exists and is checked; cold opens are now allowed by the user-set
  `StyleProfile.cold_open_clips` chronology exemption (2026-11-11). No audio check
  until a profile dimension exists. `Violation`, like `StyleDistance`, is an
  analysis result outside `schemas.py`. Checks now run after planning in `pipeline.cut`, writing
  `edl/<stem>.violations.json`; the CLI reports them and violations are non-fatal.
  Repair, fallback fixes and beat snapping remain planned. The rules planner can still
  miss the requested target when footage is insufficient; it now avoids adjacent
  duplicate groups when a suitable alternative exists (2026-11-02). Validation
  reports remaining cases without relaxing the checks.

- 2026-10-27: validate pipeline wiring — every EDL planned by `pipeline.cut` is
  checked with the analyzed shots, requested target and optional style profile.
  `validate_and_write` replaces `edl/<stem>.violations.json` via a temporary file,
  including an empty list on success, and logs each violation (or an OK info line).
  `CutResult.violations` exposes the results; the CLI prints a summary and each
  violation's clip index when present, code and message. Violations are non-fatal:
  rendering continues and a successful cut still exits 0. Repair, fallback fixes
  and beat snapping remain planned.

- 2026-10-28: beat detection — `analyze/beats.py` is a pure NumPy module over mono
  samples, with lazy imports and no schema change. Defaults: 22050 Hz reference
  rate, centered 1024-sample energy frames / 512-sample hops (durations scale to
  the supplied input rate without resampling), 60–200 BPM, and 0.08 s alignment
  tolerance. Positive log-energy differences have a centered 200 ms mean removed.
  Mean-centered autocorrelation requires at least two slowest periods and a 0.3
  normalized peak; three-tap smoothing and a shorter-period tie-break within 25%
  of the strongest peak reduce half-time errors. Parabolic peak refinement over
  up to four periods limits grid drift. A fixed-period grid maximizes summed
  onsets over phase, then snaps within 10% of a period to supported local peaks.
  Silence, short or structureless audio returns `BeatResult(None, [])`.
  `beat_aligned_ratio` uses inclusive ±80 ms by default (1e-12 s floating-point
  allowance); empty cuts or beats return `None` (not measured), distinct from 0.
  Known gaps: tempo octave errors on real music and no tempo changes. There is no
  caching or wiring into the pipeline, style profiles, planner or validation;
  beat snapping remains planned. The `MusicTrack` storage question remains open.

- 2026-10-29: reference beat alignment — `style/profile.py` decodes each reference
  proxy's own audio with the shared signed-PCM mono decoder at 22050 Hz and uses
  `analyze/beats.py` to detect beats. Sorted beat times are required in `ReferenceCuts`;
  old records without them fail validation and trigger re-analysis. `ReferenceParams.beats`
  defaults to true and participates in cache matching. Missing audio yields no beats.
  Alignment uses the same surviving, rounded boundaries as pacing and transitions,
  with inclusive ±80 ms tolerance. Pool aligned-cut counts / total-cut counts over
  references with beats, not a mean of per-reference ratios; beatless references
  contribute to pacing but are excluded from this pool. The ratio is `None` if detection
  is disabled globally or for any reference, no reference has beats, or the pool has
  zero cuts. Existing zero-cut pacing errors remain unchanged. Notes mention alignment
  only when measured; otherwise their text is unchanged. Known gaps: tempo octave
  errors on real music, no tempo changes, and footage without music gives no beats
  when no regular pulse is detected. Chosen-music analysis, planner beat snapping,
  `MusicTrack`, beat validation and speech ratio remain unwired; no schema change.

- 2026-10-30: beat snapping helper — `plan/snap.snap_to_beats` visits internal
  boundaries left to right using the already updated clips. Beats are programme
  timeline seconds; applying a music cue delay is the caller's responsibility.
  The nearest beat within inclusive 0.08 s tolerance (1e-12 s floating-point
  allowance) wins, with earlier beats preferred on ties. A local edit moves the
  outgoing out-point and incoming in-point by the timeline delta times each
  clip's speed, preserving other boundaries and the total to floating-point
  precision. Skip shifts outside shot-relative bounds, below the minimum clip
  timeline duration (default 0.25 s), or breaking adjacent dissolve lengths or
  half-window fit. Both boundaries around an unknown shot ID are skipped. The
  helper returns a new EDL, preserving transitions, audio and other metadata;
  inputs are never mutated. No schema change or pipeline wiring. Chosen-music
  analysis and `MusicTrack` remain planned.

- 2026-10-31: deterministic repair — `validate/repair.repair_edl` returns a deep copy,
  dropping unknown and quality-rejected shots while preserving retained clip order.
  The first retained clip enters with a hard cut; dropping all clips raises
  `ValueError`. A measured zero dissolve share forbids dissolves; an unmeasured mix
  or absent profile allows them. Dissolves are capped left to right at both
  neighbouring timeline durations and the remaining half-window space, using the
  checks' exact comparisons; no positive space means a hard cut. In/out points,
  speed, audio, target, notes and music cues are preserved; the computed EDL total
  follows the remaining clips. Bounds, chronology, duplicate, target and pacing
  violations remain reported, not repaired. No new selections, schema changes,
  beat snapping, LLM repair rounds or pipeline/CLI wiring.

- 2026-11-01: deterministic repair fallback — `pipeline.REPAIRABLE_CODES` triggers
  `repair_edl` only for `unknown_shot`, `rejected_shot`, `transition_not_allowed`
  or `dissolve_too_long` (oversized boundaries or overlapping half-windows).
  Repair uses the same shots and optional profile as validation. On success, the
  EDL JSON is atomically replaced via a temp file, then validation runs again with
  the same requested target and profile. The violations JSON, `CutResult.violations`
  and rendered preview all describe the repaired EDL. If repair raises `ValueError`
  because all clips would be dropped, log a warning and continue with the original
  EDL and violations. No repair is attempted without a trigger code; clean EDL JSON
  and violations output remain byte-identical. `CutResult.repaired` defaults to
  false; the CLI prints one repair line before the validation summary when true,
  with unchanged exit codes. Bounds, chronology, duplicate, target and pacing
  violations are still not repaired; no schema changes, LLM loop or beat snapping.

- 2026-11-02: rules planner adjacent duplicate avoidance — after chronological
  selection and before length sampling/fitting, `plan/rules.avoid_adjacent_duplicates`
  scans picks left to right. When neighbours share a non-None `dup_group`, replace
  the later pick with the best unselected alternative from its coverage bin
  (including bins of moments), or from all candidates for highlights. Alternatives
  must remain between the neighbouring picks and conflict with neither side.
  Higher scores win, then earlier chronological order. None groups never conflict.
  If no alternative qualifies, keep the original pick for validation to report;
  the helper never changes the pick count. It runs on each selection attempt in
  the existing length-fitting loop; sampling, fitting, placement, transitions and
  clip audio then use the final picks as before. With no shared non-None groups,
  EDL JSON is unchanged. Group computation and deterministic repair are unchanged.

- 2026-11-03: planner shot table — `plan/shot_table.format_shot_table` is a pure,
  deterministic text formatter with no pipeline wiring or model calls. Columns
  include full shot and asset IDs, source start, duration, usable range, quality
  warnings, dup_group, kept_for_audio (derived from warnings), cut-point candidates
  with kind, rejection status and reasons. Usable ranges already use shot-relative
  offsets; absolute cut-point times subtract Shot.start_s to match Clip.in_s/out_s.
  All times use two fixed decimal places. A missing usable range expands to the
  whole shot; other empty values use `-`. Tab-separated rows JSON-escape text and
  retain input shot/list order. Rejected shots are omitted by default, with a final
  omitted-count summary; include_rejected=True includes them with their reasons.
  Caption, tags and interest columns now exist before rejection status and reasons.
  Kind and ranker score columns remain planned until populated planner inputs exist.

- 2026-11-04: LLM planner core — `plan/llm.py` builds deterministic system/user
  messages with hard checks, candidate-based cut-point and usable-range preferences,
  the EDL/Clip field contract, target duration, profile pacing and measured dissolve
  share (or no profile), optional brief (or none), and the shot table without rejected
  shots. Table order defines chronology; table cut points and clip offsets are
  shot-relative. One `ModelRouter.complete("planner", messages)` call returns text;
  parsing strips whitespace and optional surrounding JSON/unlabelled Markdown fences,
  then uses `EDL.model_validate_json`. Invalid JSON or schema failures raise a one-line
  `PlannerError`; surrounding prose is not extracted. Semantic checks remain separate,
  so unknown shot IDs parse and are reported by `validate/checks.py`. No retry, repair
  loop, rules fallback, pipeline/CLI wiring or schema change. Rules remain the default;
  the prompt establishes candidate-based selection without changing execution.

- 2026-11-05: LLM repair loop — `plan/llm.plan_with_repair` uses the existing prompt
  builder, parser and hard checks with the same shots, target and profile. Each
  retry appends the previous raw reply as an assistant message and a user message
  containing ordered clip-index/code/message violation lines (or the one-line
  parse error), followed by a fixed request for complete corrected EDL JSON under
  the same contract. All calls use `ModelRouter.complete("planner", messages)`.
  `max_rounds` counts extra calls after the initial attempt, including parse
  failures; zero permits one call and negative values raise `ValueError`. Each
  round is logged. Stop at the first violation-free EDL; on exhaustion return the
  last successfully parsed EDL with its remaining violations, even if later replies
  failed to parse. If no reply parsed, raise `PlannerError`. `LLMPlanResult.rounds`
  reports the repair rounds actually used (zero for a clean first reply).
  `format_violations` preserves input order, uses `-` for absent clip indices,
  flattens message whitespace and explicitly reports empty input. The result model
  lives outside `schemas.py`; no schema change. No deterministic repair inside the
  loop, rules fallback, or pipeline/CLI wiring; rules remain the default.

- 2026-11-06: opt-in LLM pipeline and CLI — `pipeline.cut` accepts `planner`
  (`rules` by default), `router`, `brief`, and `max_repair_rounds=2`. LLM planning
  calls `plan_with_repair` with analyzed shots, target, profile and brief through
  `ModelRouter`; a missing router fails before analysis. `PlannerError` logs a
  warning and falls back to rules. The existing validation, deterministic repair,
  re-validation and render flow applies to either planner. LLM output stems append
  `_llm` after the optional style suffix; rules and fallback stems are unchanged.
  Explicit preview `--out` paths remain honored. `CutResult.planner`, `.fallback`
  and `.repair_rounds` report the producer, fallback and LLM rounds (zero for rules).
  The CLI adds `--planner {rules,llm}`, `--brief` and `--config`, loading the existing
  router YAML format via `ModelRouter.from_yaml` only for LLM runs. Missing or invalid
  config fails before analysis with a one-line error and exit 1. A planner/rounds/
  fallback summary line is printed; other exit codes and rules output bytes are
  unchanged. No prompt, parser, check, repair or schema changes; rules remain default.

- 2026-11-07: optional per-asset thread pool — `pipeline.ingest`, `analyze` and
  `cut` accept keyword-only `workers=1`; `vlogcut cut --jobs N` forwards the count.
  The default retains sequential execution; counts below 1 fail before work starts
  (CLI: one-line error, exit 1). Higher counts use a context-managed
  `ThreadPoolExecutor` for probing/proxies and shot/quality/pause analysis. Results
  and skipped-file warnings retain input order, regardless of completion order;
  planning, validation and rendering stay sequential. Fatal errors propagate in
  input order, pending futures are cancelled where possible, and running work is
  joined before returning. Progress callbacks may run on workers; CLI writes are
  locked. Per-call locks protect shared proxy, keyframe, shot, quality and pause
  cache paths when duplicate content has the same asset ID. Cache formats and keys
  are unchanged; shots, EDL, violations JSON and cache outputs are identical
  regardless of worker count. No schema change or new dependency.

- 2026-11-08: feedback — **schema change**: `Feedback` stores a non-blank variant ID
  (an EDL stem until the Run manifest exists), an optional pick/reject verdict and
  free text; a verdict or non-blank text is required. `PreferencePair` stores context
  (possibly empty) and distinct, non-blank chosen/rejected variant IDs.
  `feedback/pairs.preference_pairs` derives pairs from one run's explicit verdicts:
  ignore text-only entries, deduplicate identical verdicts in first-occurrence order,
  and raise a one-line `ValueError` naming a variant with conflicting verdicts.
  Emit every pick × reject, picks outermost and rejects innermost in their respective
  first-occurrence order, propagating context without mutating inputs. Missing picks
  or rejects yield an empty list. No pipeline/CLI wiring, HTML variants page,
  feedback-role classification or ranker yet.

- 2026-11-09: static variants page and feedback loader — `feedback/page.py` builds
  a self-contained HTML5 page with inline CSS/JS, side-by-side preview cards,
  pick/reject/unset radios (unset by default), free text and a Save feedback button.
  No server or external resources are added; supplied preview hrefs are preserved.
  A Blob/object URL downloads `feedback.json`: an array of objects with
  `variant_id`, `verdict` (pick/reject or null when unset), and `text`. Include only
  entries with an explicit verdict or non-blank text, in page order, preserving
  supplied strings. Empty variant lists and duplicate whitespace-stripped IDs fail.
  HTML contexts use `html.escape(..., quote=True)`; script data uses `json.dumps`
  with `<`, `>` and `&` escaped as `\u003c`, `\u003e` and `\u0026` to prevent
  script breakouts. Identical inputs produce byte-identical output. The UTF-8 writer
  atomically replaces the destination via a temporary file in the same directory,
  cleaning up on failure. `load_feedback` validates with `TypeAdapter(list[Feedback])`,
  accepts an empty array, preserves order/strings and feeds `preference_pairs`
  directly. Invalid JSON, non-array roots and invalid records raise a single-line
  `ValueError` naming the file. No schema changes, CLI/pipeline wiring, multi-variant
  generation, Run manifest, pair persistence, feedback classification or ranker yet.

- 2026-11-10: feedback CLI — argparse `feedback page` and `feedback pairs` reuse
  the existing page writer, feedback loader and pair derivation. Page variant IDs
  are preview file stems with one leading `preview_` removed, matching EDL stems
  until the Run manifest exists. Preview hrefs are relative to the output HTML
  directory, use POSIX separators and percent-encoding with `/` preserved. Page
  output defaults to `variants.html` in the current directory, is written atomically,
  and reports its path and variant count. Pairs preserve the helper's order and use
  `--context` (empty by default); output is a UTF-8 JSON array of model records with
  indent 1. With `-o`, replace the destination atomically via a same-directory
  temporary file, cleaning up on failure, and print the path/count to stdout.
  Without `-o`, stdout contains only JSON and stderr receives the count. Missing
  previews, duplicate IDs, missing/invalid feedback, conflicting verdicts and I/O
  errors produce one line through `_fail` and exit 1; success exits 0. Empty input
  or absent picks/rejects yields `[]`. No schema or pipeline changes, multi-variant
  generation, Run manifest, feedback classification or ranker wiring.

- 2026-11-11: cold-open allowance — **schema change**: user-editable
  `StyleProfile.cold_open_clips: int` is non-negative and defaults to 0, including
  for old profile JSON. With N > 0, chronology ignores the first
  `min(N, len(clips))` EDL clips. These may use any shots in any order, including
  shots reused later. The first remaining clip starts a fresh chronology check;
  subsequent known shots must strictly increase in supplied shot order, including
  across unknown IDs, and repeated shots in this remainder still fail. With no
  profile or N = 0, checks and violation output are unchanged. All other checks
  apply to cold-open clips, including duplicate groups between actual neighbours
  within the opening and across its boundary; violation ordering is unchanged.
  Reference analysis keeps the default; notes, summaries and style distance do not
  use this field. Opt-in rules planner support is described below; reference
  measurement remains planned. Repair, pipeline and CLI are unchanged.

- 2026-11-12: montage-burst share — **schema change**: optional
  `StyleProfile.structure` holds `StructureStats(montage_share)` with bounds [0, 1].
  Missing structure in old JSON means not measured. A burst is a maximal run of at
  least three consecutive shots strictly shorter than half the pooled profile median;
  runs never cross reference boundaries. Pool shot counts inside bursts / all shots
  using the same ordered, post-merge durations as pacing. EDLs use speed-adjusted
  timeline durations in clip order. This relative threshold measures structure
  independently of absolute speed. The pure helper accepts `rel=0.5`, `min_run=3`,
  returns `None` for zero shots, and rejects non-positive median/rel or min_run < 2.
  Existing pacing errors, including single-clip EDLs, remain unchanged. Style distance
  adds an equally weighted absolute-difference dimension, skipped and reported as
  unmeasured when either structure is `None`. Aggregation is pure: reference cache
  keys/formats and `ReferenceParams` are unchanged, as are notes and summary text.
  Planners do not produce bursts; validation and repair are unchanged. Talking/b-roll
  alternation, intro length and cold-open measurement remain planned.

- 2026-10-18: intro-vs-body pacing — **schema change**:
  `StructureStats.intro_pace_ratio: float | None = None`, strictly positive when set,
  is median(intro durations) / median(body durations). Each timeline starts at zero;
  a whole shot belongs to the intro when its start is strictly before the 20 s window
  ends, otherwise to the body. Require at least two pooled shots on each side;
  otherwise `None` means not measured, including old structure JSON without the field.
  Pool durations across references rather than averaging per-reference ratios, using
  the same timeline-ordered, post-merge durations as pacing and montage share. EDLs
  use speed-adjusted clip durations in clip order as one sequence. The pure helper
  accepts a positive `intro_s` override, defaulting to 20 s. Style distance adds
  equally weighted `intro_pace = |log2(a / b)|` in octaves, skipped and reported when
  either side is unmeasured. Reference caches, detection parameters and notes are
  unchanged. Rules planner and LLM prompt use are described below.

- 2026-10-18: rules planner intro pacing — after `assign_lengths` and before
  `fit_lengths`, the pure `plan/rules.apply_intro_pace` multiplies intro bases by
  the measured `structure.intro_pace_ratio`, clamping to each clip's floor/cap.
  Intro membership uses cumulative unscaled base starts strictly before 20 s;
  whole clips belong to one side. Bases sample one pacing distribution, so this
  is a relative bias with 1.0 neutral within bounds, not an exact median match.
  Body bases stay unchanged; no intro or no body returns unchanged lengths.
  `kept_for_audio` bases retain their existing cap assignment before the shared
  offset fit. Fitting and clip-count adjustment retain the existing ±10 % target
  tolerance when footage permits. Clamping, fitting and shifts across the 20 s
  boundary can limit the measured effect. Missing profile, structure or ratio
  bypasses scaling and preserves EDL JSON bytes. Selection, duplicate avoidance,
  placement, transition assignment and clip audio rules are unchanged. No schema
  change, burst/cold-open generation, LLM prompt change or intro-pace validation.

- 2026-10-18: feedback classification — `feedback/classify.py` defines analysis-result
  models `FeedbackClassification` (exclusive edit constraint or style delta) and
  `ProfileDelta` (at least one of duration multiplier `pace_factor`, inclusive
  [0.25, 4.0], or absolute `dissolve_share`, inclusive [0, 1]). Deterministic messages
  include verdict, text and optional profile pacing/transition measurements. Blank
  text raises before calling; otherwise exactly one `ModelRouter.complete("feedback", ...)`
  call is made, with no retries. JSON parsing accepts surrounding Markdown/JSON fences;
  invalid JSON or validation failures raise a one-line `FeedbackClassificationError`.
  Pure application returns a newly validated profile: multiply p10/median/p90,
  divide cuts per minute, and set hard-cut/dissolve shares to 1-s/s when supplied.
  Preserve all other fields and append a deterministic note without mutating inputs.
  No schema change, pipeline/CLI wiring, re-planning, persistence or ranker work.

- 2026-10-18: look measurements — `analyze/look.py` uses lazy NumPy imports and
  measures all sampled pixels equally: brightness is mean Rec.709 luma
  (0.2126 R + 0.7152 G + 0.0722 B), saturation is mean HSV (max-min)/max
  (zero for black), and warmth is mean R-B. uint8 RGB is divided by 255;
  floating-point RGB is clipped to [0, 1]. The shared ffmpeg helper decodes
  rgb24 frames at 1 fps and 64x64 by default. Across references, `pool_look`
  weights each result by its frame count and skips `None`. Zero frames or no
  measured results return `None`, meaning "not measured", distinct from black.
  `LookStats` is a local analysis result outside `schemas.py`; no schema change.
  No wiring into `StyleProfile`, distance, planners, pipeline or CLI, and no
  caching, colour matching or grading.

- 2026-10-18: captioner core — `analyze/caption.py` encodes 1–3 JPEG keyframes as
  base64 image content parts with optional context, then makes exactly one
  `ModelRouter.complete("captioner", messages)` call, without retries. Plain or
  fenced JSON validates into `CaptionResult`: stripped non-blank description,
  stripped unique non-blank tags in input order (at most 10 after normalization),
  and interest 1–5. Invalid JSON or schema failures raise a one-line `CaptionError`;
  surrounding prose is not extracted. `CaptionResult` lives outside `schemas.py`
  as a local analysis result. The core is not wired into the pipeline or CLI;
  Shot population, caching, quality/dedup filtering and privacy enforcement remain
  planned. No schema change or new dependency; router behavior is unchanged.

- 2026-10-19: cached batch captions — `analyze/caption.caption_shots` returns a
  shot-ID mapping in input order without changing Shot records. `CaptionCache`
  lives outside `schemas.py` at `<workspace>/captions/<shot_id>.json`, keyed by
  shot ID, an explicit model identifier, ordered SHA-1 hashes of each keyframe's
  bytes, stripped context and the core's prompt version. The model identifier
  labels the cache; the router's captioner role still selects the actual model.
  Missing, invalid or mismatched records trigger the existing single-shot core
  through `ModelRouter.complete("captioner", ...)`. Valid results are written via
  a `.part` file then replace, with cleanup on write failure; core errors propagate
  before any cache write. Rejected shots are skipped unless `include_rejected=True`;
  shots without keyframes are always skipped with a warning. Cache hits make no
  model call. Pipeline/CLI wiring, dedup filtering and privacy enforcement remain
  planned; no schema change, new dependency or thread pool.

- 2026-10-19: opt-in pipeline captions — `pipeline.analyze` and `cut` accept
  keyword-only `caption=False` and a router. Enabling captions without a router
  fails before analysis (before ingest in `cut`). After quality and pause analysis,
  `analyze` runs the existing `caption_shots` helper sequentially, regardless of
  `workers`, through `ModelRouter`'s `captioner` role. The configured captioner
  model name supplies the cache identity; the existing helper owns filtering and
  cache semantics under `<workspace>/captions/`. Rejected shots and shots without
  keyframes are skipped; cache hits make no new model calls. Description, tags and
  interest populate the existing Shot fields in returned results and `shots.json`.
  The CLI adds `--caption`, loads `--config` with `ModelRouter.from_yaml` once,
  shares that router with `--planner llm`, and prints `captioned N shots`.
  Missing/invalid config or missing required roles fail before analysis with a
  one-line error and exit 1. Captioning is off by default, preserving shots, EDL
  and violations output bytes. Rules planning ignores captions; planner prompts
  and shot tables are unchanged. No schema changes, new dependencies, dedup
  filtering, ranker or privacy enforcement.

- 2026-11-13: planner caption metadata — `plan/shot_table.py` exposes Shot.caption,
  tags and interest before rejection status and reasons. Caption text and ordered
  tags use the existing JSON escaping; interest is an integer and missing values
  use `-`. The LLM prompt briefly describes these columns. Rejected-shot omission,
  the summary and deterministic input order are unchanged; no schema changes.

- 2026-10-19: rules planner opt-in cold open — `rough_cut(cold_open=False)` leaves
  the existing plan byte-identical by default, and also when the profile is absent
  or `cold_open_clips` is zero. With opt-in and a positive allowance, run normal
  planning first, then `plan/rules.apply_cold_open` moves exactly one selected hook
  to the front if a safe move exists. Prefer the latter half by clip order (indices
  >= floor(clip count / 2)), then earlier clips after index zero; within each pool,
  use the existing shot scores, breaking ties toward later clips. Reuse the scores
  already computed by the planner, including a supplied custom scorer. Reject moves
  that join equal non-null duplicate groups at the front or across the removed
  hook's old position; try the next candidate, logging at debug level if none fits.
  Fewer than three clips stay unchanged. The helper does not mutate inputs; only
  order and entry transitions change. The hook, original first clip, and hook's
  original successor enter with hard cuts; all other transitions, in/out points,
  speed, audio and clip metadata are preserved. Total duration is preserved to
  floating-point precision and the remaining timeline stays chronological under
  the existing cold-open allowance. No schema, validator, pipeline, CLI, LLM planner
  or style changes; montage generation remains planned.

- 2026-10-19: pipeline and CLI cold-open opt-in — `pipeline.cut(cold_open=False)`
  and `vlogcut cut --cold-open` forward the flag to rules planning, including the
  fallback after LLM planning failure. A missing profile or zero allowance logs a
  warning and plans normally. With opt-in, the LLM planner logs one warning that
  it ignores the flag; its prompt is unchanged, and rules fallback honors the flag.
  Default output bytes and filenames, validation and repair are unchanged.

- 2026-10-19: rules planner montage bursts — pure `plan/rules.apply_montage`
  shapes bases after intro scaling, before the unchanged shared-offset fit.
  Request `round(montage_share * k)` burst clips; fill non-talking runs from the
  end backwards, taking suffixes of at least three clips, favouring the body over
  the intro without reserving a fixed intro window. Talking clips split runs and
  retain their cap-based assignment. Short bases use p10 rounded up to milliseconds
  plus 1 ms; non-burst, non-talking bases absorb any target deficit proportionally
  to their remaining capacity. Probe the existing fit with unchanged floors/caps,
  requiring burst lengths at least 1 ms below half the profile median to survive
  in/out rounding. Try smaller burst counts when capacity is insufficient; if no
  run of three fits, return the original bases. Missing/zero share, fewer than
  three requested clips, p10 >= median/2, or no representable short length are
  exact no-ops. Selection, clip-count adjustment, duplicate avoidance, cut-point
  placement, transitions and cold-open logic stay unchanged. Burst share is an
  approximation: fragmented eligible runs, limited capacity, an EDL median that
  differs from the reference, and a subsequent cold-open move can change the
  measured share. No schema, CLI, pipeline, style distance or LLM planner changes.

- 2026-10-19: LLM planner structure guidance — the deterministic `plan/llm.py`
  prompt describes measured `montage_share` (including an explicit no-bursts target
  for zero), defining bursts as at least three consecutive clips strictly shorter
  than half the profile median and giving that threshold in seconds. Measured
  `intro_pace_ratio` targets the median of clips starting before 20 s relative to
  the body, with approximate seconds computed from the profile median as a body
  estimate. New measured values use two decimal places. A positive `cold_open_clips`
  replaces strict chronology wording with an allowance for at most N leading hook
  clips, followed by a chronological remainder. Unmeasured dimensions add no lines;
  absent structure with zero allowance, and absent profiles, preserve prompt bytes.
  Validation, parsing, repair, schemas, rules planning, pipeline and CLI are unchanged.

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
