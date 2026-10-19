"""Command line: `vlogcut cut <footage folder>` -> rough-cut preview,
`vlogcut style build <reference videos>` -> style profile JSON, `style show / diff`.
`vlogcut feedback page / pairs / classify` -> variants, preferences and profile updates.

Thin wrapper over pipeline, style and feedback helpers; argument parsing and output only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from urllib.parse import quote

from vlogcut import __version__

log = logging.getLogger("vlogcut")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlogcut", description="Style-matched rough cuts for daily vlogs."
    )
    parser.add_argument("--version", action="version", version=f"vlogcut {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("cut", help="analyze a footage folder and render a rough-cut preview")
    c.add_argument("folder", type=Path, help="folder with the day's clips (not recursive)")
    c.add_argument("-t", "--target", type=float, default=180.0, help="target length, s")
    c.add_argument("-s", "--strategy", choices=["coverage", "highlights"], default="coverage")
    c.add_argument("--style", type=Path, help="style profile JSON from `vlogcut style build`")
    c.add_argument(
        "--cold-open",
        action="store_true",
        help="opt into a cold open (needs a --style profile with cold_open_clips > 0)",
    )
    c.add_argument("--planner", choices=["rules", "llm"], default="rules")
    c.add_argument("--caption", action="store_true", help="caption shots using the captioner role")
    c.add_argument("--brief", help="text brief for the LLM planner")
    c.add_argument(
        "--config", type=Path, default=Path("config.yaml"),
        help="model router config (read with --planner llm or --caption; default config.yaml)",
    )  # fmt: skip
    c.add_argument("-w", "--workspace", type=Path, help="cache dir (default ./workspace/<folder>)")
    c.add_argument("-o", "--out", type=Path, help="preview path (default inside the workspace)")
    c.add_argument(
        "--music", type=Path, metavar="PATH",
        help="user music file for beat snapping and background audio",
    )  # fmt: skip
    c.add_argument("--music-gain", type=float, default=-14.0, help="music level, dB")
    c.add_argument("--background", choices=["black", "blur"], default="black")
    c.add_argument("--no-subtitles", action="store_true", help="no debug subtitle track")
    c.add_argument("--no-render", action="store_true", help="plan only, write the EDL")
    c.add_argument("--jobs", type=int, default=1, help="per-asset analysis workers (default 1)")
    c.add_argument("-v", "--verbose", action="store_true")

    s = sub.add_parser("style", help="style profiles: build from references, show, diff")
    ss = s.add_subparsers(dest="style_command", required=True)
    b = ss.add_parser("build", help="extract a style profile from finished reference videos")
    b.add_argument("refs", type=Path, nargs="+", help="videos in the style to learn")
    b.add_argument("-o", "--out", type=Path, default=Path("profile.json"), help="profile JSON")
    b.add_argument(
        "-w", "--workspace", type=Path, default=Path("workspace") / "references",
        help="cache dir for reference proxies and cuts (default workspace/references)",
    )  # fmt: skip
    b.add_argument("-v", "--verbose", action="store_true")
    sh = ss.add_parser("show", help="print a profile's summary, pacing and transition mix")
    sh.add_argument("profile", type=Path, help="profile JSON from `vlogcut style build`")
    d = ss.add_parser("diff", help="distance between two profiles, per dimension")
    d.add_argument("a", type=Path, help="profile JSON")
    d.add_argument("b", type=Path, help="profile JSON")

    f = sub.add_parser("feedback", help="feedback: variants, preferences and profile updates")
    fs = f.add_subparsers(dest="feedback_command", required=True)
    page = fs.add_parser("page", help="create a local page comparing preview videos")
    page.add_argument("previews", type=Path, nargs="+", metavar="PREVIEW", help="preview mp4s")
    page.add_argument("-o", "--out", type=Path, default=Path("variants.html"), help="HTML path")
    pairs = fs.add_parser("pairs", help="derive preference pairs from feedback JSON")
    pairs.add_argument("feedback_json", type=Path, metavar="FEEDBACK_JSON")
    pairs.add_argument("--context", default="", help="project or brief description")
    pairs.add_argument("-o", "--out", type=Path, help="pairs JSON path (default stdout)")
    classify = fs.add_parser("classify", help="classify feedback and apply style profile deltas")
    classify.add_argument("feedback_json", type=Path, metavar="FEEDBACK_JSON")
    classify.add_argument("--profile", type=Path, required=True, metavar="PROFILE_JSON")
    classify.add_argument(
        "-o", "--out", type=Path, help="updated profile (default <profile stem>.updated.json)"
    )
    classify.add_argument("--constraints-out", type=Path, help="constraints JSON (default stdout)")
    classify.add_argument(
        "--config", type=Path, default=Path("config.yaml"), help="model router config"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verbose = getattr(args, "verbose", False)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not verbose:
        _quiet_scenedetect()
    if args.command == "cut":
        return _cut(args)
    if args.command == "feedback":
        feedback_commands = {
            "page": _feedback_page, "pairs": _feedback_pairs, "classify": _feedback_classify,
        }  # fmt: skip
        return feedback_commands[args.feedback_command](args)
    commands = {"build": _style_build, "show": _style_show, "diff": _style_diff}
    return commands[args.style_command](args)


def _cut(args: argparse.Namespace) -> int:
    if args.jobs < 1:
        return _fail("--jobs must be at least 1")
    from vlogcut.analyze.shots import ShotDetectionError
    from vlogcut.ingest import FFmpegError, FFmpegNotFoundError, ProxyError
    from vlogcut.pipeline import cut
    from vlogcut.plan import RoughCutError
    from vlogcut.render import RenderError
    from vlogcut.style import StyleError

    if args.music is not None and not args.music.is_file():
        return _fail(f"music file not found: {args.music}")
    if args.style is not None and not args.style.is_file():
        return _fail(f"style profile not found: {args.style}")
    router = None
    if args.planner == "llm" or args.caption:
        from yaml import YAMLError

        from vlogcut.llm.router import ModelRouter

        try:
            router = ModelRouter.from_yaml(args.config)
            if args.planner == "llm" and "planner" not in router.roles:
                raise ValueError("missing planner role")
            if args.caption and "captioner" not in router.roles:
                raise ValueError("missing captioner role")
        except (
            OSError,
            UnicodeError,
            YAMLError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ) as exc:
            detail = " ".join(str(exc).split())
            return _fail(f"invalid model config {args.config}: {detail}")
    workspace = args.workspace or Path("workspace") / args.folder.resolve().name
    started = time.perf_counter()
    progress_lock = Lock()

    def progress(msg: str) -> None:
        with progress_lock:
            print(msg, file=sys.stderr)

    try:
        result = cut(
            args.folder, workspace,
            target_s=args.target, strategy=args.strategy, style=args.style, music=args.music,
            planner=args.planner, router=router, caption=args.caption, brief=args.brief,
            music_gain_db=args.music_gain, out=args.out, render=not args.no_render,
            preview_params={"background": args.background, "subtitles": not args.no_subtitles},
            progress=progress, workers=args.jobs, cold_open=args.cold_open,
        )  # fmt: skip
    except FFmpegNotFoundError as exc:
        return _fail(str(exc))
    except ImportError as exc:
        detail = " ".join(str(exc).split())
        return _fail(f'{detail}; install the analysis extra: pip install -e ".[analyze]"')
    except (FileNotFoundError, RoughCutError, ShotDetectionError, StyleError) as exc:
        return _fail(str(exc))
    except (ProxyError, RenderError, FFmpegError) as exc:
        return _fail(f"ffmpeg failed: {exc}")

    usable = sum(not s.quality.rejected for s in result.shots)
    print(f"footage   {len(result.assets)} clips, {sum(a.duration_s for a in result.assets):.0f} s")
    for name, why in result.skipped:
        print(f"skipped   {name}: {why}")
    print(f"shots     {len(result.shots)} ({len(result.shots) - usable} rejected)")
    if args.caption:
        print(f"captioned {sum(s.caption is not None for s in result.shots)} shots")
    if result.profile is not None:
        print(f"style     {args.style.name}: {result.profile.notes}")
    print(f"cut       {len(result.edl.clips)} clips, {result.edl.total_duration_s:.1f} s")
    print(
        f"planner   {result.planner}, repair rounds: {result.repair_rounds}, "
        f"fallback to rules: {'yes' if result.fallback else 'no'}"
    )
    print(f"edl       {result.edl_path}")
    if result.repaired:
        print("repaired EDL")
    print(
        f"validation: {len(result.violations)} issue(s)" if result.violations else "validation: OK"
    )
    for violation in result.violations:
        clip = f"clip {violation.clip_index}: " if violation.clip_index is not None else ""
        print(f"  {clip}{violation.code}: {violation.message}")
    if result.preview_path:
        print(f"preview   {result.preview_path}")
    print(f"done in {time.perf_counter() - started:.1f} s")
    return 0


def _style_build(args: argparse.Namespace) -> int:
    from vlogcut.analyze.shots import ShotDetectionError
    from vlogcut.ingest import FFmpegError, FFmpegNotFoundError, ProbeError, ProxyError
    from vlogcut.style import StyleError, build_profile

    for ref in args.refs:
        if not ref.is_file():
            return _fail(f"reference not found: {ref}")
    started = time.perf_counter()
    try:
        profile = build_profile(
            args.refs, args.workspace, progress=lambda msg: print(msg, file=sys.stderr)
        )
    except FFmpegNotFoundError as exc:
        return _fail(str(exc))
    except ImportError as exc:
        return _fail(f'{exc}; install the analysis extra: pip install -e ".[analyze]"')
    except (ProbeError, StyleError, ShotDetectionError) as exc:
        return _fail(str(exc))
    except (ProxyError, FFmpegError) as exc:
        return _fail(f"ffmpeg failed: {exc}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(profile.model_dump_json(indent=1), encoding="utf-8")
    print(f"style     {profile.notes}")
    print(f"profile   {args.out}")
    print(f"done in {time.perf_counter() - started:.1f} s")
    return 0


def _style_show(args: argparse.Namespace) -> int:
    from vlogcut.style import StyleError, load_profile
    from vlogcut.style.distance import format_profile

    if not args.profile.is_file():
        return _fail(f"profile not found: {args.profile}")
    try:
        profile = load_profile(args.profile)
    except StyleError as exc:
        return _fail(str(exc))
    print(format_profile(args.profile.name, profile))
    return 0


def _style_diff(args: argparse.Namespace) -> int:
    from vlogcut.style import StyleError, load_profile, profile_distance
    from vlogcut.style.distance import format_diff

    for path in (args.a, args.b):
        if not path.is_file():
            return _fail(f"profile not found: {path}")
    try:
        a, b = load_profile(args.a), load_profile(args.b)
    except StyleError as exc:
        return _fail(str(exc))
    print(format_diff(args.a.name, a, args.b.name, b, profile_distance(a, b)))
    return 0


def _feedback_page(args: argparse.Namespace) -> int:
    from vlogcut.feedback import VariantEntry, write_variants_page

    try:
        variants: list[VariantEntry] = []
        for preview in args.previews:
            if not preview.is_file():
                raise FileNotFoundError(f"preview file not found: {preview}")
            relative = Path(os.path.relpath(preview, args.out.parent)).as_posix()
            variants.append(
                VariantEntry(preview.stem.removeprefix("preview_"), quote(relative, safe="/"))
            )
        write_variants_page(args.out, variants)
    except (OSError, ValueError) as exc:
        return _fail(" ".join(str(exc).split()))
    print(f"page      {args.out} ({len(variants)} variants)")
    return 0


def _feedback_pairs(args: argparse.Namespace) -> int:
    from vlogcut.feedback import load_feedback, preference_pairs

    temporary: Path | None = None
    try:
        pairs = preference_pairs(load_feedback(args.feedback_json), context=args.context)
        document = json.dumps([pair.model_dump(mode="json") for pair in pairs], indent=1)
        if args.out is not None:
            try:
                with NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="\n", dir=args.out.parent, delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(document)
                temporary.replace(args.out)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    except (OSError, ValueError) as exc:
        return _fail(" ".join(str(exc).split()))
    if args.out is None:
        print(document)
        print(f"pairs     {len(pairs)} pairs", file=sys.stderr)
    else:
        print(f"pairs     {args.out} ({len(pairs)} pairs)")
    return 0


def _feedback_classify(args: argparse.Namespace) -> int:
    from yaml import YAMLError

    from vlogcut.feedback import apply_profile_delta, classify_feedback, load_feedback
    from vlogcut.llm.router import ModelRouter
    from vlogcut.style import StyleError, load_profile

    try:
        records = load_feedback(args.feedback_json)
        profile = load_profile(args.profile)
        out = args.out or args.profile.with_name(f"{args.profile.stem}.updated.json")
        if args.out is None and out.resolve() == args.profile.resolve():
            raise ValueError("overwriting the input profile requires explicit -o")
        if args.constraints_out is not None and args.constraints_out.resolve() in {
            args.profile.resolve(),
            out.resolve(),
        }:
            raise ValueError("--constraints-out must differ from the input and output profiles")
    except (OSError, ValueError, StyleError) as exc:
        return _fail(" ".join(str(exc).split()))
    try:
        router = ModelRouter.from_yaml(args.config)
        if "feedback" not in router.roles:
            raise ValueError("missing feedback role")
    except (
        OSError,
        UnicodeError,
        YAMLError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:
        return _fail(" ".join(f"invalid model config {args.config}: {exc}".split()))

    try:
        results = [
            classify_feedback(router, record, profile) for record in records if record.text.strip()
        ]
        constraints: list[str] = []
        deltas = 0
        for result in results:
            if result.delta is not None:
                profile = apply_profile_delta(profile, result.delta)
                deltas += 1
            if result.constraint is not None:
                constraints.append(result.constraint)
        documents = [(out, profile.model_dump_json(indent=1))]
        if args.constraints_out is not None:
            documents.append((args.constraints_out, json.dumps(constraints, indent=1)))
        for path, document in documents:
            temporary: Path | None = None
            try:
                with NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(document)
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    except (OSError, ValueError) as exc:
        return _fail(" ".join(str(exc).split()))
    if args.constraints_out is None:
        for constraint in constraints:
            print(" ".join(constraint.splitlines()))
    print(
        f"classified {len(results)} records, applied {deltas} deltas, "
        f"{len(constraints)} constraints"
    )
    print(f"profile   {out}")
    return 0


def _quiet_scenedetect() -> None:
    """PySceneDetect sets its logger to INFO when imported, so import it first."""
    try:
        import scenedetect  # noqa: F401
    except ImportError:
        return  # reported later, when analysis needs it
    logging.getLogger("pyscenedetect").setLevel(logging.WARNING)


def _fail(msg: str) -> int:
    print(f"vlogcut: error: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
