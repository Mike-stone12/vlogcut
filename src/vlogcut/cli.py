"""Command line: `vlogcut cut <footage folder>` -> rough-cut preview,
`vlogcut style build <reference videos>` -> style profile JSON, `style show / diff`.

Thin wrapper over `vlogcut.pipeline` / `vlogcut.style`; argument parsing and messages only.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

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
    c.add_argument("-w", "--workspace", type=Path, help="cache dir (default ./workspace/<folder>)")
    c.add_argument("-o", "--out", type=Path, help="preview path (default inside the workspace)")
    c.add_argument("--music", type=Path, help="background music file (your own)")
    c.add_argument("--music-gain", type=float, default=-14.0, help="music level, dB")
    c.add_argument("--background", choices=["black", "blur"], default="black")
    c.add_argument("--no-subtitles", action="store_true", help="no debug subtitle track")
    c.add_argument("--no-render", action="store_true", help="plan only, write the EDL")
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
    commands = {"build": _style_build, "show": _style_show, "diff": _style_diff}
    return commands[args.style_command](args)


def _cut(args: argparse.Namespace) -> int:
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
    workspace = args.workspace or Path("workspace") / args.folder.resolve().name
    started = time.perf_counter()
    try:
        result = cut(
            args.folder, workspace,
            target_s=args.target, strategy=args.strategy, style=args.style, music=args.music,
            music_gain_db=args.music_gain, out=args.out, render=not args.no_render,
            preview_params={"background": args.background, "subtitles": not args.no_subtitles},
            progress=lambda msg: print(msg, file=sys.stderr),
        )  # fmt: skip
    except FFmpegNotFoundError as exc:
        return _fail(str(exc))
    except ImportError as exc:
        return _fail(f'{exc}; install the analysis extra: pip install -e ".[analyze]"')
    except (FileNotFoundError, RoughCutError, ShotDetectionError, StyleError) as exc:
        return _fail(str(exc))
    except (ProxyError, RenderError, FFmpegError) as exc:
        return _fail(f"ffmpeg failed: {exc}")

    usable = sum(not s.quality.rejected for s in result.shots)
    print(f"footage   {len(result.assets)} clips, {sum(a.duration_s for a in result.assets):.0f} s")
    for name, why in result.skipped:
        print(f"skipped   {name}: {why}")
    print(f"shots     {len(result.shots)} ({len(result.shots) - usable} rejected)")
    if result.profile is not None:
        print(f"style     {args.style.name}: {result.profile.notes}")
    print(f"cut       {len(result.edl.clips)} clips, {result.edl.total_duration_s:.1f} s")
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
