"""Command line: `vlogcut cut <footage folder>` -> rough-cut preview,
`vlogcut style <reference videos>` -> style profile JSON.

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
    c.add_argument("-w", "--workspace", type=Path, help="cache dir (default ./workspace/<folder>)")
    c.add_argument("-o", "--out", type=Path, help="preview path (default inside the workspace)")
    c.add_argument("--music", type=Path, help="background music file (your own)")
    c.add_argument("--music-gain", type=float, default=-14.0, help="music level, dB")
    c.add_argument("--background", choices=["black", "blur"], default="black")
    c.add_argument("--no-subtitles", action="store_true", help="no debug subtitle track")
    c.add_argument("--no-render", action="store_true", help="plan only, write the EDL")
    c.add_argument("-v", "--verbose", action="store_true")

    s = sub.add_parser("style", help="extract a style profile from finished reference videos")
    s.add_argument("refs", type=Path, nargs="+", help="videos in the style to learn")
    s.add_argument("-o", "--out", type=Path, default=Path("profile.json"), help="profile JSON")
    s.add_argument(
        "-w", "--workspace", type=Path, default=Path("workspace") / "references",
        help="cache dir for reference proxies and cuts (default workspace/references)",
    )  # fmt: skip
    s.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        _quiet_scenedetect()
    return _cut(args) if args.command == "cut" else _style(args)


def _cut(args: argparse.Namespace) -> int:
    from vlogcut.analyze.shots import ShotDetectionError
    from vlogcut.ingest import FFmpegError, FFmpegNotFoundError, ProxyError
    from vlogcut.pipeline import cut
    from vlogcut.plan import RoughCutError
    from vlogcut.render import RenderError

    if args.music is not None and not args.music.is_file():
        return _fail(f"music file not found: {args.music}")
    workspace = args.workspace or Path("workspace") / args.folder.resolve().name
    started = time.perf_counter()
    try:
        result = cut(
            args.folder, workspace,
            target_s=args.target, strategy=args.strategy, music=args.music,
            music_gain_db=args.music_gain, out=args.out, render=not args.no_render,
            preview_params={"background": args.background, "subtitles": not args.no_subtitles},
            progress=lambda msg: print(msg, file=sys.stderr),
        )  # fmt: skip
    except FFmpegNotFoundError as exc:
        return _fail(str(exc))
    except ImportError as exc:
        return _fail(f'{exc}; install the analysis extra: pip install -e ".[analyze]"')
    except (FileNotFoundError, RoughCutError, ShotDetectionError) as exc:
        return _fail(str(exc))
    except (ProxyError, RenderError, FFmpegError) as exc:
        return _fail(f"ffmpeg failed: {exc}")

    usable = sum(not s.quality.rejected for s in result.shots)
    print(f"footage   {len(result.assets)} clips, {sum(a.duration_s for a in result.assets):.0f} s")
    for name, why in result.skipped:
        print(f"skipped   {name}: {why}")
    print(f"shots     {len(result.shots)} ({len(result.shots) - usable} rejected)")
    print(f"cut       {len(result.edl.clips)} clips, {result.edl.total_duration_s:.1f} s")
    print(f"edl       {result.edl_path}")
    if result.preview_path:
        print(f"preview   {result.preview_path}")
    print(f"done in {time.perf_counter() - started:.1f} s")
    return 0


def _style(args: argparse.Namespace) -> int:
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
