"""Command-line interface — the only module that talks to the terminal."""

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nemoscribe",
        description="Multi-lingual transcriber build on Nemotron 3.5 ASR.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="transcribe audio/video files")
    t.add_argument("files", nargs="+", type=Path, help="audio or video files")
    t.add_argument(
        "--language",
        default="en-US",
        help="locale like en-US or ar-AR (default: en-US)",
    )
    t.add_argument("--device", default=None, help="cuda or cpu (default: auto)")

    args = parser.parse_args(argv)
    return _cmd_transcribe(args)


def _cmd_transcribe(args: argparse.Namespace) -> int:
    from .audio import AudioDecodeError, load
    from .engine import Transcriber
    from .vad import VadError
    from .writers import write_jsonl, write_srt, write_txt

    print("loading model...", file=sys.stderr)
    transcriber = Transcriber(device=args.device)

    status = 0
    for path in args.files:
        try:
            events = transcriber.transcribe(load(path), language=args.language)
        except (AudioDecodeError, VadError) as e:
            print(f"nemoscribe: {path}: {e}", file=sys.stderr)
            status = 1
            continue
        base = path.with_suffix("")
        base.with_suffix(".srt").write_text(write_srt(events), encoding="utf-8")
        base.with_suffix(".jsonl").write_text(
            write_jsonl(events, audio_filepath=str(path)), encoding="utf-8"
        )
        base.with_suffix(".txt").write_text(write_txt(events), encoding="utf-8")
        print(
            f"{path}: {len(events)} segments → {base}.srt / .jsonl / .txt",
            file=sys.stderr,
        )
    return status
