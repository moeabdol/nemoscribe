"""Command-line interface — the only module that talks to the terminal."""

import argparse
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nemoscribe",
        description="Multi-lingual transcriber built on Nemotron 3.5 ASR.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="transcribe audio/video files")
    t.add_argument("files", nargs="+", type=Path, help="audio or video files")
    t.add_argument(
        "--language",
        default="en-US",
        help="locale like en-US or ar-AR (default: en-US)",
    )
    t.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="inference device (default: auto-detect)",
    )

    args = parser.parse_args(argv)
    return _cmd_transcribe(args)


def _cmd_transcribe(args: argparse.Namespace) -> int:
    from .audio import SAMPLE_RATE, AudioDecodeError, load
    from .engine import EngineError, Transcriber
    from .vad import VadError
    from .writers import write_jsonl, write_srt, write_txt

    print("loading model...", file=sys.stderr)
    try:
        transcriber = Transcriber(device=args.device)
    except EngineError as e:
        print(f"nemoscribe: {e}", file=sys.stderr)
        return 2

    status = 0
    for path in args.files:
        try:
            audio = load(path)
            t0 = time.perf_counter()
            events = transcriber.transcribe(audio, language=args.language)
            work = time.perf_counter() - t0
        except (AudioDecodeError, VadError) as e:
            print(f"nemoscribe: {path}: {e}", file=sys.stderr)
            status = 1
            continue

        duration = len(audio) / SAMPLE_RATE

        base = path.with_suffix("")
        base.with_suffix(".srt").write_text(write_srt(events), encoding="utf-8")
        base.with_suffix(".jsonl").write_text(
            write_jsonl(events, audio_filepath=str(path)), encoding="utf-8"
        )
        base.with_suffix(".txt").write_text(write_txt(events), encoding="utf-8")
        print(
            f"{path}: {len(events)} segments, {duration:.0f}s audio, "
            f"RTF {work / duration:.2f} → {base}.srt / .jsonl / .txt",
            file=sys.stderr,
        )
    return status
