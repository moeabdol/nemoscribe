"""Command-line interface — the only module that talks to the terminal."""

import argparse
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nemoscribe",
        description="Multi-lingual transcriber built on Nemotron 3.5 ASR.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # transcribe command
    t = sub.add_parser("transcribe", help="transcribe audio/video files")
    _add_shared_args(t)
    t.add_argument("files", nargs="+", type=Path, help="audio or video files")

    # stream command
    s = sub.add_parser("stream", help="live-transcribe audio sources")
    _add_shared_args(s)
    s.add_argument(
        "sources",
        nargs="+",
        help="v1: file=PATH (mic and system sources arrive in later steps)",
    )
    s.add_argument(
        "--lookahead",
        type=int,
        default=6,
        choices=[3, 6, 13],
        help="right-context frames: 3=320ms 6=560ms 13=1120ms (default 6)",
    )
    s.add_argument(
        "--reset-silence-ms",
        type=int,
        default=1000,
        help="silence that ends an utterance (default: 1000)",
    )
    s.add_argument(
        "--realtime", action="store_true", help="pace file replay to the wall clock"
    )
    s.add_argument(
        "--save-audio",
        action="store_true",
        help="save captured mic audio as WAV (pairs with the JSONL manifest)",
    )

    args = parser.parse_args(argv)
    commands = {
        "transcribe": _cmd_transcribe,
        "stream": _cmd_stream,
    }
    return commands[args.command](args)


def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--language",
        default="en-US",
        help="locale like en-US or ar-AR (default: en-US)",
    )
    p.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="inference device (default: auto-detect)",
    )
    p.add_argument(
        "--max-cue-chars",
        type=int,
        default=84,
        help="max characters per SRT subtitle cue (default: 84)",
    )
    p.add_argument(
        "--cue-lead-ms",
        type=int,
        default=300,
        help="show each SRT cue this early, ms (default: 300)",
    )


def _write_outputs(
    events, path: Path, args: argparse.Namespace, audio_filepath: str | None = None
) -> Path:
    from .writers import write_jsonl, write_srt, write_txt

    base = path.with_suffix("")
    base.with_suffix(".srt").write_text(
        write_srt(
            events,
            max_cue_chars=args.max_cue_chars,
            lead_in_s=args.cue_lead_ms / 1000,
        ),
        encoding="utf-8",
    )
    base.with_suffix(".jsonl").write_text(
        write_jsonl(events, audio_filepath=audio_filepath or str(path)),
        encoding="utf-8",
    )
    base.with_suffix(".txt").write_text(write_txt(events), encoding="utf-8")
    return base


def _cmd_transcribe(args: argparse.Namespace) -> int:
    from .audio import SAMPLE_RATE, AudioDecodeError, load
    from .engine import EngineError, Transcriber
    from .vad import VadError

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

        base = _write_outputs(events, path, args)
        print(
            f"{path}: {len(events)} segments, {duration:.0f}s audio, "
            f"RTF {work / duration:.2f} → {base}.srt / .jsonl / .txt",
            file=sys.stderr,
        )

        if args.language == "auto" and events:
            tally = Counter(e.language or "unknown" for e in events)
            summary = ", ".join(f"{lang} {n}" for lang, n in tally.most_common())
            print(f"{path}: detected languages: {summary}", file=sys.stderr)

    return status


def _cmd_stream(args: argparse.Namespace) -> int:
    import numpy as np

    from .audio import save_wav
    from .engine import EngineError, Transcriber
    from .sources import file_chunks, mic_chunks
    from .streaming import StreamingSession

    if len(args.sources) != 1:
        print(
            "nemoscribe: one source at a time for now (multi-source arrives in step 9)",
            file=sys.stderr,
        )
        return 2

    spec = args.sources[0]
    if spec == "mic":
        chunks = mic_chunks()
        stem = Path(f"mic-{datetime.now():%y%m%d-%H%M%S}")  # noqa: DTZ005 — local wall-clock label, formatted and discarded; never compared
    elif spec.startswith("file="):
        stem = Path(spec[len("file=") :])
        chunks = file_chunks(stem, realtime=args.realtime)
    else:
        print(
            f"nemoscribe: unsupported source {spec!r} — supported: mic, file=PATH "
            "(system audio arrives in step 9)",
            file=sys.stderr,
        )
        return 2

    print("loading model...", file=sys.stderr)
    try:
        transcriber = Transcriber(device=args.device)
    except EngineError as e:
        print(f"nemoscribe: {e}", file=sys.stderr)
        return 2

    # stream mode inverts the stdout rule: the live text IS the product
    session = StreamingSession(
        transcriber,
        language=args.language,
        lookahead=args.lookahead,
        reset_silence_s=args.reset_silence_ms / 1000,
        on_partial=lambda piece: print(piece, end="", flush=True),
        on_event=lambda e: print(flush=True),
    )

    captured = []
    try:
        for chunk in chunks:
            session.feed(chunk)
            if args.save_audio:
                captured.append(chunk.samples)
    except KeyboardInterrupt:
        print("\nstopping...", file=sys.stderr)
    chunks.close()
    events = session.close()
    print(flush=True)

    if not events:
        print("nemoscribe: no speech detected", file=sys.stderr)
        return 0

    if args.save_audio and captured:
        save_wav(stem.with_suffix(".wav"), np.concatenate(captured))

    base = _write_outputs(
        events, stem, args, audio_filepath=str(stem.with_suffix(".wav"))
    )
    print(f"{stem}: {len(events)} events → {base}.srt / .jsonl / .txt", file=sys.stderr)
    return 0
