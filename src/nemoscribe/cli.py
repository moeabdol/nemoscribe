"""Command-line interface — the only module that talks to the terminal."""

import argparse
import sys
import time
from collections import Counter
from collections.abc import Mapping
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
        help="save each source as WAV (plus a mix when multi-source; pairs with the JSONL manifests)",
    )
    s.add_argument(
        "--split",
        action="store_true",
        help="also write srt/jsonl/txt per source (per-source manifests for fine-tuning)",
    )
    s.add_argument(
        "--max-utterance-s",
        type=float,
        default=5.0,
        help="force-finalize an utterance after this long without silence (continuous speakers; default: 5)",
    )

    args = parser.parse_args(argv)
    commands = {
        "transcribe": _cmd_transcribe,
        "stream": _cmd_stream,
    }
    return commands[args.command](args)


def _parse_source(spec: str) -> tuple[str, str, str]:
    """Parse a source spec into (kind, param, label).

    Grammar: mic[:label] | system[:label] | file=PATH (label is the file stem).
    """
    if spec.startswith("file="):
        path = spec[len("file=") :]
        return "file", path, Path(path).stem
    kind, _, label = spec.partition(":")
    if kind in ("mic", "system"):
        return kind, "", label or kind
    raise ValueError(
        f"unsupported source {spec!r} — supported: mic[:label], system[:label], file=PATH"
    )


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
        default=50,
        help="max characters per SRT subtitle cue (default: 50)",
    )
    p.add_argument(
        "--cue-lead-ms",
        type=int,
        default=500,
        help="show each SRT cue this early, ms (default: 500)",
    )


def _write_outputs(
    events,
    path: Path,
    args: argparse.Namespace,
    audio_filepath: str | Mapping[str, str] | None = None,
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
    import threading

    import numpy as np

    from .audio import AudioDecodeError, save_wav
    from .engine import EngineError, Transcriber
    from .sources import SourceError, file_chunks, mic_chunks, system_chunks
    from .streaming import StreamingSession

    try:
        parsed = [_parse_source(spec) for spec in args.sources]
    except ValueError as e:
        print(f"nemoscribe: {e}", file=sys.stderr)
        return 2

    labels = [label for _, _, label in parsed]
    if len(set(labels)) != len(labels):
        print(
            "nemoscribe: duplicate source labels — label each source uniquely, "
            "e.g. mic:me system:them",
            file=sys.stderr,
        )
        return 2
    multi = len(parsed) > 1

    if not multi and parsed[0][0] == "file":
        stem = Path(parsed[0][1])
    elif not multi and parsed[0][0] == "mic":
        stem = Path(f"mic-{datetime.now():%y%m%d-%H%M%S}")  # noqa: DTZ005 - local label
    else:
        stem = Path(f"stream-{datetime.now():%y%m%d-%H%M%S}")  # noqa: DTZ005 - local label

    print("loading model...", file=sys.stderr)
    try:
        transcriber = Transcriber(device=args.device)
    except EngineError as e:
        print(f"nemoscribe: {e}", file=sys.stderr)
        return 2

    def make_chunks(kind: str, param: str):
        if kind == "mic":
            return mic_chunks()
        if kind == "system":
            return system_chunks()
        return file_chunks(Path(param), realtime=args.realtime)

    # single source streams partials live; multi prints one labeled line per event
    def show_partial(piece: str) -> None:
        print(piece, end="", flush=True)

    def show_event(event) -> None:
        if multi:
            print(f"[{event.source}] {event.text}", flush=True)
        else:
            print(flush=True)

    stop = threading.Event()
    captures: dict[str, list] = {label: [] for label in labels}
    sessions: list[tuple[str, StreamingSession]] = []
    feeders: list[threading.Thread] = []
    errors: list[str] = []

    try:
        for kind, param, label in parsed:
            session = StreamingSession(
                transcriber,
                language=args.language,
                lookahead=args.lookahead,
                reset_silence_s=args.reset_silence_ms / 1000,
                max_utterance_s=args.max_utterance_s,
                source=label,
                on_partial=(lambda p: None) if multi else show_partial,
                on_event=show_event,
            )
            chunks = make_chunks(kind, param)
            sessions.append((label, session))

            def feeder(chunks=chunks, session=session, label=label):
                try:
                    for chunk in chunks:
                        session.feed(chunk)
                        if args.save_audio:
                            captures[label].append(chunk.samples)
                        if stop.is_set():
                            break
                except (AudioDecodeError, SourceError) as e:
                    print(f"\nnemoscribe: {label}: {e}", file=sys.stderr)
                    errors.append(label)
                    stop.set()
                finally:
                    chunks.close()

            feeders.append(threading.Thread(target=feeder, daemon=True))
    except SourceError as e:
        print(f"nemoscribe: {e}", file=sys.stderr)
        return 2

    for th in feeders:
        th.start()
    try:
        for th in feeders:
            th.join()  # file sources end on their own
    except KeyboardInterrupt:  # live sources end here
        print("\nstopping...", file=sys.stderr)
        stop.set()
        for th in feeders:
            th.join()

    events = []
    for _, session in sessions:
        events.extend(session.close())
    print(flush=True)

    status = 1 if errors else 0

    audio_map = (
        {label: f"{stem.name}-{label}.wav" for label in labels} if multi else None
    )
    if args.save_audio:
        arrays = {}
        for label, parts in captures.items():
            if parts:
                arrays[label] = np.concatenate(parts)
                wav_name = (
                    audio_map[label] if audio_map else stem.with_suffix(".wav").name
                )
                save_wav(stem.parent / wav_name, arrays[label])
        if multi and arrays:
            longest = max(len(a) for a in arrays.values())
            mix = np.zeros(longest, dtype=np.float32)
            for a in arrays.values():
                mix[: len(a)] += a
            save_wav(stem.with_suffix(".wav"), mix)

    if not events:
        print("nemoscribe: no speech detected", file=sys.stderr)
        return status

    base = _write_outputs(
        events,
        stem,
        args,
        audio_filepath=audio_map
        if multi
        else (str(stem.with_suffix(".wav")) if parsed[0][0] == "mic" else None),
    )
    if args.split:
        sbase = stem.with_suffix("")  # single-file stems carry ".wav"; strip once
        for label in labels:
            own = [e for e in events if e.source == label]
            if own:
                per = sbase.parent / f"{sbase.name}-{label}"
                _write_outputs(own, per, args, audio_filepath=f"{per.name}.wav")
    print(f"{stem}: {len(events)} events → {base}.srt / .jsonl / .txt", file=sys.stderr)
    return status
