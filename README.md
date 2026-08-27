# NeMoscribe

A multi-lingual CLI transcriber (batch + live streaming) built on NVIDIA
Nemotron 3.5 ASR. NeMoscribe has two subcommands `transcribe` and `stream`. The
transcribe subcommand can read audio or video files and generate transcriptions
in `TXT`, `JSONL` and `SRT` formats. The stream subcommand can capture multiple
live sources like `mic`, `system` and also `file` in a single run then generates
the same transcription formats. NeMoscribe is a wrapper around NVIDIA's Nemotron
ASR model which supports 40 language-locales in three quality tiers (en-US,
en-GB, ar-AR are "transcription-ready"). NeMoscribe was built and tested for
English and Arabic (--language accepts any supported language by the model).

## Requirements

- Python >= 3.12

| Dependency       | Needed for                        | Linux                                           | Windows                                          |
| ---------------- | --------------------------------- | ----------------------------------------------- | ------------------------------------------------ |
| ffmpeg + ffprobe | All audio loading, system capture | apt install ffmpeg / pacman -S ffmpeg           | winget install ffmpeg                            |
| PortAudio        | mic source                        | apt install libportaudio2 / pacman -S portaudio | bundled in the sounddevice wheel — nothing to do |
| PipeWire/Pulse   | system source                     | usually preinstalled                            | not supported (see matrix)                       |

## Installation

## Quick Start

Transcribe `talk.mp4` and generate `talk.txt` `talk.jsonl` and `talk.srt`.

```bash
nemoscribe transcribe talk.mp4
```

Transcribe a live stream from your mic. This will show your speech as you talk
`Ctrl-C` ends it.

```bash
nemoscribe stream mic
```

Transcribe a meeting live, save the audio and generate transcription files. This
is the full set, and the naming law: every `.wav` pairs with the same-named
`.srt/.jsonl/.txt; stream-X.wav` is the playable session — `mpv stream-X.wav`
auto-loads its subtitles — and `stream-X-me.wav` etc. are clean per-speaker
stems.

```bash
nemoscribe stream mic:me system:them --split --save-audio
```

## Support Matrix

## Performance

## Caches

## Fine-Tuning

## Development
