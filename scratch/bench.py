import sys
import time

from nemoscribe.audio import SAMPLE_RATE, load
from nemoscribe.engine import Transcriber

path, device = sys.argv[1], sys.argv[2]
language = sys.argv[3] if len(sys.argv) > 3 else "en-US"
dtype = sys.argv[4] if len(sys.argv) > 4 else "fp32"

audio = load(path)
duration = len(audio) / SAMPLE_RATE

# measure model load time
t0 = time.perf_counter()
transcriber = Transcriber(device=device)

if dtype == "fp16":
    transcriber.model.half()

load_s = time.perf_counter() - t0

# warm-up: 10 s slice
transcriber.transcribe(audio[: SAMPLE_RATE * 10], language=language)

# measure work time
t0 = time.perf_counter()
events = transcriber.transcribe(audio, language=language)
work_s = time.perf_counter() - t0

print(
    f"{device}/{dtype}: load {load_s:.1f}s | {duration:.1f}s audio "
    f"in {work_s:.1f}s | RTF {work_s / duration:.3f} | {len(events)} segments"
)

if device == "cuda":
    import torch

    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
