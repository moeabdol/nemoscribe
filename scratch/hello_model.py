import torch
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.audio_utils import load_audio

model_id = "nvidia/nemotron-3.5-asr-streaming-0.6b"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)  # the front end: waveform → log-mel
model = AutoModelForRNNT.from_pretrained(model_id).to(device)  # encoder + RNNT

sr = processor.feature_extractor.sampling_rate
audio = load_audio(  # decodes + downsamples to 16 kHz mono
    "scratch/hello.wav",
    sampling_rate=sr,
)

inputs = processor(audio, sampling_rate=sr, language="en-US")
inputs.to(model.device, dtype=model.dtype)  # move tensors to GPU, match dtype

output = model.generate(  # the RNNT emission loop
    **inputs,
    return_dict_in_generate=True,
)
print(output.keys())
print(
    processor.decode(
        output.sequences, durations=output.durations, skip_special_tokens=False
    ),
)
