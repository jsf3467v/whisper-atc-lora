"""Transcribe arbitrary ATC audio with the fine-tuned (or any) Whisper model.

Accepts a single audio file or a folder of files, and writes the transcript plus
the extracted callsign for each clip to a JSONL results file. Reuses the project's
decode (transcribe.waveform), model loading (transcribe.whisper), and callsign
extraction (scoring.callsign), so nothing is duplicated and the output is scored
the same way the evaluation is.

WAV/FLAC/OGG/AIFF read directly through soundfile; MP3/M4A need a separate decode
step (e.g. convert to WAV first).

    python infer.py <audio_file_or_folder> [model_dir] [output.jsonl]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from transcribe import device, root, transcripts, waveform, whisper

sys.path.insert(0, str(root / "Evaluation"))
from scoring import callsign

AUDIO_EXT = (".wav", ".flac", ".ogg", ".aiff")
DEFAULT_MODEL = str(root / "models" / "whisper-small-lora")


def clips(target):
    path = Path(target)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in AUDIO_EXT)
    return [path]


def main(target, model_id=DEFAULT_MODEL, out="transcripts.jsonl", batch=16):
    files = clips(target)
    if not files:
        raise FileNotFoundError(f"No audio files found at {target}")
    dev = device()
    model, processor = whisper(model_id, dev)
    model.config.use_cache = True
    out_path = Path(out)
    print(f"device {dev.type} | {model_id} | {len(files)} file(s)")
    with out_path.open("w") as f:
        for start in range(0, len(files), batch):
            chunk = files[start:start + batch]
            texts = transcripts(model, processor, [waveform({"path": str(p)}) for p in chunk], dev)
            for p, text in zip(chunk, texts, strict=True):
                hyp = text.strip()
                f.write(json.dumps({"file": p.name, "transcript": hyp,
                                    "callsign": " ".join(callsign(hyp))}) + "\n")
            f.flush()
            if dev.type == "mps":
                torch.mps.empty_cache()
            elif dev.type == "cuda":
                torch.cuda.empty_cache()
    print(f"wrote {len(files)} transcript(s) to {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit("usage: python infer.py <audio_file_or_folder> [model_dir] [output.jsonl]")
    main(*args)