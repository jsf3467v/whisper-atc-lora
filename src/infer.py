"""Transcribe an audio file or folder and write each transcript with its callsign to JSONL."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transcribe import MODEL_DIR, decoded, device, root, whisper

from audio import AUDIO_EXT  # isort: skip

sys.path.insert(0, str(root / "Evaluation"))
from scoring import callsign

USAGE = "usage: python src/infer.py <audio_file_or_folder> [model_dir] [output.jsonl]"


def clips(target):
    path = Path(target)
    if not path.exists():
        raise FileNotFoundError(f"{target} does not exist")
    found = sorted(path.iterdir()) if path.is_dir() else [path]
    return [p for p in found if p.suffix.lower() in AUDIO_EXT]


def main(target, model_id=MODEL_DIR, out=root / "predictions" / "transcripts.jsonl", batch=16):
    files = clips(target)
    if not files:
        raise FileNotFoundError(f"No {'/'.join(AUDIO_EXT)} files found at {target}")
    dev = device()
    model, processor = whisper(model_id, dev)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [files[s : s + batch] for s in range(0, len(files), batch)]
    audio = ([{"path": str(p)} for p in chunk] for chunk in chunks)
    print(f"device {dev.type} | {model_id} | {len(files)} file(s)")
    with out_path.open("w") as f:
        for chunk, texts in zip(chunks, decoded(model, processor, dev, audio), strict=True):
            f.writelines(
                json.dumps({"file": p.name, "transcript": t, "callsign": " ".join(callsign(t))})
                + "\n"
                for p, t in zip(chunk, texts, strict=True)
            )
            f.flush()
    print(f"wrote {len(files)} transcript(s) to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(USAGE)
    main(*sys.argv[1:4])
