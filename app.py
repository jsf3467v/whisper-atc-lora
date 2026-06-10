"""Gradio demo for the LoRA-fine-tuned ATC speech recognizer.

Transcribes one air-traffic-control clip and surfaces the callsign, e.g.
"united four seven zero" (airline style) or "november one two three alpha"
(registration style). The demo reuses the project's shared audio decoder,
text normalizer, and callsign extractor,

"""

from __future__ import annotations

import sys
from pathlib import Path
from functools import lru_cache

import torch
import gradio as gr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "SRC"))
sys.path.insert(0, str(HERE / "Evaluation"))

from transcribe import device, whisper, transcripts, waveform, root
from scoring import callsign

MODEL_DIR = str(root / "models" / "whisper-small-lora")


@lru_cache(maxsize=1)
def asr():
    """Load the model and processor once on the fastest backend, then keep them resident."""
    dev = device()
    model, processor = whisper(MODEL_DIR, dev)
    model.config.use_cache = True
    print(f"ATC ASR ready on {dev.type}")
    return model, processor, dev


def recognition(path):
    """One clip in, transcript and callsign out."""
    if not path:
        return "Upload or record a clip to begin.", ""
    model, processor, dev = asr()
    text = transcripts(model, processor, [waveform({"path": str(path)})], dev)[0].strip()
    if dev.type == "mps":
        torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()
    return text, " ".join(callsign(text)) or "(none detected)"


def clips():
    """Relative example clips for the UI, if an examples/ folder is present."""
    folder = HERE / "examples"
    found = sorted(str(p) for p in folder.glob("*.wav")) if folder.exists() else []
    return [[p] for p in found] or None


def demo():
    """Two-field interface: clip in, transcript and callsign out."""
    info = ("Upload or record an air-traffic-control clip. "
            "WAV/FLAC/OGG/AIFF read directly; convert MP3/M4A to WAV first.")
    inp = gr.Audio(sources=["upload", "microphone"], type="filepath", label="ATC clip")
    out = [gr.Textbox(label="Transcript"), gr.Textbox(label="Callsign")]
    return gr.Interface(fn=recognition, inputs=inp, outputs=out, examples=clips(),
                        title="ATC Speech Recognition (Whisper-small + LoRA)", description=info)


if __name__ == "__main__":
    demo().launch()