"""Gradio demo that transcribes one air traffic control clip and extracts its callsign."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "Evaluation"))

from scoring import callsign

from transcribe import BASE, MODEL_DIR, device, features, release, transcripts, whisper

from audio import AUDIO_EXT  # isort: skip

MODEL = os.environ.get("ATC_MODEL", MODEL_DIR)


@lru_cache(maxsize=1)
def asr():
    dev = device()
    model, processor = whisper(MODEL, dev)
    print(f"ATC ASR ready on {dev.type}")
    return model, processor, dev


def recognition(path):
    if not path:
        return "Upload or record a clip to begin.", ""
    model, processor, dev = asr()
    text = transcripts(model, processor, features(processor, [{"path": str(path)}]), dev)[0]
    release(dev)
    return text, " ".join(callsign(text)) or "(none detected)"


def examples():
    folder = HERE / "examples"
    found = sorted(str(p) for p in folder.glob("*.wav")) if folder.exists() else []
    return [[p] for p in found] or None


def interface():
    formats = "/".join(ext.lstrip(".").upper() for ext in AUDIO_EXT)
    info = f"Upload or record an air traffic control clip. Accepted formats are {formats}."
    inp = gr.Audio(sources=["upload", "microphone"], type="filepath", label="ATC clip")
    out = [gr.Textbox(label="Transcript"), gr.Textbox(label="Callsign")]
    return gr.Interface(
        fn=recognition,
        inputs=inp,
        outputs=out,
        examples=examples(),
        cache_examples=True,
        cache_mode="lazy",
        title=f"ATC Speech Recognition ({Path(BASE).name} + LoRA)",
        description=info,
    )


demo = interface()

if __name__ == "__main__":
    demo.launch()
