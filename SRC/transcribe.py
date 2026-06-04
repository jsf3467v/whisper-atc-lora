"""Transcribe audio with a Whisper model and save hypotheses for evaluation.

Inference only. Works for the zero-shot baseline and, unchanged, for a fine-tuned
checkpoint by passing a different model id and tag.

Predictions land in <project>/predictions/<split>-<tag>.jsonl, one record per
utterance: index, source, reference text, hypothesis. The tag in the filename
keeps every model's output distinct in a single folder.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio

HERE = Path(__file__).resolve().parent
root = next((p for p in [HERE, *HERE.parents] if (p / "Datasets" / "data.py").exists()), None)
if root is None:
    raise FileNotFoundError("Could not find the Datasets folder; keep this inside the ATC project.")
sys.path.insert(0, str(root / "Datasets"))

from data import corpora
from audio import waveform, TARGET_SR


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def whisper(model_id, dev):
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(dev).eval()
    return model, processor


def transcripts(model, processor, arrays, dev):
    feats = processor(arrays, sampling_rate=TARGET_SR, return_tensors="pt").input_features.to(dev)
    with torch.inference_mode():
        ids = model.generate(feats, language="en", task="transcribe", max_new_tokens=128)
    out = processor.batch_decode(ids.cpu(), skip_special_tokens=True)
    del feats, ids
    return out


def completed(path):
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["i"] for line in f if line.strip()}


def hypotheses(split, name, model, processor, dev, tag, batch=16):
    out_path = root / "predictions" / f"{name}-{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = split.cast_column("audio", Audio(decode=False))
    refs, sources = raw["text_raw"], raw["source"]
    pending = [i for i in range(len(raw)) if i not in completed(out_path)]
    with out_path.open("a") as f:
        for start in range(0, len(pending), batch):
            chunk = pending[start:start + batch]
            guesses = transcripts(model, processor, [waveform(raw[i]["audio"]) for i in chunk], dev)
            for i, hyp in zip(chunk, guesses):
                f.write(json.dumps({"i": i, "source": sources[i], "ref": refs[i], "hyp": hyp}) + "\n")
            f.flush()
            if dev.type == "mps":
                torch.mps.empty_cache()
            elif dev.type == "cuda":
                torch.cuda.empty_cache()


def main(model_id="openai/whisper-small", tag="whisper-small", batch=16,
         splits=("test_indomain", "test_ood")):
    torch.manual_seed(0)
    dev = device()
    model, processor = whisper(model_id, dev)
    data = corpora()
    print(f"device {dev.type} | {model_id} | tag {tag}")
    for name in splits:
        hypotheses(data[name], name, model, processor, dev, tag, batch)
        print(f"{name}: {len(data[name])} utterances written")


if __name__ == "__main__":
    main()