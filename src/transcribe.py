"""Transcribe the test splits with a Whisper model and save hypotheses for evaluation."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "Datasets"))

from audio import TARGET_SR, waveform
from data import TEST_SPLITS, corpora

BASE = "openai/whisper-small"
LANGUAGE, TASK = "en", "transcribe"
logging.get_logger("transformers.generation.utils").setLevel(logging.ERROR)


def adapted(name):
    return f"{Path(name).name}-lora"


MODEL_DIR = root / "models" / adapted(BASE)


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def release(dev):
    if dev.type == "mps":
        torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()


def whisper(model_id, dev):
    local = (root / model_id).resolve()
    found = next((p for p in (local, Path(model_id)) if p.exists()), None)
    if found is None and local.is_relative_to(root / "models"):
        raise FileNotFoundError(f"No model at {local.relative_to(root)}; run src/train.py first.")
    source = str(found or model_id)
    processor = WhisperProcessor.from_pretrained(source)
    model = WhisperForConditionalGeneration.from_pretrained(source)
    if model.config.tie_word_embeddings:
        model.proj_out.weight = model.get_input_embeddings().weight
    model = model.to(dev).eval()
    model.config.use_cache = True
    return model, processor


def built(model):
    config = Path(model.name_or_path) / "config.json"
    return config.stat().st_mtime if config.exists() else 0.0


def within(split, limit):
    return split.filter(lambda s: s <= limit, input_columns="seconds")


def cap(model, processor, texts):
    """Allow the longest training transcript plus its end token as new tokens after the prompt."""
    ids = processor.tokenizer(list(texts), add_special_tokens=False).input_ids
    prompt = len(processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)) + 1
    limit = min(max(map(len, ids)) + 1, model.config.max_target_positions - prompt)
    model.generation_config.max_length = None
    model.generation_config.max_new_tokens = limit
    return limit


def features(processor, items):
    arrays = [waveform(item) for item in items]
    return processor(arrays, sampling_rate=TARGET_SR, return_tensors="pt").input_features


def transcripts(model, processor, feats, dev):
    try:
        with torch.inference_mode():
            ids = model.generate(feats.to(dev), language=LANGUAGE, task=TASK)
    except RuntimeError as err:
        if len(feats) == 1 or "out of memory" not in str(err).lower():
            raise
        release(dev)
        half = len(feats) // 2
        return transcripts(model, processor, feats[:half], dev) + transcripts(
            model, processor, feats[half:], dev
        )
    texts = processor.batch_decode(
        ids.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return [text.strip() for text in texts]


def decoded(model, processor, dev, groups):
    """Transcribe groups of audio items while the next group's features are extracted."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = (pool.submit(features, processor, group) for group in groups)
        ahead = next(pending, None)
        while ahead is not None:
            current, ahead = ahead, next(pending, None)
            texts = transcripts(model, processor, current.result(), dev)
            release(dev)
            yield texts


def completed(path, refs, since):
    if not path.exists():
        return set()
    if path.stat().st_mtime < since:
        raise ValueError(f"{path.relative_to(root)} predates the model; delete it.")
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if any(not 0 <= r["i"] < len(refs) or r["ref"] != refs[r["i"]] for r in records):
        raise ValueError(f"{path.relative_to(root)} does not match the current split; delete it.")
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return {r["i"] for r in records}


def hypotheses(split, name, model, processor, dev, tag, batch):
    out_path = root / "predictions" / f"{name}-{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    refs, sources, seconds = (list(split[c]) for c in ("text_raw", "source", "seconds"))
    done = completed(out_path, refs, built(model))
    pending = sorted(set(range(len(split))) - done, key=seconds.__getitem__, reverse=True)
    chunks = [pending[s : s + batch] for s in range(0, len(pending), batch)]
    audio = (split.select(chunk)["audio"] for chunk in chunks)
    with out_path.open("a") as f:
        for chunk, guesses in zip(chunks, decoded(model, processor, dev, audio), strict=True):
            f.writelines(
                json.dumps({"i": i, "source": sources[i], "ref": refs[i], "hyp": hyp}) + "\n"
                for i, hyp in zip(chunk, guesses, strict=True)
            )
            f.flush()
    return len(pending)


def main(model_id=BASE, tag=None, batch=16, splits=TEST_SPLITS):
    tag = tag or Path(model_id).name
    dev = device()
    model, processor = whisper(model_id, dev)
    data = corpora()
    window = processor.feature_extractor.chunk_length
    limit = cap(model, processor, data["train"]["text_norm"])
    print(f"device {dev.type} | {model_id} | tag {tag} | max new tokens {limit}")
    for name in splits:
        over = len(data[name]) - len(within(data[name], window))
        written = hypotheses(data[name], name, model, processor, dev, tag, batch)
        print(f"{name}: {written} new of {len(data[name])} utterances | over {window} s {over}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
