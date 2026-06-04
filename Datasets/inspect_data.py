"""Data inventory / EDA.  Run from the Datasets folder with python inspect_data.py

Reports split sizes, source counts, sampled-hour estimates, and raw-to-normalized
transcript samples so the normalizer can be checked before anything is built on it.
Durations are read from audio headers and previews touch only text, so this runs
without an audio-decode backend.
"""
from __future__ import annotations

import io
import random
from collections import Counter

import soundfile as sf
from datasets import Audio

from data import corpora

PREVIEW_SPLITS = ("train", "test_indomain", "test_ood")
TEXT_FIELDS = ("source", "text_raw", "text_norm")


def span(item):
    source = item["path"] or io.BytesIO(item["bytes"])
    info = sf.info(source)
    return info.frames / info.samplerate


def hours(split, n=150):
    total = len(split)
    if total == 0:
        return 0.0
    raw = split.cast_column("audio", Audio(decode=False))
    idx = random.Random(0).sample(range(total), min(n, total))
    spans = [span(raw[i]["audio"]) for i in idx]
    return sum(spans) / len(spans) * total / 3600


def main():
    data = corpora()

    print("split sizes")
    for name, split in data.items():
        print(f"  {name:<14} {len(split):>7,}")

    print("source counts")
    for name, split in data.items():
        counts = Counter(split["source"])
        print(f"  {name:<14} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    print("hours (sampled estimate)")
    for name, split in data.items():
        print(f"  {name:<14} ~{hours(split):.2f}")

    print("samples (raw -> norm)")
    rng = random.Random(1)
    for name in PREVIEW_SPLITS:
        split = data.get(name)
        if not split or len(split) == 0:
            continue
        view = split.select_columns(list(TEXT_FIELDS))
        for i in rng.sample(range(len(view)), min(3, len(view))):
            row = view[i]
            print(f"  [{name}/{row['source']}] {row['text_raw']!r} -> {row['text_norm']!r}")


if __name__ == "__main__":
    main()