"""Report split sizes, hours, source counts, and normalized samples."""

from __future__ import annotations

import random
from collections import Counter

from data import TEST_SPLITS, corpora

PREVIEW_SPLITS = ("train", *TEST_SPLITS)
TEXT_FIELDS = ("source", "text_raw", "text_norm")


def samples(data, k=3):
    rng = random.Random(1)
    for name in PREVIEW_SPLITS:
        split = data.get(name)
        if not split:
            continue
        view = split.select_columns(list(TEXT_FIELDS))
        for i in rng.sample(range(len(view)), min(k, len(view))):
            row = view[i]
            print(f"  [{name}/{row['source']}] {row['text_raw']!r} -> {row['text_norm']!r}")


def main():
    data = corpora()
    print("split sizes, hours, source counts")
    for name, split in data.items():
        counts = ", ".join(f"{k}={v}" for k, v in sorted(Counter(split["source"]).items()))
        print(f"  {name:<14} {len(split):>7,}  {sum(split['seconds']) / 3600:6.2f} h  {counts}")
    print("samples (raw -> normalized)")
    samples(data)


if __name__ == "__main__":
    main()
