"""Unify the free ATC corpora into one DatasetDict.

Splits returned with train, validation, test_indomain, test_ood.
The large ATCO2-PL set and LDC-ATCC are paid and are not used; ATCO2-1h is held
out as the out-of-distribution probe with real radio, unseen airports.

A processed snapshot is written to ./processed on first run; later runs reload
it and skip both download and assembly. A crash mid-stage resumes cleanly.
The Hugging Face download itself caches to ./cache and resumes on its own.
"""
from __future__ import annotations

from pathlib import Path

from normalize import normalize

from datasets import (
    Audio,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
PROCESSED = HERE / "processed"

SAMPLE_RATE = 16_000
TEXT_NAMES = ("text", "transcript", "transcription", "sentence")
KEEP = ("audio", "text_raw", "text_norm", "source", "split_orig", "ood")
TRAIN_SPLITS = ("train", "training", "validation", "valid", "dev")

# (repo, source tag, is out-of-distribution). Dataset identities are the only
# values that genuinely must be named here.
SOURCES = (
    ("Jzuluaga/atcosim_corpus", "atcosim", False),
    ("Jzuluaga/uwb_atcc", "uwb_atcc", False),
    ("Jzuluaga/atco2_corpus_1h", "atco2_1h", True),
)


def text_column(ds):
    for name in TEXT_NAMES:
        if name in ds.column_names:
            return name
    raise KeyError(f"No text column in {ds.column_names}; update TEXT_NAMES.")


def unified(ds, source, split_orig, ood):
    col = text_column(ds)

    def row(example):
        raw = example[col] or ""
        return {
            "text_raw": raw,
            "text_norm": normalize(raw),
            "source": source,
            "split_orig": split_orig,
            "ood": ood,
        }

    ds = ds.map(row)
    extra = [c for c in ds.column_names if c not in KEEP]
    ds = ds.remove_columns(extra)
    return ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))


def parts(repo, source, ood):
    data = load_dataset(repo, cache_dir=str(CACHE))
    train, test = [], []
    for name, split in data.items():
        bucket = train if name.lower() in TRAIN_SPLITS else test
        bucket.append(unified(split, source, name, ood))
    return train, test


def corpora(val=0.05, seed=13):
    if PROCESSED.exists():
        return load_from_disk(str(PROCESSED))

    train, indomain, ood = [], [], []
    for repo, source, is_ood in SOURCES:
        tr, te = parts(repo, source, is_ood)
        if is_ood:
            ood += tr + te
        else:
            train += tr
            indomain += te

    train_all = concatenate_datasets(train)
    if not len(train_all):
        raise RuntimeError("No training data found; check split names in data.py.")

    halves = train_all.train_test_split(test_size=val, seed=seed)
    out = {"train": halves["train"], "validation": halves["test"]}
    if indomain:
        out["test_indomain"] = concatenate_datasets(indomain)
    if ood:
        out["test_ood"] = concatenate_datasets(ood)

    out = DatasetDict(out)
    out.save_to_disk(str(PROCESSED))
    return out