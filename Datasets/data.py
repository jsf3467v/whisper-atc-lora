"""Unify the free ATC corpora into train, validation, test_indomain, and test_ood splits."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from audio import handle
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from normalize import normalize

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
PROCESSED = HERE / "processed"
VAL = 0.05
SEED = 13

INDOMAIN, OOD = "test_indomain", "test_ood"
TEST_SPLITS = (INDOMAIN, OOD)
TEXT_NAMES = ("text", "transcript", "transcription", "sentence")
KEEP = ("id", "audio", "text_raw", "text_norm", "source", "seconds", "sample_rate")
TRAIN_SPLITS = ("train", "training", "validation", "valid", "dev")
SEGMENT = re.compile(r"(_\d{3})?_\d+_\d+(_[A-Z]+)?$|(-[A-Z])?__\d+-\d+$")

SOURCES = (
    ("Jzuluaga/atcosim_corpus", "atcosim", False),
    ("Jzuluaga/uwb_atcc", "uwb_atcc", False),
    ("Jzuluaga/atco2_corpus_1h", "atco2_1h", True),
)


def recording(uid):
    """Recording or speaker session of an ATCOSIM, UWB-ATCC, or ATCO2 utterance id."""
    return SEGMENT.sub("", uid, count=1)


def text_column(ds):
    for name in TEXT_NAMES:
        if name in ds.column_names:
            return name
    raise KeyError(f"No text column in {ds.column_names}; update TEXT_NAMES.")


def unified(ds, source):
    def rows(texts, audios):
        raw = [t or "" for t in texts]
        infos = [sf.info(handle(a)) for a in audios]
        return {
            "text_raw": raw,
            "text_norm": [normalize(t) for t in raw],
            "source": [source] * len(raw),
            "seconds": [i.frames / i.samplerate for i in infos],
            "sample_rate": [i.samplerate for i in infos],
        }

    if "id" not in ds.column_names:
        raise KeyError(f"No id column in {source}; recordings cannot be grouped.")
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.map(rows, input_columns=[text_column(ds), "audio"], batched=True)
    return ds.remove_columns([c for c in ds.column_names if c not in KEEP])


def parts(repo, source):
    data = load_dataset(repo, cache_dir=str(CACHE))
    train, test = [], []
    for name, split in data.items():
        bucket = train if name.lower() in TRAIN_SPLITS else test
        bucket.append(unified(split, source))
    return train, test


def held_out(ds, val, seed):
    """Whole recordings in seeded order, up to the validation share of utterances."""
    groups = np.unique([recording(u) for u in ds["id"]], return_inverse=True)[1]
    sizes = np.bincount(groups)
    if len(sizes) < 2:
        raise ValueError("A corpus needs at least two recordings to hold out validation.")
    order = np.random.default_rng(seed).permutation(len(sizes))
    count = np.searchsorted(np.cumsum(sizes[order]), val * len(groups), side="right")
    chosen = np.isin(groups, order[: min(max(count, 1), len(sizes) - 1)])
    return ds.select(np.flatnonzero(~chosen)), ds.select(np.flatnonzero(chosen))


def stratified(groups, val, seed):
    halves = [held_out(concatenate_datasets(g), val, seed) for g in groups]
    return (
        concatenate_datasets([fit for fit, _ in halves]),
        concatenate_datasets([held for _, held in halves]),
    )


def assembled(val, seed):
    train, indomain, ood = [], [], []
    for repo, source, is_ood in SOURCES:
        tr, te = parts(repo, source)
        if is_ood:
            ood += tr + te
        else:
            train += [tr] if tr else []
            indomain += te
    if not train:
        raise RuntimeError("No training data found; check TRAIN_SPLITS in data.py.")
    fit, held = stratified(train, val, seed)
    out = {"train": fit, "validation": held}
    if indomain:
        out[INDOMAIN] = concatenate_datasets(indomain)
    if ood:
        out[OOD] = concatenate_datasets(ood)
    return DatasetDict(out)


def fingerprint(val=VAL, seed=SEED):
    code = b"".join((HERE / f).read_bytes() for f in ("audio.py", "data.py", "normalize.py"))
    return f"val{val}-seed{seed}-{hashlib.sha256(code).hexdigest()[:10]}"


def corpora(val=VAL, seed=SEED):
    snapshot = PROCESSED / fingerprint(val, seed)
    if snapshot.exists():
        return load_from_disk(str(snapshot))
    data = assembled(val, seed)
    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)
    partial = PROCESSED / f"{snapshot.name}.partial"
    data.save_to_disk(str(partial))
    partial.rename(snapshot)
    return data
