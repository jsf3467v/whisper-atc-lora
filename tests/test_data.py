"""Unit tests for corpus unification, the recording-level split, and the snapshot in Datasets/data.py."""

import io

import data
import numpy as np
import pytest
import soundfile as sf
from data import held_out, recording
from datasets import Audio, Dataset, DatasetDict


def corpus(sessions, per_session):
    ids = [
        f"atcosim_sm{s}_01_{k:03d}_000000_000100"
        for s in range(sessions)
        for k in range(per_session)
    ]
    return Dataset.from_dict({"id": ids})


def test_disjoint():
    fit, held = held_out(corpus(40, 25), 0.05, 13)
    train_keys = {recording(u) for u in fit["id"]}
    held_keys = {recording(u) for u in held["id"]}
    assert train_keys.isdisjoint(held_keys)
    assert len(fit) + len(held) == 1000 and 0 < len(held) <= 50


def test_seeded():
    first, second = (list(held_out(corpus(40, 25), 0.05, 13)[1]["id"]) for _ in range(2))
    assert first == second


def test_both_sides():
    fit, held = held_out(corpus(3, 100), 0.01, 0)
    assert len(held) == 100 and len(fit) == 200


def test_single_recording():
    with pytest.raises(ValueError):
        held_out(corpus(1, 10), 0.05, 0)


def wav(seconds, sr):
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(seconds * sr)), sr, format="WAV")
    return {"bytes": buffer.getvalue(), "path": None}


def raw(texts, column="transcript"):
    ds = Dataset.from_dict(
        {
            "id": [f"atcosim_sm{k}_01_001_000000_000100" for k in range(len(texts))],
            "audio": [wav(0.5, 8_000) for _ in texts],
            column: texts,
            "speaker": ["s"] * len(texts),
        }
    )
    return ds.cast_column("audio", Audio(decode=False))


def test_text_column():
    assert data.text_column(raw(["a"], "sentence")) == "sentence"
    with pytest.raises(KeyError):
        data.text_column(Dataset.from_dict({"words": ["a"]}))


def test_unified():
    ds = data.unified(raw(["Delta 470", None]), "atcosim")
    assert set(ds.column_names) == set(data.KEEP)
    assert list(ds["text_norm"]) == ["delta four seven zero", ""]
    assert list(ds["seconds"]) == [0.5, 0.5] and list(ds["sample_rate"]) == [8_000, 8_000]


def test_unified_needs_id():
    with pytest.raises(KeyError, match="id"):
        data.unified(raw(["a"]).remove_columns("id"), "atcosim")


def test_parts(monkeypatch):
    splits = {"train": raw(["a"]), "Validation": raw(["b"]), "test": raw(["c"])}
    monkeypatch.setattr(data, "load_dataset", lambda repo, cache_dir: splits)
    train, test = data.parts("repo", "atcosim")
    assert [len(t) for t in train] == [1, 1] and len(test) == 1


def test_stratified():
    fit, held = data.stratified([[corpus(40, 25)], [corpus(20, 5)]], 0.05, 13)
    assert len(fit) + len(held) == 1100 and len(held) > 25


def test_assembled(monkeypatch):
    fake = {"atcosim": ([corpus(40, 25)], [corpus(2, 5)]), "atco2_1h": ([], [corpus(3, 2)])}
    monkeypatch.setattr(data, "parts", lambda repo, source: fake.get(source, ([], [])))
    out = data.assembled(0.05, 13)
    assert set(out) == {"train", "validation", data.INDOMAIN, data.OOD}
    assert len(out[data.INDOMAIN]) == 10 and len(out[data.OOD]) == 6


def test_assembled_needs_train(monkeypatch):
    monkeypatch.setattr(data, "parts", lambda repo, source: ([], [corpus(2, 5)]))
    with pytest.raises(RuntimeError, match="No training data"):
        data.assembled(0.05, 13)


def test_fingerprint():
    key = data.fingerprint(0.05, 13)
    assert key.startswith("val0.05-seed13-") and key == data.fingerprint(0.05, 13)
    assert key != data.fingerprint(0.1, 13)


def test_snapshot(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(data, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(
        data, "assembled", lambda val, seed: calls.append(1) or DatasetDict({"train": corpus(2, 2)})
    )
    (tmp_path / "processed" / "stale").mkdir(parents=True)
    first, second = data.corpora(), data.corpora()
    assert len(calls) == 1 and list(first["train"]["id"]) == list(second["train"]["id"])
    assert [p.name for p in (tmp_path / "processed").iterdir()] == [data.fingerprint()]
