"""Unit tests for file discovery and transcript output in src/infer.py."""

import json

import pytest

import infer


@pytest.fixture
def folder(tmp_path):
    for name in ("b.wav", "a.FLAC", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    return tmp_path


def test_clips(folder):
    assert [p.name for p in infer.clips(folder)] == ["a.FLAC", "b.wav"]
    assert [p.name for p in infer.clips(folder / "b.wav")] == ["b.wav"]
    assert infer.clips(folder / "notes.txt") == []


def test_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        infer.clips(tmp_path / "absent.wav")


def test_no_audio(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="No"):
        infer.main(tmp_path)


def test_main(folder, monkeypatch):
    texts = {"a.FLAC": "lufthansa 456 climb", "b.wav": "roger"}

    def decoded(model, processor, dev, groups):
        for group in groups:
            yield [texts[item["path"].rsplit("/", 1)[-1]] for item in group]

    monkeypatch.setattr(infer, "whisper", lambda model_id, dev: (None, None))
    monkeypatch.setattr(infer, "decoded", decoded)
    out = folder / "out" / "transcripts.jsonl"
    infer.main(folder, out=out, batch=1)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows == [
        {"file": "a.FLAC", "transcript": texts["a.FLAC"], "callsign": "lufthansa four five six"},
        {"file": "b.wav", "transcript": "roger", "callsign": ""},
    ]
