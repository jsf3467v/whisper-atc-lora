"""Unit tests for the Gradio demo callbacks in app.py."""

import torch

import app


def test_no_clip():
    assert app.recognition(None) == ("Upload or record a clip to begin.", "")


def test_recognition(monkeypatch):
    monkeypatch.setattr(app, "asr", lambda: (None, None, torch.device("cpu")))
    monkeypatch.setattr(app, "features", lambda processor, items: items)
    for text, sign in (("csa 1 delta zulu descend", "csa one delta zulu"), ("roger", None)):
        monkeypatch.setattr(app, "transcripts", lambda m, p, feats, d, t=text: [t])
        assert app.recognition("clip.wav") == (text, sign or "(none detected)")


def test_examples():
    found = app.examples()
    assert found and all(len(row) == 1 and row[0].endswith(".wav") for row in found)


def test_interface():
    assert app.demo.title.endswith("+ LoRA)")


def test_asr_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "whisper", lambda model, dev: calls.append(model) or ("m", "p"))
    app.asr.cache_clear()
    assert app.asr() is app.asr() and calls == [app.MODEL]
    app.asr.cache_clear()
