"""Unit tests for model loading, decoding, and resumable transcription in src/transcribe.py."""

import json
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers import WhisperConfig, WhisperForConditionalGeneration

import transcribe

CPU = torch.device("cpu")


class Tokenizer:
    def __call__(self, texts, add_special_tokens=False):
        return SimpleNamespace(input_ids=[t.split() for t in texts])


class Processor:
    tokenizer = Tokenizer()
    feature_extractor = SimpleNamespace(chunk_length=30)

    def get_decoder_prompt_ids(self, language, task):
        return [(1, 0), (2, 0), (3, 0)]

    def batch_decode(self, ids, **kwargs):
        return [f" {int(row[0])} " for row in ids]


class Model:
    def __init__(self, positions=448):
        self.config = SimpleNamespace(max_target_positions=positions)
        self.generation_config = SimpleNamespace(max_length=448, max_new_tokens=None)

    def generate(self, feats, language, task):
        if len(feats) > 1:
            raise RuntimeError("MPS backend out of memory")
        return feats


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(transcribe, "root", tmp_path)
    return tmp_path


def test_adapted():
    assert transcribe.adapted("openai/whisper-small") == "whisper-small-lora"


def test_device():
    assert transcribe.device().type in {"mps", "cuda", "cpu"}
    transcribe.release(CPU)


def test_within():
    split = Dataset.from_dict({"seconds": [1.0, 30.0, 30.5]})
    assert list(transcribe.within(split, 30)["seconds"]) == [1.0, 30.0]


def test_cap():
    model = Model()
    assert transcribe.cap(model, Processor(), ["a b c", "a"]) == 4
    assert (
        model.generation_config.max_new_tokens == 4 and model.generation_config.max_length is None
    )


def test_cap_ceiling():
    assert transcribe.cap(Model(positions=8), Processor(), ["a " * 20]) == 4


def test_missing_model(project):
    with pytest.raises(FileNotFoundError, match="train.py"):
        transcribe.whisper("models/absent", CPU)


def test_built(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert transcribe.built(SimpleNamespace(name_or_path=str(tmp_path))) > 0
    assert transcribe.built(SimpleNamespace(name_or_path="openai/whisper-small")) == 0.0


def test_oom_split():
    feats = torch.arange(4.0).reshape(4, 1)
    assert transcribe.transcripts(Model(), Processor(), feats, CPU) == ["0", "1", "2", "3"]


def test_errors_raised():
    class Broken(Model):
        def __init__(self, message):
            super().__init__()
            self.message = message

        def generate(self, feats, language, task):
            raise RuntimeError(self.message)

    with pytest.raises(RuntimeError, match="shape"):
        transcribe.transcripts(Broken("shape mismatch"), Processor(), torch.zeros(2, 1), CPU)
    with pytest.raises(RuntimeError, match="out of memory"):
        transcribe.transcripts(Broken("out of memory"), Processor(), torch.zeros(1, 1), CPU)


def test_decoded(monkeypatch):
    monkeypatch.setattr(transcribe, "features", lambda processor, group: group)
    monkeypatch.setattr(transcribe, "transcripts", lambda m, p, feats, d: [x * 2 for x in feats])
    groups = ([1, 2], [3], [])
    assert list(transcribe.decoded(None, None, CPU, groups)) == [[2, 4], [6], []]


def test_completed(project):
    path = project / "preds.jsonl"
    assert transcribe.completed(path, ["a"], 0.0) == set()
    lines = [json.dumps({"i": 1, "ref": "b"}), json.dumps({"i": 0, "ref": "a"}), '{"i": 2']
    path.write_text("\n".join(lines))
    assert transcribe.completed(path, ["a", "b", "c"], 0.0) == {0, 1}
    assert len(path.read_text().splitlines()) == 2


def test_completed_guards(project):
    path = project / "preds.jsonl"
    path.write_text(json.dumps({"i": 0, "ref": "a"}) + "\n")
    with pytest.raises(ValueError, match="does not match"):
        transcribe.completed(path, ["changed"], 0.0)
    with pytest.raises(ValueError, match="does not match"):
        transcribe.completed(path, [], 0.0)
    with pytest.raises(ValueError, match="predates"):
        transcribe.completed(path, ["a"], path.stat().st_mtime + 1)


def test_hypotheses(project, monkeypatch):
    def decoded(model, processor, dev, groups):
        for group in groups:
            yield [item["path"] for item in group]

    monkeypatch.setattr(transcribe, "decoded", decoded)
    split = Dataset.from_dict(
        {
            "text_raw": ["a", "b", "c"],
            "source": ["x", "x", "y"],
            "seconds": [1.0, 3.0, 2.0],
            "audio": [{"path": p} for p in ("p0", "p1", "p2")],
        }
    )
    model = SimpleNamespace(name_or_path=str(project / "none"))
    assert transcribe.hypotheses(split, "test", model, None, CPU, "tag", 2) == 3
    out = project / "predictions" / "test-tag.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["i"] for r in rows] == [1, 2, 0]
    assert all(r["hyp"] == f"p{r['i']}" and r["ref"] == "abc"[r["i"]] for r in rows)
    assert transcribe.hypotheses(split, "test", model, None, CPU, "tag", 2) == 0


def tiny():
    config = WhisperConfig(
        vocab_size=51865, d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
        decoder_ffn_dim=32, max_source_positions=1500, max_target_positions=32,
    )  # fmt: skip
    return WhisperForConditionalGeneration(config)


def test_whisper(project, monkeypatch):
    sources = []
    monkeypatch.setattr(
        transcribe.WhisperProcessor, "from_pretrained", lambda src: sources.append(src) or "p"
    )
    monkeypatch.setattr(
        transcribe.WhisperForConditionalGeneration, "from_pretrained", lambda src: tiny()
    )
    (project / "models" / "m").mkdir(parents=True)
    model, processor = transcribe.whisper("models/m", CPU)
    assert processor == "p" and not model.training and model.config.use_cache
    assert model.proj_out.weight is model.get_input_embeddings().weight
    transcribe.whisper("openai/whisper-small", CPU)
    assert sources == [str(project / "models" / "m"), "openai/whisper-small"]


def test_main(project, monkeypatch, capsys):
    calls = []
    split = Dataset.from_dict({"seconds": [1.0, 31.0], "text_norm": ["a b", "a"]})
    data = DatasetDict({"train": split, "test_indomain": split, "test_ood": split})
    monkeypatch.setattr(transcribe, "whisper", lambda model_id, dev: (Model(), Processor()))
    monkeypatch.setattr(transcribe, "corpora", lambda: data)
    monkeypatch.setattr(
        transcribe,
        "hypotheses",
        lambda s, name, m, p, d, tag, batch: calls.append((name, tag)) or 2,
    )
    transcribe.main("models/whisper-small-lora")
    assert calls == [("test_indomain", "whisper-small-lora"), ("test_ood", "whisper-small-lora")]
    out = capsys.readouterr().out
    assert "max new tokens 3" in out and out.count("2 new of 2 utterances | over 30 s 1") == 2
