"""Unit tests for the collator, run signature, precision, and memory fallback in src/train.py."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

import train

START, END = 50, 99
CPU = torch.device("cpu")


class Tokenizer:
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        ids = [[START, *range(len(t.split())), END][:max_length] for t in texts]
        width = max(map(len, ids))
        return SimpleNamespace(
            input_ids=torch.tensor([r + [END] * (width - len(r)) for r in ids]),
            attention_mask=torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in ids]),
        )


class Trainer:
    calls = []

    def __init__(self, model, args, **kwargs):
        self.args = args

    def train(self, resume_from_checkpoint):
        args = self.args
        Trainer.calls.append((args.per_device_train_batch_size, args.gradient_accumulation_steps))
        if args.per_device_train_batch_size > 4:
            raise RuntimeError("CUDA out of memory")


@pytest.fixture
def processor(monkeypatch):
    monkeypatch.setattr(train, "waveform", lambda item: np.zeros(8_000, dtype=np.float32))
    return SimpleNamespace(feature_extractor=WhisperFeatureExtractor(), tokenizer=Tokenizer())


@pytest.fixture
def trainer(monkeypatch):
    model = SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=START))
    model.config.max_target_positions = 448
    monkeypatch.setattr(train, "base_model", lambda name, dev: model)
    monkeypatch.setattr(train, "Seq2SeqTrainer", Trainer)
    Trainer.calls = []
    return Trainer


def test_collator(processor):
    rows = [{"audio": {}, "text_norm": "a b"}, {"audio": {}, "text_norm": "a"}]
    batch = train.Collator(processor, START, 448)(rows)
    assert batch["input_features"].shape == (2, 80, 3000)
    assert batch["labels"].tolist() == [[0, 1, END], [0, END, -100]]


def test_collator_truncates(processor):
    batch = train.Collator(processor, START, 3)([{"audio": {}, "text_norm": "a b c d"}])
    assert batch["labels"].tolist() == [[0, 1]]


def test_collator_keeps_start(processor):
    batch = train.Collator(processor, -1, 448)([{"audio": {}, "text_norm": "a"}])
    assert batch["labels"].tolist() == [[START, 0, END]]


def test_signature():
    base = train.signature("m", 16, 1, 3, 1e-3)
    assert base == train.signature("m", 8, 2, 3, 1e-3)
    assert base != train.signature("m", 16, 1, 3, 5e-4)
    assert base != train.signature("m", 16, 1, 2, 1e-3)


def test_precision():
    assert train.precision(CPU) == {"bf16": False, "fp16": False}


def test_settings(tmp_path):
    args = train.settings(tmp_path, CPU, 8, 2, 3, 1e-3, 10)
    assert args.eval_steps == args.save_steps == 10 and args.load_best_model_at_end
    assert args.metric_for_best_model == "eval_loss" and not args.greater_is_better


def test_memory_fallback(tmp_path, trainer):
    processor = SimpleNamespace(feature_extractor=None, tokenizer=None)
    train.fit("m", processor, list(range(64)), [], tmp_path, CPU, 16, 1, 1, 1e-3)
    assert trainer.calls == [(16, 1), (8, 2), (4, 4)]


def test_odd_batch_raises(tmp_path, trainer):
    processor = SimpleNamespace(feature_extractor=None, tokenizer=None)
    with pytest.raises(RuntimeError, match="out of memory"):
        train.fit("m", processor, list(range(64)), [], tmp_path, CPU, 5, 1, 1, 1e-3)
    assert trainer.calls == [(5, 1)]
