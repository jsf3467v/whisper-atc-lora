"""LoRA fine-tune Whisper on the ATC train split and save the merged model to models/<tag>/."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from transcribe import BASE, LANGUAGE, TASK, adapted, cap, device, release, root, within

from audio import TARGET_SR, waveform  # isort: skip
from data import corpora, fingerprint  # isort: skip

LORA = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
)
SAVES_PER_EPOCH = 3
WARMUP_STEPS = 50
LOG_STEPS = 50
SEED = 0
WORKERS = max(1, (os.cpu_count() or 2) // 2)


class Collator:
    def __init__(self, processor, start_id, max_tokens):
        self.extractor = processor.feature_extractor
        self.tokenizer = processor.tokenizer
        self.start_id = start_id
        self.max_tokens = max_tokens

    def __call__(self, rows):
        feats = self.extractor(
            [waveform(r["audio"]) for r in rows], sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features
        batch = self.tokenizer(
            [r["text_norm"] for r in rows],
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        labels = batch.input_ids.masked_fill(batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.start_id).all():
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}


class CacheClear(TrainerCallback):
    def __init__(self, dev):
        self.dev = dev

    def on_evaluate(self, args, state, control, **kwargs):
        release(self.dev)


def base_model(name, dev):
    torch.manual_seed(SEED)
    model = WhisperForConditionalGeneration.from_pretrained(name)
    model.config.use_cache = False
    return get_peft_model(model, LORA).to(dev)


def precision(dev):
    bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    return {"bf16": bf16, "fp16": dev.type == "cuda" and not bf16}


def signature(name, batch, accum, epochs, lr):
    spec = {
        "model": name,
        "effective_batch": batch * accum,
        "epochs": epochs,
        "lr": lr,
        "warmup": WARMUP_STEPS,
        "seed": SEED,
        "lora": [LORA.r, LORA.lora_alpha, sorted(LORA.target_modules), LORA.lora_dropout],
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]


def settings(out_dir, dev, batch, accum, epochs, lr, every):
    return Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=epochs,
        eval_strategy="steps",
        eval_steps=every,
        save_strategy="steps",
        save_steps=every,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        logging_steps=LOG_STEPS,
        dataloader_num_workers=WORKERS,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=dev.type == "cuda",
        report_to="none",
        seed=SEED,
        **precision(dev),
    )


def fit(name, processor, train_ds, val_ds, out_dir, dev, batch, accum, epochs, lr):
    every = max(1, math.ceil(len(train_ds) / (batch * accum) / SAVES_PER_EPOCH))
    while True:
        model = base_model(name, dev)
        collator = Collator(
            processor, model.config.decoder_start_token_id, model.config.max_target_positions
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=settings(out_dir, dev, batch, accum, epochs, lr, every),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            callbacks=[CacheClear(dev)],
        )
        try:
            trainer.train(resume_from_checkpoint=any(out_dir.glob("checkpoint-*")))
            return trainer
        except RuntimeError as err:
            if batch % 2 or "out of memory" not in str(err).lower():
                raise
        del trainer, model
        gc.collect()
        release(dev)
        batch, accum = batch // 2, accum * 2
        print(f"out of memory; retrying with batch {batch} and accumulation {accum}")


def main(name=BASE, tag=None, batch=16, epochs=3, lr=1e-3, accum=1):
    tag = tag or adapted(name)
    dev = device()
    processor = WhisperProcessor.from_pretrained(name, language=LANGUAGE, task=TASK)
    data = corpora()
    limit = processor.feature_extractor.chunk_length
    train_ds, val_ds = within(data["train"], limit), within(data["validation"], limit)
    dropped = len(data["train"]) + len(data["validation"]) - len(train_ds) - len(val_ds)
    run = f"{fingerprint()}-{signature(name, batch, accum, epochs, lr)}"
    print(
        f"device {dev.type} | {name} -> LoRA r={LORA.r} | train {len(train_ds)} val {len(val_ds)} "
        f"| over {limit} s dropped {dropped} | workers {WORKERS} | run {run}"
    )
    out_dir = root / "checkpoints" / tag / run
    trainer = fit(name, processor, train_ds, val_ds, out_dir, dev, batch, accum, epochs, lr)
    final = root / "models" / tag
    merged = trainer.model.merge_and_unload()
    cap(merged, processor, data["train"]["text_norm"])
    merged.save_pretrained(str(final))
    processor.save_pretrained(str(final))
    print(f"saved fine-tuned model to {final.relative_to(root)}")


if __name__ == "__main__":
    main()
