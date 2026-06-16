"""LoRA fine-tune Whisper on the ATC train split.

Device-agnostic by construction where the model detects MPS, CUDA, or CPU and picks precision to
match (fp32 on MPS, bf16 on CUDA), so the same file runs locally or on a rented GPU
with no edits. Reuses the verified audio decode from transcribe.py.

Output: a merged model at <project>/models/<tag>/, which transcribe.py runs as-is
(model_id=that path, tag="whisper-small-lora") to produce the "after" hypotheses.
"""
from __future__ import annotations

import torch
from data import corpora
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from datasets import Audio
from transcribe import TARGET_SR, device, root, waveform  # sets up the Datasets path


class Collator:
    def __init__(self, processor):
        self.extractor = processor.feature_extractor
        self.tokenizer = processor.tokenizer

    def __call__(self, rows):
        feats = self.extractor([waveform(r["audio"]) for r in rows],
                               sampling_rate=TARGET_SR, return_tensors="pt").input_features
        batch = self.tokenizer([r["text_raw"] for r in rows], padding=True, return_tensors="pt")
        labels = batch.input_ids.masked_fill(batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}


class CacheClear(TrainerCallback):
    def __init__(self, dev):
        self.dev = dev

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.dev.type == "mps":
            torch.mps.empty_cache()
        elif self.dev.type == "cuda":
            torch.cuda.empty_cache()


def adapter(model):
    config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                        lora_dropout=0.05, bias="none")
    return get_peft_model(model, config)


def base_model(name, dev):
    processor = WhisperProcessor.from_pretrained(name, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    return processor, adapter(model).to(dev)


def precision(dev):
    return {"bf16": dev.type == "cuda", "fp16": False}


def settings(out_dir, dev, batch, epochs, lr, accum):
    return Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        warmup_steps=50,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        remove_unused_columns=False,
        label_names=["labels"],
        logging_steps=50,
        dataloader_num_workers=2,
        report_to="none",
        seed=0,
        **precision(dev),
    )


def resumable(out_dir):
    return out_dir.exists() and any(out_dir.glob("checkpoint-*"))


def main(name="openai/whisper-small", tag="whisper-small-lora", batch=16,
         epochs=3, lr=1e-3, accum=1):
    torch.manual_seed(0)
    dev = device()
    processor, model = base_model(name, dev)
    data = corpora()
    train_ds = data["train"].cast_column("audio", Audio(decode=False))
    val_ds = data["validation"].cast_column("audio", Audio(decode=False))
    out_dir = root / "checkpoints" / tag
    trainer = Seq2SeqTrainer(
        model=model,
        args=settings(out_dir, dev, batch, epochs, lr, accum),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(processor),
        callbacks=[CacheClear(dev)],
    )
    print(f"device {dev.type} | {name} -> LoRA r=16 | train {len(train_ds)} val {len(val_ds)}")
    trainer.train(resume_from_checkpoint=resumable(out_dir))

    final = root / "models" / tag
    final.mkdir(parents=True, exist_ok=True)
    trainer.model.merge_and_unload().save_pretrained(str(final))
    processor.save_pretrained(str(final))
    print(f"saved fine-tuned model to {final}")


if __name__ == "__main__":
    main()