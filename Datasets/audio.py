"""Shared audio decode for the ATC pipeline.

One soundfile + scipy decoder, reused by training, inference, evaluation, and the
EDA so every stage reads waveforms identically. Deliberately free of any model
dependency, which keeps lightweight callers such as the EDA off torch.
"""
from __future__ import annotations

import io
import math

import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16_000


def waveform(item):
    src = io.BytesIO(item["bytes"]) if item.get("bytes") else item["path"]
    audio, sr = sf.read(src, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        g = math.gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sr // g)
    return audio.astype("float32")