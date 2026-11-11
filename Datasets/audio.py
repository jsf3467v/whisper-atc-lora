"""Shared audio decoding, free of model dependencies, used by every stage to read waveforms identically."""

from __future__ import annotations

import io
import math

import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16_000
AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff")


def handle(item):
    return io.BytesIO(item["bytes"]) if item.get("bytes") else item["path"]


def waveform(item):
    audio, sr = sf.read(handle(item), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        g = math.gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sr // g)
    return audio.astype("float32")
