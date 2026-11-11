"""Unit tests for the shared audio reader in Datasets/audio.py."""

import io

import numpy as np
import soundfile as sf
from audio import TARGET_SR, handle, waveform


def clip(tmp_path, sr, channels=1, seconds=0.5):
    data = np.random.default_rng(0).uniform(-0.5, 0.5, (int(sr * seconds), channels))
    path = tmp_path / f"clip_{sr}_{channels}.wav"
    sf.write(path, data, sr)
    return path


def test_handle(tmp_path):
    path = clip(tmp_path, TARGET_SR)
    assert handle({"path": str(path), "bytes": None}) == str(path)
    assert isinstance(handle({"path": None, "bytes": path.read_bytes()}), io.BytesIO)


def test_native_rate(tmp_path):
    audio = waveform({"path": str(clip(tmp_path, TARGET_SR))})
    assert audio.dtype == np.float32 and audio.shape == (TARGET_SR // 2,)


def test_resampled(tmp_path):
    for sr in (8_000, 32_000):
        audio = waveform({"bytes": clip(tmp_path, sr).read_bytes()})
        assert audio.dtype == np.float32 and audio.shape == (TARGET_SR // 2,)


def test_mono(tmp_path):
    path = clip(tmp_path, TARGET_SR, channels=2)
    stereo, _ = sf.read(path, dtype="float32")
    np.testing.assert_allclose(waveform({"path": str(path)}), stereo.mean(axis=1), atol=1e-6)
