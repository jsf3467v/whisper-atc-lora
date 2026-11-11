"""Usage guards of the command-line entry points."""

import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script", ["src/infer.py", "Evaluation/evaluate.py", "Evaluation/error_analysis.py"]
)
def test_usage(script, monkeypatch):
    monkeypatch.setattr(sys, "argv", [script])
    with pytest.raises(SystemExit, match="usage"):
        runpy.run_path(str(ROOT / script), run_name="__main__")
