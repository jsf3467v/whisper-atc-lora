"""Making the project's modules importable when running pytest from the repo root.
The normalize.py file lives in Datasets/, scoring.py in Evaluation/; both are normally put
on sys.path by the runner scripts. Tests need the same wiring, nothing more.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("Datasets", "Evaluation"):
    sys.path.insert(0, str(ROOT / sub))
