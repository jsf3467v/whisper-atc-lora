"""Put Datasets/, Evaluation/, src/, and the project root on the import path for pytest."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("", "Datasets", "Evaluation", "src"):
    sys.path.insert(0, str(ROOT / sub))
