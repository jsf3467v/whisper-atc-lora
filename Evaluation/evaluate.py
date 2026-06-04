"""Evaluation

Reads predictions/<split>-<tag>.jsonl, scores each source through the shared
scoring module, and prints the table with callsign recall and precision first,
then exact-match, coverage, and WER. In-domain is shown by source, overall, and
leak-free (utterances whose transcript also appears in training are dropped, since
those numbers are optimistic). Out-of-distribution (ATCO2) is the honest signal.

    python evaluate.py [tag]      # tag defaults to whisper-small
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
root = next((p for p in [HERE, *HERE.parents] if (p / "Datasets" / "data.py").exists()), None)
if root is None:
    raise FileNotFoundError("Could not find the Datasets folder; keep this inside the ATC project.")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(root / "Datasets"))

from scoring import scores
from normalize import normalize

COLUMNS = ("callsign_recall", "callsign_precision", "callsign_exact", "callsign_coverage", "wer")
HEADS = ("recall", "prec", "exact", "cov", "wer")


def records(split, tag):
    path = root / "predictions" / f"{split}-{tag}.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def seen():
    from data import corpora
    return set(corpora()["train"]["text_norm"])


def row(label, recs):
    if not recs:
        return (label, 0, None)
    return (label, len(recs), scores([r["ref"] for r in recs], [r["hyp"] for r in recs]))


def table(rows):
    print(" ".join([f"{'split':24}", f"{'n':>6}"] + [f"{h:>8}" for h in HEADS]))
    for label, n, s in rows:
        if s is None:
            print(f"{label:24} {n:>6}" + "       -" * len(HEADS))
            continue
        cells = " ".join(f"{s[c]:>8.3f}" for c in COLUMNS)
        print(f"{label:24} {n:>6} {cells}")


def report(tag="whisper-small"):
    indomain = records("test_indomain", tag)
    ood = records("test_ood", tag)
    train = seen()
    rows = [row(f"in-domain {src}", [r for r in indomain if r["source"] == src])
            for src in sorted({r["source"] for r in indomain})]
    rows.append(row("in-domain overall", indomain))
    rows.append(row("in-domain leak-free", [r for r in indomain if normalize(r["ref"]) not in train]))
    rows.append(row("ood atco2_1h", ood))
    print(f"\n{tag}\n")
    table(rows)


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "whisper-small")