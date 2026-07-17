"""Shared scoring for ATC transcription: one normalizer, one set of metrics,
imported by every evaluation file so numbers stay comparable.

WER over normalized text (jiwer). Callsign accuracy by token precision/recall
with exact match and coverage. Bootstrap confidence intervals for WER and for a
paired improvement over a baseline.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import jiwer
import numpy as np

root = next((p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
             if (p / "Datasets" / "data.py").exists()), None)
if root is None:
    raise FileNotFoundError("Could not find the Datasets folder; keep this inside the ATC project.")
sys.path.insert(0, str(root / "Datasets"))

from normalize import DIGITS, normalize

# Phonetics are folded through the normalizer so the set always matches its output
# vocabulary (e.g. alfa and alpha collapse to one form).
PHONETIC = {normalize(w) for w in (
    "alfa", "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    "hotel", "india", "juliet", "juliett", "kilo", "lima", "mike", "november",
    "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor",
    "whiskey", "xray", "yankee", "zulu",
)}
VALUE_WORDS = set(DIGITS.values()) | {"point"}
OPERATOR_MAX = 2
CALLSIGN_MAX = 6


def predictions(split, tag):
    path = root / "predictions" / f"{split}-{tag}.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def callsign(text):
    """Leading callsign span. A short operator name then its digit/phonetic run,
    or a leading phonetic run when there is no operator."""
    toks = normalize(text).split()
    i, operator = 0, []
    while i < len(toks) and toks[i] not in VALUE_WORDS and toks[i] not in PHONETIC and len(operator) < OPERATOR_MAX:
        operator.append(toks[i])
        i += 1
    span = operator[:]
    while i < len(toks) and (toks[i] in VALUE_WORDS or toks[i] in PHONETIC) and len(span) < CALLSIGN_MAX:
        span.append(toks[i])
        i += 1
    return span if any(t not in VALUE_WORDS for t in span) else []


def callsign_scores(refs, hyps):
    tp = ref_n = hyp_n = exact = covered = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        gold = callsign(ref)
        if not gold:
            continue
        covered += 1
        guess = callsign(hyp)
        tp += sum((Counter(gold) & Counter(guess)).values())
        ref_n += len(gold)
        hyp_n += len(guess)
        exact += gold == guess
    return {
        "callsign_recall": tp / ref_n if ref_n else 0.0,
        "callsign_precision": tp / hyp_n if hyp_n else 0.0,
        "callsign_exact": exact / covered if covered else 0.0,
        "callsign_coverage": covered / len(refs) if refs else 0.0,
    }


def count(ref, hyp):
    """Edit count (S+D+I) and reference length for one utterance; (0, 0) when the
    reference normalizes to empty, which callers drop as undefined WER."""
    r = normalize(ref)
    if not r:
        return 0, 0
    o = jiwer.process_words(r, normalize(hyp))
    return o.substitutions + o.deletions + o.insertions, o.hits + o.substitutions + o.deletions


def counts(refs, hyps):
    e, n = [], []
    for ref, hyp in zip(refs, hyps, strict=True):
        edits, reflen = count(ref, hyp)
        if reflen:
            e.append(edits)
            n.append(reflen)
    return np.array(e, dtype=np.int64), np.array(n, dtype=np.int64)


def wer(edits, reflen):
    total = int(reflen.sum())
    return int(edits.sum()) / total if total else 0.0


def scores(refs, hyps):
    e, n = counts(refs, hyps)
    return {**callsign_scores(refs, hyps), "wer": wer(e, n)}


def interval(edits, reflen, n_boot=10000, seed=0):
    """Corpus WER and its 95% bootstrap CI over resampled utterances."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, edits.size, size=(n_boot, edits.size))
    boot = edits[idx].sum(axis=1) / reflen[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return wer(edits, reflen), float(lo), float(hi)


def delta(edits_a, reflen_a, edits_b, reflen_b, n_boot=10000, seed=0):
    """WER(model b) minus WER(baseline a), paired, with 95% bootstrap CI.
    A point and interval below zero mean the model improved."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, edits_a.size, size=(n_boot, edits_a.size))
    wa = edits_a[idx].sum(axis=1) / reflen_a[idx].sum(axis=1)
    wb = edits_b[idx].sum(axis=1) / reflen_b[idx].sum(axis=1)
    lo, hi = np.percentile(wb - wa, [2.5, 97.5])
    point = wer(edits_b, reflen_b) - wer(edits_a, reflen_a)
    return point, float(lo), float(hi)