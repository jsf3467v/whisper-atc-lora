"""Scoring for ATC transcription. Shared by every evaluation file so numbers stay
comparable (one normalizer, one set of metrics).

Two metrics:
  - WER over normalized text (jiwer).
  - Callsign accuracy, led by token precision/recall with exact-match alongside.

Callsign extraction reads ATC structure rather than a stopword lexicon. A 1-2 word
operator name followed by its digit/phonetic run (airline form, e.g. "delta four
seven zero"), or a leading phonetic-letter run when there is no operator
(registration form, e.g. "hotel golf echo"). The same extractor runs on reference
and hypothesis, so the metric measures whether the model reproduced the callsign
span. Utterances with no extractable callsign are excluded and reported as coverage;
content-first readbacks where the callsign sits mid-sentence are the known gap.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import jiwer

HERE = Path(__file__).resolve().parent
root = next((p for p in [HERE, *HERE.parents] if (p / "Datasets" / "data.py").exists()), None)
if root is None:
    raise FileNotFoundError("Could not find the Datasets folder; keep this inside the ATC project.")
sys.path.insert(0, str(root / "Datasets"))

from normalize import DIGITS, normalize

PHONETIC = {
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu",
}
VALUE_WORDS = set(DIGITS.values()) | {"point"}
OPERATOR_MAX = 2
CALLSIGN_MAX = 6


def callsign(text):
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


def wer(refs, hyps):
    return jiwer.wer([normalize(r) for r in refs], [normalize(h) for h in hyps])


def scores(refs, hyps):
    return {**callsign_scores(refs, hyps), "wer": wer(refs, hyps)}