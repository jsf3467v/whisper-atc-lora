"""Shared scoring imported by every evaluation file, keeping all reported numbers comparable."""

import json
import sys
import warnings
import zlib
from pathlib import Path

import jiwer
import numpy as np
from transformers.models.whisper.english_normalizer import EnglishNumberNormalizer

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "Datasets"))

from normalize import DIGITS, normalize

PHONETIC = {normalize(w) for w in (
    "alfa", "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    "hotel", "india", "juliet", "juliett", "kilo", "lima", "mike", "november",
    "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor",
    "whiskey", "xray", "yankee", "zulu",
)}  # fmt: skip
VALUE_WORDS = set(DIGITS.values()) | {"point"}
CODE_WORDS = PHONETIC | VALUE_WORDS
FILLERS = {
    "roger", "wilco", "hello", "hi", "good", "morning", "afternoon", "evening", "day",
    "ok", "okay", "thanks", "thank", "you",
}  # fmt: skip
STATIONS = {"radar", "tower", "approach", "ground", "control", "director", "information"}
NON_OPERATORS = STATIONS | {
    "heading", "climb", "climbing", "descend", "descending", "maintain", "squawk", "qnh",
    "contact", "frequency", "flight", "level", "altitude", "runway", "wind", "speed", "knots",
    "feet", "cleared", "turn", "left", "right", "reduce", "increase", "direct", "report",
    "taxi", "hold", "via", "number", "traffic", "expect", "ils", "fl",
    "to", "on", "at", "and", "for", "the",
}  # fmt: skip
TAIL_STOPS = CODE_WORDS | NON_OPERATORS | FILLERS | {
    "land", "takeoff", "clear", "leaving", "passing", "until", "again", "say", "wait", "copy",
    "yeah", "line", "up", "final", "short", "point", "position", "later", "bye", "goodbye",
    "ahoj", "dobry", "den", "correction", "establish", "established", "available", "ready",
    "visual", "or", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand",
}  # fmt: skip
OPERATOR_MAX = 2
CALLSIGN_MAX = 6
REGISTRATION_MIN = 3
REPETITION_RATIO = 2.4
N_BOOT = 10_000
NUMBER_NORMALIZER = EnglishNumberNormalizer()

METRICS = (
    "wer", "sub", "del", "ins", "ser", "cer", "wer_numeric",
    "callsign_precision", "callsign_recall", "callsign_f1", "callsign_wer",
    "callsign_false_alarm", "coverage", "repetition",
)  # fmt: skip
HIGHER_BETTER = np.array(
    [m in {"callsign_precision", "callsign_recall", "callsign_f1"} for m in METRICS]
)
DESCRIPTIVE = {"coverage"}


def predictions(split, tag):
    path = root / "predictions" / f"{split}-{tag}.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def opening(tokens):
    k = next((k for k, t in enumerate(tokens) if t not in FILLERS), len(tokens))
    if k + 1 < len(tokens) and tokens[k + 1] in STATIONS:
        return opening(tokens[k + 2 :])
    return tokens[k:]


def leading(tokens):
    """Optional operator of up to two words followed by a run of digit or phonetic words."""
    tokens = opening(tokens)[:CALLSIGN_MAX]
    i = 0
    while i < min(len(tokens), OPERATOR_MAX) and tokens[i] not in CODE_WORDS:
        if tokens[i] in NON_OPERATORS:
            return []
        i += 1
    j = i
    while j < len(tokens) and tokens[j] in CODE_WORDS:
        j += 1
    span = tokens[:j]
    return span if j > i and any(t not in VALUE_WORDS for t in span) else []


def trailing(tokens):
    """Callsign closing a readback: a registration, or an operator before a code run."""
    j = len(tokens)
    while j > max(len(tokens) - CALLSIGN_MAX, 0) and tokens[j - 1] in CODE_WORDS:
        j -= 1
    code = tokens[j:]
    letters = [k for k, t in enumerate(code) if t in PHONETIC]
    if len(letters) >= REGISTRATION_MIN:
        return [] if tokens[j - 1 : j] == ["via"] else code[letters[0] :]
    i = j
    while i > max(j - OPERATOR_MAX, 0) and tokens[i - 1] not in TAIL_STOPS:
        i -= 1
    return tokens[i:] if code and i < j and tokens[i - 1 : i] != ["contact"] else []


def extracted(tokens):
    return leading(tokens) or trailing(tokens)


def callsign(text):
    return extracted(normalize(text).split())


def edits(ref, hyp):
    o = jiwer.process_words(ref, hyp)
    return o.substitutions, o.deletions, o.insertions, o.hits + o.substitutions + o.deletions


def repetition(text):
    data = text.encode()
    return len(data) / len(zlib.compress(data)) > REPETITION_RATIO if data else False


def callsign_parts(ref, hyp):
    gold, guess = extracted(ref.split()), extracted(hyp.split())
    exact = bool(gold) and gold == guess
    span = sum(edits(" ".join(gold), " ".join(guess))[:3]) if gold else 0
    num = (exact, exact, 2 * exact, span, bool(guess) and not gold, bool(gold))
    den = (bool(guess), bool(gold), bool(guess) + bool(gold), len(gold), not gold, 1)
    return num, den


def canonical(text):
    return normalize(NUMBER_NORMALIZER(text))


def numeric_parts(ref, hyp):
    ref, hyp = canonical(ref), canonical(hyp)
    if not ref:
        return 0, 0
    sub, dele, ins, words = edits(ref, hyp)
    return sub + dele + ins, words


def utterance(ref, hyp):
    sub, dele, ins, words = edits(ref, hyp)
    errors = sub + dele + ins
    c = jiwer.process_characters(ref, hyp)
    char_errors, char_len = (
        c.substitutions + c.deletions + c.insertions,
        c.hits + c.substitutions + c.deletions,
    )
    cnum, cden = callsign_parts(ref, hyp)
    numeric_errors, numeric_words = numeric_parts(ref, hyp)
    num = (errors, sub, dele, ins, errors > 0, char_errors, numeric_errors, *cnum, repetition(hyp))
    den = (words, words, words, words, 1, char_len, numeric_words, *cden, 1)
    return num, den


def statistics(refs, hyps):
    """Per-utterance metric matrices, dropping utterances whose reference normalizes to empty."""
    texts = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps, strict=True)]
    kept = [k for k, (ref, _) in enumerate(texts) if ref]
    rows = [utterance(*texts[k]) for k in kept]
    num = np.array([r[0] for r in rows], dtype=np.float64).reshape(-1, len(METRICS))
    den = np.array([r[1] for r in rows], dtype=np.float64).reshape(-1, len(METRICS))
    return num, den, kept, [texts[k] for k in kept]


def ratios(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def weights(rng, groups, strata, size):
    """Multinomial counts per utterance from resampling groups within each stratum."""
    w = np.zeros((size, len(groups)))
    for s in np.unique(strata):
        members = strata == s
        local = np.unique(groups[members], return_inverse=True)[1]
        g = int(local.max()) + 1
        w[:, members] = rng.multinomial(g, np.full(g, 1 / g), size=size)[:, local]
    return w


def resampled(pairs, groups=None, strata=None, n_boot=N_BOOT, seed=0, chunk=1_000):
    """Bootstrap corpus metrics for paired (num, den) matrices over shared resamples."""
    n = len(pairs[0][0])
    if not n:
        raise ValueError("No scorable utterances to resample.")
    if any(len(num) != n or len(den) != n for num, den in pairs):
        raise ValueError("Paired matrices must cover the same utterances.")
    groups = np.arange(n) if groups is None else np.asarray(groups)
    strata = np.zeros(n) if strata is None else np.asarray(strata)
    rng = np.random.default_rng(seed)
    out = np.empty((len(pairs), n_boot, len(METRICS)))
    for start in range(0, n_boot, chunk):
        w = weights(rng, groups, strata, min(chunk, n_boot - start))
        for k, (num, den) in enumerate(pairs):
            out[k, start : start + len(w)] = ratios(w @ num, w @ den)
    return out


def bounds(boots):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(boots, [2.5, 97.5], axis=0)


def finite(x):
    return None if np.isnan(x) else float(x)


def corpus(num, den):
    return ratios(num.sum(0), den.sum(0))


def estimate(point, boots):
    lo, hi = bounds(boots)
    used = (~np.isnan(boots)).sum(0)
    return {
        m: {
            "value": finite(point[k]),
            "lo": finite(lo[k]),
            "hi": finite(hi[k]),
            "resamples": int(used[k]),
        }
        for k, m in enumerate(METRICS)
    }


def change(point_base, point_model, boots_base, boots_model):
    """Model minus baseline, its interval, and probability of improvement (Bisani and Ney, 2004)."""
    diff = boots_model - boots_base
    lo, hi = bounds(diff)
    valid = ~np.isnan(diff)
    better = np.where(HIGHER_BETTER, diff > 0, diff < 0) & valid
    with np.errstate(invalid="ignore", divide="ignore"):
        win, tie = better.sum(0) / valid.sum(0), ((diff == 0) & valid).sum(0) / valid.sum(0)
    delta = point_model - point_base
    return {
        m: {
            "delta": finite(delta[k]),
            "lo": finite(lo[k]),
            "hi": finite(hi[k]),
            "p_improve": finite(win[k]),
            "p_tie": finite(tie[k]),
        }
        for k, m in enumerate(METRICS)
        if m not in DESCRIPTIVE
    }
