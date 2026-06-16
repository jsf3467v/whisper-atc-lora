"""ATC transcript normalization.

The same function is applied to references and hypotheses before scoring, so WER
and callsign metrics stay comparable across every evaluation file.

The corpora references are already lowercased, unpunctuated, and digit-spelled, so
this mostly reshapes model output to match with spoken-digit expansion  of 290 -> two nine
zero, decimals to "point", and ICAO spelling variants to one canonical token.
Known residual are grouped values such as "one hundred" or aircraft types "three
twenty" are left for a later pass once real model errors are observed.
"""
from __future__ import annotations

import re

DIGITS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

# Radio/ICAO spellings mapped to one canonical word - applied to refs and hyps.
WORD_MAP = {
    "niner": "nine", "tree": "three", "fife": "five", "fower": "four",
    "decimal": "point", "alfa": "alpha", "juliett": "juliet", "whisky": "whiskey",
}

BRACKETS = re.compile(r"[\[\(<][^\]\)>]*[\]\)>]")
NON_SPEECH = re.compile(
    r"\b(hnoise|noise|unintelligible|unk|sil|spk|fragment|empty|offtalk|speaker)\b"
)
NUMBER = re.compile(r"\d+(?:\.\d+)?")
PUNCT = re.compile(r"[^\w\s']")
SPACES = re.compile(r"\s+")


def spoken(token):
    return " ".join("point" if c == "." else DIGITS[c] for c in token)


def normalize(text, *, digits=True):
    if not text:
        return ""
    t = text.lower()
    t = BRACKETS.sub(" ", t)
    t = NON_SPEECH.sub(" ", t)
    if digits:
        t = NUMBER.sub(lambda m: f" {spoken(m.group())} ", t)
    t = PUNCT.sub(" ", t)
    t = " ".join(WORD_MAP.get(w, w) for w in t.split())
    return SPACES.sub(" ", t).strip()