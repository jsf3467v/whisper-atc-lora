"""Air traffic control transcript normalization, applied identically to references and hypotheses."""

from __future__ import annotations

import re

DIGITS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}  # fmt: skip

WORD_MAP = {
    "niner": "nine", "tree": "three", "fife": "five", "fower": "four",
    "decimal": "point", "alfa": "alpha", "juliett": "juliet", "whisky": "whiskey",
}  # fmt: skip

BRACKETS = re.compile(r"[\[\(<][^\]\)>]*[\]\)>]")
NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")
PUNCT = re.compile(r"[^\w\s]")
XRAY = re.compile(r"\bx ray\b")


def spoken(token):
    return " ".join("point" if c == "." else DIGITS[c] for c in token)


def normalize(text):
    if not text:
        return ""
    t = BRACKETS.sub(" ", text.lower())
    t = PUNCT.sub(" ", NUMBER.sub(lambda m: f" {spoken(m.group())} ", t))
    t = XRAY.sub("xray", " ".join(t.split()))
    return " ".join(WORD_MAP.get(w, w) for w in t.split())
