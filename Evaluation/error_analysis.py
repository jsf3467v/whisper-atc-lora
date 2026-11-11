"""Worst utterances, callsign misses, false alarms, and repeated output for one split and model."""

import json
import sys

from scoring import METRICS, extracted, predictions, root, statistics

from data import OOD  # isort: skip

WER, REPETITION = METRICS.index("wer"), METRICS.index("repetition")


def rows(recs):
    num, den, kept, texts = statistics([r["ref"] for r in recs], [r["hyp"] for r in recs])
    return [
        {
            "i": recs[pos]["i"],
            "ref": ref,
            "hyp": hyp,
            "wer": num[k, WER] / den[k, WER],
            "repeated": bool(num[k, REPETITION]),
            "gold": extracted(ref.split()),
            "guess": extracted(hyp.split()),
        }
        for k, (pos, (ref, hyp)) in enumerate(zip(kept, texts, strict=True))
    ]


def worst(rs, k):
    ranked = sorted(rs, key=lambda r: r["wer"], reverse=True)[:k]
    return [{key: r[key] for key in ("i", "wer", "ref", "hyp")} for r in ranked]


def misses(rs, k):
    bad = [r for r in rs if r["gold"] and r["gold"] != r["guess"]]
    examples = [{"i": r["i"], "gold": r["gold"], "guess": r["guess"]} for r in bad[:k]]
    return {"count": len(bad), "covered": sum(bool(r["gold"]) for r in rs), "examples": examples}


def alarms(rs, k):
    hits = [r for r in rs if r["guess"] and not r["gold"]]
    examples = [{"i": r["i"], "ref": r["ref"], "guess": r["guess"]} for r in hits[:k]]
    return {"count": len(hits), "uncovered": sum(not r["gold"] for r in rs), "examples": examples}


def repeats(rs, k):
    hits = [r for r in rs if r["repeated"]]
    return {"count": len(hits), "examples": [{"i": r["i"], "hyp": r["hyp"]} for r in hits[:k]]}


def report(tag, split=OOD, worst_k=12):
    rs = rows(predictions(split, tag))
    k = int(worst_k)
    result = {
        "split": split,
        "tag": tag,
        "n": len(rs),
        "worst": worst(rs, k),
        "callsign_misses": misses(rs, k),
        "false_alarms": alarms(rs, k),
        "repetitions": repeats(rs, k),
    }
    out = root / "results" / f"error-{split}-{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    m, a, r = result["callsign_misses"], result["false_alarms"], result["repetitions"]
    print(
        f"{split} | {tag} | n={len(rs)} | callsign misses {m['count']} of {m['covered']} "
        f"| false alarms {a['count']} of {a['uncovered']} | repeated outputs {r['count']}"
    )
    print(f"wrote {out.relative_to(root)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python Evaluation/error_analysis.py <tag> [split] [worst_k]")
    report(*sys.argv[1:4])
