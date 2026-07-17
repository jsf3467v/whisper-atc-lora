"""OOD error analysis. Reads predictions/<split>-<tag>.jsonl, reports corpus WER
with a 95% CI, the paired WER change over a baseline, and the worst utterances and
callsign misses behind the numbers. Uses the shared scorer, so figures match
evaluate.py. Writes JSON plus a compact console summary.

    python error_analysis.py [split] [tag] [baseline] [worst_k]
    # defaults: test_ood  whisper-small-lora  whisper-small  12
"""

import json
import sys

import numpy as np

from scoring import callsign, count, delta, interval, predictions, root, wer

from normalize import normalize


def rows(recs):
    out = []
    for r in recs:
        edits, reflen = count(r["ref"], r["hyp"])
        if not reflen:
            continue
        out.append({"i": r.get("i"), "ref": normalize(r["ref"]), "hyp": normalize(r["hyp"]),
                    "edits": edits, "reflen": reflen, "wer": edits / reflen,
                    "gold": callsign(r["ref"]), "guess": callsign(r["hyp"])})
    return out


def arrays(rs):
    return (np.array([r["edits"] for r in rs], dtype=np.int64),
            np.array([r["reflen"] for r in rs], dtype=np.int64))


def matched(rows_a, rows_b):
    a = {r["i"]: r for r in rows_a if r["i"] is not None}
    b = {r["i"]: r for r in rows_b if r["i"] is not None}
    keys = sorted(set(a) & set(b))
    ea, na = arrays([a[k] for k in keys])
    eb, nb = arrays([b[k] for k in keys])
    return ea, na, eb, nb, len(keys)


def worst(rs, k):
    top = sorted(rs, key=lambda r: r["wer"], reverse=True)[:k]
    return [{"wer": r["wer"], "ref": r["ref"], "hyp": r["hyp"]} for r in top]


def misses(rs, k):
    bad = [r for r in rs if r["gold"] and r["gold"] != r["guess"]]
    return {"count": len(bad), "covered": sum(1 for r in rs if r["gold"]),
            "examples": [{"gold": r["gold"], "guess": r["guess"]} for r in bad[:k]]}


def analysis(split, tag, baseline, worst_k):
    rs = rows(predictions(split, tag))
    edits, reflen = arrays(rs)
    point, lo, hi = interval(edits, reflen)
    result = {"split": split, "tag": tag, "n": len(rs), "wer": point, "ci": [lo, hi],
              "baseline": None, "delta": None}
    try:
        base = rows(predictions(split, baseline))
    except FileNotFoundError:
        base = None
    if base:
        ea, na, eb, nb, m = matched(base, rs)
        if m:
            dp, dlo, dhi = delta(ea, na, eb, nb)
            result["baseline"] = baseline
            result["delta"] = {"point": dp, "ci": [dlo, dhi], "n": m}
    result["worst"] = worst(rs, worst_k)
    result["callsign_misses"] = misses(rs, worst_k)
    return result


def summary(result):
    print(f"\n{result['split']} | {result['tag']} (n={result['n']}, 10k bootstrap)\n")
    print(f"  wer {result['wer']:.3f}  95% ci [{result['ci'][0]:.3f}, {result['ci'][1]:.3f}]")
    d = result["delta"]
    if d:
        print(f"  vs {result['baseline']}: {d['point']:+.3f}  "
              f"95% ci [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]  (paired n={d['n']}; below 0 = better)")


def report(split="test_ood", tag="whisper-small-lora", baseline="whisper-small", worst_k=12):
    result = analysis(split, tag, baseline, worst_k)
    out = root / "results" / f"error-{split}-{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    summary(result)
    print(f"\nwrote {out.relative_to(root)}")


if __name__ == "__main__":
    a = sys.argv[1:]
    report(*a[:3], *([int(a[3])] if len(a) > 3 else []))