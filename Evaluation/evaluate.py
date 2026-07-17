"""Score predictions/<split>-<tag>.jsonl through the shared scoring module and
write a JSON table plus a compact console summary. In-domain is shown by source,
overall, and leak-free (references also present in training are dropped).
Out-of-distribution (ATCO2) is the honest signal.

    python evaluate.py [tag]      # tag defaults to whisper-small
"""

import json
import sys

from scoring import predictions, root, scores

from data import corpora
from normalize import normalize

METRICS = ("recall", "precision", "exact", "coverage", "wer")
SHORT = ("recall", "prec", "exact", "cov", "wer")
KEYS = ("callsign_recall", "callsign_precision", "callsign_exact", "callsign_coverage", "wer")


def train_texts():
    return set(corpora()["train"]["text_norm"])


def row(label, recs):
    if not recs:
        return {"split": label, "n": 0}
    s = scores([r["ref"] for r in recs], [r["hyp"] for r in recs])
    return {"split": label, "n": len(recs), **{m: s[k] for m, k in zip(METRICS, KEYS)}}


def table(rows):
    lines = [f"{'split':22}{'n':>6}" + "".join(f"{s:>9}" for s in SHORT)]
    for r in rows:
        if r["n"] == 0:
            lines.append(f"{r['split']:22}{0:>6}" + "        -" * len(METRICS))
        else:
            lines.append(f"{r['split']:22}{r['n']:>6}" + "".join(f"{r[m]:>9.3f}" for m in METRICS))
    return "\n".join(lines)


def report(tag="whisper-small"):
    indomain = predictions("test_indomain", tag)
    ood = predictions("test_ood", tag)
    train = train_texts()
    rows = [row(f"in-domain {src}", [r for r in indomain if r["source"] == src])
            for src in sorted({r["source"] for r in indomain})]
    rows.append(row("in-domain overall", indomain))
    rows.append(row("in-domain leak-free", [r for r in indomain if normalize(r["ref"]) not in train]))
    rows.append(row("ood atco2_1h", ood))
    out = root / "results" / f"eval-{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tag": tag, "rows": rows}, indent=2))
    print(f"\n{tag}\n")
    print(table(rows))
    print(f"\nwrote {out.relative_to(root)}")


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "whisper-small")