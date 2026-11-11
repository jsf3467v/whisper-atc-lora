"""Score a model on every metric in scoring.METRICS with recording-level bootstrap intervals."""

import json
import sys

import numpy as np
from scoring import METRICS, change, corpus, estimate, predictions, resampled, root, statistics

from data import INDOMAIN, OOD, corpora, recording  # isort: skip

SPLITS = {INDOMAIN: "in-domain", OOD: "ood"}
SHOWN = ("cer", "ser", "wer_numeric", "callsign_f1", "callsign_false_alarm", "repetition")
COMPARED = ("wer", "cer", "wer_numeric", "callsign_f1")
TERCILES = ("short", "medium", "long")
PRIMARY = ("ood overall", "wer")
POOLED = "in-domain overall"
WORDS = METRICS.index("wer")


def records(tag):
    found = {}
    for split in SPLITS:
        try:
            found[split] = {r["i"]: r for r in predictions(split, tag)}
        except FileNotFoundError:
            continue
    return found or None


def aligned(split, preds, meta):
    if preds.keys() != set(range(len(meta["id"]))):
        raise ValueError(f"{split} predictions are incomplete; rerun src/transcribe.py.")
    if any(r["ref"] != meta["text_raw"][i] for i, r in preds.items()):
        raise ValueError(f"{split} predictions do not match the current data.")


def utterances(model, base, data):
    if base and base.keys() != model.keys():
        raise ValueError("Model and baseline predictions cover different splits.")
    rows = []
    for split in sorted(model):
        meta = data[split].select_columns(["id", "seconds", "text_raw"]).to_dict()
        for preds in (model[split], base[split]) if base else (model[split],):
            aligned(split, preds, meta)
        rows += [
            {
                "domain": SPLITS[split],
                "source": model[split][i]["source"],
                "recording": recording(meta["id"][i]),
                "seconds": meta["seconds"][i],
                "ref": model[split][i]["ref"],
                "hyp": model[split][i]["hyp"],
                "base": base[split][i]["hyp"] if base else None,
            }
            for i in range(len(meta["id"]))
        ]
    return rows


def terciles(data):
    edges = {}
    for split in SPLITS.keys() & data.keys():
        cols = data[split].select_columns(["source", "seconds"]).to_dict()
        source, seconds = np.array(cols["source"]), np.array(cols["seconds"], dtype=np.float64)
        edges |= {s: np.quantile(seconds[source == s], [1 / 3, 2 / 3]) for s in set(cols["source"])}
    return edges


def durations(source, seconds, edges):
    out = []
    for s, (a, b) in sorted(edges.items()):
        bins = np.digitize(seconds, (a, b))
        spans = (f"< {a:.1f} s", f"{a:.1f}-{b:.1f} s", f">= {b:.1f} s")
        out += [
            (f"{s} {t} {span}", (source == s) & (bins == k))
            for k, (t, span) in enumerate(zip(TERCILES, spans, strict=True))
        ]
    return out


def slices(rows, texts, seen, edges):
    domain, source, rec = (
        np.array([r[k] for r in rows]) for k in ("domain", "source", "recording")
    )
    seconds = np.array([r["seconds"] for r in rows], dtype=np.float64)
    leak_free = np.array([ref not in seen["texts"] for ref, _ in texts], dtype=bool)
    unseen = ~np.isin(rec, list(seen["recordings"]))
    indomain, ood = domain == "in-domain", domain == "ood"
    sources = [(f"in-domain {s}", indomain & (source == s)) for s in sorted(set(source[indomain]))]
    out = sources + [
        (POOLED, indomain),
        ("in-domain leak-free", indomain & leak_free),
        ("in-domain unseen recordings", indomain & unseen),
    ]
    ood_sources = sorted(set(source[ood]))
    out += [(f"ood {s}", ood & (source == s)) for s in ood_sources] if len(ood_sources) > 1 else []
    return out + [("ood overall", ood)] + durations(source, seconds, edges), {
        label for label, _ in sources
    }


def result(label, n, words, points, boots):
    out = {"split": label, "n": n, "words": words, "metrics": estimate(points[0], boots[0])}
    if len(points) > 1:
        out["baseline"] = estimate(points[1], boots[1])
        out["change"] = change(points[1], points[0], boots[1], boots[0])
    return out


def summary(label, mask, matrices, groups, strata, rng):
    pairs = [(num[mask], den[mask]) for num, den in matrices]
    boots = resampled(pairs, groups[mask], strata[mask], seed=rng)
    points = [corpus(*p) for p in pairs]
    words = int(pairs[0][1][:, WORDS].sum())
    return result(label, int(mask.sum()), words, points, boots), (points, boots)


def macro(label, parts):
    points = [np.mean([p[k] for p, _ in parts], axis=0) for k in range(len(parts[0][0]))]
    return result(label, None, None, points, np.mean([b for _, b in parts], axis=0))


def evaluated(masks, per_source, matrices, groups, strata):
    results, parts, rng = [], [], np.random.default_rng(0)
    for label, mask in masks:
        if mask.any():
            out, part = summary(label, mask, matrices, groups, strata, rng)
            results.append(out)
            parts += [part] if label in per_source else []
    if len(parts) > 1:
        after = next(k for k, r in enumerate(results) if r["split"] == POOLED) + 1
        results.insert(after, macro("in-domain macro", parts))
    return results


def cell(x, width=8, signed=False):
    return f"{'-':>{width}}" if x is None else f"{x:>+{width}.3f}" if signed else f"{x:>{width}.3f}"


def table(results):
    names = [m.replace("_", " ") for m in SHOWN]
    widths = [max(9, len(v) + 2) for v in names]
    header = "".join(f"{v:>{w}}" for v, w in zip(names, widths, strict=True))
    lines = [f"{'split':34}{'n':>6}{'wer 95% ci':>22}{header}"]
    for r in results:
        w = r["metrics"]["wer"]
        ci = f"{cell(w['value'], 6)} [{cell(w['lo'], 5)},{cell(w['hi'], 6)}]"
        n = "" if r["n"] is None else r["n"]
        values = "".join(
            cell(r["metrics"][m]["value"], w) for m, w in zip(SHOWN, widths, strict=True)
        )
        lines.append(f"{r['split']:34}{n:>6}{ci:>22}{values}")
    return "\n".join(lines)


def comparison(results):
    lines = [f"{'split':34}" + "".join(f"{m.replace('_', ' '):>40}" for m in COMPARED)]
    for r in results:
        parts = [
            f"{cell(c['delta'], 7, True)} [{cell(c['lo'], 6, True)},{cell(c['hi'], 6, True)}]"
            f" P(improve){cell(c['p_improve'], 6)}"
            for c in (r["change"][m] for m in COMPARED)
        ]
        lines.append(f"{r['split']:34}" + "".join(f"{p:>40}" for p in parts))
    return "\n".join(lines)


def scored(rows, key):
    return statistics([r["ref"] for r in rows], [r[key] for r in rows])


def seen(data):
    train = data["train"].select_columns(["id", "text_norm"]).to_dict()
    return {"texts": set(train["text_norm"]), "recordings": {recording(u) for u in train["id"]}}


def report(tag, baseline=None):
    model = records(tag)
    if model is None:
        raise FileNotFoundError(f"No predictions for {tag}; run src/transcribe.py first.")
    base = records(baseline) if baseline and baseline != tag else None
    if baseline and baseline != tag and base is None:
        print(f"no predictions for baseline {baseline}; reporting {tag} without a baseline")
    data = corpora()
    rows = utterances(model, base, data)
    num, den, kept, texts = scored(rows, "hyp")
    matrices = [(num, den)] + ([scored(rows, "base")[:2]] if base else [])
    rows = [rows[k] for k in kept]
    groups, strata = (np.array([r[k] for r in rows]) for k in ("recording", "source"))
    masks, per_source = slices(rows, texts, seen(data), terciles(data))
    write(tag, baseline if base else None, evaluated(masks, per_source, matrices, groups, strata))


def write(tag, baseline, results):
    primary = next((r for r in results if r["split"] == PRIMARY[0]), None)
    key = "change" if baseline else "metrics"
    payload = {
        "tag": tag,
        "baseline": baseline,
        "resampling": "recordings within each corpus",
        "primary": {
            "split": PRIMARY[0],
            "metric": PRIMARY[1],
            key: primary[key][PRIMARY[1]] if primary else None,
        },
        "slices": results,
    }
    out = root / "results" / f"eval-{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n{tag}\n\n{table(results)}")
    if baseline:
        print(f"\nchange vs {baseline} (95% ci, recordings resampled within each corpus)")
        print(f"\n{comparison(results)}")
    print(f"\nwrote {out.relative_to(root)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python Evaluation/evaluate.py <tag> [baseline]")
    report(*sys.argv[1:3])
