"""End-to-end tests of Evaluation/evaluate.py and Evaluation/error_analysis.py on synthetic data."""

import json

import data
import error_analysis
import evaluate
import pytest
import scoring
from datasets import Dataset

INDOMAIN = [
    ("lufthansa 456 climb flight level 340", "atcosim", 2.0, "atcosim_sm1_01_001_000000_000200"),
    ("csa 12 descend 4000 feet", "uwb_atcc", 3.5, "uwb-atcc_TWR-AAA_000000_000350_AT"),
    ("roger", "uwb_atcc", 1.0, "uwb-atcc_TWR-AAA_000400_000500_PI"),
    ("air berlin 298 contact praha 127.1", "uwb_atcc", 6.0, "uwb-atcc_TWR-BBB_000000_000600_AT"),
    ("swiss 7 turn left heading 270", "atcosim", 4.5, "atcosim_zf1_05_146_000000_000450"),
    ("[noise]", "atcosim", 0.5, "atcosim_zf1_05_147_000000_000050"),
]
OOD = [
    (
        "hotel golf echo cleared to land",
        "atco2_1h",
        3.0,
        "atco2_test-set-1h_LKPR_X_20201028-A__000000-000300",
    ),
    (
        "oscar kilo alfa taxi to holding point",
        "atco2_1h",
        5.0,
        "atco2_test-set-1h_LKPR_X_20201028-G__000300-000800",
    ),
    ("wilco", "atco2_1h", 0.8, "atco2_test-set-1h_LKTB_Y_20201026-A__000000-000080"),
]
TRAIN = [
    ("roger", "uwb-atcc_TWR-AAA_000600_000700_PI"),
    ("csa 12 descend 4000 feet", "atcosim_sm2_09_001_000000_000100"),
]
BASELINE = {"lufthansa 456": "lufthansa 4 5 7", "hotel golf": "hotel go", "csa 12": "csa 1 2 3"}


def write(tmp_path, tag, split, rows, edit):
    folder = tmp_path / "predictions"
    folder.mkdir(exist_ok=True)
    lines = [
        json.dumps({"i": i, "source": s, "ref": ref, "hyp": edit(ref)})
        for i, (ref, s, _, _) in enumerate(rows)
    ]
    (folder / f"{split}-{tag}.jsonl").write_text("\n".join(lines) + "\n")


def degrade(ref):
    return next((ref.replace(k, v) for k, v in BASELINE.items() if ref.startswith(k)), ref + " uh")


def split(rows):
    columns = zip(*rows, strict=True)
    return Dataset.from_dict(
        dict(zip(("text_raw", "source", "seconds", "id"), columns, strict=True))
    )


@pytest.fixture
def project(tmp_path, monkeypatch):
    for module in (scoring, evaluate, error_analysis):
        monkeypatch.setattr(module, "root", tmp_path)
    for tag, edit in (("model", lambda r: r), ("base", degrade)):
        write(tmp_path, tag, "test_indomain", INDOMAIN, edit)
        write(tmp_path, tag, "test_ood", OOD, edit)
    data = {
        "train": Dataset.from_dict(
            {"id": [u for _, u in TRAIN], "text_norm": [scoring.normalize(t) for t, _ in TRAIN]}
        ),
        "test_indomain": split(INDOMAIN),
        "test_ood": split(OOD),
    }
    monkeypatch.setattr(evaluate, "corpora", lambda: data)
    return tmp_path


def slices(project, tag="model", baseline="base"):
    evaluate.report(tag, baseline)
    result = json.loads((project / "results" / f"eval-{tag}.json").read_text())
    return result, {s["split"]: s for s in result["slices"]}


@pytest.mark.parametrize(
    "uid,key",
    [
        ("atcosim_sm1_01_001_000000_000329", "atcosim_sm1_01"),
        ("uwb-atcc_TWR-34720N_001179_001375_PI", "uwb-atcc_TWR-34720N"),
        (
            "atco2_test-set-1h_YSSY_SYDNEY_Tower_120_5MHz_20210503_223154-B__000444-000721",
            "atco2_test-set-1h_YSSY_SYDNEY_Tower_120_5MHz_20210503_223154",
        ),
    ],
)
def test_recordings(uid, key):
    assert data.recording(uid) == key


def test_slices(project):
    result, s = slices(project)
    assert s["in-domain overall"]["n"] == 5 and s["in-domain overall"]["words"] > 0
    assert s["in-domain leak-free"]["n"] == 3
    assert s["in-domain unseen recordings"]["n"] == 3
    assert s["ood overall"]["metrics"]["wer"]["value"] == 0.0
    assert s["in-domain overall"]["change"]["wer"]["delta"] < 0
    assert s["in-domain macro"]["n"] is None
    assert result["primary"]["change"] == s["ood overall"]["change"]["wer"]
    durations = [k for k in s if k.split()[0] in ("atcosim", "uwb_atcc")]
    assert sum(s[k]["n"] for k in durations) == 5


def test_macro(project):
    result, s = slices(project, tag="base", baseline=None)
    sources = [s["in-domain atcosim"]["metrics"], s["in-domain uwb_atcc"]["metrics"]]
    for m in ("wer", "cer"):
        mean = sum(x[m]["value"] for x in sources) / 2
        assert mean > 0 and s["in-domain macro"]["metrics"][m]["value"] == pytest.approx(mean)
    order = [x["split"] for x in result["slices"]]
    assert order.index("in-domain macro") == order.index("in-domain overall") + 1


def test_stale(project):
    path = project / "predictions" / "test_ood-model.jsonl"
    path.write_text(path.read_text().replace("wilco", "roger"))
    with pytest.raises(ValueError):
        evaluate.report("model", "base")


def test_no_baseline(project):
    result, s = slices(project, baseline=None)
    assert result["baseline"] is None and "metrics" in result["primary"]
    assert all("change" not in x for x in s.values())


def test_missing(project):
    with pytest.raises(FileNotFoundError):
        evaluate.report("absent", "base")


def test_error_analysis(project):
    error_analysis.report("base", "test_indomain", 2)
    result = json.loads((project / "results" / "error-test_indomain-base.json").read_text())
    assert result["n"] == 5
    assert len(result["worst"]) == 2 and result["worst"][0]["wer"] >= result["worst"][1]["wer"]
    assert result["callsign_misses"]["count"] == 2
    assert result["false_alarms"]["count"] == 0 and result["false_alarms"]["uncovered"] == 1
    assert result["repetitions"]["count"] == 0


def test_incomplete(project):
    path = project / "predictions" / "test_ood-model.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))
    with pytest.raises(ValueError, match="incomplete"):
        evaluate.report("model", "base")


def test_split_mismatch(project):
    (project / "predictions" / "test_ood-base.jsonl").unlink()
    with pytest.raises(ValueError, match="different splits"):
        evaluate.report("model", "base")
