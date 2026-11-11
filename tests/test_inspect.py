"""Unit tests for the split report in Datasets/inspect_data.py."""

import inspect_data
from datasets import Dataset, DatasetDict


def split(n, source):
    return Dataset.from_dict(
        {
            "source": [source] * n,
            "text_raw": [f"Delta {k}" for k in range(n)],
            "text_norm": [f"delta {k}" for k in range(n)],
            "seconds": [1800.0] * n,
        }
    )


def test_report(monkeypatch, capsys):
    data = DatasetDict({"train": split(4, "atcosim"), "test_ood": split(2, "atco2_1h")})
    monkeypatch.setattr(inspect_data, "corpora", lambda: data)
    inspect_data.main()
    out = capsys.readouterr().out
    assert "2.00 h  atcosim=4" in out and "1.00 h  atco2_1h=2" in out
    assert out.count("[train/atcosim]") == 3 and out.count("[test_ood/atco2_1h]") == 2
