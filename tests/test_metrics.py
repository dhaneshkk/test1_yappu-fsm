import math

from yappu_fsm import data_path
from yappu_fsm.clinical import evaluate, mcc


def binary_mcc(tp, tn, fp, fn):
    return (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


def test_mcc_binary_matches_formula():
    y_true = ["1"] * 6 + ["0"] * 4
    y_pred = ["1"] * 5 + ["0"] + ["0"] * 3 + ["1"]
    assert abs(mcc(y_true, y_pred) - binary_mcc(tp=5, tn=3, fp=1, fn=1)) < 1e-12


def test_mcc_extremes():
    assert mcc(list("abcabc"), list("abcabc")) == 1.0
    assert mcc(list("1100"), list("0011")) == -1.0
    assert mcc(list("aaaa"), list("aaaa")) == 0.0  # undefined -> 0


def test_gold_set_regression(linker):
    r = evaluate(linker, data_path("gold_assertions.jsonl"))
    assert r["assertion_mcc"] >= 0.95
    assert r["laterality_mcc"] >= 0.99
    assert r["codable_mcc"] >= 0.90
