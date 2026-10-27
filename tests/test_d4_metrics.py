"""
Specification test for decision D4 (docs/AUDIT.md section 5). Fails until src.fl.utils.d4_metrics is implemented.
sklearn is the independent reference implementation.
"""

import pytest
from sklearn.metrics import f1_score

from src.fl.utils import d4_metrics

LABELS = [0, 1, 2, 3]   # class 3 appears in no client's y_true but is predicted once
Y_TRUE = {
    "A": [0, 0, 0, 1, 1, 1, 2],          # qualifying (>=3): {0, 1}; class 2 has one example
    "B": [0, 0, 0, 2, 2, 2, 2],          # qualifying: {0, 2}
    "C": [1, 2],                         # no class reaches 3 -> macro_f1 None
}
Y_PRED = {
    "A": [0, 0, 1, 1, 1, 0, 2],
    "B": [0, 0, 3, 2, 2, 1, 2],
    "C": [1, 2],
}


def _result():
    return d4_metrics(Y_TRUE, Y_PRED, LABELS, min_support=3)


def test_pooled_macro_f1_over_explicit_label_list():
    result = _result()
    pooled_true = sum((Y_TRUE[c] for c in Y_TRUE), [])
    pooled_pred = sum((Y_PRED[c] for c in Y_PRED), [])
    expected = f1_score(pooled_true, pooled_pred, labels=LABELS, average="macro", zero_division=0)
    assert result["pooled_macro_f1"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("client,qualifying", [("A", [0, 1]), ("B", [0, 2])])
def test_per_client_restricted_to_qualifying_classes(client, qualifying):
    r = _result()["per_client"][client]
    assert sorted(r["qualifying_classes"]) == qualifying
    assert r["n_qualifying_classes"] == len(qualifying)
    expected = f1_score(Y_TRUE[client], Y_PRED[client], labels=qualifying, average="macro", zero_division=0)
    assert r["macro_f1"] == pytest.approx(expected, abs=1e-12)


def test_client_without_qualifying_class_is_none_and_excluded_from_mean():
    result = _result()
    assert result["per_client"]["C"]["macro_f1"] is None
    assert result["per_client"]["C"]["n_qualifying_classes"] == 0
    expected_mean = (result["per_client"]["A"]["macro_f1"] + result["per_client"]["B"]["macro_f1"]) / 2
    assert result["client_mean_macro_f1"] == pytest.approx(expected_mean, abs=1e-12)
