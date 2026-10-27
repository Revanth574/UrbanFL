import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from src.fl.models import MLPHead, class_weights
from src.fl.utils import accuracy, macro_f1, per_class_f1, per_class_report

Y_TRUE = [0, 0, 1, 1, 1, 2, 2, 0, 1, 2, 2, 2]
Y_PRED = [0, 1, 1, 1, 0, 2, 1, 0, 1, 2, 0, 3]   # predicts a class (3) absent from y_true
LABELS = [0, 1, 2, 3, 4]                         # 4 never appears at all


def test_macro_f1_matches_sklearn_with_explicit_labels():
    expected = f1_score(Y_TRUE, Y_PRED, labels=LABELS, average="macro", zero_division=0)
    assert macro_f1(Y_TRUE, Y_PRED, LABELS) == pytest.approx(expected, abs=1e-12)


def test_per_class_f1_and_report_match_sklearn():
    p, r, f, s = precision_recall_fscore_support(Y_TRUE, Y_PRED, labels=LABELS, zero_division=0)
    np.testing.assert_allclose(per_class_f1(Y_TRUE, Y_PRED, LABELS), f, atol=1e-12)
    rep = per_class_report(Y_TRUE, Y_PRED, LABELS)
    np.testing.assert_allclose([rep[str(k)]["precision"] for k in LABELS], p, atol=1e-12)
    np.testing.assert_allclose([rep[str(k)]["recall"] for k in LABELS], r, atol=1e-12)
    assert [rep[str(k)]["support"] for k in LABELS] == list(s)


def test_accuracy_matches_sklearn():
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(accuracy_score(Y_TRUE, Y_PRED))


def test_class_weights_inverse_frequency():
    w = class_weights([0, 0, 0, 1], n_classes=3)
    assert w.tolist() == pytest.approx([4 / 6, 4 / 2, 0.0])


def test_mlp_head_shape():
    assert MLPHead(384, 9)(torch.zeros(5, 384)).shape == (5, 9)
