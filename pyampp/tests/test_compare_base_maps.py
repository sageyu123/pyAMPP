from __future__ import annotations

import numpy as np

from pyampp.tests.compare_base_maps import _best_simple_transform, _metrics


def test_metrics_identity():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = _metrics(a, a.copy())
    assert m.mae == 0.0
    assert m.rmse == 0.0
    assert m.max_abs == 0.0
    assert np.isclose(m.corr, 1.0)


def test_best_transform_detects_transpose():
    sav = np.arange(12, dtype=float).reshape(3, 4)
    h5 = sav.T
    name, m = _best_simple_transform(h5, sav, signed=False)
    assert name == "T"
    assert m.mae == 0.0


def test_best_transform_detects_sign_flip():
    sav = np.array([[2.0, -1.0], [0.5, 4.0]])
    h5 = -np.fliplr(sav)
    name, m = _best_simple_transform(h5, sav, signed=True)
    assert name == "fliplr_neg"
    assert m.mae == 0.0
