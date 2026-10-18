"""Tests for surrogate.metrics, surrogate.models and surrogate.experiments (synthetic data, fast).

Run from the repo root:  python -m unittest discover ml/tests -v
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surrogate.data import TERMS, make_targets  # noqa: E402
from surrogate.experiments import COLUMNS, log_experiment  # noqa: E402
from surrogate.metrics import curve_errors, summarize, target_errors  # noqa: E402
from surrogate.models import PCAGP, StaticShapeGP, TermScaler, knn_model  # noqa: E402


def synthetic(n=40, n_freq=12, seed=0):
    """Smooth impedance-like curves driven by 3 inputs; the 8 learned terms differ in scale."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 3))
    f = np.linspace(0.5, 30, n_freq)
    K = np.zeros((n, n_freq, 6, 6), dtype=complex)
    for k, (_, i, j) in enumerate(TERMS):
        scale = 1e5 if k < 3 else (1e8 if k < 6 else 3e5)
        curve = (2 + X[:, :1]) * (1 + 0.02 * f * (1 + 0.5 * X[:, 1:2])) + 1j * 0.05 * f * (1 + 0.3 * X[:, 2:3])
        K[:, :, i, j] = K[:, :, j, i] = scale * curve
    return X, f, K


class TestMetrics(unittest.TestCase):
    def test_zero_for_identical_curves(self):
        _, f, K = synthetic()
        Y, spec = make_targets(K)
        np.testing.assert_array_equal(target_errors(Y, Y, spec), 0)

    def test_value_and_band(self):
        Kt = np.ones((1, 4, 1)) * np.array([1, 2, 4, 2])[None, :, None]       # max |K| = 4
        Kp = Kt.copy()
        Kp[0, 3, 0] += 0.4                                                     # 10 % of max |K|, at f = 4
        f = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(float(curve_errors(Kt, Kp)[0, 0]), 10.0)
        self.assertAlmostEqual(float(curve_errors(Kt, Kp, f, band=(1, 3))[0, 0]), 0.0)

    def test_summary(self):
        e = np.array([[1.0, 10.0], [3.0, 30.0]])
        s = summarize(e, ["a", "b"])
        self.assertAlmostEqual(s["err_a"], np.percentile([1, 3], 95))
        self.assertAlmostEqual(s["err_worst"], s["err_b"])
        self.assertAlmostEqual(s["err_mean"], 11.0)


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.f, K = synthetic()
        cls.Y, cls.spec = make_targets(K)

    def test_term_scaler(self):
        sc = TermScaler(self.spec).fit(self.Y)
        Z = sc.transform(self.Y)
        np.testing.assert_allclose(sc.inverse_transform(Z), self.Y)
        # every term's median max |value| becomes 1 (complex modulus over Re and Im columns)
        K8 = Z.reshape(len(Z), 8, 2, -1)
        mod = np.hypot(K8[:, :, 0], K8[:, :, 1]).max(axis=2)
        np.testing.assert_allclose(np.median(mod, axis=0), 1.0)

    def test_knn_shapes_and_exact_at_training_points(self):
        m = knn_model(self.spec, k=3).fit(self.X, self.Y)
        P = m.predict(self.X)
        self.assertEqual(P.shape, self.Y.shape)
        np.testing.assert_allclose(P, self.Y, rtol=1e-6)        # distance weighting: a point is its own neighbour

    def test_pcagp_fits_a_smooth_function(self):
        m = PCAGP(self.spec, n_components=4).fit(self.X[:30], self.Y[:30])
        P, S = m.predict(self.X[30:], return_std=True)
        self.assertEqual(P.shape, self.Y[30:].shape)
        self.assertEqual(S.shape, self.Y[30:].shape)
        self.assertTrue(np.all(S >= 0))
        self.assertLess(target_errors(self.Y[30:], P, self.spec).max(), 2.0)    # smooth data: small error
        self.assertEqual(m.length_scales().shape, (4, 3))

    def test_static_shape_gp(self):
        m = StaticShapeGP(self.spec, n_components=4).fit(self.X[:30], self.Y[:30])
        P = m.predict(self.X[30:])
        self.assertEqual(P.shape, self.Y[30:].shape)
        self.assertLess(target_errors(self.Y[30:], P, self.spec).max(), 5.0)   # 30 points, 3 inputs
        s = m.predict_static(self.X[30:])
        self.assertTrue(np.all(s > 0))                      # synthetic statics are all positive

    def test_static_shape_rejects_sign_change(self):
        Y = self.Y.copy()
        Y[0, self.spec.columns("Ux", "re")] *= -1           # one case with a negative static Ux
        with self.assertRaises(ValueError):
            StaticShapeGP(self.spec, n_components=2).fit(self.X, Y)

    def test_pcagp_caps_components_at_case_count(self):
        m = PCAGP(self.spec, n_components=40).fit(self.X[:6], self.Y[:6])
        self.assertEqual(m.pca_.n_components_, 6)


class TestExperiments(unittest.TestCase):
    def test_log_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "exp.csv"
            log_experiment("A", path=path, model="first", err_mean=1.234567)
            log_experiment("B", path=path, model="other")
            log_experiment("A", path=path, model="second")
            with open(path, newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([r["exp_id"] for r in rows], ["B", "A"])
            self.assertEqual(rows[1]["model"], "second")
            self.assertEqual(list(rows[0].keys()), COLUMNS)

    def test_unknown_field_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                log_experiment("A", path=Path(d) / "exp.csv", modle="typo")


if __name__ == "__main__":
    unittest.main()
