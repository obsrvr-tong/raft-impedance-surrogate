"""Tests for surrogate.gpu (GPyTorch engine). Skipped if gpytorch is not installed.
Runs on the GPU if one is available, otherwise on the CPU."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import gpytorch  # noqa: F401
    HAVE_GPYTORCH = True
except ImportError:
    HAVE_GPYTORCH = False

from surrogate.data import make_targets  # noqa: E402
from surrogate.metrics import target_errors  # noqa: E402
from test_models import synthetic  # noqa: E402


@unittest.skipUnless(HAVE_GPYTORCH, "gpytorch not installed")
class TestTorchModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, _, K = synthetic(n=60)
        cls.Y, cls.spec = make_targets(K)

    def test_pcagp(self):
        from surrogate.gpu import TorchPCAGP
        m = TorchPCAGP(self.spec, n_components=4).fit(self.X[:45], self.Y[:45])
        P, S = m.predict(self.X[45:], return_std=True)
        self.assertEqual(P.shape, self.Y[45:].shape)
        self.assertEqual(S.shape, self.Y[45:].shape)
        self.assertLess(target_errors(self.Y[45:], P, self.spec).max(), 2.0)
        self.assertEqual(m.length_scales().shape, (4, 3))

    def test_static_shape(self):
        from surrogate.gpu import TorchStaticShapeGP
        m = TorchStaticShapeGP(self.spec, n_components=4).fit(self.X[:45], self.Y[:45])
        P = m.predict(self.X[45:])
        self.assertLess(target_errors(self.Y[45:], P, self.spec).max(), 2.0)
        self.assertEqual(m.predict_static(self.X[45:]).shape, (15, 8))
        r = m.fit_report()
        self.assertEqual(r["precision"], "float32")
        self.assertAlmostEqual(r["noise_floor"], 1e-4)

    def test_noise_floor_grows_with_n(self):
        from surrogate.gpu import float32_noise_floor
        self.assertEqual(float32_noise_floor(300), 1e-4)
        self.assertAlmostEqual(float32_noise_floor(5120), 1.024e-3)

    def test_failed_float32_group_is_refitted_in_float64(self):
        import torch
        from linear_operator.utils.errors import NotPSDError
        from surrogate import gpu
        original = gpu.TorchGPEngine._fit_part

        def fail_in_float32(engine, X, T, dtype):
            if dtype == torch.float32:
                raise NotPSDError("simulated failure")
            return original(engine, X, T, dtype)

        gpu.TorchGPEngine._fit_part = fail_in_float32
        try:
            m = gpu.TorchStaticShapeGP(self.spec, n_components=4, n_iter=100, max_batch=6).fit(self.X[:45], self.Y[:45])
        finally:
            gpu.TorchGPEngine._fit_part = original
        r = m.fit_report()
        self.assertEqual(r["refit_in_float64"], 2)                 # 12 GPs in groups of 6
        self.assertEqual(m.predict(self.X[45:]).shape, self.Y[45:].shape)


if __name__ == "__main__":
    unittest.main()
