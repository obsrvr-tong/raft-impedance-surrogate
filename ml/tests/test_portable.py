"""Tests for the portable model, the command line and the API (synthetic data, fast).

Run from the repo root:  python -m unittest discover ml/tests -v
The API tests are skipped when fastapi (and httpx, for its test client) is not installed.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surrogate.data import TERMS, make_targets  # noqa: E402
from surrogate.models import StaticShapeGP  # noqa: E402
from surrogate.portable import INPUTS, ImpedanceSurrogate, export, features  # noqa: E402

try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
    HAVE_API = True
except ImportError:
    HAVE_API = False


def synthetic_cases(n=60, n_freq=10, seed=1):
    """Impedance-like curves driven by the 10 real inputs: stiffness grows with g and raft size."""
    rng = np.random.default_rng(seed)
    lo = np.array([4, 3, 10, 2, 12, 10, 30, 0.667, 30, 15])
    hi = np.array([10, 7, 18, 5, 22, 18, 46, 1.5, 80, 31])
    X = lo + (hi - lo) * rng.uniform(size=(n, 10))
    f = np.linspace(0.45, 15, n_freq)
    H = X[:, :7].sum(1, keepdims=True)
    K = np.zeros((n, n_freq, 6, 6), dtype=complex)
    for k, (_, i, j) in enumerate(TERMS):
        size = (X[:, 8:9] * X[:, 9:10]) ** (0.5 if k < 3 else (1.5 if k < 6 else 1.0))
        sign = -1.0 if k == 7 else 1.0
        curve = 1 - 0.002 * f ** 2 * (H / 90) + 1j * 0.03 * f * (1 + 0.2 * X[:, :1] / 10)
        K[:, :, i, j] = K[:, :, j, i] = sign * 1e3 * X[:, 7:8] * size * curve
    return X, f, K


class TestPortable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.f, K = synthetic_cases()
        Y, cls.spec = make_targets(K)
        cls.model = StaticShapeGP(cls.spec, n_components=4, trend="all").fit(features(cls.X[:50]), Y[:50])
        cls.sur = export(cls.model, cls.f, meta={"name": "test"}, X_check=cls.X[50:])

    def test_export_reproduces_the_model(self):
        self.assertLess(self.sur.meta["export_check_max_dev_pct"], 1e-6)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.sur.save(Path(d) / "m.npz")
            again = ImpedanceSurrogate.load(path)
        np.testing.assert_allclose(again.predict(self.X[50:]).K, self.sur.predict(self.X[50:]).K, rtol=1e-12)
        self.assertEqual(again.meta["name"], "test")

    def test_inputs_as_dict_list_or_array(self):
        one = dict(zip(INPUTS, self.X[55]))
        a = self.sur.predict(one).K
        b = self.sur.predict([one, one]).K
        c = self.sur.predict(self.X[55:56]).K
        np.testing.assert_allclose(a[0], c[0])
        self.assertEqual(b.shape[0], 2)
        with self.assertRaises(ValueError):
            self.sur.predict({k: 1.0 for k in INPUTS[:-1]})

    def test_std_with_all_neighbours_is_the_exact_gp_std(self):
        Xs = self.model.x_scaler_.transform(features(self.X[50:]))
        _, S = self.sur.gp_outputs(self.X[50:], return_std=True, n_neighbours=10_000)
        for c, gp in enumerate(self.model.gps_[:6]):
            _, sk = gp.predict(Xs, return_std=True)            # includes the white-noise variance
            noise = gp.kernel_.k2.noise_level * gp._y_train_std ** 2
            np.testing.assert_allclose(S[:, c] ** 2, sk ** 2 - noise, rtol=1e-4, atol=1e-12)

    def test_fewer_neighbours_never_reduce_the_std(self):
        _, exact = self.sur.gp_outputs(self.X[50:], return_std=True, n_neighbours=10_000)
        _, few = self.sur.gp_outputs(self.X[50:], return_std=True, n_neighbours=12)
        self.assertTrue(np.all(few >= exact * (1 - 1e-6)))

    def test_band_contains_the_prediction(self):
        p = self.sur.predict(self.X[50:], band=True)
        for part in (np.real, np.imag):
            self.assertTrue(np.all(part(p.lo) <= part(p.K) + 1e-9 * np.abs(part(p.K))))
            self.assertTrue(np.all(part(p.K) <= part(p.hi) + 1e-9 * np.abs(part(p.K))))

    def test_truncation_error_is_exported_and_widens_the_band(self):
        from surrogate.data import targets_to_K, terms_to_targets
        rstd = self.sur.a["shape_resid_std"]
        np.testing.assert_allclose(rstd, self.model.shape_resid_std_)
        K8 = targets_to_K(make_targets(synthetic_cases()[2])[0][:50], self.spec)
        shapes = terms_to_targets(K8 / K8[:, 0, :].real[:, None, :])
        again = ImpedanceSurrogate(dict(self.sur.a), dict(self.sur.meta)).set_truncation_error(shapes)
        np.testing.assert_allclose(again.a["shape_resid_std"], rstd, rtol=1e-6, atol=1e-12)
        G, S = self.sur.gp_outputs(self.X[50:], return_std=True)
        lo1, hi1 = self.sur.band_from_gp(G, S, truncation=False)
        lo2, hi2 = self.sur.band_from_gp(G, S, truncation=True)
        self.assertGreater(np.mean(hi2.real - lo2.real), np.mean(hi1.real - lo1.real))

    def test_range_warnings(self):
        case = dict(zip(INPUTS, self.X[0]), L_x=200.0)
        w = self.sur.predict(case).warnings[0]
        self.assertEqual(len(w), 1)
        self.assertIn("L_x", w[0])
        self.assertEqual(self.sur.predict(dict(zip(INPUTS, self.X[0]))).warnings[0], [])

    def test_case_dict_is_json(self):
        d = self.sur.predict(self.X[50:52], band=True).case(1)
        json.dumps(d)
        self.assertEqual(set(d["terms"]), set(self.spec.terms))
        self.assertEqual(len(d["terms"]["Uz"]["re_lo"]), len(self.f))


class TestCommandLine(unittest.TestCase):
    def test_predict_prints_and_writes_csv(self):
        from surrogate.__main__ import main
        X, f, K = synthetic_cases(n=40)
        Y, spec = make_targets(K)
        sur = export(StaticShapeGP(spec, n_components=3).fit(features(X), Y), f, meta={"name": "cli-test"})
        with tempfile.TemporaryDirectory() as d:
            model = sur.save(Path(d) / "m.npz")
            case = Path(d) / "case.json"
            case.write_text(json.dumps(dict(zip(INPUTS, X[3].tolist()))))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                main(["predict", str(case), "--model", str(model), "--band", "--csv", str(Path(d) / "c.csv")])
            lines = (Path(d) / "c.csv").read_text().splitlines()
        self.assertIn("cli-test", out.getvalue())
        self.assertEqual(len(lines), 1 + len(f))
        self.assertEqual(len(lines[0].split(",")), 2 + 8 * 6)


class TestAnalysis(unittest.TestCase):
    def test_sobol_on_the_ishigami_function(self):
        from surrogate.analysis import sobol_indices
        a, b = 7.0, 0.1
        f = lambda X: np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])
        S = sobol_indices(f, [-np.pi] * 3, [np.pi] * 3, n=8192, n_boot=20)
        np.testing.assert_allclose(S["S1"][:, 0], [0.314, 0.442, 0.0], atol=0.03)
        np.testing.assert_allclose(S["ST"][:, 0], [0.558, 0.442, 0.244], atol=0.03)

    def test_monte_carlo_percentiles_are_ordered(self):
        from surrogate.analysis import monte_carlo
        X, f, K = synthetic_cases(n=40)
        Y, spec = make_targets(K)
        sur = export(StaticShapeGP(spec, n_components=3).fit(features(X), Y), f)
        base = dict(zip(INPUTS, X[0]))
        mc = monte_carlo(sur, base, n=300)
        self.assertTrue(np.all(np.diff(mc["re"], axis=0) >= 0))
        self.assertEqual(mc["re"].shape, (5, len(f), 8))


@unittest.skipUnless(HAVE_API, "fastapi / httpx not installed")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, f, K = synthetic_cases(n=40)
        Y, spec = make_targets(K)
        cls.tmp = tempfile.TemporaryDirectory()
        path = export(StaticShapeGP(spec, n_components=3).fit(features(X), Y), f,
                      meta={"name": "api-test"}).save(Path(cls.tmp.name) / "m.npz")
        os.environ["IMPEDANCE_SURROGATE_MODEL"] = str(path)
        from fastapi.testclient import TestClient
        from surrogate import api
        api.get_model.cache_clear()
        api._viewer_html.cache_clear()
        cls.client = TestClient(api.app)
        cls.case = dict(zip(INPUTS, X[5].tolist()))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_health_and_model_card(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        card = self.client.get("/model").json()
        self.assertEqual(card["name"], "api-test")
        self.assertEqual([i["name"] for i in card["inputs"]], list(INPUTS))

    def test_predict(self):
        r = self.client.post("/predict", json={"cases": [self.case], "band": True})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        res = body["results"][0]
        self.assertEqual(len(res["terms"]["Ux"]["re"]), len(body["freqs"]))
        self.assertIn("re_lo", res["terms"]["Ux"])
        self.assertEqual(res["warnings"], [])

    def test_out_of_range_warns_and_invalid_is_rejected(self):
        r = self.client.post("/predict", json={"cases": [dict(self.case, L_x=500.0)]})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(any("L_x" in w for w in r.json()["results"][0]["warnings"]))
        self.assertNotIn("re_lo", r.json()["results"][0]["terms"]["Ux"])
        self.assertEqual(self.client.post("/predict", json={"cases": [dict(self.case, L_x=-1)]}).status_code, 422)
        self.assertEqual(self.client.post("/predict", json={"cases": []}).status_code, 422)

    def test_viewer_page(self):
        r = self.client.get("/viewer")
        self.assertEqual(r.status_code, 200)
        self.assertIn("<title>", r.text)


if __name__ == "__main__":
    unittest.main()
