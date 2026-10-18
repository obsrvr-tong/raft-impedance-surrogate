"""Dataset v1 generation: site file, Vs -> E, sampling, and a full case build."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from codeaster_auto.sampling import halton, random_points  # noqa: E402
from codeaster_auto.strata import layers_for, read_site, youngs_modulus  # noqa: E402
from make_dataset import plan  # noqa: E402
from make_case import make_case  # noqa: E402

CFG = json.loads((HERE / "inputs" / "dataset_v1.json").read_text())


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.soil, self.bedrock = read_site(HERE / CFG["site"])

    def test_site_v1(self):
        self.assertEqual([s.name for s in self.soil], ["L1", "L2", "L3", "L4", "L5", "L6", "L7"])
        self.assertEqual(self.bedrock.name, "bedrock")

    def test_vs_to_E(self):
        self.assertAlmostEqual(youngs_modulus(1000, 2.18, 0.34), 5842.4)

    def test_one_layer_per_stratum_and_g_scaling(self):
        t = [s.t_ref for s in self.soil]
        L1 = layers_for(self.soil, self.bedrock, t, 1.0)
        L2 = layers_for(self.soil, self.bedrock, t, 1.5)
        self.assertEqual(len(L1), 8)
        self.assertEqual([l.thickness for l in L1[:-1]], t)
        for a, b in zip(L1[:-1], L2[:-1]):
            self.assertAlmostEqual(b.E / a.E, 1.5)          # G (and E) scale with g
        self.assertEqual(L1[-1].E, L2[-1].E)                # bedrock not scaled

    def test_halton_is_deterministic_and_extensible(self):
        self.assertEqual(halton(10, 8, 7), halton(10, 8, 7))
        self.assertEqual(halton(10, 8, 7)[6:], halton(4, 8, 7, skip=6))
        pts = halton(64, 8, 7) + random_points(16, 8, 8)
        self.assertTrue(all(0 <= x < 1 for p in pts for x in p))

    def test_plan_stays_in_ranges(self):
        rows = plan(CFG, self.soil, 32, 8)
        self.assertEqual(sum(r[0] == "train" for r in rows), 32)
        self.assertEqual(sum(r[0] == "test" for r in rows), 8)
        for _, _, _, t, g in rows:
            for s, ti in zip(self.soil, t):
                self.assertTrue(s.t_min <= ti <= s.t_max)
            self.assertTrue(CFG["g_factor_min"] - 1e-3 <= g <= CFG["g_factor_max"] + 1e-3)

    def test_case_build_uses_settings_v1(self):
        t = [s.t_ref for s in self.soil]
        with tempfile.TemporaryDirectory() as d:
            mesh = Path(d) / "raft.med"
            mesh.write_bytes(b"dummy mesh: make_case only copies it")
            case = make_case("x", mesh,
                             layers_for(self.soil, self.bedrock, t), out_dir=d,
                             freq_min=CFG["freq_min"], freq_max=CFG["freq_max"], freq_step=CFG["freq_step"])
            comm = (case / "x.comm").read_text()
            self.assertEqual(len(re.findall(r"NUME_MATE=", comm)), 8)
            self.assertIn("FREQ_PAS=0.25", comm)
            self.assertIn("FREQ_MAX=33.2", comm)


if __name__ == "__main__":
    unittest.main()
