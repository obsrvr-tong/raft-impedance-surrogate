"""collect_dataset.py on two fake finished cases (real impedance file, fake .mess)."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "numpy not installed")
class TestCollect(unittest.TestCase):
    def test_collect_two_cases_and_skip_a_failed_one(self):
        cfg = json.loads((HERE / "inputs" / "dataset_v1.json").read_text())
        with tempfile.TemporaryDirectory() as d:
            cases = Path(d) / "cases"
            for name, diag in (("m_train0001", "<A>_ALARM"), ("m_test0001", "OK"), ("m_train0002", "<F>_ERROR")):
                c = cases / name
                c.mkdir(parents=True)
                (c / f"{name}.export").write_text(f"F mess {name}.mess R 6\nF libr {name}_impe.txt R 7\n")
                (c / f"{name}.mess").write_text(f"--- DIAGNOSTIC JOB : {diag}\n")
                shutil.copy(HERE / "tests" / "data" / "test_impe.txt", c / f"{name}_impe.txt")
                inputs = {"t_L1": 7.0, "t_L2": 5.0, "t_L3": 14.0, "t_L4": 3.5, "t_L5": 17.0,
                          "t_L6": 14.0, "t_L7": 38.0, "g_factor": 1.0}
                rec = {"split": "test" if "test" in name else "train", "inputs": inputs, "config": cfg,
                       "mesh": {"mesh_id": "m", "L_x": "33", "L_y": "31"}, "git_commit": "abc1234"}
                (c / "case.json").write_text(json.dumps(rec))
            out = Path(d) / "ds"
            r = subprocess.run([sys.executable, str(HERE / "collect_dataset.py"), "--cases", str(cases),
                                "--out", str(out)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            z = np.load(out.with_suffix(".npz"), allow_pickle=False)
            self.assertEqual(z["K"].shape, (2, 51, 6, 6))
            self.assertAlmostEqual(float(z["K"][0, 0, 0, 0].real), 13267.0, places=0)
            self.assertAlmostEqual(float(z["K"][0, 0, 0, 0].imag), -266.15, places=1)
            self.assertEqual(list(z["x_names"])[-2:], ["L_x", "L_y"])
            self.assertEqual(sorted(z["split"]), ["test", "train"])
            H, vs_avg, f0 = z["D"][0]
            self.assertAlmostEqual(H, 98.5, places=6)
            self.assertAlmostEqual(f0, vs_avg / (4 * H), places=9)
            man = json.loads((Path(d) / "ds_manifest.json").read_text())
            self.assertEqual(man["n_cases"], 2)
            self.assertEqual(man["skipped"][0]["case"], "m_train0002")


if __name__ == "__main__":
    unittest.main()
