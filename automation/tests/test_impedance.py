"""Impedance file parser checks (synthetic file in the CHAMP U/T IMPOSE layout)."""
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from codeaster_auto.impedance import read_impe, write_csv  # noqa: E402


def synthetic(path, n=6, freqs=(0.2, 1.2, 2.2)):
    with open(path, "w") as f:
        f.write(" header text\n")
        for u in range(1, n + 1):
            for t in range(1, n + 1):
                f.write(f"\n CHAMP U IMPOSE {u:4d}  CHAMP T IMPOSE {t:4d}\n")
                for fr in freqs:
                    # value encodes (row t, column u, frequency) so placement can be checked
                    f.write(f"  {fr:12.5E}  {t * 10 + u + fr:14.6E}  {-(t * 10 + u):14.6E}\n")


class TestImpedance(unittest.TestCase):
    def test_parse_places_terms_row_T_column_U(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x_impe.txt"
            synthetic(p)
            data = read_impe(p)
            self.assertEqual(data["n"], 6)
            self.assertEqual(data["freqs"], [0.2, 1.2, 2.2])
            self.assertEqual(data["missing"], [])
            # K_ij = row i (T), column j (U): K15 -> re = 15 + f
            self.assertAlmostEqual(data["re"][1][0][4], 15 + 1.2)
            self.assertAlmostEqual(data["re"][1][4][0], 51 + 1.2)
            self.assertAlmostEqual(data["im"][2][2][3], -34)
            write_csv(data, Path(d) / "x.csv")
            header = (Path(d) / "x.csv").read_text().splitlines()[0].split(",")
            self.assertEqual(header[:5], ["frequency", "K11_re", "K11_im", "K12_re", "K12_im"])

    def test_no_blocks_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "empty.txt"
            p.write_text("nothing here\n")
            with self.assertRaises(ValueError):
                read_impe(p)


class TestRealFile(unittest.TestCase):
    """tests/data/test_impe.txt: a real CALC_MISS impedance file supplied by Roy."""

    def test_real_file(self):
        d = read_impe(HERE / "tests" / "data" / "test_impe.txt")
        self.assertEqual((d["n"], len(d["freqs"]), d["missing"]), (6, 51, []))
        self.assertAlmostEqual(d["freqs"][0], 0.2)
        self.assertAlmostEqual(d["freqs"][-1], 50.2)
        self.assertAlmostEqual(d["re"][0][0][0], 0.13267e5)      # K11 at 0.2 Hz
        self.assertAlmostEqual(d["im"][0][0][0], -0.26615e3)
        # sway-rocking terms: K15 ~ K51 (near-symmetric), K24 of opposite sign
        self.assertAlmostEqual(d["re"][0][0][4] / d["re"][0][4][0], 1.0, places=2)
        self.assertLess(d["re"][0][1][3] * d["re"][0][0][4], 0)


if __name__ == "__main__":
    unittest.main()
