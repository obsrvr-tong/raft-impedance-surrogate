"""Tests for surrogate.data.   Run from the repo root:  python -m unittest discover ml/tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # works with or without pip install -e
from surrogate.data import (DEFAULT_NPZ, FMAX, FMIN, Q_ROT90, TERMS, concat, load_dataset,  # noqa: E402
                            make_targets, nradier_offsets, rotate_90, symmetrize_K, targets_to_K,
                            transfer_K, transfer_matrix)

RX, RY = 1.171, 0.807          # ref33x31: NRADIER relative to the centroid


def random_symmetric_K(rng, shape=(4, 7)):
    A = rng.normal(size=shape + (6, 6)) + 1j * rng.normal(size=shape + (6, 6))
    return A + np.swapaxes(A, -1, -2)


def coupling_ratio(K, i, j):
    """max over cases and frequencies of |K_ij| / sqrt(|K_ii| |K_jj|)."""
    return float((np.abs(K[..., i, j]) / np.sqrt(np.abs(K[..., i, i]) * np.abs(K[..., j, j]))).max())


class TestTransfer(unittest.TestCase):
    def test_rotation_moves_reference_node(self):
        # a unit rotation about z at the centroid moves N by theta x r = (-ry, rx, 0)
        q_n = transfer_matrix(RX, RY) @ np.array([0, 0, 0, 0, 0, 1.0])
        np.testing.assert_allclose(q_n, [-RY, RX, 0, 0, 0, 1])

    def test_rocking_moves_reference_node_vertically(self):
        # rotations about x and y at the centroid lift N by ry*Rx - rx*Ry
        q_n = transfer_matrix(RX, RY) @ np.array([0, 0, 0, 1.0, 2.0, 0])
        np.testing.assert_allclose(q_n, [0, 0, RY - 2 * RX, 1, 2, 0])

    def test_zero_offset_is_identity(self):
        K = random_symmetric_K(np.random.default_rng(0)).astype(np.complex64)
        np.testing.assert_array_equal(transfer_K(K, 0.0, 0.0), K)

    def test_transfer_back_and_forth(self):
        # T(-r) is the inverse of T(r), so K_N = T(-r)^T K_C T(-r) must transfer back to K_C
        K_c = random_symmetric_K(np.random.default_rng(1))
        T_inv = transfer_matrix(-RX, -RY)
        np.testing.assert_allclose(T_inv @ transfer_matrix(RX, RY), np.eye(6), atol=1e-15)
        K_n = T_inv.T @ K_c @ T_inv
        np.testing.assert_allclose(transfer_K(K_n, RX, RY), K_c, rtol=1e-12, atol=1e-12)

    def test_transfer_keeps_symmetry_and_dtype(self):
        K = random_symmetric_K(np.random.default_rng(2)).astype(np.complex64)
        K_c = transfer_K(K, RX, RY)
        self.assertEqual(K_c.dtype, np.complex64)
        np.testing.assert_allclose(K_c, np.swapaxes(K_c, -1, -2), rtol=1e-5, atol=1e-4)

    def test_offsets_from_meshes_csv(self):
        off = nradier_offsets()
        np.testing.assert_allclose(off["ref33x31"], (RX, RY), atol=1e-9)
        self.assertEqual(off["m80x20"], (0.0, 0.0))


class TestTargets(unittest.TestCase):
    def setUp(self):
        self.K = random_symmetric_K(np.random.default_rng(3), shape=(3, 5))
        self.Y, self.spec = make_targets(self.K)

    def test_shape(self):
        self.assertEqual(self.Y.shape, (3, 8 * 2 * 5))
        self.assertEqual(self.spec.n_columns, 80)
        self.assertEqual(self.Y.dtype, np.float64)

    def test_round_trip(self):
        rows, cols = [t[1] for t in TERMS], [t[2] for t in TERMS]
        np.testing.assert_allclose(targets_to_K(self.Y, self.spec), self.K[:, :, rows, cols])

    def test_columns_pick_the_right_curve(self):
        np.testing.assert_allclose(self.Y[:, self.spec.columns("Uz", "im")], self.K[:, :, 2, 2].imag)
        np.testing.assert_allclose(self.Y[:, self.spec.columns("Ux-Ry", "re")], self.K[:, :, 0, 4].real)

    def test_symmetrize(self):
        A = np.random.default_rng(4).normal(size=(2, 6, 6))
        S = symmetrize_K(A)
        np.testing.assert_allclose(S, np.swapaxes(S, -1, -2))
        np.testing.assert_allclose(np.diagonal(S, axis1=1, axis2=2), np.diagonal(A, axis1=1, axis2=2))


@unittest.skipUnless(DEFAULT_NPZ.is_file(), "dataset_v1.npz not present")
class TestRealDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_dataset(to_centroid=False, fmin=None, fmax=None)   # as computed: NRADIER, 133 frequencies
        cls.full = load_dataset(fmin=None, fmax=None)                     # at the centroid, 133 frequencies
        cls.ds = load_dataset()                                           # the modelling default

    def test_counts(self):
        self.assertEqual(len(self.ds), 3232)
        self.assertEqual(len(self.ds.select(split="test")), 160)
        self.assertEqual(len(self.ds.select(split="train", mesh="ref33x31")), 1024)
        self.assertEqual(self.full.K.shape[1:], (133, 6, 6))

    def test_default_band_is_0_45_to_15_Hz(self):
        self.assertEqual((FMIN, FMAX), (0.45, 15.0))
        self.assertAlmostEqual(float(self.ds.freqs[0]), 0.45, places=6)
        self.assertAlmostEqual(float(self.ds.freqs[-1]), 14.95, places=6)
        self.assertEqual(self.ds.K.shape[1:], (59, 6, 6))
        np.testing.assert_array_equal(self.ds.K, self.full.K[:, 1:60])
        self.assertTrue(any("below 0.45 Hz dropped" in n for n in self.ds.notes))
        self.assertTrue(any("above 15 Hz dropped" in n for n in self.ds.notes))

    def test_head_per_mesh_is_balanced(self):
        sub = self.ds.select(split="train").head_per_mesh(100)
        self.assertEqual(len(sub), 500)
        for m in ["m30x15", "ref33x31"]:
            names = sub.select(mesh=m).case
            self.assertEqual((names[0], names[-1]), (f"{m}_train0001", f"{m}_train0100"))
        # crossed design: the same soils on every footprint
        np.testing.assert_allclose(sub.select(mesh="m30x15").X[:, :8], sub.select(mesh="ref33x31").X[:, :8])

    def test_only_ref33x31_is_moved(self):
        for m in ["m30x15", "m45x15", "m60x15", "m80x20"]:
            np.testing.assert_array_equal(self.full.select(mesh=m).K, self.raw.select(mesh=m).K)
        self.assertTrue(any("ref33x31" in n for n in self.full.notes))

    def test_offset_couplings_vanish_at_the_centroid(self):
        # Uz-Rx, Uz-Ry, Ux-Rz, Uy-Rz come from the off-centre node (7-15 % before the transfer).
        # On the structured, centred meshes they are exactly zero by symmetry. On ref33x31's
        # unstructured TRIA3 mesh, Uz-Rx and Ux-Rz keep ~0.4 %: the data put its stiffness centre
        # ~4 cm from the geometric centroid in y. That is below the mesh's own numerical noise.
        raw = self.raw.select(mesh="ref33x31").K
        moved = self.full.select(mesh="ref33x31").K
        for i, j in [(2, 3), (2, 4), (0, 5), (1, 5)]:
            before, after = coupling_ratio(raw, i, j), coupling_ratio(moved, i, j)
            with self.subTest(term=f"{i}{j}"):
                self.assertGreater(before, 0.05)
                self.assertLess(after, 0.1 * before)
                self.assertLess(after, 0.01)

    def test_centred_meshes_have_no_offset_couplings(self):
        centred = self.ds.select(mesh=["m30x15", "m45x15", "m60x15", "m80x20"]).K
        for i, j in [(2, 3), (2, 4), (0, 5), (1, 5)]:
            self.assertLess(coupling_ratio(centred, i, j), 1e-6)

    def test_rotation(self):
        np.testing.assert_allclose(np.linalg.matrix_power(Q_ROT90, 4), np.eye(6))      # four quarter turns
        np.testing.assert_allclose(Q_ROT90 @ Q_ROT90.T, np.eye(6))                     # orthogonal
        sub = self.ds.select(mesh="m30x15", split="test")
        rot = rotate_90(sub)
        np.testing.assert_allclose(rot.X[:, 8:10], sub.X[:, [9, 8]])
        np.testing.assert_allclose(rot.K[:, :, 0, 0], sub.K[:, :, 1, 1], rtol=1e-6)    # Kx' = Ky
        np.testing.assert_allclose(rot.K[:, :, 3, 3], sub.K[:, :, 4, 4], rtol=1e-6)    # Krx' = Kry
        np.testing.assert_allclose(rot.K[:, :, 0, 4], -sub.K[:, :, 1, 3], rtol=1e-6)   # Ux-Ry' = -(Uy-Rx)
        # half a turn: a centred, doubly symmetric raft must look the same
        twice = rotate_90(rot)
        for i, j in [(0, 0), (2, 2), (4, 4), (0, 4), (1, 3)]:
            np.testing.assert_allclose(twice.K[:, :, i, j], sub.K[:, :, i, j], rtol=1e-6)
        both = concat(sub, rot)
        self.assertEqual(len(both), 2 * len(sub))
        self.assertTrue(str(both.mesh_id[-1]).endswith("_rot90"))

    def test_halton_order_kept(self):
        names = self.ds.select(split="train", mesh="m30x15").case
        self.assertEqual(names[0], "m30x15_train0001")
        self.assertEqual(names[-1], "m30x15_train0512")


if __name__ == "__main__":
    unittest.main()
