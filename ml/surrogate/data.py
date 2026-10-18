"""Dataset v1 for the impedance surrogate: loading, reference-point transfer, target arrays.

    from surrogate.data import load_dataset, make_targets, targets_to_K
    ds = load_dataset()                              # 3,232 cases, 0.45-14.95 Hz, ref33x31 at its centroid
    train = ds.select(split="train", mesh="ref33x31")
    Y, spec = make_targets(train.K)                  # (n, 944) real rows for scikit-learn / PyTorch
    K8 = targets_to_K(Y, spec)                       # back to (n, 59, 8) complex

Conventions (the same as in automation/):
  units MN, m, s;  DOF order Ux, Uy, Uz, Rx, Ry, Rz;
  K[case, freq, i, j] is the 6x6 complex impedance at frequency freqs[freq].
Within each mesh and split, cases keep their case-number order, which for the training cases
is the Halton order: the first N training cases of a mesh are themselves a space-filling design.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = Path(os.environ.get("IMPEDANCE_DATASET",
                                  REPO / "automation" / "datasets" / "v1" / "dataset_v1.npz"))
MESHES_CSV = REPO / "automation" / "inputs" / "meshes.csv"

DOF = ("Ux", "Uy", "Uz", "Rx", "Ry", "Rz")

# Frequency band used for modelling (decided 4 October 2026):
#  FMIN: the 0.2 Hz point is numerically unreliable in a minority of cases (notebook 01, §3a).
#  FMAX: impedances above 15-20 Hz fluctuate strongly, in MISS3D (notebook 02, §5) as in full FE
#        models, and the band that matters for soil-structure interaction of these buildings lies
#        below 15 Hz. Targets therefore run 0.45-14.95 Hz (59 frequencies).
FMIN = 0.45
FMAX = 15.0

# The 8 terms the surrogate learns: (name, row, column) in the 6x6 matrix.
# The six diagonal terms, plus the two horizontal-rocking couplings.
TERMS = (
    ("Ux", 0, 0), ("Uy", 1, 1), ("Uz", 2, 2),
    ("Rx", 3, 3), ("Ry", 4, 4), ("Rz", 5, 5),
    ("Ux-Ry", 0, 4), ("Uy-Rx", 1, 3),
)


# --------------------------------------------------------------------------------------------
# The dataset container
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Dataset:
    """All arrays of dataset v1, one row (first axis) per case."""
    K: np.ndarray            # (n, n_freq, 6, 6) complex impedance
    freqs: np.ndarray        # (n_freq,) Hz
    X: np.ndarray            # (n, 10) sampled inputs and footprint, see x_names
    x_names: tuple
    D: np.ndarray            # (n, 3) derived features: H_bedrock, vs_avg, f0
    d_names: tuple
    case: np.ndarray         # (n,) case names, e.g. "m30x15_train0001"
    split: np.ndarray        # (n,) "train" or "test"
    mesh_id: np.ndarray      # (n,) footprint, e.g. "ref33x31"
    notes: tuple = ()        # what has been done to K since loading (provenance)

    def __len__(self):
        return len(self.case)

    def take(self, idx) -> "Dataset":
        """The cases selected by idx: a boolean mask, an index array or a slice."""
        return replace(self, K=self.K[idx], X=self.X[idx], D=self.D[idx],
                       case=self.case[idx], split=self.split[idx], mesh_id=self.mesh_id[idx])

    def head_per_mesh(self, n) -> "Dataset":
        """The first n cases of every mesh (in case order, i.e. the Halton order for training cases):
        a balanced subset in which every footprint sees the same n soils."""
        idx = np.concatenate([np.flatnonzero(self.mesh_id == m)[:n] for m in np.unique(self.mesh_id)])
        return self.take(np.sort(idx))

    def select(self, split=None, mesh=None) -> "Dataset":
        """Cases of the given split(s) and mesh(es); each argument is a string or a list."""
        keep = np.ones(len(self), dtype=bool)
        if split is not None:
            keep &= np.isin(self.split, _as_list(split))
        if mesh is not None:
            keep &= np.isin(self.mesh_id, _as_list(mesh))
        return self.take(keep)

    def frame(self):
        """Inputs, derived features and labels as a pandas DataFrame (one row per case)."""
        import pandas as pd
        df = pd.DataFrame(np.hstack([self.X, self.D]), columns=list(self.x_names) + list(self.d_names))
        df.insert(0, "split", self.split)
        df.insert(0, "mesh_id", self.mesh_id)
        df.insert(0, "case", self.case)
        return df


def _input_names(stored):
    """Input names: the first seven columns are the layer thicknesses from the top down, named
    t_L1 ... t_L7 (the published, anonymised layer names), then g_factor, L_x, L_y."""
    names = [str(n) for n in stored.tolist()]
    return tuple([f"t_L{i}" for i in range(1, 8)] + names[7:])


def _as_list(v):
    return [v] if isinstance(v, str) else list(v)


def load_dataset(path=DEFAULT_NPZ, to_centroid=True, symmetrize=False, fmin=FMIN, fmax=FMAX) -> Dataset:
    """Load dataset_v1.npz.

    to_centroid  transfer every K to its raft's centroid (only ref33x31 needs it; see
                 move_to_centroid). Leave True unless you want to see the raw data.
    symmetrize   replace K by (K + K^T) / 2, averaging out the small numerical asymmetry.
    fmin, fmax   keep only frequencies within [fmin, fmax] Hz. The defaults (FMIN = 0.45,
                 FMAX = 15) give the modelling band; fmin=None, fmax=None keep all 133.
    """
    with np.load(path) as z:
        ds = Dataset(
            K=z["K"], freqs=z["freqs"], X=z["X"], x_names=_input_names(z["x_names"]),
            D=z["D"], d_names=tuple(z["d_names"].tolist()),
            case=z["case"], split=z["split"], mesh_id=z["mesh_id"],
            notes=(f"loaded {Path(path).name}",),
        )
    if to_centroid:
        ds = move_to_centroid(ds)
    if symmetrize:
        ds = replace(ds, K=symmetrize_K(ds.K), notes=ds.notes + ("symmetrized: K = (K + K^T) / 2",))
    if fmin is not None:
        keep = ds.freqs >= fmin - 1e-9
        ds = replace(ds, K=ds.K[:, keep], freqs=ds.freqs[keep],
                     notes=ds.notes + (f"frequencies below {fmin:g} Hz dropped ({int((~keep).sum())})",))
    if fmax is not None:
        keep = ds.freqs <= fmax + 1e-9
        ds = replace(ds, K=ds.K[:, keep], freqs=ds.freqs[keep],
                     notes=ds.notes + (f"frequencies above {fmax:g} Hz dropped ({int((~keep).sum())})",))
    return ds


# --------------------------------------------------------------------------------------------
# Reference-point transfer (rigid raft)
# --------------------------------------------------------------------------------------------
def transfer_matrix(rx, ry):
    """T such that q_N = T q_C for a rigid raft.

    q = (Ux, Uy, Uz, Rx, Ry, Rz); C is the centroid, N the reference node, r = N - C = (rx, ry, 0).
    A rigid rotation theta moves N by theta x r, which gives the off-diagonal entries.
    """
    T = np.eye(6)
    T[0, 5] = -ry      # Ux_N = Ux_C - ry * Rz
    T[1, 5] = rx       # Uy_N = Uy_C + rx * Rz
    T[2, 3] = ry       # Uz_N = Uz_C + ry * Rx - rx * Ry
    T[2, 4] = -rx
    return T


def transfer_K(K, rx, ry):
    """Impedance at N -> impedance at C:  K_C = T^T K_N T, for every case and frequency at once.

    K has shape (..., 6, 6). The matrix product @ works on the last two axes and repeats
    ("broadcasts") over all leading ones, so no loop over cases or frequencies is needed.
    The product is formed in double precision and returned in K's own dtype.
    """
    T = transfer_matrix(rx, ry)
    return (T.T @ K.astype(np.complex128) @ T).astype(K.dtype)


def nradier_offsets(path=MESHES_CSV) -> dict:
    """{mesh_id: (rx, ry)}: position of the reference node NRADIER relative to the centroid."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {row["mesh_id"]: (float(row["nradier_x"]) - float(row["centroid_x"]),
                                 float(row["nradier_y"]) - float(row["centroid_y"]))
                for row in csv.DictReader(f)}


def move_to_centroid(ds: Dataset, offsets=None) -> Dataset:
    """Transfer K of every mesh whose NRADIER is off the centroid (in v1: ref33x31 only).

    Offsets come from automation/inputs/meshes.csv; a mesh missing there raises KeyError.
    """
    offsets = nradier_offsets() if offsets is None else offsets
    K = ds.K.copy()
    notes = []
    for m in np.unique(ds.mesh_id):
        rx, ry = offsets[str(m)]
        if rx == 0 and ry == 0:
            continue
        sel = ds.mesh_id == m
        K[sel] = transfer_K(K[sel], rx, ry)
        notes.append(f"{m}: K moved from NRADIER to the centroid (r = {rx:.3f}, {ry:.3f} m)")
    return replace(ds, K=K, notes=ds.notes + tuple(notes))


# 90-degree rotation of the axes about the vertical: x' = y, y' = -x (z unchanged). Displacements
# and rotations transform alike: q' = Q q, so K' = Q K Q^T.
Q_ROT90 = np.zeros((6, 6))
Q_ROT90[0, 1] = Q_ROT90[3, 4] = 1.0           # Ux' = Uy,  Rx' = Ry
Q_ROT90[1, 0] = Q_ROT90[4, 3] = -1.0          # Uy' = -Ux, Ry' = -Rx
Q_ROT90[2, 2] = Q_ROT90[5, 5] = 1.0           # Uz' = Uz,  Rz' = Rz


def rotate_90(ds: Dataset) -> Dataset:
    """The same cases described in axes rotated by 90 degrees: L_x and L_y swap, K -> Q K Q^T.

    Exact for a rigid raft on horizontally layered soil, whose response cannot depend on the
    compass direction. A 30 x 15 m raft becomes a 15 x 30 m one, so a training set gains a
    rotated twin of every footprint at no simulation cost (data augmentation from a physical
    symmetry). Mesh ids get the suffix "_rot90". In the learned terms, the new Ux-Ry is minus
    the old Uy-Rx, and the new Uy-Rx is minus the old Ux-Ry.
    """
    Q = Q_ROT90
    K = (Q @ ds.K.astype(np.complex128) @ Q.T).astype(ds.K.dtype)
    X = ds.X.copy()
    X[:, [8, 9]] = X[:, [9, 8]]
    tag = np.char.add
    return replace(ds, K=K, X=X, case=tag(ds.case.astype(str), "_rot90"),
                   mesh_id=tag(ds.mesh_id.astype(str), "_rot90"), notes=ds.notes + ("rotated by 90 degrees",))


def concat(*parts: Dataset) -> Dataset:
    """Cases of several datasets (same frequencies) stacked into one."""
    first = parts[0]
    cat = lambda name: np.concatenate([getattr(p, name) for p in parts])
    return replace(first, K=cat("K"), X=cat("X"), D=cat("D"), case=cat("case"), split=cat("split"),
                   mesh_id=cat("mesh_id"), notes=first.notes + (f"concatenated {len(parts)} sets",))


def symmetrize_K(K):
    """(K + K^T) / 2 over the last two axes."""
    return 0.5 * (K + np.swapaxes(K, -1, -2))


# --------------------------------------------------------------------------------------------
# Targets: complex curves -> one real row per case, and back
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetSpec:
    """Layout of a target row: term after term; within a term, Re then Im; within each, frequency.

    For 8 terms and 59 frequencies a row has 8 x 2 x 59 = 944 numbers:
      [Re Ux(f1..f59), Im Ux(f1..f59), Re Uy(...), Im Uy(...), ..., Re Uy-Rx, Im Uy-Rx]
    """
    terms: tuple             # term names
    ij: tuple                # (row, column) of each term in the 6x6 matrix
    n_freq: int

    @property
    def n_columns(self):
        return len(self.terms) * 2 * self.n_freq

    def columns(self, term, part="re") -> slice:
        """Columns of one curve, e.g. Y[:, spec.columns("Uz", "im")] is Im K_zz at all frequencies."""
        k = self.terms.index(term)
        p = {"re": 0, "im": 1}[part]
        start = (2 * k + p) * self.n_freq
        return slice(start, start + self.n_freq)


def make_targets(K, terms=TERMS):
    """K (n, n_freq, 6, 6) complex -> Y (n, n_terms * 2 * n_freq) float64, plus its TargetSpec."""
    rows = [t[1] for t in terms]
    cols = [t[2] for t in terms]
    K8 = K[:, :, rows, cols]                          # (n, n_freq, n_terms): pairs (rows[k], cols[k])
    spec = TargetSpec(terms=tuple(t[0] for t in terms), ij=tuple((t[1], t[2]) for t in terms),
                      n_freq=K.shape[1])
    return terms_to_targets(K8), spec


def terms_to_targets(K8):
    """Complex terms (n, n_freq, n_terms) -> target rows (n, n_terms * 2 * n_freq), float64."""
    parts = np.stack([K8.real, K8.imag], axis=-1)     # (n, n_freq, n_terms, 2)
    return parts.transpose(0, 2, 3, 1).reshape(len(K8), -1).astype(np.float64)


def targets_to_K(Y, spec: TargetSpec):
    """Y (n, n_columns) -> complex terms (n, n_freq, n_terms), the layout of K[:, :, rows, cols]."""
    a = np.asarray(Y).reshape(len(Y), len(spec.terms), 2, spec.n_freq)
    return (a[:, :, 0] + 1j * a[:, :, 1]).transpose(0, 2, 1)
