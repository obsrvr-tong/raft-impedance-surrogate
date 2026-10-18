"""The trained surrogate as one portable file: numpy only, no scikit-learn, PyTorch or dataset.

    from surrogate.portable import ImpedanceSurrogate
    sur = ImpedanceSurrogate.load()                         # ml/models/impedance_surrogate_v1.npz
    p = sur.predict({"t_L1": 7.0, "t_L2": 5.0, "t_L3": 14.0, "t_L4": 3.5, "t_L5": 17.0,
                     "t_L6": 14.0, "t_L7": 38.0, "g_factor": 1.0, "L_x": 30, "L_y": 15}, band=True)
    p.K[0, :, 0]          # K_xx(f), complex, at p.freqs
    p.lo, p.hi            # 95 % band (2.5 and 97.5 percentiles of Re and Im)
    p.warnings[0]         # inputs outside the training range, if any

Why a separate format: the trained model object depends on scikit-learn or GPyTorch (and a GPU),
which a user of the model does not need. A GP prediction only needs a few arrays: the training
inputs, one weight vector per GP (alpha = (K + noise I)^-1 y, the "kriging weights" of the
training outputs), the hyperparameters, the trend and the PCA basis. `export()` extracts them
once; `ImpedanceSurrogate` predicts from them with plain numpy, on any machine, in milliseconds.

Uncertainty. The band has two parts.
1. The GPs' own uncertainty about the 28 numbers. The exact GP variance at a new input needs the
   n x n Cholesky factor of every GP (28 GPs x 5,120^2 numbers = 6 GB for the release model),
   too much to ship. Here the variance of each GP is computed from its `n_neighbours` (128)
   training points nearest to the new input (nearest in that GP's own length-scale units), with
   the same hyperparameters. Conditioning a GP on fewer points can only leave more variance, so
   this is an upper bound on the exact variance (about 1.35 times the exact standard deviation on
   the release model).
2. What the PCA leaves out. The shape is rebuilt from 20 components, so even perfect GPs would
   miss the rest: on the release model this truncation is 2.4 of the 2.9 % mean error.
   `shape_resid_std` holds, for every value of the shape row, the standard deviation of that
   residual on the training shapes; it enters as an independent normal term. (A pointwise band
   only needs each value's marginal distribution, so correlations along frequency do not matter.)
The parts are combined by sampling, so the exp() of the static part is handled exactly. Without
part 2 the release model's 95 % band held 90 % of the hold-out values; with it, 98 %
(notebook 05). The uncertainty of the trend coefficients is not included.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL = MODELS_DIR / "impedance_surrogate_v1.npz"
FORMAT = "impedance-surrogate/1"
N_NEIGHBOURS = 128      # training cases per GP for the variance (notebook 05: the speed / tightness trade-off)

# Model inputs, in this order. The thicknesses are of the seven soil strata above the bedrock,
# from the ground surface down (automation/inputs/site_v1.csv); g_factor scales the shear modulus
# of all seven strata (Vs x sqrt(g)); L_x, L_y are the raft's plan dimensions.
INPUTS = ("t_L1", "t_L2", "t_L3", "t_L4", "t_L5", "t_L6", "t_L7", "g_factor", "L_x", "L_y")
UNITS = dict(**{n: "m" for n in INPUTS[:7]}, g_factor="-", L_x="m", L_y="m")
NOMINAL = {"t_L1": 7.0, "t_L2": 5.0, "t_L3": 14.0, "t_L4": 3.5, "t_L5": 17.0, "t_L6": 14.0,
           "t_L7": 38.0, "g_factor": 1.0, "L_x": 45.0, "L_y": 15.0}     # mid-range soil, 45 x 15 m raft


def features(inputs) -> np.ndarray:
    """Raw inputs (n, 10) in INPUTS order -> model features: thicknesses, log g, log L_x, log L_y
    (the same as surrogate.models.input_features(ds, footprint=True))."""
    X = np.array(inputs, dtype=float, ndmin=2)
    if X.shape[1] != len(INPUTS):
        raise ValueError(f"expected {len(INPUTS)} inputs per case ({', '.join(INPUTS)}), got {X.shape[1]}")
    if np.any(X[:, 7:] <= 0):
        raise ValueError("g_factor, L_x and L_y must be positive")
    X = X.copy()
    X[:, 7:] = np.log(X[:, 7:])
    return X


def as_array(inputs) -> np.ndarray:
    """One case as a dict, several as a list of dicts, or an array (n, 10) -> array (n, 10)."""
    if isinstance(inputs, dict):
        inputs = [inputs]
    if len(inputs) and isinstance(inputs[0], dict):
        missing = [k for k in INPUTS if k not in inputs[0]]
        if missing:
            raise ValueError(f"missing inputs: {', '.join(missing)}")
        return np.array([[float(case[k]) for k in INPUTS] for case in inputs])
    return np.array(inputs, dtype=float, ndmin=2)


def _unit(name):
    return "" if UNITS[name] == "-" else f" {UNITS[name]}"


def matern52(r):
    """Matern 5/2 correlation at scaled distance r: (1 + sqrt5 r + 5/3 r^2) exp(-sqrt5 r)."""
    s = np.sqrt(5.0) * r
    return (1.0 + s + s * s / 3.0) * np.exp(-s)


def _sq_dist(A, B):
    """Squared Euclidean distances between the rows of A (m, d) and B (n, d), via one matrix product."""
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d2, 0.0)


@dataclass
class Prediction:
    """Predicted impedance curves for n cases."""
    freqs: np.ndarray                 # (n_freq,) Hz
    terms: tuple                      # the 8 term names
    K: np.ndarray                     # (n, n_freq, n_terms) complex: Re = stiffness, Im = damping part
    static: np.ndarray                # (n, n_terms) static values (Re K at the lowest frequency)
    lo: np.ndarray | None = None      # (n, n_freq, n_terms) complex: lower band edge of Re and of Im
    hi: np.ndarray | None = None      # upper band edge
    gp_std: np.ndarray | None = None  # (n, n_gp) standard deviation of every GP output (log static, PCA weights)
    warnings: list = field(default_factory=list)   # one list of strings per case

    def case(self, i=0) -> dict:
        """Case i as plain lists, ready for JSON: {"freqs", "static", "terms": {name: {"re", "im", ...}}}."""
        out = {"freqs": self.freqs.tolist(), "static": dict(zip(self.terms, self.static[i].tolist())),
               "terms": {}, "warnings": list(self.warnings[i])}
        for k, t in enumerate(self.terms):
            d = {"re": self.K[i, :, k].real.tolist(), "im": self.K[i, :, k].imag.tolist()}
            if self.lo is not None:
                d.update(re_lo=self.lo[i, :, k].real.tolist(), re_hi=self.hi[i, :, k].real.tolist(),
                         im_lo=self.lo[i, :, k].imag.tolist(), im_hi=self.hi[i, :, k].imag.tolist())
            out["terms"][t] = d
        return out


class ImpedanceSurrogate:
    """Two-part GP surrogate (static x shape) evaluated with numpy. Build with export() or load()."""

    def __init__(self, arrays: dict, meta: dict):
        self.a = {k: np.asarray(v) for k, v in arrays.items()}
        self.a.setdefault("shape_resid_std", np.zeros(self.a["pca_mean"].shape))   # files exported before 6 Oct 2026 lack it
        self.meta = meta
        self.freqs = self.a["freqs"]
        self.terms = tuple(meta["terms"])
        self.n_terms = len(self.terms)
        self.n_freq = len(self.freqs)
        self.n_gp = len(self.a["alpha"])
        self.x_min, self.x_max = self.a["x_min"], self.a["x_max"]

    # ---------------------------------------------------------------- file
    def save(self, path=DEFAULT_MODEL):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=np.array(json.dumps(self.meta)), **self.a)
        return path

    @classmethod
    def load(cls, path=DEFAULT_MODEL):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found: train the release model first with "
                                    "`python -m surrogate train` (needs the dataset), or pass another file")
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("format") != FORMAT:
                raise ValueError(f"{path}: not an {FORMAT} file")
            return cls({k: z[k] for k in z.files if k != "meta"}, meta)

    # ---------------------------------------------------------------- prediction
    def _scaled(self, X_raw):
        return (features(X_raw) - self.a["x_mean"]) / self.a["x_scale"]

    def gp_outputs(self, X_raw, return_std=False, n_neighbours=N_NEIGHBOURS, chunk=2048):
        """The 28 GP outputs (log |static| per term, then the PCA weights of the shape), (n, n_gp),
        with the trend included; optionally their standard deviations (nearest-neighbour bound)."""
        Xs = self._scaled(X_raw)
        a = self.a
        mean = np.empty((len(Xs), self.n_gp))
        std = np.empty_like(mean) if return_std else None
        for lo in range(0, len(Xs), chunk):                       # bounded memory for large batches
            xs = Xs[lo:lo + chunk]
            for c in range(self.n_gp):
                ls = a["length_scale"][c]
                Xt = a["X_train"] / ls
                r2 = _sq_dist(xs / ls, Xt)
                k = a["signal_var"][c] * matern52(np.sqrt(r2))
                mean[lo:lo + chunk, c] = a["mean0"][c] + k @ a["alpha"][c]
                if return_std:
                    std[lo:lo + chunk, c] = self._nn_std(c, r2, k, Xt, n_neighbours)
        A1 = np.column_stack([np.ones(len(Xs)), Xs])
        mean = a["y_mean"] + a["y_std"] * mean + A1 @ a["trend_coef"]
        return (mean, std * a["y_std"]) if return_std else mean

    def _nn_std(self, c, r2, k, Xt, m):
        """Std of GP c (standardised units) conditioned on the m training points nearest to each input."""
        a = self.a
        sv, nz = a["signal_var"][c], a["noise"][c] + 1e-10
        if m >= len(Xt):                                                     # all points: the exact variance
            Kf = sv * matern52(np.sqrt(_sq_dist(Xt, Xt))) + nz * np.eye(len(Xt))
            v = np.linalg.solve(np.linalg.cholesky(Kf), k.T)
            return np.sqrt(np.maximum(sv - (v * v).sum(0), 0.0))
        idx = np.argpartition(r2, m - 1, axis=1)[:, :m]                    # (p, m) nearest points
        var = np.empty(len(idx))
        step = max(1, int(2e7 // (m * m)))                                   # bounded memory
        for s in range(0, len(idx), step):
            P = Xt[idx[s:s + step]]                                          # (q, m, d)
            sq = (P * P).sum(-1)
            d2 = np.maximum(sq[:, :, None] + sq[:, None, :] - 2 * np.einsum("qid,qjd->qij", P, P), 0.0)
            Ks = sv * matern52(np.sqrt(d2))
            Ks[:, np.arange(m), np.arange(m)] += nz
            ks = np.take_along_axis(k[s:s + step], idx[s:s + step], axis=1)[..., None]   # (q, m, 1)
            v = np.linalg.solve(Ks, ks)
            var[s:s + step] = sv - (ks * v).sum((1, 2))
        return np.sqrt(np.maximum(var, 0.0))

    def curves(self, G):
        """GP outputs (..., n_gp) -> K (..., n_freq, n_terms) complex and static values (..., n_terms)."""
        a = self.a
        lead = G.shape[:-1]
        G = G.reshape(-1, self.n_gp)
        s = a["sign"] * np.exp(G[:, :self.n_terms])
        rows = a["pca_mean"] + G[:, self.n_terms:] @ a["pca_components"]
        shape = rows.reshape(len(G), self.n_terms, 2, self.n_freq)
        shape = (shape[:, :, 0] + 1j * shape[:, :, 1]).transpose(0, 2, 1)   # (n, n_freq, n_terms)
        K = shape * s[:, None, :]
        return K.reshape(*lead, self.n_freq, self.n_terms), s.reshape(*lead, self.n_terms)

    def predict(self, inputs, band=False, level=95.0, n_samples=400, n_neighbours=N_NEIGHBOURS, seed=0) -> Prediction:
        """Impedance curves for one case (dict), several (list of dicts) or an array (n, 10).

        band=True adds a `level` % band: the 28 GP outputs are sampled from their predictive
        distributions (n_samples draws) and the band edges are percentiles of Re K and Im K."""
        X = as_array(inputs)
        if band:
            G, S = self.gp_outputs(X, return_std=True, n_neighbours=n_neighbours)
        else:
            G, S = self.gp_outputs(X), None
        K, s = self.curves(G)
        p = Prediction(self.freqs, self.terms, K, s, gp_std=S, warnings=self.range_warnings(X))
        if band:
            p.lo, p.hi = self.band_from_gp(G, S, level=level, n_samples=n_samples, seed=seed)
        return p

    def band_from_gp(self, G, S, level=95.0, n_samples=400, seed=0, truncation=True):
        """Band edges (lo, hi), each (n, n_freq, n_terms) complex, from GP means G and standard
        deviations S (n, n_gp): sample the GP outputs, add the PCA truncation term, take percentiles."""
        rng = np.random.default_rng(seed)
        draws = G[:, None, :] + S[:, None, :] * rng.standard_normal((len(G), n_samples, self.n_gp))
        a = self.a
        s = a["sign"] * np.exp(draws[..., :self.n_terms])                  # (n, n_samples, n_terms)
        rows = a["pca_mean"] + draws[..., self.n_terms:] @ a["pca_components"]
        if truncation:
            rows = rows + a["shape_resid_std"] * rng.standard_normal(rows.shape)
        sh = rows.reshape(len(G), n_samples, self.n_terms, 2, self.n_freq)
        Kd = (sh[..., 0, :] + 1j * sh[..., 1, :]).transpose(0, 1, 3, 2) * s[:, :, None, :]
        q = [(100 - level) / 2, (100 + level) / 2]
        re, im = np.percentile(Kd.real, q, axis=1), np.percentile(Kd.imag, q, axis=1)
        return re[0] + 1j * im[0], re[1] + 1j * im[1]

    def set_truncation_error(self, shapes):
        """Set shape_resid_std from training shape rows (n, n_columns), with this model's PCA basis.
        For files exported before the trained model recorded it (see release.calibrate)."""
        a = self.a
        A = (np.asarray(shapes) - a["pca_mean"]) @ a["pca_components"].T
        self.a["shape_resid_std"] = (shapes - (a["pca_mean"] + A @ a["pca_components"])).std(axis=0)
        return self

    def range_warnings(self, X_raw):
        """For each case, the inputs outside the training range (where the model extrapolates)."""
        X = as_array(X_raw)
        out = []
        for row in X:
            w = [f"{n} = {v:g} is outside the training range {lo:g} to {hi:g}{_unit(n)}"
                 for n, v, lo, hi in zip(INPUTS, row, self.x_min, self.x_max) if v < lo - 1e-9 or v > hi + 1e-9]
            out.append(w)
        return out

    def info(self) -> dict:
        """Model card: inputs with ranges, terms, frequencies, training and test summary."""
        inputs = [{"name": n, "unit": UNITS[n], "min": float(lo), "max": float(hi), "nominal": NOMINAL[n]}
                  for n, lo, hi in zip(INPUTS, self.x_min, self.x_max)]
        return {**self.meta, "inputs": inputs, "freqs": self.freqs.tolist()}


# ---------------------------------------------------------------------------- export
def _gp_parts(model):
    """Training inputs, standardised training outputs and hyperparameters of the model's GPs,
    from either engine: scikit-learn (StaticShapeGP) or GPyTorch (TorchStaticShapeGP)."""
    gps = model.gps_
    if isinstance(gps, list):                                   # scikit-learn: one GaussianProcessRegressor per output
        X = np.asarray(gps[0].X_train_, dtype=float)
        rows = []
        for gp in gps:
            k = gp.kernel_
            rows.append(dict(y=np.asarray(gp.y_train_, float).ravel(), y_mean=float(np.ravel(gp._y_train_mean)[0]),
                             y_std=float(np.ravel(gp._y_train_std)[0]), mean0=0.0,
                             signal_var=float(k.k1.k1.constant_value),
                             length_scale=np.broadcast_to(np.asarray(k.k1.k2.length_scale, float), X.shape[1]),
                             noise=float(k.k2.noise_level) + float(gp.alpha)))
        return X, rows, "scikit-learn"
    rows, X = [], None                                          # GPyTorch: batches of GPs (surrogate.gpu)
    for part in gps.parts_:
        m = part["model"]
        Xp = m.train_inputs[0][0].detach().double().cpu().numpy()
        X = Xp if X is None else X
        Y = m.train_targets.detach().double().cpu().numpy()                 # (b, n), standardised
        ls = m.covar_module.base_kernel.lengthscale.detach().double().cpu().numpy().reshape(len(Y), -1)
        sv = m.covar_module.outputscale.detach().double().cpu().numpy().reshape(-1)
        nz = m.likelihood.noise.detach().double().cpu().numpy().reshape(-1)
        c0 = m.mean_module.constant.detach().double().cpu().numpy().reshape(-1)
        ym = part["y_mean"].double().cpu().numpy().reshape(-1)
        ys = part["y_std"].double().cpu().numpy().reshape(-1)
        for j in range(len(Y)):
            rows.append(dict(y=Y[j], y_mean=ym[j], y_std=ys[j], mean0=c0[j], signal_var=sv[j],
                             length_scale=ls[j], noise=nz[j]))
    return X, rows, "GPyTorch"


def _alpha(X, r):
    """alpha = (K + noise I)^-1 (y - mean0) in double precision, by Cholesky factorisation."""
    Xt = X / r["length_scale"]
    K = r["signal_var"] * matern52(np.sqrt(_sq_dist(Xt, Xt)))
    jitter = 0.0
    for _ in range(6):
        try:
            L = np.linalg.cholesky(K + (r["noise"] + jitter) * np.eye(len(X)))
            break
        except np.linalg.LinAlgError:
            jitter = max(1e-10, 10 * jitter)
    else:
        raise np.linalg.LinAlgError("kernel matrix not positive definite even with jitter")
    z = np.linalg.solve(L, r["y"] - r["mean0"])
    return np.linalg.solve(L.T, z)


def export(model, freqs, meta=None, X_check=None) -> ImpedanceSurrogate:
    """Extract a fitted StaticShapeGP / TorchStaticShapeGP (trained on input_features(ds, footprint=True))
    into an ImpedanceSurrogate. X_check: raw inputs (n, 10) on which to compare the two."""
    X, rows, engine = _gp_parts(model)
    spec = model.spec
    n_gp = len(rows)
    d = X.shape[1]
    trend = model.trend_coef_ if model.trend_coef_ is not None else np.zeros((d + 1, n_gp))
    raw = X * model.x_scaler_.scale_ + model.x_scaler_.mean_                 # training features
    raw[:, 7:] = np.exp(raw[:, 7:])                                          # back to g, L_x, L_y
    footprints = sorted({(round(float(lx), 3), round(float(ly), 3)) for lx, ly in raw[:, 8:10]})
    arrays = dict(
        freqs=np.asarray(freqs, float), x_mean=model.x_scaler_.mean_, x_scale=model.x_scaler_.scale_,
        X_train=X, alpha=np.array([_alpha(X, r) for r in rows]),
        length_scale=np.array([r["length_scale"] for r in rows]),
        signal_var=np.array([r["signal_var"] for r in rows]), noise=np.array([r["noise"] for r in rows]),
        mean0=np.array([r["mean0"] for r in rows]), y_mean=np.array([r["y_mean"] for r in rows]),
        y_std=np.array([r["y_std"] for r in rows]), trend_coef=np.asarray(trend, float),
        sign=model.sign_, pca_mean=model.pca_.mean_, pca_components=model.pca_.components_,
        shape_resid_std=getattr(model, "shape_resid_std_", np.zeros(model.pca_.mean_.shape)),
        x_min=raw.min(0), x_max=raw.max(0),
    )
    info = dict(format=FORMAT, terms=list(spec.terms), term_ij=[list(ij) for ij in spec.ij], engine=engine,
                n_train=int(len(X)), n_gp=n_gp, n_components=int(model.pca_.n_components_),
                trend=model.trend, footprints_trained=[list(f) for f in footprints],
                representation="K_t(f) = static_t x shape_t(f); log|static| and PCA weights of the shape, one GP each",
                kernel="Matern 5/2, one length scale per input")
    info.update(meta or {})
    sur = ImpedanceSurrogate(arrays, info)
    if X_check is not None:
        from .data import targets_to_K
        K_ref = targets_to_K(model.predict(features(X_check)), spec)
        K_new = sur.predict(X_check).K
        dev = np.abs(K_new - K_ref).max(axis=1) / np.abs(K_ref).max(axis=1)
        sur.meta["export_check_max_dev_pct"] = float(100 * dev.max())
    return sur
