"""Surrogate models. Every model has the scikit-learn interface:

    model.fit(X, Y)            X: (n, n_inputs) inputs, Y: (n, n_columns) target rows (make_targets)
    model.predict(X) -> Y      in the same physical units as the training targets

so they can be swapped, cross-validated and timed the same way.

    from surrogate.models import input_features, knn_model, PCAGP
    X, names = input_features(ds)             # 7 thicknesses + log(g) (+ log L_x, log L_y if asked)
    model = PCAGP(spec, n_components=40).fit(X_train, Y_train)
    Y_pred, Y_std = model.predict(X_test, return_std=True)
"""
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Dataset, TargetSpec, targets_to_K, terms_to_targets

SOIL_INPUTS = ["t_L1", "t_L2", "t_L3", "t_L4", "t_L5", "t_L6", "t_L7", "log_g"]


def input_features(ds: Dataset, footprint=False, derived=False):
    """Model inputs: the 7 thicknesses and log(g_factor); with footprint=True also log L_x, log L_y;
    with derived=True also log f0 and log H_bedrock (computed from the soil inputs, notebook 01 §5).

    g was sampled log-uniformly (x1.5 and /1.5 equally likely), so log(g) is the natural scale:
    equal steps in log(g) are equally likely, and distances in input space treat them alike.
    The footprint enters as logarithms because stiffness grows as a power of size (roughly L for
    translation, L^3 for rocking): in log space those power laws become straight lines.
    Returns (X, names).
    """
    X = ds.X[:, :8].copy()
    X[:, 7] = np.log(X[:, 7])
    names = list(SOIL_INPUTS)
    if footprint:
        X = np.hstack([X, np.log(ds.X[:, 8:10])])
        names += ["log_Lx", "log_Ly"]
    if derived:
        X = np.hstack([X, np.log(ds.D[:, [2, 0]])])           # f0, H_bedrock
        names += ["log_f0", "log_H"]
    return X, names


class TermScaler(TransformerMixin, BaseEstimator):
    """Divide each term's columns by one constant: the training median of that term's max |K|.

    Brings rocking (~1e8) and translations (~1e5) to the same order, so that a squared-error fit
    does not spend all its effort on rocking. One constant per term keeps each curve's shape and
    the differences between cases intact.
    """

    def __init__(self, spec: TargetSpec):
        self.spec = spec

    def fit(self, Y, y=None):
        K8 = targets_to_K(Y, self.spec)                             # (n, n_freq, n_terms)
        self.term_scale_ = np.median(np.abs(K8).max(axis=1), axis=0)
        self.col_scale_ = np.repeat(self.term_scale_, 2 * self.spec.n_freq)
        return self

    def transform(self, Y):
        return np.asarray(Y) / self.col_scale_

    def inverse_transform(self, Z):
        return np.asarray(Z) * self.col_scale_


def knn_model(spec: TargetSpec, k=5):
    """k-nearest neighbours: the prediction is a distance-weighted average of the k training
    curves whose inputs are closest (inputs standardised first, targets scaled per term)."""
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k, weights="distance")),
        transformer=TermScaler(spec), check_inverse=False)


def _fit_gp(kernel, X, y, n_restarts, random_state):
    # Optimiser warnings ("length scale close to the upper bound", "lbfgs failed to converge")
    # are silenced here: with dozens of GPs they would flood the output. A length scale at its
    # upper bound is information, not a fault (the input barely matters); length_scales() shows it.
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return gp.fit(X, y)


class _TrendMixin:
    """Optional linear trend under the GPs ("universal kriging", kriging with a drift).

    A plain GP reverts to the mean of its training data away from it, which is wrong when a
    target follows a clear trend: log static stiffness grows almost linearly with log L, so for
    a footprint larger than any in training the GP alone would predict a stiffness that is far
    too low. With a trend, each chosen target is first fitted by least squares as a linear
    function of the (standardised) inputs, and the GP models only what is left. Inside the data
    the GP corrects the trend; outside it the trend carries the prediction.
    """

    def _fit_targets(self, Xs, T, trend_cols=()):
        T = np.asarray(T, dtype=float)
        self.trend_coef_ = None
        trend_cols = list(trend_cols)
        if trend_cols:
            A = np.column_stack([np.ones(len(Xs)), Xs])
            coef = np.zeros((A.shape[1], T.shape[1]))
            coef[:, trend_cols] = np.linalg.lstsq(A, T[:, trend_cols], rcond=None)[0]
            self.trend_coef_ = coef
            T = T - A @ coef
        self.gps_ = self._fit_gps(Xs, T)

    def _predict_targets(self, Xs, return_std=False):
        P, S = self._predict_gps(Xs, return_std)
        if self.trend_coef_ is not None:
            P = P + np.column_stack([np.ones(len(Xs)), Xs]) @ self.trend_coef_
        return P, S


class PCAGP(_TrendMixin, RegressorMixin, BaseEstimator):
    """PCA of the scaled curves, then one Gaussian process per PCA coefficient.

    fit:      inputs -> standardised;  targets -> TermScaler -> PCA (n_components coefficients)
              one GP (Matern 5/2 kernel, one length scale per input, plus a small white-noise
              term) is fitted to each coefficient; n_jobs fits several GPs at once.
    predict:  GP means -> inverse PCA -> inverse scaling. return_std=True also returns the
              standard deviation of each target value, propagated from the GPs through the PCA
              basis (it does not include the PCA truncation error).
    """

    def __init__(self, spec: TargetSpec, n_components=40, nu=2.5, n_restarts=0, n_jobs=None,
                 random_state=0, trend=None):
        self.spec = spec
        self.n_components = n_components
        self.nu = nu
        self.n_restarts = n_restarts
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.trend = trend            # None, or "all": a linear trend under every coefficient's GP

    def _kernel(self, n_inputs):
        return (ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=np.ones(n_inputs), length_scale_bounds=(1e-2, 1e3), nu=self.nu)
                + WhiteKernel(1e-6, (1e-10, 1e-1)))

    def fit(self, X, Y):
        self.x_scaler_ = StandardScaler().fit(X)
        self.y_scaler_ = TermScaler(self.spec).fit(Y)
        Z = self.y_scaler_.transform(Y)
        n = min(self.n_components, len(X), Z.shape[1])      # PCA cannot have more components than cases
        self.pca_ = PCA(n_components=n, random_state=self.random_state).fit(Z)
        A = self.pca_.transform(Z)                           # (n_cases, n) coefficients
        self._fit_targets(self.x_scaler_.transform(X), A, range(n) if self.trend == "all" else ())
        return self

    # --- the GP engine: one independent GP per column of T. Overridden in surrogate.gpu. ---
    def _fit_gps(self, Xs, T):
        return Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_gp)(self._kernel(Xs.shape[1]), Xs, T[:, c], self.n_restarts, self.random_state)
            for c in range(T.shape[1]))

    def _predict_gps(self, Xs, return_std=False):
        out = [gp.predict(Xs, return_std=return_std) for gp in self.gps_]
        if not return_std:
            return np.column_stack(out), None
        return np.column_stack([m for m, _ in out]), np.column_stack([s for _, s in out])

    def _length_scales(self):
        return np.array([gp.kernel_.k1.k2.length_scale for gp in self.gps_])

    def predict(self, X, return_std=False):
        A, S = self._predict_targets(self.x_scaler_.transform(X), return_std)
        Y = self.y_scaler_.inverse_transform(self.pca_.inverse_transform(A))
        if not return_std:
            return Y
        # independent coefficients: var(z_j) = sum_c std_c^2 * phi_cj^2, then undo the scaling
        var_z = (S ** 2) @ (self.pca_.components_ ** 2)
        return Y, np.sqrt(var_z) * self.y_scaler_.col_scale_

    def length_scales(self):
        """(n_components, n_inputs): the fitted length scale of every input, for every GP.

        In standardised input units: a large value means the coefficient barely changes along
        that input (automatic relevance determination)."""
        return self._length_scales()


class StaticShapeGP(_TrendMixin, RegressorMixin, BaseEstimator):
    """Two-part surrogate: the static value of each term, and the curve's shape relative to it.

        K_t(f) = s_t * shape_t(f),   s_t = Re K_t at the lowest frequency (0.45 Hz)

    - log |s_t|: one GP per term. Static stiffness follows power laws in footprint size and soil
      stiffness, which are close to linear in log space (log L, log g), so this part is smooth.
    - shape_t(f) = K_t(f) / s_t: dimensionless (= 1 + small imaginary part at 0.45 Hz), similar in
      size for every term and footprint; PCA + one GP per coefficient, as in PCAGP.

    This is the same split as the classical dynamic-stiffness form K = K_static * (k + i a0 c):
    the footprint mostly scales the curve, and the shape varies more gently.
    The sign of each s_t is taken from the training data (it must not change within a term).

    trend: None; "static" (a linear trend under the log-static GPs, so that the static stiffness
    of a footprint outside the training range follows the power law); or "all" (also under the
    shape coefficients).
    """

    def __init__(self, spec: TargetSpec, n_components=20, nu=2.5, n_restarts=0, n_jobs=None,
                 random_state=0, trend=None):
        self.spec = spec
        self.n_components = n_components
        self.nu = nu
        self.n_restarts = n_restarts
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.trend = trend

    _kernel = PCAGP._kernel
    _fit_gps = PCAGP._fit_gps
    _predict_gps = PCAGP._predict_gps
    _length_scales = PCAGP._length_scales

    def fit(self, X, Y):
        K8 = targets_to_K(Y, self.spec)                          # (n, n_freq, n_terms)
        s = K8[:, 0, :].real                                     # static value per case and term
        self.sign_ = np.sign(np.median(s, axis=0))
        if np.any(np.sign(s) != self.sign_):
            raise ValueError("a static value changes sign within a term")
        shape = terms_to_targets(K8 / s[:, None, :])             # dimensionless rows
        self.x_scaler_ = StandardScaler().fit(X)
        Xs = self.x_scaler_.transform(X)
        n = min(self.n_components, len(X), shape.shape[1])
        self.pca_ = PCA(n_components=n, random_state=self.random_state).fit(shape)
        A = self.pca_.transform(shape)
        # what the n components leave out, per value of the shape row (PCA truncation error): used
        # by the portable model's uncertainty band
        self.shape_resid_std_ = (shape - self.pca_.inverse_transform(A)).std(axis=0)
        self.n_terms_ = s.shape[1]
        T = np.column_stack([np.log(np.abs(s)), A])              # log statics, then shape coefficients
        trend_cols = {None: (), "static": range(self.n_terms_), "all": range(T.shape[1])}[self.trend]
        self._fit_targets(Xs, T, trend_cols)
        return self

    def _predict_parts(self, X):
        P, _ = self._predict_targets(self.x_scaler_.transform(X))
        return self.sign_ * np.exp(P[:, :self.n_terms_]), P[:, self.n_terms_:]

    def predict_static(self, X):
        """Predicted static values s_t, (n, n_terms)."""
        return self._predict_parts(X)[0]

    def predict(self, X):
        s, A = self._predict_parts(X)
        shape = targets_to_K(self.pca_.inverse_transform(A), self.spec)
        return terms_to_targets(shape * s[:, None, :])

    def length_scales(self):
        """(n_terms + n_components, n_inputs): length scales of the static GPs, then the shape GPs."""
        return self._length_scales()
