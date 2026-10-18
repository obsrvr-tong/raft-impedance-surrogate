"""The same surrogates with the Gaussian processes trained on the GPU (PyTorch + GPyTorch).

    from surrogate.gpu import TorchStaticShapeGP, device_info
    print(device_info())                         # e.g. "cuda: NVIDIA GeForce RTX 5070 Ti"
    model = TorchStaticShapeGP(spec, n_components=20).fit(X, Y)      # same interface as StaticShapeGP
    print(model.fit_report())                    # precision, noise floor, numerical retries

Why: an exact GP needs a Cholesky factorisation of an n x n matrix, so its cost grows as n^3.
At 1,000 training cases scikit-learn is fine; at 2,500-5,000 (all footprints, all soils) each
fit takes many minutes on the CPU. Here the independent GPs (one per PCA coefficient and per
static term) are stacked into a *batch* and trained together, as a few large matrix operations
that a GPU executes in parallel.

Differences from the scikit-learn engine: the hyperparameters are found with the Adam optimiser
over a fixed number of steps (n_iter) instead of L-BFGS until convergence, and the arithmetic is
in single precision (float32: consumer GPUs are ~60x slower in double precision).

Single precision and the noise floor. A float32 Cholesky factorisation fails when the matrix is
too ill-conditioned: roughly when cond = lambda_max / lambda_min exceeds 10^7. On dataset v1
the largest eigenvalue of the kernel matrix is close to n (long length scales: most cases are
strongly correlated), and the smallest is the noise variance, which the GPs push to its lower
bound because MISS3D is deterministic. So the noise floor must grow with n:
noise >= 2e-7 n (and >= 1e-4) on standardised outputs, i.e. 1e-4 up to 500 cases and 1e-3 at
5,120 (float32_noise_floor). A test at 320 cases gave the same accuracy with floors 1e-5 and
1e-4, and in double precision. If a factorisation still fails, that group
of GPs is refitted in double precision automatically, and fit_report() says so. GPyTorch's own
fallback (adding "jitter" to the diagonal, with a NumericalWarning) is counted, not printed.
If the GPU runs out of memory, set max_batch to train the GPs in smaller groups.
Needs: pip install gpytorch
"""
import contextlib
import re
import warnings

import numpy as np
import torch
import gpytorch
from linear_operator.utils.errors import NanError, NotPSDError
from linear_operator.utils.warnings import NumericalWarning

from .models import PCAGP, StaticShapeGP


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_info():
    d = default_device()
    return f"{d}: {torch.cuda.get_device_name(0)}" if d.type == "cuda" else "cpu (no CUDA device found)"


def float32_noise_floor(n):
    """Smallest noise variance (standardised outputs) that keeps a float32 Cholesky of n cases reliable."""
    return max(1e-4, 2e-7 * n)


@contextlib.contextmanager
def _full_precision_matmul():
    """Matrix products in true float32 (no TF32 shortcut), restored afterwards."""
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(old)


@contextlib.contextmanager
def _count_jitter(report):
    """Record GPyTorch's NumericalWarnings ("added jitter") in report instead of printing them;
    any other warning is passed on."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for w in caught:
        if issubclass(w.category, NumericalWarning):
            report["jitter_events"] += 1
            m = re.search(r"jitter of ([0-9.eE+-]+)", str(w.message))
            if m:
                report["max_jitter"] = max(report["max_jitter"], float(m.group(1)))
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


class _BatchGP(gpytorch.models.ExactGP):
    """b independent GPs sharing the training inputs: constant mean, scaled Matern ARD kernel."""

    def __init__(self, X, Y, likelihood, nu):
        super().__init__(X, Y, likelihood)
        batch = torch.Size([Y.shape[0]])
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=X.shape[-1], batch_shape=batch),
            batch_shape=batch)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class TorchGPEngine:
    """Fit/predict one independent GP per column of T, trained together in batches of max_batch.

    noise_floor: lower bound of the noise variance on standardised outputs; None chooses
    float32_noise_floor(n) in single precision and 1e-6 in double precision."""

    def __init__(self, nu=2.5, n_iter=1000, lr=0.1, device=None, dtype=torch.float32, noise_floor=None,
                 max_batch=None):
        self.nu, self.n_iter, self.lr = nu, n_iter, lr
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype, self.noise_floor, self.max_batch = dtype, noise_floor, max_batch

    def _floor(self, n, dtype):
        if self.noise_floor is not None:
            return float(self.noise_floor)
        return float32_noise_floor(n) if dtype == torch.float32 else 1e-6

    def _fit_part(self, X, T, dtype):
        X = X.to(dtype)
        Y = torch.as_tensor(np.asarray(T).T, dtype=dtype, device=self.device)   # (b, n)
        y_mean, y_std = Y.mean(1, keepdim=True), Y.std(1, keepdim=True)
        Yn = (Y - y_mean) / y_std                                # each output standardised
        b, n = Yn.shape
        floor = self._floor(n, dtype)
        Xb = X.unsqueeze(0).expand(b, n, X.shape[1])             # the same inputs for every GP
        lik = gpytorch.likelihoods.GaussianLikelihood(
            batch_shape=torch.Size([b]), noise_constraint=gpytorch.constraints.GreaterThan(floor))
        model = _BatchGP(Xb, Yn, lik, self.nu).to(device=self.device, dtype=dtype)
        model.likelihood.noise = max(1e-3, 2 * floor)
        model.covar_module.base_kernel.lengthscale = 1.0
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        losses = []
        with gpytorch.settings.max_cholesky_size(100_000):      # exact Cholesky, no iterative solver
            for _ in range(self.n_iter):
                opt.zero_grad()
                loss = -mll(model(Xb), Yn).sum()
                loss.backward()
                opt.step()
                losses.append(loss.item() / b)
        model.eval()
        return {"model": model, "y_mean": y_mean, "y_std": y_std, "loss": losses, "dtype": dtype,
                "noise_floor": floor}

    def fit(self, Xs, T):
        X = torch.as_tensor(np.asarray(Xs, dtype=np.float64), device=self.device)
        T = np.asarray(T)
        step = self.max_batch or T.shape[1]
        self.report_ = {"precision": str(self.dtype).replace("torch.", ""), "noise_floor": None,
                        "jitter_events": 0, "max_jitter": 0.0, "refit_in_float64": 0, "predict_in_float64": 0}
        self.parts_ = []
        with _full_precision_matmul(), _count_jitter(self.report_):
            for i in range(0, T.shape[1], step):
                try:
                    part = self._fit_part(X, T[:, i:i + step], self.dtype)
                except (NotPSDError, NanError):
                    if self.dtype == torch.float64:
                        raise
                    self.report_["refit_in_float64"] += 1           # this group again, in double precision
                    part = self._fit_part(X, T[:, i:i + step], torch.float64)
                self.parts_.append(part)
        self.report_["noise_floor"] = max(p["noise_floor"] for p in self.parts_)
        self.loss_ = np.mean([p["loss"] for p in self.parts_], axis=0)    # mean loss per GP, per step
        return self

    def _predict_part(self, p, X, return_std):
        b = p["y_mean"].shape[0]
        Xp = X.to(p["dtype"])
        post = p["model"](Xp.unsqueeze(0).expand(b, *Xp.shape))         # latent function, no noise
        mean = (post.mean * p["y_std"] + p["y_mean"]).T.double().cpu().numpy()
        std = (post.variance.clamp_min(0).sqrt() * p["y_std"]).T.double().cpu().numpy() if return_std else None
        return mean, std

    def predict(self, Xs, return_std=False):
        X = torch.as_tensor(np.asarray(Xs, dtype=np.float64), device=self.device)
        report = getattr(self, "report_", {"jitter_events": 0, "max_jitter": 0.0, "predict_in_float64": 0})
        means, stds = [], []
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(100_000), _full_precision_matmul(), \
                _count_jitter(report):
            for p in self.parts_:
                try:
                    m, s = self._predict_part(p, X, return_std)
                except (NotPSDError, NanError):
                    if p["dtype"] == torch.float64:
                        raise
                    p["model"] = p["model"].double()                # same hyperparameters, double precision
                    p["y_mean"], p["y_std"], p["dtype"] = p["y_mean"].double(), p["y_std"].double(), torch.float64
                    report["predict_in_float64"] += 1
                    m, s = self._predict_part(p, X, return_std)
                means.append(m)
                stds.append(s)
        return np.hstack(means), (np.hstack(stds) if return_std else None)

    def length_scales(self):
        return np.vstack([p["model"].covar_module.base_kernel.lengthscale.detach().squeeze(1).double().cpu().numpy()
                          for p in self.parts_])


class _TorchHooks:
    """Replaces the scikit-learn GP engine of PCAGP / StaticShapeGP with TorchGPEngine."""

    def _fit_gps(self, Xs, T):
        return TorchGPEngine(nu=self.nu, n_iter=self.n_iter, lr=self.lr, device=self.device,
                             dtype=torch.float64 if self.double else torch.float32,
                             noise_floor=self.noise_floor, max_batch=self.max_batch).fit(Xs, T)

    def _predict_gps(self, Xs, return_std=False):
        return self.gps_.predict(Xs, return_std)

    def _length_scales(self):
        return self.gps_.length_scales()

    def fit_report(self):
        """Precision, noise floor, GPyTorch jitter events and double-precision fallbacks of the last fit."""
        return dict(self.gps_.report_)


class TorchPCAGP(_TorchHooks, PCAGP):
    """PCAGP with the GPs trained in batches on the GPU."""

    def __init__(self, spec, n_components=40, nu=2.5, n_iter=1000, lr=0.1, device=None, max_batch=None,
                 random_state=0, trend=None, noise_floor=None, double=False):
        self.spec, self.n_components, self.nu = spec, n_components, nu
        self.n_iter, self.lr, self.device = n_iter, lr, device
        self.max_batch, self.random_state, self.trend = max_batch, random_state, trend
        self.noise_floor, self.double = noise_floor, double


class TorchStaticShapeGP(_TorchHooks, StaticShapeGP):
    """StaticShapeGP with the GPs trained in batches on the GPU."""

    def __init__(self, spec, n_components=20, nu=2.5, n_iter=1000, lr=0.1, device=None, max_batch=None,
                 random_state=0, trend=None, noise_floor=None, double=False):
        self.spec, self.n_components, self.nu = spec, n_components, nu
        self.n_iter, self.lr, self.device = n_iter, lr, device
        self.max_batch, self.random_state, self.trend = max_batch, random_state, trend
        self.noise_floor, self.double = noise_floor, double


def format_report(r):
    """One line for notebooks: precision, noise floor, numerical events."""
    s = f"{r['precision']}, noise floor {r['noise_floor']:.1e}"
    if r["jitter_events"]:
        s += f", jitter added {r['jitter_events']}x (max {r['max_jitter']:.0e})"
    if r["refit_in_float64"]:
        s += f", {r['refit_in_float64']} group(s) refitted in float64"
    if r["predict_in_float64"]:
        s += f", {r['predict_in_float64']} group(s) predicted in float64"
    return s
