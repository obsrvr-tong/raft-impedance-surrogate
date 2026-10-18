"""Train, test and save the release model (Milestone 5).

    python -m surrogate train                      # GPU if available: 512 soils x 5 footprints + rotated twins
    python -m surrogate train --soils 128 --engine cpu --out ml/models/small.npz

The recipe is the one notebooks 03 and 04 settled on: two-part model (log static value + PCA
of the shape, 20 components), a linear trend under every GP, rotated twins of every footprint.
The trained model is exported to the portable format (surrogate.portable), tested on the 160
hold-out simulations (32 soils per footprint, never used in training), and saved with its test
results in the file, so that the model card travels with the model.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import time
from pathlib import Path

import numpy as np

from .data import DEFAULT_NPZ, concat, load_dataset, make_targets, rotate_90, targets_to_K, terms_to_targets
from .experiments import git_commit
from .metrics import summarize, target_errors
from .models import StaticShapeGP, input_features
from .portable import DEFAULT_MODEL, N_NEIGHBOURS, export


def _sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_engine(engine="auto"):
    """'gpu' if PyTorch sees a CUDA device (for 'auto'), else 'cpu' (scikit-learn)."""
    if engine != "auto":
        return engine
    try:
        import torch
        return "gpu" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def coverage(sur, X, K_true, level=95.0, n_neighbours=N_NEIGHBOURS, truncation=True):
    """Share of true Re and Im values that fall inside the predicted band (all terms, frequencies).
    truncation=False leaves out the PCA truncation term (the GPs' uncertainty only)."""
    G, S = sur.gp_outputs(X, return_std=True, n_neighbours=n_neighbours)
    lo, hi = sur.band_from_gp(G, S, level=level, truncation=truncation)     # as in sur.predict(band=True)
    inside = lambda t, lo, hi: (t >= lo - 1e-12 * abs(t)) & (t <= hi + 1e-12 * abs(t))
    ok_re = inside(K_true.real, lo.real, hi.real)
    ok_im = inside(K_true.imag, lo.imag, hi.imag)
    return {"all": float(np.mean([ok_re.mean(), ok_im.mean()])),
            **{t: float(np.mean([ok_re[..., k].mean(), ok_im[..., k].mean()])) for k, t in enumerate(sur.terms)}}


def train_release(n_soils=512, engine="auto", n_iter=1000, max_batch=None, n_jobs=None, rotate=True,
                  trend="all", n_components=20, validation_soils=None, ds=None, data_path=DEFAULT_NPZ,
                  verbose=True):
    """Fit the release recipe on the first n_soils training soils of every footprint, export it,
    test it on the hold-out cases. Returns (ImpedanceSurrogate, report dict).

    validation_soils=v tests on the next v training soils of every footprint instead, so that
    settings (e.g. n_components) can be compared without touching the hold-out cases."""
    say = print if verbose else (lambda *a, **k: None)
    ds = load_dataset(data_path) if ds is None else ds
    pool = ds.select(split="train")
    train = pool.head_per_mesh(n_soils)
    if validation_soils:
        bigger = pool.head_per_mesh(n_soils + validation_soils)
        test = bigger.take(~np.isin(bigger.case, train.case))
        if len(test) < validation_soils * len(np.unique(ds.mesh_id)):
            raise ValueError(f"not enough unused training soils for {validation_soils} per footprint")
    else:
        test = None
    if rotate:
        train = concat(train, rotate_90(train))
    X, _ = input_features(train, footprint=True)
    Y, spec = make_targets(train.K)
    engine = resolve_engine(engine)
    say(f"training on {len(X)} cases ({n_soils} soils x {len(np.unique(ds.mesh_id))} footprints"
        f"{' + rotated twins' if rotate else ''}), engine: {engine}")

    t0 = time.perf_counter()
    if engine == "gpu":
        from .gpu import TorchStaticShapeGP, device_info, format_report
        device = device_info()
        mb = max_batch or (7 if len(X) > 3000 else None)
        model = TorchStaticShapeGP(spec, n_components=n_components, n_iter=n_iter, max_batch=mb, trend=trend)
        recipe = f"GPyTorch batch GPs, Adam {n_iter} steps"
    else:
        import os
        device = f"cpu ({os.cpu_count()} logical cores)"
        model = StaticShapeGP(spec, n_components=n_components, n_jobs=n_jobs, trend=trend)
        recipe = "scikit-learn GPs, L-BFGS"
    model.fit(X, Y)
    fit_s = time.perf_counter() - t0
    say(f"fitted in {fit_s:.0f} s on {device}; exporting")
    training = dict(n_soils_per_footprint=n_soils, rotated_twins=rotate, n_cases=int(len(X)),
                    engine=recipe, device=device, fit_s=round(fit_s, 1))
    if engine == "gpu":
        training["gpu_fit"] = model.fit_report()          # precision, noise floor, numerical fallbacks
        say("GPU fit:", format_report(training["gpu_fit"]))
    training["n_components"] = int(model.pca_.n_components_)
    return package(model, ds, training, data_path=data_path, verbose=verbose, test=test,
                   test_kind="validation" if validation_soils else "holdout")


def package(model, ds, training, name="impedance_surrogate_v1", data_path=DEFAULT_NPZ, verbose=True,
            test=None, test_kind="holdout"):
    """Export a fitted model, test the exported model on the hold-out cases of ds (or on `test`)
    and store the results in its model card. Returns (ImpedanceSurrogate, report dict)."""
    say = print if verbose else (lambda *a, **k: None)
    test = ds.select(split="test") if test is None else test
    spec = model.spec
    t0 = time.perf_counter()
    meta = dict(
        name=name, created=_dt.date.today().isoformat(), git_commit=git_commit(),
        dataset=Path(data_path).name,
        dataset_sha256=_sha256(data_path) if Path(data_path).exists() else None,
        band_hz=[float(ds.freqs[0]), float(ds.freqs[-1])], training=training,
    )
    sur = export(model, ds.freqs, meta=meta, X_check=test.X[:32])
    say(f"exported in {time.perf_counter() - t0:.0f} s "
        f"(max deviation from the trained model: {sur.meta['export_check_max_dev_pct']:.2g} %)")

    # hold-out test with the exported model, i.e. exactly what users will get
    Y_t, _ = make_targets(test.K)
    t0 = time.perf_counter()
    K_p = sur.predict(test.X).K
    pred_ms = 1000 * (time.perf_counter() - t0) / len(test)
    errors = target_errors(Y_t, terms_to_targets(K_p), spec)
    s = summarize(errors, list(spec.terms))
    cov = coverage(sur, test.X, targets_to_K(Y_t, spec))
    per_mesh = {str(m): round(float(errors[test.mesh_id == m].mean()), 3) for m in np.unique(test.mesh_id)}
    sur.meta["band"] = ("GP variance from the 128 nearest training cases (an upper bound) "
                        "+ PCA truncation error estimated on the training shapes")
    sur.meta[test_kind] = dict(
        n_cases=int(len(test)),
        description=("32 random soils per footprint, never used in training" if test_kind == "holdout"
                     else "training soils beyond those used, per footprint: for choosing settings"),
        metric="e = 100 max_f|dK| / max_f|K| per term; err_<term> = 95th percentile over cases",
        err_mean=round(s["err_mean"], 3), err_worst=round(s["err_worst"], 3),
        err_95th={t: round(s[f"err_{t}"], 3) for t in spec.terms}, err_mean_per_footprint=per_mesh,
        band95_coverage=round(cov["all"], 4), predict_ms_per_case=round(pred_ms, 3))
    say(f"{'hold-out' if test_kind == 'holdout' else 'validation'} ({len(test)} cases): "
        f"mean error {s['err_mean']:.2f} %, worst term {s['err_worst']:.2f} % "
        f"(95th percentile), 95 % band covers {100 * cov['all']:.1f} % of true values; "
        f"{pred_ms:.2f} ms per case")
    return sur, dict(summary=s, errors=errors, fit_s=training.get("fit_s"), pred_ms=pred_ms, coverage=cov,
                     model=model, n_train=training.get("n_cases"), n_test=len(test), recipe=training.get("engine"))


def training_shapes(sur, ds):
    """The training shape rows of a saved release model, rebuilt from its model card and the dataset."""
    info = sur.meta["training"]
    train = ds.select(split="train").head_per_mesh(info["n_soils_per_footprint"])
    if info["rotated_twins"]:
        train = concat(train, rotate_90(train))
    K8 = targets_to_K(*make_targets(train.K))
    s = K8[:, 0, :].real
    shapes = terms_to_targets(K8 / s[:, None, :])
    if not np.allclose(shapes.mean(axis=0), sur.a["pca_mean"], rtol=0, atol=1e-8):
        raise ValueError("the dataset's training cases do not match this model's PCA (different data or settings)")
    return shapes


def calibrate(model_path=DEFAULT_MODEL, data_path=DEFAULT_NPZ, log=False, verbose=True):
    """Add the PCA truncation error to the band of a model saved before 6 October 2026, re-test the
    band on the hold-out cases, and save the model. Seconds; no retraining. Safe to repeat: both
    coverages are recomputed from the model each time."""
    from .portable import ImpedanceSurrogate
    say = print if verbose else (lambda *a, **k: None)
    sur = ImpedanceSurrogate.load(model_path)
    ds = load_dataset(data_path)
    h = sur.meta["holdout"]
    sur.set_truncation_error(training_shapes(sur, ds))
    test = ds.select(split="test")
    Y_t, spec = make_targets(test.K)
    K_t = targets_to_K(Y_t, spec)
    cov = coverage(sur, test.X, K_t)
    before = coverage(sur, test.X, K_t, truncation=False)["all"]
    h["band95_coverage_gp_only"] = round(before, 4)
    h["band95_coverage"] = round(cov["all"], 4)
    sur.meta["band"] = ("GP variance from the 128 nearest training cases (an upper bound) "
                        "+ PCA truncation error estimated on the training shapes")
    sur.meta["calibrated"] = _dt.date.today().isoformat()
    sur.save(model_path)
    say(f"95 % band on the {len(test)} hold-out cases: holds {100 * cov['all']:.1f} % of the simulated values "
        f"(GP uncertainty only: {100 * before:.1f} %); saved {model_path}")
    if log:
        import csv
        from .experiments import COLUMNS, RESULTS, log_experiment
        with open(RESULTS, newline="", encoding="utf-8") as f:
            row = next((r for r in csv.DictReader(f) if r["exp_id"] == "M5-release"), None)
        if row is None:
            say("no M5-release row in the experiment table; nothing logged")
        else:
            fields = {c: row[c] for c in COLUMNS if c not in ("date", "exp_id", "git_commit")}
            fields["notes"] = (f"release model; 95 % band coverage {100 * cov['all']:.1f} % with the PCA "
                               f"truncation term ({100 * before:.1f} % without)")
            log_experiment("M5-release", **fields)
            say("row M5-release updated in ml/results/experiments.csv")
    return sur, cov


def log_release(report, exp_id="M5-release", band=None):
    """One row in results/experiments.csv for the release model."""
    from .experiments import log_experiment
    log_experiment(exp_id, milestone="M5", footprints="all 5", validation="hold-out 32 soils per footprint",
                   band_hz=band or "0.45-14.95", n_train=report["n_train"], n_test=report["n_test"],
                   representation="static (log) + shape, PCA 20",
                   model=report["recipe"] + ", Matern 5/2 ARD, linear trend, rotated twins; portable export",
                   fit_s=report["fit_s"], predict_ms_per_case=report["pred_ms"],
                   notes=f"release model; 95 % band coverage {100 * report['coverage']['all']:.1f} %",
                   **report["summary"])


def save_release(sur, path=DEFAULT_MODEL):
    return sur.save(path)
