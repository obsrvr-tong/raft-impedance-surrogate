"""The experiment table, ml/results/experiments.csv: one row per experiment.

    from surrogate.experiments import log_experiment
    log_experiment("M2-pcagp40", milestone="M2", model="PCA(40) + GP Matern 5/2", ...,
                   **summarize(errors, spec.terms))

Logging is idempotent: a row is keyed by exp_id, and logging the same exp_id again replaces it,
so re-running a notebook updates its rows instead of piling up duplicates. Every row records
the date and the git commit of the package code (with "+dirty" if ml/surrogate has uncommitted
changes), so a number can always be traced back to the code that produced it.
"""
import csv
import subprocess
from datetime import date

from .data import REPO

RESULTS = REPO / "ml" / "results" / "experiments.csv"
TERM_NAMES = ["Ux", "Uy", "Uz", "Rx", "Ry", "Rz", "Ux-Ry", "Uy-Rx"]
COLUMNS = (["date", "exp_id", "milestone", "footprints", "validation", "band_hz", "n_train", "n_test",
            "representation", "model"]
           + [f"err_{t}" for t in TERM_NAMES]
           + ["err_worst", "err_mean", "fit_s", "predict_ms_per_case", "git_commit", "notes"])


def git_commit():
    """Short hash of HEAD, plus '+dirty' if ml/surrogate has uncommitted changes."""
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10)
    try:
        head = run("rev-parse", "--short", "HEAD").stdout.strip()
        dirty = run("status", "--porcelain", "--", "ml/surrogate").stdout.strip()
        return (head or "unknown") + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else v


def log_experiment(exp_id, path=RESULTS, **fields):
    """Add or replace the row exp_id. Unknown field names raise ValueError. Returns the row."""
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown experiment fields: {sorted(unknown)}")
    row = {c: "" for c in COLUMNS}
    row.update({k: _fmt(v) for k, v in fields.items()})
    row.update(date=date.today().isoformat(), exp_id=exp_id, git_commit=git_commit())
    rows = []
    if path.is_file():
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("exp_id") != exp_id]
    rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return row


def read_experiments(path=RESULTS):
    """The table as a pandas DataFrame."""
    import pandas as pd
    return pd.read_csv(path)
