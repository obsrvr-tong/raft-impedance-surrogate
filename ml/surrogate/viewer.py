"""The interactive surrogate viewer: one self-contained HTML page with the model inside.

    python -m surrogate viewer                       # ml/viewer/index.html, with the MISS3D hold-out cases
    python -m surrogate viewer --no-holdout          # without them (no dataset needed)

The page evaluates the portable model in the browser (JavaScript, the same arithmetic as
surrogate.portable): no server, no installation, works offline and can be shared as one file.
Sliders set the soil profile and the raft; the eight impedance curves and their 95 % band
update as you drag. The hold-out simulations (never used in training) can be loaded to show
surrogate and MISS3D side by side. The API serves the same page at /viewer.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .portable import DEFAULT_MODEL, INPUTS, N_NEIGHBOURS, NOMINAL, ImpedanceSurrogate

TEMPLATE = Path(__file__).with_name("viewer_template.html")

# The site of dataset v1 (automation/inputs/site_v1.csv): strata from the surface down, shear-wave
# velocity before the g factor, and the bedrock below them.
STRATA = (("L1", 1000), ("L2", 1200), ("L3", 1100), ("L4", 1000), ("L5", 500), ("L6", 500), ("L7", 900))
BEDROCK = ("Rock", 1600)


def _b64(a, dtype="<f8"):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def load_holdout(sur: ImpedanceSurrogate, path=None):
    """The hold-out cases of dataset v1 as target rows in the model's layout, or None if the
    dataset is not available."""
    from .data import DEFAULT_NPZ, load_dataset, make_targets
    path = DEFAULT_NPZ if path is None else path
    if not Path(path).exists():
        return None
    ds = load_dataset(path).select(split="test")
    if not np.allclose(ds.freqs, sur.freqs):
        raise ValueError("dataset frequencies differ from the model's")
    Y, spec = make_targets(ds.K)
    if tuple(spec.terms) != sur.terms:
        raise ValueError("dataset terms differ from the model's")
    return {"case": [str(c) for c in ds.case], "mesh": [str(m) for m in ds.mesh_id],
            "X": _b64(ds.X, "<f4"), "Y": _b64(Y, "<f4"), "n": int(len(ds))}


def viewer_data(sur: ImpedanceSurrogate, holdout=None) -> dict:
    a = sur.a
    model = {
        "n": int(a["X_train"].shape[0]), "d": int(a["X_train"].shape[1]), "n_gp": int(sur.n_gp),
        "n_terms": sur.n_terms, "n_comp": int(a["pca_components"].shape[0]), "n_neighbours": N_NEIGHBOURS,
        "X": _b64(a["X_train"]), "alpha": _b64(a["alpha"]), "ls": _b64(a["length_scale"]),
        "trend": _b64(a["trend_coef"]), "pca_mean": _b64(a["pca_mean"]), "pca_comp": _b64(a["pca_components"]),
        "rstd": _b64(a["shape_resid_std"]),
        **{k: np.asarray(a[k], float).tolist() for k in
           ("signal_var", "noise", "mean0", "y_mean", "y_std", "sign", "x_mean", "x_scale")},
    }
    inputs = [{"name": n, "min": float(lo), "max": float(hi), "nominal": NOMINAL[n]}
              for n, lo, hi in zip(INPUTS, sur.x_min, sur.x_max)]
    meta = {k: sur.meta.get(k) for k in ("name", "created", "n_train", "engine", "training", "holdout",
                                         "footprints_trained", "band_hz", "git_commit")}
    return {"meta": meta, "freqs": sur.freqs.tolist(), "terms": list(sur.terms), "inputs": inputs,
            "strata": [list(s) for s in STRATA], "bedrock": list(BEDROCK), "model": model, "holdout": holdout}


def viewer_html(sur: ImpedanceSurrogate, holdout=None, full=True) -> str:
    """The page as a string. full=False leaves out the <!doctype>/<html>/<head>/<body> skeleton
    (for hosts that add their own)."""
    body = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(viewer_data(sur, holdout), separators=(",", ":")).replace("</", "<\\/")
    body = body.replace("__VIEWER_DATA__", payload)
    if not full:
        return body
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            "<style>html{color-scheme:light}body{margin:0;font:14px system-ui,sans-serif}"
            "img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n"
            + body + "\n</body>\n</html>\n")


def build_viewer(model_path=DEFAULT_MODEL, out=None, holdout=True, full=True) -> Path:
    sur = ImpedanceSurrogate.load(model_path)
    out = Path(out) if out is not None else Path(model_path).parents[1] / "viewer" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(viewer_html(sur, load_holdout(sur) if holdout else None, full=full), encoding="utf-8")
    return out
