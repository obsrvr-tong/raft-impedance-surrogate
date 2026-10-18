"""Command line for the impedance surrogate.

    python -m surrogate predict ml/examples/nominal_45x15.json            # print the static values
    python -m surrogate predict ml/examples/nominal_45x15.json --band --csv curves.csv
    python -m surrogate train  [--soils 512] [--engine auto|gpu|cpu]       # retrain the release model
    python -m surrogate train  --soils 256 --components 40 --validate 64   # compare settings (validation set)
    python -m surrogate calibrate [--log]                                  # add PCA error to an older model's band
    python -m surrogate viewer [--out ml/viewer/index.html]                # build the interactive viewer
    python -m surrogate serve  [--port 8000]                               # start the prediction API

`predict` reads one case (a JSON object) or several (a JSON list) with the 10 inputs
t_L1, t_L2, t_L3, t_L4, t_L5, t_L6, t_L7 [m], g_factor [-], L_x, L_y [m].
"""
import argparse
import csv
import json
import sys
from pathlib import Path

UNITS = {"Ux": "MN/m", "Uy": "MN/m", "Uz": "MN/m", "Rx": "MN.m/rad", "Ry": "MN.m/rad", "Rz": "MN.m/rad",
         "Ux-Ry": "MN/rad", "Uy-Rx": "MN/rad"}


def _predict(args):
    from .portable import INPUTS, ImpedanceSurrogate
    sur = ImpedanceSurrogate.load(args.model)
    text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
    cases = json.loads(text)
    cases = cases if isinstance(cases, list) else [cases]
    p = sur.predict(cases, band=args.band)
    h = sur.meta.get("holdout", {})
    print(f"{sur.meta.get('name', 'surrogate')}: {sur.meta['n_train']} training cases, "
          f"{p.freqs[0]:g}-{p.freqs[-1]:g} Hz"
          + (f"; hold-out error {h['err_mean']:.1f} % mean, {h['err_worst']:.1f} % worst term" if h else ""))
    for i, case in enumerate(cases):
        print(f"\ncase {i + 1}: " + ", ".join(f"{k}={case[k]:g}" for k in INPUTS))
        for w in p.warnings[i]:
            print(f"  warning: {w} (extrapolation: treat with care)")
        print(f"  {'term':6s} {'static Re K(0.45 Hz)':>21s}  {'unit':9s}" + ("  95 % band" if args.band else ""))
        for k, t in enumerate(p.terms):
            line = f"  {t:6s} {p.static[i, k]:21.4e}  {UNITS.get(t, ''):9s}"
            if args.band:
                line += f"  {p.lo[i, 0, k].real:.4e} to {p.hi[i, 0, k].real:.4e}"
            print(line)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            parts = ["re", "im"] + (["re_lo", "re_hi", "im_lo", "im_hi"] if args.band else [])
            w.writerow(["case", "freq_hz"] + [f"{t}_{q}" for t in p.terms for q in parts])
            for i in range(len(cases)):
                for j, fr in enumerate(p.freqs):
                    row = [i + 1, f"{fr:g}"]
                    for k in range(len(p.terms)):
                        vals = [p.K[i, j, k].real, p.K[i, j, k].imag]
                        if args.band:
                            vals += [p.lo[i, j, k].real, p.hi[i, j, k].real, p.lo[i, j, k].imag, p.hi[i, j, k].imag]
                        row += [f"{v:.6e}" for v in vals]
                    w.writerow(row)
        print(f"\ncurves ({len(p.freqs)} frequencies x {len(p.terms)} terms) written to {args.csv}")


def _train(args):
    from .portable import MODELS_DIR
    from .release import log_release, train_release
    out = args.out
    if out is None:                       # a validation run never overwrites the release model
        out = (MODELS_DIR / f"validation_s{args.soils}_c{args.components}.npz" if args.validate
               else MODELS_DIR / "impedance_surrogate_v1.npz")
    sur, report = train_release(n_soils=args.soils, engine=args.engine, n_iter=args.n_iter, n_jobs=args.jobs,
                                max_batch=args.max_batch, n_components=args.components,
                                validation_soils=args.validate)
    path = sur.save(out)
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    if args.log and not args.validate:
        log_release(report)
        print("row M5-release written to ml/results/experiments.csv")


def _calibrate(args):
    from .release import calibrate
    calibrate(args.model, log=args.log)


def _viewer(args):
    from .viewer import build_viewer
    path = build_viewer(args.model, args.out, holdout=not args.no_holdout)
    print(f"viewer written to {path} ({path.stat().st_size / 1e6:.1f} MB): open it in a browser")


def _serve(args):
    import os
    import uvicorn
    os.environ["IMPEDANCE_SURROGATE_MODEL"] = str(args.model)
    print(f"API docs at http://{args.host}:{args.port}/docs, viewer at http://{args.host}:{args.port}/viewer")
    uvicorn.run("surrogate.api:app", host=args.host, port=args.port)


def main(argv=None):
    from .portable import DEFAULT_MODEL
    ap = argparse.ArgumentParser(prog="python -m surrogate", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="impedance curves for the cases in a JSON file")
    p.add_argument("input", help="JSON file with one case (object) or several (list); '-' reads stdin")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--band", action="store_true", help="add the 95 %% uncertainty band")
    p.add_argument("--csv", help="write the curves to this CSV file")
    p.set_defaults(func=_predict)

    p = sub.add_parser("train", help="train, test and save the release model (needs the dataset)")
    p.add_argument("--soils", type=int, default=512, help="training soils per footprint (default 512)")
    p.add_argument("--engine", default="auto", choices=["auto", "gpu", "cpu"])
    p.add_argument("--n-iter", type=int, default=1000, help="Adam steps (GPU engine)")
    p.add_argument("--jobs", type=int, default=None, help="parallel GP fits (CPU engine)")
    p.add_argument("--max-batch", type=int, default=None, help="GPs per GPU batch (lower it if out of memory)")
    p.add_argument("--components", type=int, default=20, help="PCA components of the shape (default 20)")
    p.add_argument("--validate", type=int, default=None, metavar="V",
                   help="test on the next V training soils per footprint instead of the hold-out cases")
    p.add_argument("--out", default=None, help="model file (default: the release model, or validation_*.npz)")
    p.add_argument("--log", action="store_true", help="also write row M5-release to results/experiments.csv")
    p.set_defaults(func=_train)

    p = sub.add_parser("calibrate", help="add the PCA truncation error to a saved model's band (seconds)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--log", action="store_true", help="also update row M5-release in results/experiments.csv")
    p.set_defaults(func=_calibrate)

    p = sub.add_parser("viewer", help="build the standalone interactive viewer (HTML)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=Path(DEFAULT_MODEL).parents[1] / "viewer" / "index.html")
    p.add_argument("--no-holdout", action="store_true", help="leave out the MISS3D hold-out cases")
    p.set_defaults(func=_viewer)

    p = sub.add_parser("serve", help="start the prediction API (needs fastapi and uvicorn)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_serve)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
