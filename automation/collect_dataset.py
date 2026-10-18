"""Collect finished dataset cases into ML-ready files.

    python3 collect_dataset.py                          # all cases in datasets/v1/cases
    python3 collect_dataset.py --mesh ref33x31          # one mesh only
    python3 collect_dataset.py --cases datasets/v1/cases --out datasets/v1/dataset_v1

Reads every case folder that has a case.json (written by make_dataset.py), keeps the
cases whose .mess says OK and whose impedance file is complete, and writes:

  <out>.npz            numpy arrays (load with np.load(path, allow_pickle=False)):
                         K        complex64 (n_cases, n_freq, 6, 6)  impedance K[i][j]
                         freqs    (n_freq,)                          Hz
                         X        (n_cases, n_inputs)                sampled inputs + footprint
                         x_names  (n_inputs,)                        column names of X
                         D        (n_cases, 3)                       derived features
                         d_names  ('H_bedrock', 'vs_avg', 'f0')
                         case, split, mesh_id   (n_cases,)            labels
  <out>_cases.csv      the same inputs and derived features as a table (pandas / Excel)
  <out>_manifest.json  what went in: counts, skipped cases, frequency grid, settings,
                       git commits, creation time

Derived features (physics-based, often more informative than raw inputs):
  H_bedrock  depth to bedrock = sum of stratum thicknesses [m]
  vs_avg     travel-time average Vs of the soil above bedrock = H / sum(t_i / Vs_i) [m/s]
  f0         fundamental frequency of the soil column ~ vs_avg / (4 H) [Hz]
Needs numpy (the case generator itself does not).
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("collect_dataset.py needs numpy (e.g. in a venv: pip install numpy)")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from codeaster_auto.impedance import read_impe  # noqa: E402
from codeaster_auto.runner import case_paths, is_success, read_mess  # noqa: E402
from codeaster_auto.strata import read_site  # noqa: E402

D_NAMES = ["H_bedrock", "vs_avg", "f0"]


def derived(inputs, soil):
    """H_bedrock, travel-time average Vs and f0 for one case."""
    g = inputs["g_factor"]
    t = [inputs[f"t_{s.name}"] for s in soil]
    vs = [s.vs * g ** 0.5 for s in soil]
    H = sum(t)
    vs_avg = H / sum(ti / vi for ti, vi in zip(t, vs))
    return [H, vs_avg, vs_avg / (4 * H)]


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--cases", default=str(HERE / "datasets" / "v1" / "cases"))
    a.add_argument("--out", default=None, help="output prefix (default <cases>/../dataset_v1[_<mesh>])")
    a.add_argument("--mesh", nargs="+", help="only these mesh_ids")
    args = a.parse_args()

    root = Path(args.cases)
    folders = sorted(d for d in root.iterdir() if (d / "case.json").is_file())
    if not folders:
        sys.exit(f"no case folders with case.json in {root}")

    records, K_list, skipped = [], [], []
    freqs, x_names, sites = None, None, {}
    for n, d in enumerate(folders, 1):
        rec = json.loads((d / "case.json").read_text(encoding="utf-8"))
        mesh_id = rec.get("mesh", {}).get("mesh_id", "")
        if args.mesh and mesh_id not in args.mesh:
            continue
        _, mess, _ = case_paths(d)
        status = read_mess(mess)["status"]
        impe = sorted(d.glob("*_impe.txt"))
        if not is_success(status) or not impe:
            skipped.append({"case": d.name, "reason": status if not is_success(status) else "no impedance file"})
            continue
        data = read_impe(impe[0])
        if data["missing"] or data["n"] != 6:
            skipped.append({"case": d.name, "reason": f"incomplete impedance matrix ({len(data['missing'])} terms missing)"})
            continue
        if freqs is None:
            freqs = data["freqs"]
        elif len(data["freqs"]) != len(freqs) or max(abs(a_ - b) for a_, b in zip(data["freqs"], freqs)) > 1e-6:
            skipped.append({"case": d.name, "reason": "different frequency grid"})
            continue
        inputs = rec["inputs"]
        names = list(inputs) + ["L_x", "L_y"]
        if x_names is None:
            x_names = names
        elif names != x_names:
            skipped.append({"case": d.name, "reason": "different input columns"})
            continue
        site_path = rec["config"]["site"]
        if site_path not in sites:
            sites[site_path] = read_site(HERE / site_path)[0]
        K_list.append(np.asarray(data["re"], dtype=np.float32) + 1j * np.asarray(data["im"], dtype=np.float32))
        records.append({
            "case": d.name, "split": rec["split"], "mesh_id": mesh_id,
            "x": [float(inputs[k]) for k in inputs] + [float(rec["mesh"]["L_x"]), float(rec["mesh"]["L_y"])],
            "d": derived(inputs, sites[site_path]),
            "git_commit": rec.get("git_commit", "unknown"), "config": rec.get("config", {}),
        })
        if n % 200 == 0:
            print(f"  read {n}/{len(folders)} folders ...")

    if not records:
        sys.exit("no usable cases found")
    suffix = f"_{'_'.join(args.mesh)}" if args.mesh else ""
    out = Path(args.out) if args.out else root.parent / f"dataset_{records[0]['config'].get('dataset', 'v1')}{suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)

    K = np.stack(K_list).astype(np.complex64)
    X = np.array([r["x"] for r in records], dtype=float)
    D = np.array([r["d"] for r in records], dtype=float)
    np.savez_compressed(out.with_suffix(".npz"), K=K, freqs=np.array(freqs), X=X, x_names=np.array(x_names),
                        D=D, d_names=np.array(D_NAMES), case=np.array([r["case"] for r in records]),
                        split=np.array([r["split"] for r in records]),
                        mesh_id=np.array([r["mesh_id"] for r in records]))

    csv_path = out.parent / f"{out.name}_cases.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["case", "split", "mesh_id"] + x_names + D_NAMES)
        for r in records:
            w.writerow([r["case"], r["split"], r["mesh_id"]] + [f"{v:g}" for v in r["x"]] + [f"{v:.6g}" for v in r["d"]])

    splits = {s: sum(r["split"] == s for r in records) for s in sorted({r["split"] for r in records})}
    meshes = {m: sum(r["mesh_id"] == m for r in records) for m in sorted({r["mesh_id"] for r in records})}
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"), "source": str(root.resolve()),
        "n_cases": len(records), "splits": splits, "meshes": meshes, "skipped": skipped,
        "n_freq": len(freqs), "freq_min": freqs[0], "freq_max": freqs[-1],
        "x_names": x_names, "d_names": D_NAMES, "K_shape": list(K.shape), "K_dtype": str(K.dtype),
        "git_commits": sorted({r["git_commit"] for r in records}), "config": records[0]["config"],
    }
    man_path = out.parent / f"{out.name}_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"{len(records)} cases collected ({', '.join(f'{k} {v}' for k, v in splits.items())}; "
          f"meshes: {', '.join(f'{k} {v}' for k, v in meshes.items())}); {len(skipped)} skipped")
    print(f"K {K.shape} complex64, {len(freqs)} frequencies {freqs[0]:g}-{freqs[-1]:g} Hz; X {X.shape}: {', '.join(x_names)}")
    for name, col in zip(x_names + D_NAMES, np.hstack([X, D]).T):
        print(f"  {name:<10} {col.min():10.4g} - {col.max():<10.4g}")
    k0 = np.abs(K[:, 0, :, :]).diagonal(axis1=1, axis2=2)
    print("static |K_ii| at the lowest frequency (min-max): " +
          ", ".join(f"{lbl} {k0[:, i].min():.3g}-{k0[:, i].max():.3g}" for i, lbl in enumerate(["Ux", "Uy", "Uz", "Rx", "Ry", "Rz"])))
    for p in (out.with_suffix(".npz"), csv_path, man_path):
        print(f"wrote {p}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError) as e:
        sys.exit(f"INPUT ERROR: {e}")
