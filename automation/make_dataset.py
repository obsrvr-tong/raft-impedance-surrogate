"""Build the case folders of a training dataset for one mesh.

    python3 make_dataset.py --mesh ref33x31 --dry-run     # show the plan and run-time estimate
    python3 make_dataset.py --mesh ref33x31               # write datasets/v1/cases/...
    python3 make_dataset.py --mesh m120x75 --n-train 0 --n-test 1   # just one hold-out case
    python3 run_cases.py --cases datasets/v1/cases        # run them (resumable)

Everything is defined by a frozen config (default inputs/dataset_v1.json):
  site     inputs/site_v1.csv   strata (thickness ranges, Vs, rho, nu, damping) + bedrock
  meshes   inputs/meshes.csv    mesh_id -> .med file and footprint geometry
  sampling n_train points of a scrambled Halton sequence (space-filling) and n_test
           uniform random points (hold-out), fixed seed -> reproducible
Inputs per case: one thickness per stratum (uniform in [t_min, t_max]) and a shear-modulus
factor g applied to all soil strata (log-uniform in [g_min, g_max]; Vs scales with sqrt(g)).

Writes, for mesh M:
  datasets/v1/cases/M_train0001 ... M_test0001 ...   case folders (mesh, .comm, .export, case.json)
  datasets/v1/samples_M.csv                          one row per case: all inputs and geometry
Use --extend N to add N more training points (continues the same Halton sequence after the
highest training index already built; can be repeated: 128 -> 512 -> 1024 ...).
--n-train / --n-test override the config's counts (the points are the same: the first N of the
same sequences), e.g. a single timing case per new mesh before a batch is planned.
Case folders that already exist are left as they are, so building more later is safe.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from codeaster_auto.sampling import halton, random_points  # noqa: E402
from codeaster_auto.strata import layers_for, read_site  # noqa: E402
from make_case import make_case  # noqa: E402


def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def read_meshes(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["mesh_id"].strip(): r for r in csv.DictReader(f)}


def plan(cfg, soil, n_train, n_test, skip=0):
    """[(split, index, u, thicknesses, g)] for the requested points."""
    dim = len(soil) + 1
    lg0, lg1 = math.log(cfg["g_factor_min"]), math.log(cfg["g_factor_max"])
    rows = []
    for split, pts, start in (("train", halton(n_train, dim, cfg["seed"], skip), skip),
                              ("test", random_points(n_test, dim, cfg["seed"] + 1), 0)):
        for k, u in enumerate(pts, start + 1):
            t = [round(s.t_min + ui * (s.t_max - s.t_min), 2) for s, ui in zip(soil, u)]
            g = round(math.exp(lg0 + u[-1] * (lg1 - lg0)), 4)
            rows.append((split, k, u, t, g))
    return rows


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default=str(HERE / "inputs" / "dataset_v1.json"))
    a.add_argument("--mesh", required=True, help="mesh_id from the meshes file")
    a.add_argument("--out", default=None, help="dataset folder (default datasets/<dataset>)")
    a.add_argument("--n-train", type=int, default=None, metavar="N",
                   help="training points to build (default: n_train of the config)")
    a.add_argument("--n-test", type=int, default=None, metavar="N",
                   help="hold-out points to build (default: n_test of the config)")
    a.add_argument("--extend", type=int, default=0, metavar="N",
                   help="add N training points after the existing n_train (no new test points)")
    a.add_argument("--time-limit", type=int, default=None, metavar="S",
                   help="time_limit of the .export in seconds (default: make_case's 15300)")
    a.add_argument("--memory-limit", type=int, default=None, metavar="MB",
                   help="memory_limit of the .export in MB (default: make_case's 2048)")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--force", action="store_true")
    args = a.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    soil, bedrock = read_site(HERE / cfg["site"])
    meshes = read_meshes(HERE / cfg["meshes"])
    if args.mesh not in meshes:
        sys.exit(f"INPUT ERROR: mesh_id {args.mesh!r} not in {cfg['meshes']} (known: {sorted(meshes)})")
    mesh = meshes[args.mesh]
    mesh_file = HERE / mesh["file"]
    if not mesh_file.is_file():
        sys.exit(f"INPUT ERROR: mesh file not found: {mesh_file}")
    out = Path(args.out) if args.out else HERE / "datasets" / cfg["dataset"]

    if args.extend:
        # continue the Halton sequence after the highest training index already built
        existing = [int(d.name.rsplit("_train", 1)[1]) for d in (out / "cases").glob(f"{args.mesh}_train*")
                    if d.name.rsplit("_train", 1)[1].isdigit()]
        start = max(existing, default=0)
        if start == 0:
            sys.exit("INPUT ERROR: --extend needs an existing dataset for this mesh; build it first")
        rows = plan(cfg, soil, args.extend, 0, skip=start)
    else:
        n_train = cfg["n_train"] if args.n_train is None else args.n_train
        n_test = cfg["n_test"] if args.n_test is None else args.n_test
        if n_train < 0 or n_test < 0:
            sys.exit("INPUT ERROR: --n-train and --n-test must be >= 0")
        rows = plan(cfg, soil, n_train, n_test)
    if not rows:
        sys.exit("nothing to build (0 training and 0 hold-out points)")

    n_freq = int(round((cfg["freq_max"] - cfg["freq_min"]) / cfg["freq_step"])) + 1
    est = len(rows) * (4 + 0.65 * n_freq)
    H = [sum(t) for *_, t, _ in rows]
    g = [r[4] for r in rows]
    print(f"dataset {cfg['dataset']}  mesh {args.mesh} ({mesh['L_x']} x {mesh['L_y']} m)  "
          f"{sum(r[0] == 'train' for r in rows)} train + {sum(r[0] == 'test' for r in rows)} test cases")
    print(f"strata: {', '.join(f'{s.name} {s.t_min:g}-{s.t_max:g} m' for s in soil)}; bedrock {bedrock.name} Vs {bedrock.vs:g}")
    print(f"depth to bedrock {min(H):.1f}-{max(H):.1f} m; G factor {min(g):.3f}-{max(g):.3f}")
    est_txt = (f"{est / 3600:.1f} h" if est >= 3600 else
               f"{est / 60:.0f} min" if est >= 90 else f"{est:.0f} s")
    print(f"{n_freq} frequencies per case; estimated run time about {est_txt} on one job "
          f"(reference-mesh timing; larger meshes are slower)")
    if args.dry_run:
        print("dry run: nothing written")
        return

    commit = git_commit()
    limits = {k: v for k, v in (("time_limit", args.time_limit), ("memory_limit", args.memory_limit))
              if v is not None}
    cases_dir = out / "cases"
    table = []
    for split, k, u, t, gf in rows:
        name = f"{args.mesh}_{split}{k:04d}"
        inputs = {f"t_{s.name}": ti for s, ti in zip(soil, t)}
        inputs["g_factor"] = gf
        record = {
            "dataset": cfg["dataset"], "split": split, "index": k, "u": [round(x, 6) for x in u],
            "inputs": inputs, "mesh": mesh, "git_commit": commit, "config": cfg,
            "profile": {"description": f"{split} #{k}: H {sum(t):.1f} m, G x{gf:.2f}"},
        }
        try:
            make_case(name, mesh_file, layers_for(soil, bedrock, t, gf, cfg["substratum_thickness"]),
                      out_dir=cases_dir, template=HERE / cfg["template"],
                      freq_min=cfg["freq_min"], freq_max=cfg["freq_max"], freq_step=cfg["freq_step"],
                      force=args.force, extra_record=record, **limits)
            status = "built"
        except FileExistsError:
            status = "exists"
        table.append({"case": name, "split": split, "index": k, "mesh_id": args.mesh,
                      "L_x": mesh["L_x"], "L_y": mesh["L_y"], **inputs,
                      "H_bedrock": round(sum(t), 2), "status": status})

    samples = out / f"samples_{args.mesh}.csv"
    existing = []
    if samples.is_file():
        with open(samples, newline="", encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f) if r["case"] not in {x["case"] for x in table}]
    with open(samples, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        for r in existing + table:
            w.writerow(r)
    built = sum(r["status"] == "built" for r in table)
    print(f"{built} case folders written to {cases_dir} ({len(table) - built} already existed)")
    print(f"sample table: {samples}")
    print(f"next: python3 run_cases.py --cases {cases_dir}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError) as e:
        sys.exit(f"INPUT ERROR: {e}")
