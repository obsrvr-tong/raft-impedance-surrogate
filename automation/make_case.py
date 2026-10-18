"""Build one code_aster case folder: mesh + .comm + .export (+ inputs record).

Example (from the automation/ folder):
    python make_case.py --name ref1_check \
        --mesh meshes/m30x15.med \
        --soil my_soil.csv

Creates cases/ref1_check/ containing:
    mesh.med, soil_profile.csv    copies of the inputs
    ref1_check.comm               filled template
    ref1_check.export             run instructions for run_aster
    case.json                     what was used to build the case
Run it in the VM with:
    cd cases/ref1_check && aster run_aster ref1_check.export
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from codeaster_auto import export as ex  # noqa: E402
from codeaster_auto.comm import render_comm, write_comm  # noqa: E402
from codeaster_auto.soil import read_profile, write_profile, defi_sol_miss_keywords  # noqa: E402

DEFAULT_TEMPLATE = HERE / "templates" / "miss_impedance_shallow.comm"


def make_case(name, mesh, soil, out_dir=HERE / "cases", template=DEFAULT_TEMPLATE,
              freq_min=0.2, freq_max=50.2, freq_step=1.0,
              time_limit=15300, memory_limit=2048, absolute_paths=False, force=False,
              extra_record=None):
    case_dir = Path(out_dir) / name
    if case_dir.exists() and not force:
        raise FileExistsError(f"{case_dir} exists (use force=True / --force to overwrite)")
    case_dir.mkdir(parents=True, exist_ok=True)

    # soil: a CSV path, or a list of SoilLayer already built (e.g. from the Excel input)
    if isinstance(soil, (list, tuple)):
        layers = list(soil)
        for i, layer in enumerate(layers, 1):
            layer.check(i)
        write_profile(case_dir / "soil_profile.csv", layers)
        soil_source = "(generated)"
    else:
        layers = read_profile(soil)
        shutil.copyfile(soil, case_dir / "soil_profile.csv")
        soil_source = str(soil)
    shutil.copyfile(mesh, case_dir / "mesh.med")

    comm = render_comm(template, {
        "SOIL_KEYWORDS": defi_sol_miss_keywords(layers),
        "FREQ_MIN": repr(float(freq_min)),
        "FREQ_MAX": repr(float(freq_max)),
        "FREQ_PAS": repr(float(freq_step)),
    })
    write_comm(case_dir / f"{name}.comm", comm)

    p = (lambda f: (case_dir / f).resolve()) if absolute_paths else (lambda f: f)
    files = [
        ("comm", p(f"{name}.comm"), "D", ex.UNIT_COMM),
        ("mmed", p("mesh.med"), "D", ex.UNIT_MESH),
        ("libr", p(f"{name}_impe.txt"), "R", ex.UNIT_IMPE),
        ("libr", p(f"{name}_forc.txt"), "R", ex.UNIT_FORC),
        ("mess", p(f"{name}.mess"), "R", ex.UNIT_MESS),
    ]
    ex.write_export(case_dir / f"{name}.export", files,
                    time_limit=time_limit, memory_limit=memory_limit)

    record = {
        "name": name, "created": datetime.now().isoformat(timespec="seconds"),
        "template": Path(template).name, "mesh_source": str(mesh), "soil_source": soil_source,
        "n_layers": len(layers), "freq_min": freq_min, "freq_max": freq_max,
        "freq_step": freq_step, "time_limit": time_limit, "memory_limit": memory_limit,
    }
    if extra_record:
        record.update(extra_record)
    (case_dir / "case.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return case_dir


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--name", required=True)
    a.add_argument("--mesh", required=True)
    a.add_argument("--soil", required=True)
    a.add_argument("--out", default=str(HERE / "cases"))
    a.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    a.add_argument("--freq-min", type=float, default=0.2)
    a.add_argument("--freq-max", type=float, default=50.2)
    a.add_argument("--freq-step", type=float, default=1.0)
    a.add_argument("--time-limit", type=int, default=15300)
    a.add_argument("--memory-limit", type=int, default=2048)
    a.add_argument("--absolute-paths", action="store_true",
                   help="write absolute paths in the .export (default: relative to the case folder)")
    a.add_argument("--force", action="store_true", help="overwrite an existing case folder")
    args = a.parse_args()
    d = make_case(args.name, args.mesh, args.soil, args.out, args.template,
                  args.freq_min, args.freq_max, args.freq_step,
                  args.time_limit, args.memory_limit, args.absolute_paths, args.force)
    print(f"case written to {d}")


if __name__ == "__main__":
    main()
