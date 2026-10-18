# automation/: the simulation pipeline

Generates, runs and collects code_aster / MISS3D impedance cases of a rigid surface raft on a
layered soil. Standard-library Python (the collector also needs numpy), written for a Linux
machine with the salome_meca container.

The raft meshes and the reference model are not included, so the pipeline cannot be run as is:
it needs your own meshes (one `.med` file per footprint, with an element group `RADIER` for the
raft and a node group `NRADIER` for the reference node) listed in `inputs/meshes.csv`.

## Layout
- `make_dataset.py` + `codeaster_auto/strata.py` + `codeaster_auto/sampling.py`: the case
  generator. Reads the site (`inputs/site_v1.csv`: thickness range, Vs, density, Poisson's ratio
  and damping per layer, bedrock last) and the frozen settings (`inputs/dataset_v1.json`), draws
  a scrambled Halton design for training and uniform random points for the hold-out set (fixed
  seeds, extensible with `--extend`), and writes one case folder per sample.
- `make_case.py` + `codeaster_auto/comm.py`, `export.py`, `soil.py`: one case folder (mesh,
  `.comm` from `templates/miss_impedance_shallow.comm`, `.export`).
- `run_cases.py` + `codeaster_auto/runner.py`: the batch runner. Runs every case folder with
  `--jobs N` in parallel, reads code_aster's diagnostic, records run time and peak memory in
  `run_summary.csv`, resumes where it stopped, interleaves meshes (`--order index`), and reports
  one line per mesh (`--status`).
- `collect_dataset.py` + `codeaster_auto/impedance.py`: the collector. Turns the finished cases
  into one `dataset_v1.npz` (complex 6 × 6 impedance per case and frequency, inputs, derived
  features, labels) plus a case table and a manifest.
- `bin/aster`: runs a command inside the salome_meca container (`aster run_aster job.export`).
- `inputs/probe_v1.json`: a 5-frequency run-time probe for new meshes.
- `tests/`: generator, runner (with a fake `aster` command), collector and impedance-reader tests.

## Use (from automation/)
    python3 -m unittest discover tests
    python3 make_dataset.py --mesh m30x15 --dry-run                 # plan + run-time estimate
    python3 make_dataset.py --mesh m30x15                           # datasets/v1/cases/
    python3 run_cases.py --cases datasets/v1/cases --jobs 8 --order index
    python3 run_cases.py --cases datasets/v1/cases --status
    python3 collect_dataset.py                                      # datasets/v1/dataset_v1.npz

Dataset v1 used five footprints (30 × 15, 45 × 15, 60 × 15, 80 × 20 and 33 × 31 m): 3,232 cases,
about 18 h with 6–8 parallel jobs on a 20-vCPU machine.
