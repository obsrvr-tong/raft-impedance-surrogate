# Raft Impedance Surrogate

A machine-learning surrogate for seismic soil–structure interaction. It predicts the 6 × 6
frequency-dependent impedance of a rigid surface raft on a layered soil (8 terms, 0.45–15 Hz)
in milliseconds, instead of a 20 s to 7 min MISS3D boundary-element run, and says how sure it is.

**[Open the interactive viewer](https://obsrvr-tong.github.io/raft-impedance-surrogate/)**: drag the
soil profile and the raft size, watch the curves and their 95 % band follow, and overlay any of
the 160 held-out simulations.

## Results

Tested on 160 MISS3D simulations that were never used for training (32 random soils on each of
five raft footprints):

| | |
|---|---|
| Mean error | **2.9 %** (largest deviation over 0.45–15 Hz, as a share of the curve's largest value) |
| Worst term, 95 % of soils | 8.6 % (sway–rocking coupling); rocking within 3.4 % |
| 95 % uncertainty band | holds 97.9 % of the simulated values |
| Prediction | 3.6 ms per case (5,000 to 100,000 times faster than MISS3D) |
| Training | 5,120 cases, 27 min on one RTX 5070 Ti (GPyTorch) |

The full story, from the simulation campaign to the error budget, is in
[`docs/ml_writeup.md`](docs/ml_writeup.md). The notebooks are saved with their outputs.

## How it works

```
soil + raft inputs ──► automation/: code_aster + MISS3D cases ──► batch runner ──► collector
 (7 layer thicknesses,     (generated from a template,           (parallel,          (one dataset
  stiffness factor,         space-filling designs)                resumable)          file)
  raft length, width)                                                                    │
                                                                                         ▼
          CLI · FastAPI · browser viewer  ◄──  portable model  ◄──  ml/: Gaussian processes
                                               (numpy only)         (static × shape split, PCA,
                                                                     trend, rotated rafts, GPU)
```

- **Data engine** (`automation/`): standard-library Python that writes code_aster case files
  for MISS3D impedance runs, runs them in parallel with resume and memory tracking, and collects
  the finished runs into one dataset. 3,232 simulations, about 18 h of batch time.
- **Surrogate** (`ml/`): each impedance term is split into its static value and a dimensionless
  shape; PCA compresses the shapes; 28 Gaussian processes (Matérn 5/2, one length scale per
  input, linear trend) predict the static values and shape weights. Rafts turned by 90° are added
  as free training data (exact on layered soil). Trained in batches on the GPU.
- **Shipping**: the trained model is exported to a 1.4 MB numpy-only file, used by a command
  line, a FastAPI service (with a Dockerfile) and a browser viewer that runs the model client-side.
- **Studies**: Sobol sensitivity indices and a Monte Carlo on soil uncertainty, each in about a
  minute instead of days of simulation (notebook 05).

## Quick start

    conda env create -f ml/environment.yml
    conda activate ml
    pip install -e ml
    python -m surrogate predict ml/examples/nominal_45x15.json --band     # one prediction
    python -m surrogate serve                                             # API: http://127.0.0.1:8000/docs
    python -m unittest discover ml/tests                                  # 49 tests

The model file is included, so prediction, the API and the viewer work without the training data.

## What is not included

- **The training data** (3,232 simulations, 91 MB) is not in this repository. The notebooks are
  saved with their outputs so that the results can be read; retraining and re-running the
  notebooks need the dataset.
- **The raft meshes and the hand-made reference model** used to build and check the case
  template are not included, so the simulation pipeline cannot be run as is. Running it also
  needs code_aster with MISS3D (the salome_meca container).
- **The site** is an imaginary seven-layer profile over bedrock; the layers are named L1 to L7.

## Repository layout

- `ml/`: the `surrogate` package, tests, notebooks 01–05, examples, the release model, the
  experiment table. See [`ml/README.md`](ml/README.md).
- `automation/`: the simulation pipeline (case generator, batch runner, collector) and its tests.
  See [`automation/README.md`](automation/README.md).
- `docs/`: the write-up, and the viewer (`docs/index.html`, served by GitHub Pages).

## How it was built

I defined the problem and the engineering decisions, built and ran the simulation campaign,
trained the models on my own GPU and checked the results against my experience in
soil–structure interaction. I used Claude (Anthropic) as a coding partner; it wrote most of the
code.

## Copyright

© 2026 Lipeng (Roy) Tong. All rights reserved. This repository is shared for viewing as a
portfolio project; no licence is granted to copy, modify or redistribute the code.
