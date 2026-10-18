# ml/: the impedance surrogate

Machine-learning part of the repository: Gaussian-process surrogates trained on dataset v1
(3,232 MISS3D impedance cases, five raft footprints) that predict the 6 × 6 complex impedance
of a rigid surface raft on layered soil, 0.45–15 Hz, in milliseconds, with an uncertainty band.
The story and the numbers are in `docs/ml_writeup.md`.

## A prediction in one command

    conda env create -f ml/environment.yml && conda activate ml && pip install -e ml
    python -m surrogate predict ml/examples/nominal_45x15.json --band

This prints the static stiffness of the 8 terms with their 95 % band; `--csv curves.csv` writes
the full curves. The inputs are a JSON object (one case) or list (several) with the 10 inputs
`t_L1, t_L2, t_L3, t_L4, t_L5, t_L6, t_L7` (stratum thicknesses, m), `g_factor` (shear-modulus
factor on all strata) and `L_x, L_y` (raft size, m). Inputs outside the training range are
flagged. The model file `models/impedance_surrogate_v1.npz` needs only numpy.

Other ways to use the same model:

    python -m surrogate serve        # prediction API: http://127.0.0.1:8000/docs (POST /predict)
    python -m surrogate viewer       # ml/viewer/index.html: interactive page, works offline, one file
    docker build -t impedance-api ml && docker run --rm -p 8000:8000 impedance-api

and from Python:

    from surrogate.portable import ImpedanceSurrogate
    sur = ImpedanceSurrogate.load()
    p = sur.predict({"t_L1": 7.0, "t_L2": 5.0, "t_L3": 14.0, "t_L4": 3.5, "t_L5": 17.0,
                     "t_L6": 14.0, "t_L7": 38.0, "g_factor": 1.0, "L_x": 45, "L_y": 15}, band=True)
    p.K, p.lo, p.hi, p.warnings     # (n, 59, 8) complex curves, band edges, range warnings

To retrain the release model (needs the dataset; about 30 minutes on the RTX 5070 Ti, far longer
on a CPU):

    python -m surrogate train --log  # 512 soils x 5 footprints + rotated twins; tests and saves it

To compare settings without touching the hold-out cases, test on unused training soils instead
(the model goes to `models/validation_*.npz`, never over the release model):

    python -m surrogate train --soils 256 --components 40 --validate 64

A model saved before 6 October 2026 lacks the PCA truncation term of the band; add it (seconds):

    python -m surrogate calibrate --log

## Layout
- `surrogate/`: the package (imported as `surrogate`)
  - `data.py`: load `dataset_v1.npz`, move ref33x31 to its centroid, optional symmetrising and
    frequency cut, rotated twins, and the complex 6×6 matrices as real target rows (and back)
  - `style.py`: shared plot style (one colour per footprint, one sequential ramp)
  - `metrics.py`: the error metric (max |ΔK| / max |K| per term) and its summaries
  - `models.py`: input features, per-term target scaling, kNN, PCA + GP (`PCAGP`), the two-part
    static × shape model (`StaticShapeGP`), optional linear trend under the GPs
  - `gpu.py`: the same GP models trained as batches on the GPU (GPyTorch)
  - `experiments.py`: idempotent logging to `results/experiments.csv`, with the git commit
  - `portable.py`: the trained model as one numpy-only file; predictions with a 95 % band
    (each GP's variance from its 128 nearest training cases, an upper bound on the exact one,
    plus the PCA truncation error measured on the training curves)
  - `release.py`: train, export, test and save the release model (`python -m surrogate train`);
    validation runs; `calibrate` for older model files
  - `analysis.py`: Sobol sensitivity indices and Monte Carlo through the surrogate
  - `viewer.py`, `viewer_template.html`: the interactive page (the model runs in the browser)
  - `api.py`: the FastAPI app; `__main__.py`: the command line
- `notebooks/`: one notebook per step
  - `01_look.ipynb`: data card, input coverage, physics checks, curves, findings for Milestone 2
  - `02_first_surrogate.ipynb`: Milestone 2 on ref33x31: baselines, PCA, GP, error by band,
    learning curves, length scales
  - `03_footprints.ipynb`: Milestone 3, five footprints: two representations, leave-one-footprint-out,
    two fixes for new footprints (linear trend, rotated twins); CPU, ~15-30 min
  - `04_gpu_scaling.ipynb`: Milestone 3b, the fixed recipe on the GPU with all the soils
  - `05_using_the_surrogate.ipynb`: Milestone 5: model card, surrogate against MISS3D, calibration
    of the band, Sobol indices, Monte Carlo on soil uncertainty, speed
- `examples/`: input files for `python -m surrogate predict`
- `models/impedance_surrogate_v1.npz`: the release model (in git; other files in `models/` are not)
- `viewer/index.html`: the built viewer (`python -m surrogate viewer`)
- `results/experiments.csv`: one row per experiment, from the first baseline on
- `tests/`: `python -m unittest discover ml/tests -v` (run from the repo root)
- `environment.yml`, `pyproject.toml`: the environment recipe and the package definition
- `Dockerfile`, `requirements-api.txt`: the API as a container (numpy + FastAPI only)

## Set-up (Windows)
    conda env create -f ml/environment.yml      # or, in the existing `ml` environment:
                                                #   pip install fastapi==0.142.2 uvicorn==0.53.0 httpx==0.28.1
    conda activate ml
    pip install -e ml                           # once: makes `import surrogate` work everywhere
    python -m unittest discover ml/tests -v

The training data are not included in this repository (see the top-level README). Prediction,
the API and the viewer need only the model file. Training and re-running the notebooks need the
dataset `dataset_v1.npz` (SHA-256 `9718d80bf8ee132d4e9c717ae2c7d46bccb5f4b68fa8f1a6966cd32d8534fe20`)
at `automation/datasets/v1/` or at the path in the environment variable `IMPEDANCE_DATASET`.

## Conventions
- Units MN, m, s. DOF order Ux, Uy, Uz, Rx, Ry, Rz. `K[case, freq, i, j]`.
- **Frequencies:** the dataset holds 133 (0.2–33.2 Hz, step 0.25 Hz). Modelling uses 59, from
  0.45 to 14.95 Hz (decided 4 October 2026): the 0.2 Hz point is numerically unreliable in a
  minority of cases (notebook 01, §3a), and above 15–20 Hz impedances fluctuate strongly, in
  MISS3D (notebook 02, §5) as in full FE models, while the band that matters for SSI of these
  buildings lies below 15 Hz. `load_dataset()` applies this by default (`FMIN`, `FMAX`);
  `fmin=None, fmax=None` keeps all 133. Notebook 02 used 0.45–33.2 Hz.
- After `load_dataset()`, every K refers to its raft's centroid (only ref33x31 is transferred).
- **Rotated twins:** `rotate_90(ds)` gives every case as seen in axes turned by 90° (L_x and L_y
  swap, K → Q K Qᵀ); exact on horizontally layered soil, used as data augmentation.
- **Targets:** 8 terms (Ux, Uy, Uz, Rx, Ry, Rz, Ux–Ry, Uy–Rx). A target row lists them term
  after term; within a term, Re then Im; within each, the 59 frequencies: 944 numbers per case.
  `TargetSpec.columns(term, part)` finds any curve.
- **Inputs:** `X` = t_L1, t_L2, t_L3, t_L4, t_L5, t_L6, t_L7, g_factor, L_x, L_y;
  derived `D` = H_bedrock, vs_avg, f0.
- **Im K < 0:** MISS3D uses the time convention e^(−iωt).

## Experiment table
`results/experiments.csv` gets one row per experiment, written by `log_experiment` (re-running a
notebook replaces its rows, keyed by `exp_id`). Columns:

- `footprints`, `validation`, `band_hz`, `n_train`, `n_test`: what was trained and how it was tested;
- `representation`, `model`: the target representation and the regressor;
- `err_<term>`: for each of the 8 terms, the 95th percentile over the test cases of
  e = 100 · max_f |K̂ − K| / max_f |K| (in %), the deviation measured within `band_hz`;
- `err_worst`: the largest `err_<term>`; `err_mean`: the mean of e over all test cases and terms;
- `fit_s`, `predict_ms_per_case`: timing on the machine that ran it;
- `git_commit`: HEAD when the row was written, `+dirty` if `ml/surrogate` had uncommitted changes.
