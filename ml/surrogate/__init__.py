"""Surrogate models for MISS3D soil-impedance matrices (dataset v1).

Modules:
  data         loading dataset v1, reference-point transfer, target arrays for the models
  models       input features, scaling, kNN, PCA + GP, the two-part static x shape model (scikit-learn)
  gpu          the same GP models trained in batches on the GPU (GPyTorch)
  metrics      the curve error metric and its summaries
  experiments  the experiment table (results/experiments.csv)
  portable     the trained model as one numpy-only file: predict curves with an uncertainty band
  release      train, test and save the release model
  viewer       the interactive HTML viewer;  api: the prediction API (FastAPI)
  analysis     sensitivity indices (Sobol) and Monte Carlo through the surrogate

Command line: python -m surrogate {predict, train, viewer, serve} --help
"""
__version__ = "1.0.0"
