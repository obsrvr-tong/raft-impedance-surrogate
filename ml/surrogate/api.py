"""Prediction API: soil and raft inputs as JSON in, impedance curves out.

    python -m surrogate serve                     # or: uvicorn surrogate.api:app
    # then open http://127.0.0.1:8000/docs for the interactive documentation

    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
         -d '{"cases": [{"t_L1": 7.0, "t_L2": 5.0, "t_L3": 14.0, "t_L4": 3.5, "t_L5": 17.0,
                         "t_L6": 14.0, "t_L7": 38.0, "g_factor": 1.0, "L_x": 45, "L_y": 15}],
              "band": true}'

Endpoints: GET /health, GET /model (model card: inputs and their training ranges, terms,
frequencies, test results), POST /predict, GET /viewer (the interactive page).
The model is the portable file (surrogate.portable), loaded once, on the first request; set
IMPEDANCE_SURROGATE_MODEL to use another file. Needs only numpy, fastapi and uvicorn.

Design choices: inputs outside the training range are accepted, but every such input is listed
in the case's "warnings" (the GP then extrapolates, and its band widens); non-positive sizes or
stiffness factors are rejected by validation (HTTP 422).
"""
from __future__ import annotations

import os
import time
from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .portable import DEFAULT_MODEL, NOMINAL, ImpedanceSurrogate

_thk = "Thickness of stratum {} [m]"


class Case(BaseModel):
    """One soil profile and raft. Training ranges are listed by GET /model."""
    t_L1: float = Field(gt=0, description=_thk.format("L1 (top)"), examples=[NOMINAL["t_L1"]])
    t_L2: float = Field(gt=0, description=_thk.format("L2"), examples=[NOMINAL["t_L2"]])
    t_L3: float = Field(gt=0, description=_thk.format("L3"), examples=[NOMINAL["t_L3"]])
    t_L4: float = Field(gt=0, description=_thk.format("L4"), examples=[NOMINAL["t_L4"]])
    t_L5: float = Field(gt=0, description=_thk.format("L5"), examples=[NOMINAL["t_L5"]])
    t_L6: float = Field(gt=0, description=_thk.format("L6"), examples=[NOMINAL["t_L6"]])
    t_L7: float = Field(gt=0, description=_thk.format("L7 (above the bedrock)"), examples=[NOMINAL["t_L7"]])
    g_factor: float = Field(gt=0, description="Shear-modulus factor on all seven strata (Vs x sqrt(g)) [-]",
                            examples=[1.0])
    L_x: float = Field(gt=0, description="Raft length along x [m]", examples=[45.0])
    L_y: float = Field(gt=0, description="Raft width along y [m]", examples=[15.0])


class PredictRequest(BaseModel):
    cases: list[Case] = Field(min_length=1, max_length=1000)
    band: bool = Field(False, description="Add an uncertainty band (slower: ~0.1-0.5 s per case)")
    level: float = Field(95.0, gt=0, lt=100, description="Band level in %")


class TermCurves(BaseModel):
    re: list[float]
    im: list[float]
    re_lo: list[float] | None = None
    re_hi: list[float] | None = None
    im_lo: list[float] | None = None
    im_hi: list[float] | None = None


class CaseResult(BaseModel):
    static: dict[str, float] = Field(description="Re K at the lowest frequency, per term")
    terms: dict[str, TermCurves]
    warnings: list[str]


class PredictResponse(BaseModel):
    model: str
    freqs: list[float]
    units: dict[str, str]
    results: list[CaseResult]
    elapsed_ms: float


UNITS = {"Ux": "MN/m", "Uy": "MN/m", "Uz": "MN/m", "Rx": "MN.m/rad", "Ry": "MN.m/rad", "Rz": "MN.m/rad",
         "Ux-Ry": "MN/rad", "Uy-Rx": "MN/rad"}


@lru_cache(maxsize=1)
def get_model() -> ImpedanceSurrogate:
    return ImpedanceSurrogate.load(os.environ.get("IMPEDANCE_SURROGATE_MODEL", DEFAULT_MODEL))


app = FastAPI(title="Soil impedance surrogate",
              description="Gaussian-process surrogate of MISS3D impedance matrices of rigid surface rafts "
                          "on layered soil: 8 terms (6 diagonal + 2 rocking couplings), 0.45-15 Hz.",
              version="1.0")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return ("<h1>Soil impedance surrogate</h1><p><a href='/docs'>API documentation</a> &middot; "
            "<a href='/viewer'>interactive viewer</a> &middot; <a href='/model'>model card</a></p>")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_card():
    return get_model().info()


@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True)
def predict(req: PredictRequest):
    sur = get_model()
    t0 = time.perf_counter()
    p = sur.predict([c.model_dump() for c in req.cases], band=req.band, level=req.level)
    results = []
    for i in range(len(req.cases)):
        d = p.case(i)
        results.append({"static": d["static"], "terms": d["terms"], "warnings": d["warnings"]})
    return {"model": sur.meta.get("name", "surrogate"), "freqs": p.freqs.tolist(), "units": UNITS,
            "results": results, "elapsed_ms": 1000 * (time.perf_counter() - t0)}


@app.get("/viewer", response_class=HTMLResponse, include_in_schema=False)
def viewer():
    return _viewer_html()


@lru_cache(maxsize=1)
def _viewer_html():
    from .viewer import viewer_html
    return viewer_html(get_model(), holdout=None)
