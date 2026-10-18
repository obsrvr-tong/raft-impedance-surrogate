"""Stratum-based ground description -> MISS3D layers (settings v1: no sub-layering).

A site file (e.g. inputs/site_v1.csv) has one row per stratum, top to bottom, and the
bedrock as the last row (its thickness columns left blank):

    name, t_min, t_max, t_ref, vs, rho, nu, damping
      t_*     thickness range and reference value [m]
      vs      shear-wave velocity [m/s]
      rho     density [t/m3]
      nu      Poisson's ratio [-]
      damping written to DEFI_SOL_MISS as AMOR_HYST, unchanged

Each stratum becomes exactly one MISS3D layer; the bedrock becomes the substratum.
E [MPa] = 2 * rho * Vs^2 * (1 + nu) / 1000 ; rho in the model = rho [t/m3] * 1e-3.
"""
import csv
from dataclasses import dataclass

from .soil import SoilLayer


@dataclass
class Stratum:
    name: str
    vs: float
    rho: float          # t/m3
    nu: float
    damping: float
    t_min: float = None
    t_max: float = None
    t_ref: float = None

    def check(self):
        if self.vs <= 0 or self.rho <= 0:
            raise ValueError(f"{self.name}: vs and rho must be > 0")
        if not 0 <= self.nu < 0.5:
            raise ValueError(f"{self.name}: nu must be in [0, 0.5)")
        if self.t_min is not None and not 0 < self.t_min <= self.t_max:
            raise ValueError(f"{self.name}: need 0 < t_min <= t_max")


def youngs_modulus(vs, rho, nu):
    """E [MPa] from Vs [m/s], rho [t/m3], nu."""
    return 2.0 * rho * vs ** 2 * (1.0 + nu) / 1000.0


def read_site(path):
    """-> (list of soil Stratum top to bottom, bedrock Stratum)."""
    num = lambda v: float(v) if v not in (None, "") else None
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            s = Stratum(r["name"].strip(), float(r["vs"]), float(r["rho"]), float(r["nu"]),
                        float(r["damping"]), num(r["t_min"]), num(r["t_max"]), num(r["t_ref"]))
            s.check()
            rows.append(s)
    if len(rows) < 2:
        raise ValueError(f"{path}: need at least one stratum and the bedrock")
    soil, bedrock = rows[:-1], rows[-1]
    missing = [s.name for s in soil if s.t_min is None or s.t_max is None]
    if missing:
        raise ValueError(f"{path}: thickness range missing for {missing}")
    return soil, bedrock


def layers_for(soil, bedrock, thicknesses, g_factor=1.0, substratum_thickness=100.0):
    """One SoilLayer per stratum (Vs scaled by sqrt(g_factor), i.e. G scaled by g_factor),
    then the bedrock (not scaled) as the substratum."""
    if len(thicknesses) != len(soil):
        raise ValueError("one thickness per stratum is required")
    layers = []
    for s, t in zip(soil, thicknesses):
        vs = s.vs * g_factor ** 0.5
        layers.append(SoilLayer(float(t), youngs_modulus(vs, s.rho, s.nu), s.nu, s.rho * 1e-3, s.damping))
    layers.append(SoilLayer(substratum_thickness, youngs_modulus(bedrock.vs, bedrock.rho, bedrock.nu),
                            bedrock.nu, bedrock.rho * 1e-3, bedrock.damping))
    for i, layer in enumerate(layers, 1):
        layer.check(i)
    return layers
