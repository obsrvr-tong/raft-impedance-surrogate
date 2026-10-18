"""Soil profile -> DEFI_SOL_MISS keywords.

A soil profile is a CSV file with one row per layer, top to bottom:

    thickness,E,nu,rho,damping
    2.5,5561.0,0.34,0.00218,0.01
    ...
    100.0,16474.0,0.45,0.0022,0.01      <- last row = substratum (half-space)

Units must be consistent with the rest of the model. The reference model uses
MN, m, s: E in MPa (MN/m2), rho in Mg/m3 x 1e-3 (i.e. t/m3 / 1000). Check before use.
`damping` is AMOR_HYST (hysteretic damping ratio).
"""
import csv
from dataclasses import dataclass

COLUMNS = ("thickness", "E", "nu", "rho", "damping")


@dataclass
class SoilLayer:
    thickness: float
    E: float
    nu: float
    rho: float
    damping: float

    def check(self, i):
        if self.thickness <= 0:
            raise ValueError(f"layer {i}: thickness must be > 0")
        if self.E <= 0 or self.rho <= 0:
            raise ValueError(f"layer {i}: E and rho must be > 0")
        if not 0 <= self.nu < 0.5:
            raise ValueError(f"layer {i}: nu must be in [0, 0.5)")
        if not 0 <= self.damping < 1:
            raise ValueError(f"layer {i}: damping must be in [0, 1)")


def read_profile(path):
    """Read a soil profile CSV into a list of SoilLayer (top to bottom)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = set(COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
        layers = [SoilLayer(*(float(row[c]) for c in COLUMNS))
                  for row in reader if any((v or "").strip() for v in row.values())]
    if len(layers) < 2:
        raise ValueError(f"{path}: need at least one layer plus the substratum")
    for i, layer in enumerate(layers, 1):
        layer.check(i)
    return layers


def write_profile(path, layers):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for L in layers:
            w.writerow([repr(L.thickness), repr(L.E), repr(L.nu), repr(L.rho), repr(L.damping)])


def _num(x):
    return repr(float(x))


def defi_sol_miss_keywords(layers, indent=23):
    """Return the COUCHE=(...) and MATERIAU=(...) text for DEFI_SOL_MISS.

    One material per layer (NUME_MATE = layer number), as in the reference model.
    Layer 1 is RECEPTEUR and SOURCE (surface foundation); the last layer
    is the SUBSTRATUM.
    """
    pad = " " * indent
    n = len(layers)
    couche = []
    for i, L in enumerate(layers, 1):
        top = "OUI" if i == 1 else "NON"
        sub = "OUI" if i == n else "NON"
        couche.append(
            f"_F(EPAIS={_num(L.thickness)}, NUME_MATE={i}, "
            f"RECEPTEUR='{top}', SOURCE='{top}', SUBSTRATUM='{sub}')")
    mater = [f"_F(E={_num(L.E)}, NU={_num(L.nu)}, RHO={_num(L.rho)}, "
             f"AMOR_HYST={_num(L.damping)})" for L in layers]
    sep = ",\n" + pad + "    "
    return (f"COUCHE=(\n{pad}    " + sep.join(couche) + f",\n{pad}),\n"
            f"{pad}MATERIAU=(\n{pad}    " + sep.join(mater) + f",\n{pad})")
