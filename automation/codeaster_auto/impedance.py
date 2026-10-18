"""Read CALC_MISS impedance files (UNITE_RESU_IMPE, TYPE='ASCII').

File layout (as parsed by the earlier old-tool/impe2csv.py):

    ... CHAMP U IMPOSE  <j>  CHAMP T IMPOSE  <i> ...    block header
    <freq>  <real>  <imag>                               one line per frequency
    ...

Block (U=j, T=i) is the impedance term K[i][j]: force/moment i due to a unit
displacement/rotation j of the reference node. DOF order is that of the
MODE_STATIQUE static modes on NRADIER: 1..6 = DX, DY, DZ, DRX, DRY, DRZ (assumed).
Units in our models (MN, m, s): MN/m, MN/rad, MN.m/rad.
"""
import csv
import re

_HEADER = re.compile(r"CHAMP\s+U\s+IMPOSE\s+(\d+)\s+CHAMP\s+T\s+IMPOSE\s+(\d+)")
DOF_LABELS = ["Ux", "Uy", "Uz", "Rx", "Ry", "Rz"]


def _num(s):
    return float(s.replace("D", "E").replace("d", "e"))


def read_impe(path):
    """Return dict(freqs, n, re, im, missing).

    re[k][i][j], im[k][i][j]: real/imaginary part of K_ij at freqs[k] (0-based i, j).
    missing: list of (i, j) 1-based terms absent from the file (left as 0.0).
    """
    blocks, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _HEADER.search(line)
            if m:
                u, t = int(m.group(1)), int(m.group(2))
                cur = (t, u)                      # (row i = T, column j = U)
                if cur in blocks:
                    raise ValueError(f"{path}: term K{t}{u} appears twice")
                blocks[cur] = []
                continue
            if cur is None:
                continue
            parts = line.split()
            if len(parts) == 3:
                try:
                    blocks[cur].append(tuple(_num(p) for p in parts))
                except ValueError:
                    pass                          # a text line that happens to have 3 words
    if not blocks:
        raise ValueError(f"{path}: no 'CHAMP U IMPOSE ... CHAMP T IMPOSE ...' blocks found")

    n = max(max(k) for k in blocks)
    ref_key = min(blocks)
    freqs = [v[0] for v in blocks[ref_key]]
    for key, rows in blocks.items():
        fk = [v[0] for v in rows]
        if len(fk) != len(freqs) or any(abs(a - b) > 1e-6 * max(1.0, abs(b)) for a, b in zip(fk, freqs)):
            raise ValueError(f"{path}: term K{key[0]}{key[1]} has a different frequency list "
                             f"({len(fk)} values) from K{ref_key[0]}{ref_key[1]} ({len(freqs)})")
    re_ = [[[0.0] * n for _ in range(n)] for _ in freqs]
    im_ = [[[0.0] * n for _ in range(n)] for _ in freqs]
    for (i, j), rows in blocks.items():
        for k, (_, r, im) in enumerate(rows):
            re_[k][i - 1][j - 1] = r
            im_[k][i - 1][j - 1] = im
    missing = [(i, j) for i in range(1, n + 1) for j in range(1, n + 1) if (i, j) not in blocks]
    return {"freqs": freqs, "n": n, "re": re_, "im": im_, "missing": missing}


def write_csv(data, path):
    """One row per frequency: frequency, K11_re, K11_im, K12_re, ... (row-major)."""
    n = data["n"]
    cols = ["frequency"] + [f"K{i}{j}_{p}" for i in range(1, n + 1) for j in range(1, n + 1)
                            for p in ("re", "im")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for k, fr in enumerate(data["freqs"]):
            row = [fr]
            for i in range(n):
                for j in range(n):
                    row += [data["re"][k][i][j], data["im"][k][i][j]]
            w.writerow(row)
