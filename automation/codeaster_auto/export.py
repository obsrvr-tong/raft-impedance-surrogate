"""Write code_aster .export files.

An .export file tells run_aster how to run a job (documented in U1.04.00):

    P <parameter> <value>                   job parameters (time, memory, cpus)
    F <type> <path> <D|R|DR> <unit>         files: D = input (data), R = result,
                                            DR = both; <unit> is the Fortran
                                            logical unit the .comm refers to
                                            (e.g. LIRE_MAILLAGE(UNITE=20))

Unit numbers used by our MISS3D impedance template (they must match the .comm):
"""
from pathlib import Path

UNIT_COMM = 1    # command file
UNIT_MESS = 6    # message file (log)
UNIT_IMPE = 7    # CALC_MISS UNITE_RESU_IMPE -> impedance functions
UNIT_FORC = 8    # CALC_MISS UNITE_RESU_FORC -> seismic forces
UNIT_MESH = 20   # LIRE_MAILLAGE UNITE=20 -> .med mesh

VALID_MODES = ("D", "R", "DR")


def export_text(files, time_limit=15300, memory_limit=2048,
                ncpus=1, mpi_nbcpu=1, mpi_nbnoeud=1):
    """Return the text of an .export file.

    files: list of (ftype, path, mode, unit) tuples,
           e.g. ("mmed", "mesh.med", "D", 20).
    time_limit: seconds; memory_limit: MB.
    """
    lines = [
        f"P time_limit {int(time_limit)}",
        f"P memory_limit {int(memory_limit)}",
        f"P ncpus {int(ncpus)}",
        f"P mpi_nbcpu {int(mpi_nbcpu)}",
        f"P mpi_nbnoeud {int(mpi_nbnoeud)}",
        "",
    ]
    units_seen = set()
    for ftype, path, mode, unit in files:
        path = Path(path).as_posix()      # forward slashes, even if built on Windows
        if mode not in VALID_MODES:
            raise ValueError(f"{path}: mode must be one of {VALID_MODES}, got {mode!r}")
        if " " in path:
            raise ValueError(f"{path}: spaces are not allowed in .export paths")
        if unit in units_seen:
            raise ValueError(f"{path}: logical unit {unit} is used twice")
        units_seen.add(unit)
        lines.append(f"F {ftype} {path} {mode} {int(unit)}")
    return "\n".join(lines) + "\n"


def write_export(path, files, **params):
    """Write an .export file with Linux line endings."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(export_text(files, **params))
