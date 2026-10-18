"""Run code_aster case folders and read their outcome from the .mess file.

A case folder = a folder with exactly one .export file (as written by make_case.py).
The job's result is taken from the last line code_aster writes in the .mess file:
    --- DIAGNOSTIC JOB : <A>_ALARM
"""
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# DIAGNOSTIC JOB value -> short status used in summaries
STATUS_OF = {
    "OK": "OK",
    "<A>_ALARM": "OK (alarms)",
    "NOOK_TEST_RESU": "RESULT CHECK FAILED",
    "NO_TEST_RESU": "OK",
}

_DIAG = re.compile(r"DIAGNOSTIC JOB\s*:\s*(\S+)")
_ALARM = re.compile(r"<A>\s*<([A-Z0-9_]+)>")
_ERROR = re.compile(r"<(?:F|S|E|EXCEPTION)>\s*<([A-Z0-9_]+)>")


def export_files(export_path):
    """Return {ftype: [(path, mode, unit), ...]} from an .export file."""
    files = {}
    for line in Path(export_path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] == "F":
            _, ftype, path, mode, unit = parts
            files.setdefault(ftype, []).append((path, mode, int(unit)))
    return files


def find_cases(root):
    """Case folders under root (one level), sorted by name."""
    cases = []
    for d in sorted(Path(root).iterdir()):
        if d.is_dir():
            exports = list(d.glob("*.export"))
            if len(exports) == 1:
                cases.append(d)
    return cases


def case_paths(case_dir):
    """(export, mess, result files) for a case folder; paths resolved against the folder."""
    case_dir = Path(case_dir)
    export = next(case_dir.glob("*.export"))
    files = export_files(export)
    resolve = lambda p: (case_dir / p) if not Path(p).is_absolute() else Path(p)
    mess = resolve(files["mess"][0][0]) if "mess" in files else None
    results = [resolve(p) for ftype, lst in files.items() if ftype != "mess"
               for p, mode, _ in lst if "R" in mode]
    return export, mess, results


def read_mess(mess_path):
    """Return dict(diagnostic, status, alarms, errors) from a .mess file."""
    out = {"diagnostic": "", "status": "NOT RUN", "alarms": "", "errors": ""}
    if mess_path is None or not Path(mess_path).is_file():
        return out
    text = Path(mess_path).read_text(encoding="utf-8", errors="replace")
    diags = _DIAG.findall(text)
    if not diags:
        out["status"] = "INCOMPLETE (no diagnostic)"
    else:
        d = diags[-1]
        out["diagnostic"] = d
        out["status"] = STATUS_OF.get(d, "FAILED")
    out["alarms"] = " ".join(sorted(set(_ALARM.findall(text))))
    out["errors"] = " ".join(sorted(set(_ERROR.findall(text))))
    return out


def is_success(status):
    return status.startswith("OK")


# A case being run holds a RUNNING marker file (pid, host, start time), so a second runner
# started on the same folder skips it instead of running it twice at the same time.
MARKER = "RUNNING"


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running_pid(case_dir):
    """PID of the live process running this case (from its RUNNING marker), else None."""
    marker = Path(case_dir) / MARKER
    try:
        pid, host = marker.read_text(encoding="utf-8").split()[:2]
        pid = int(pid)
    except (OSError, ValueError):
        return None
    return pid if host == socket.gethostname() and _alive(pid) else None


def claim(case_dir):
    """Atomically mark the case as run by this process. False if a live process already has it."""
    marker = Path(case_dir) / MARKER
    for _ in range(2):
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if running_pid(case_dir) is not None:
                return False
            try:
                marker.unlink()                  # stale marker of a killed run
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()} {socket.gethostname()} {datetime.now():%Y-%m-%dT%H:%M:%S}\n")
        return True
    return False


def release(case_dir):
    try:
        (Path(case_dir) / MARKER).unlink()
    except FileNotFoundError:
        pass


def run_command(cmd, cwd, log, timeout=None):
    """Run cmd in cwd, output to the open file log, and wait for it.

    Returns (return code, or "timeout", and peak memory in MB or None). The peak is the
    largest resident memory reached by any single process of the job: on Linux, os.wait4
    reports it for the whole process tree (wrapper, run_aster, aster and the MISS3D solver
    it starts). On a timeout only the direct child (the wrapper) is killed.
    """
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    expired = threading.Event()
    timer = None
    if timeout:
        def stop():
            expired.set()
            proc.kill()
        timer = threading.Timer(timeout, stop)
        timer.daemon = True
        timer.start()
    peak = None
    try:
        if hasattr(os, "wait4"):
            _, status, usage = os.wait4(proc.pid, 0)
            if timer:
                timer.cancel()
            proc.returncode = os.waitstatus_to_exitcode(status)
            if sys.platform.startswith("linux"):
                peak = round(usage.ru_maxrss / 1024, 1)      # Linux reports KiB
        else:                                                 # Windows: no rusage
            proc.wait()
    finally:
        if timer:
            timer.cancel()
    return ("timeout" if expired.is_set() else proc.returncode), peak


def run_case(case_dir, aster_cmd, timeout=None):
    """Run one case with `<aster_cmd> run_aster <name>.export` inside the case folder.

    Console output goes to run.log in the case folder. Returns a result dict, including
    elapsed_s (wall-clock seconds) and peak_mem_mb (see run_command).
    """
    case_dir = Path(case_dir)
    export, mess, results = case_paths(case_dir)
    if not claim(case_dir):
        return {"case": case_dir.name, "status": "RUNNING ELSEWHERE"}
    try:
        if mess is not None and mess.exists():
            mess.unlink()                 # never read a stale .mess from an earlier run
        started = datetime.now()
        t0 = time.time()
        with open(case_dir / "run.log", "w", encoding="utf-8") as log:
            rc, peak = run_command([str(aster_cmd), "run_aster", export.name], case_dir, log, timeout)
        elapsed = time.time() - t0
    finally:
        release(case_dir)
    res = read_mess(mess)
    if rc == "timeout":
        res["status"] = "TIMEOUT"
    res.update({
        "case": case_dir.name,
        "return_code": rc,
        "elapsed_s": round(elapsed, 1),
        "peak_mem_mb": "" if peak is None else peak,
        "started": started.isoformat(timespec="seconds"),
        "results_ok": all(p.is_file() and p.stat().st_size > 0 for p in results),
    })
    if is_success(res["status"]) and not res["results_ok"]:
        res["status"] = "FAILED (missing results)"
    return res
