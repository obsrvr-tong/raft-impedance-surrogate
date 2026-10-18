"""Run every case folder in cases/ with code_aster, one after another (or N at a time).

    python3 run_cases.py                      # run all cases not yet finished successfully
    python3 run_cases.py --only S200_H10_R1500 S400_H20_R1500
    python3 run_cases.py --only 'm120x75_*'   # shell-style patterns (keep the quotes)
    python3 run_cases.py --rerun              # run everything again
    python3 run_cases.py --jobs 2             # two cases at the same time
    python3 run_cases.py --order index        # sample 1 of every mesh, then sample 2, ...
    python3 run_cases.py --status             # just report the status of every case

Each case runs as  aster run_aster <case>.export  inside its folder; console output
goes to <case>/run.log. The outcome is read from the DIAGNOSTIC JOB line of the .mess.
Progress is written to cases/run_summary.csv after every case (with run time and peak
memory), so an interrupted batch (Ctrl+C) can be resumed by simply running the command
again. Rows of cases outside the current selection are kept, so runs with different
--only selections (even two at the same time) can share one cases folder.

A case being run holds a RUNNING marker file; other runners skip it. After a crash or a
reboot, stale markers are ignored automatically (their process no longer exists).

--order index interleaves dataset cases (<mesh>_train0001, <mesh>_test0001, ...): every
mesh gets its sample 1, then its sample 2, and so on, largest mesh first within each round.
If a batch has to be stopped early, every mesh then has the first N points of its
(space-filling) sequence, and the large and small meshes share the memory over the night.
"""
import argparse
import csv
import fnmatch
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

try:
    import fcntl
except ImportError:                 # Windows: no file locking (the runner is meant for Linux)
    fcntl = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from codeaster_auto.runner import (case_paths, find_cases, is_success,  # noqa: E402
                                   read_mess, run_case, running_pid)

COLS = ["case", "status", "diagnostic", "elapsed_s", "peak_mem_mb", "started", "return_code",
        "results_ok", "alarms", "errors"]
KEEP = ("elapsed_s", "peak_mem_mb", "started", "return_code", "results_ok")
SAMPLE = re.compile(r"^(.*)_(train|test)(\d+)$")      # dataset case names
WAITING = ("NOT RUN", "QUEUED", "RUNNING")


def fmt_time(s):
    s = int(round(float(s or 0)))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def fmt_mem(mb):
    return f"{float(mb):.0f} MB" if mb not in (None, "") else "-"


def group_of(name):
    """Mesh part of a dataset case name (m120x75_train0007 -> m120x75); other cases: '(other)'."""
    m = SAMPLE.match(name)
    return m.group(1) if m else "(other)"


def index_key(case_dir):
    """Sort key for --order index: sample number, then the largest mesh first, then the name."""
    m = SAMPLE.match(case_dir.name)
    idx = int(m.group(3)) if m else 0
    cells = 0
    try:
        rec = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        cells = int(float(rec.get("mesh", {}).get("n_cells") or 0))
    except (OSError, ValueError, AttributeError):
        pass
    return (idx, -cells, case_dir.name)


def read_summary(path):
    if not Path(path).is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["case"]: r for r in csv.DictReader(f)}


@contextmanager
def locked(path):
    """Exclusive lock (Linux) around the read-merge-write of the summary file."""
    with open(path, "a") as fh:
        if fcntl:
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl:
                fcntl.flock(fh, fcntl.LOCK_UN)


def write_summary(path, rows, root):
    """Merge rows into the summary CSV. Rows of other cases already in the file are kept
    (while their folder exists), so runs with different selections don't wipe each other."""
    path = Path(path)
    with locked(path.with_name(path.name + ".lock")):
        merged = {k: v for k, v in read_summary(path).items() if (Path(root) / k).is_dir()}
        merged.update(rows)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            for name in sorted(merged):
                w.writerow(merged[name])
        os.replace(tmp, path)


def current_status(case_dir):
    _, mess, _ = case_paths(case_dir)
    r = read_mess(mess)
    r["case"] = case_dir.name
    pid = running_pid(case_dir)
    if pid is not None:
        r["status"] = f"RUNNING (pid {pid})"
    return r


def print_table(rows):
    print(f"\n{'case':<24}{'status':<28}{'time':>8}{'memory':>10}  alarms / errors")
    for name in sorted(rows):
        r = rows[name]
        t = fmt_time(r["elapsed_s"]) if r.get("elapsed_s") not in (None, "") else "-"
        print(f"{name:<24}{r['status']:<28}{t:>8}{fmt_mem(r.get('peak_mem_mb')):>10}  "
              f"{r.get('errors') or r.get('alarms') or ''}")


def print_groups(rows):
    """One line per mesh: case counts, and run time / peak memory of the successful cases."""
    groups = defaultdict(list)
    for name, r in rows.items():
        groups[group_of(name)].append(r)
    print(f"\n{'mesh':<14}{'cases':>6}{'OK':>6}{'failed':>7}{'running':>8}{'to run':>7}"
          f"{'mean time':>11}{'max time':>10}{'peak mem':>10}")
    for g in sorted(groups):
        rs = groups[g]
        ok = [r for r in rs if is_success(r["status"])]
        running = [r for r in rs if r["status"].startswith("RUNNING")]
        waiting = [r for r in rs if r["status"].startswith(WAITING)]
        t = [float(r["elapsed_s"]) for r in ok if r.get("elapsed_s") not in (None, "")]
        m = [float(r["peak_mem_mb"]) for r in ok if r.get("peak_mem_mb") not in (None, "")]
        print(f"{g:<14}{len(rs):>6}{len(ok):>6}{len(rs) - len(ok) - len(waiting):>7}{len(running):>8}"
              f"{len(waiting) - len(running):>7}"
              f"{fmt_time(sum(t) / len(t)) if t else '-':>11}{fmt_time(max(t)) if t else '-':>10}"
              f"{fmt_mem(max(m)) if m else '-':>10}")


def report(rows, full):
    """Per-case table (all cases if few or --table, else only the failed ones) + per-mesh lines."""
    if full or len(rows) <= 40:
        print_table(rows)
    else:
        failed = {n: r for n, r in rows.items()
                  if not is_success(r["status"]) and not r["status"].startswith(WAITING)}
        if failed:
            print("\nCases that did not finish OK:")
            print_table(failed)
    print_groups(rows)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--cases", default=str(HERE / "cases"), help="folder containing the case folders")
    a.add_argument("--only", nargs="+", metavar="CASE", help="case names or patterns such as 'm80x20_*'")
    a.add_argument("--rerun", action="store_true", help="also re-run cases that already succeeded")
    a.add_argument("--jobs", type=int, default=1, help="cases run at the same time (default 1)")
    a.add_argument("--order", choices=("name", "index"), default="name",
                   help="run order: by folder name (default) or interleaved by sample index")
    a.add_argument("--timeout", type=float, default=None, help="seconds per case (default: none)")
    a.add_argument("--aster", default=str(HERE / "bin" / "aster"), help="the aster wrapper")
    a.add_argument("--status", action="store_true", help="only report the status of each case")
    a.add_argument("--table", action="store_true", help="always print the full per-case table")
    args = a.parse_args()

    root = Path(args.cases)
    if not root.is_dir():
        sys.exit(f"cases folder not found: {root}")
    cases = find_cases(root)
    if args.only:
        unmatched = [p for p in args.only if not any(fnmatch.fnmatchcase(c.name, p) for c in cases)]
        if unmatched:
            sys.exit(f"--only: no case folder in {root} matches {unmatched}")
        cases = [c for c in cases if any(fnmatch.fnmatchcase(c.name, p) for p in args.only)]
    if not cases:
        sys.exit(f"no case folders (with one .export) found in {root}")

    rows = {c.name: current_status(c) for c in cases}
    summary = root / "run_summary.csv"
    # keep timings of earlier runs where available
    for name, old in read_summary(summary).items():
        if name in rows and rows[name]["status"] == old.get("status"):
            rows[name].update({k: old.get(k, "") for k in KEEP})

    if args.status:
        report(rows, args.table)
        return

    busy = [c for c in cases if rows[c.name]["status"].startswith("RUNNING")]
    todo = [c for c in cases if c not in busy and (args.rerun or not is_success(rows[c.name]["status"]))]
    if args.order == "index":
        todo.sort(key=index_key)
    skipped = len(cases) - len(todo) - len(busy)
    if busy:
        print(f"{len(busy)} case(s) are being run by another runner and are left to it: "
              f"{', '.join(c.name for c in busy[:5])}{' ...' if len(busy) > 5 else ''}")
    print(f"{len(todo)} case(s) to run, {skipped} already done (use --rerun to repeat them), "
          f"{args.jobs} at a time, order: {args.order}. Started {datetime.now():%H:%M:%S}. Ctrl+C to stop.")
    t0 = time.time()
    done = 0
    pool = ThreadPoolExecutor(max_workers=max(1, args.jobs))
    futures = {}
    try:
        for c in todo:
            rows[c.name]["status"] = "QUEUED"
            futures[pool.submit(run_case, c, args.aster, args.timeout)] = c
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                res = fut.result()
            except Exception as e:                      # runner problem, not code_aster
                res = {"case": c.name, "status": f"RUNNER ERROR: {e}"}
            rows[c.name] = res
            done += 1
            write_summary(summary, rows, root)
            wall = time.time() - t0
            left = wall / done * (len(todo) - done)
            print(f"[{done}/{len(todo)}] {c.name:<24}{res['status']:<28}"
                  f"{fmt_time(res.get('elapsed_s', 0)):>8}{fmt_mem(res.get('peak_mem_mb')):>10}"
                  f"   {fmt_time(wall)} elapsed, ~{fmt_time(left)} left", flush=True)
        pool.shutdown()
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)   # queued cases are not started
        for fut, c in futures.items():
            if rows[c.name]["status"] != "QUEUED":
                continue                                  # already reported above
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                rows[c.name] = fut.result()               # finished just as Ctrl+C came
            else:
                rows[c.name]["status"] = "NOT RUN (interrupted)"
        print("\nInterrupted. Finished cases are saved; run the same command to resume.")
    write_summary(summary, rows, root)
    report(rows, args.table)
    ok = sum(is_success(r["status"]) for r in rows.values())
    print(f"\n{ok}/{len(rows)} case(s) OK. Batch time {fmt_time(time.time() - t0)}. Summary: {summary}")


if __name__ == "__main__":
    main()
