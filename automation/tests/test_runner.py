"""Batch runner checks, using a fake 'aster' command instead of code_aster."""
import csv
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from codeaster_auto.runner import read_mess, run_case  # noqa: E402
from make_case import make_case  # noqa: E402
from run_cases import index_key  # noqa: E402
from codeaster_auto.strata import layers_for, read_site  # noqa: E402


# Fake aster: reads the .export, writes the .mess (and result files) like code_aster would.
# The diagnostic to write is taken from a file FAKE_DIAG in the case folder (default <A>_ALARM).
# FAKE_SLEEP (seconds) makes it slow; FAKE_MEM (MB) makes a child process (like MISS3D) use memory.
FAKE = r'''#!/usr/bin/env python3
import sys, pathlib, subprocess, time
export = pathlib.Path(sys.argv[2])
diag = pathlib.Path("FAKE_DIAG").read_text().strip() if pathlib.Path("FAKE_DIAG").exists() else "<A>_ALARM"
if pathlib.Path("FAKE_SLEEP").exists():
    time.sleep(float(pathlib.Path("FAKE_SLEEP").read_text()))
if pathlib.Path("FAKE_MEM").exists():
    mb = int(pathlib.Path("FAKE_MEM").read_text())
    subprocess.run([sys.executable, "-c", "b = b'x' * (%d * 1024 * 1024)" % mb], check=True)
for line in export.read_text().splitlines():
    p = line.split()
    if len(p) == 5 and p[0] == "F" and "R" in p[3]:
        if p[1] == "mess":
            pathlib.Path(p[2]).write_text("<A> <MESH3_2>\n--- DIAGNOSTIC JOB : %s\n" % diag)
        elif diag != "<F>_ERROR":
            pathlib.Path(p[2]).write_text("results\n")
print("fake run of", export)
'''


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.fake = t / "fake_aster"
        self.fake.write_text(FAKE)
        self.fake.chmod(self.fake.stat().st_mode | stat.S_IEXEC)
        self.cases = t / "cases"
        mesh = t / "raft.med"
        mesh.write_bytes(b"dummy mesh: make_case only copies it")
        soil, bedrock = read_site(HERE / "inputs" / "site_v1.csv")
        layers = layers_for(soil, bedrock, [s.t_ref for s in soil])
        for name in ("c1", "c2", "c3"):
            make_case(name, mesh, layers, out_dir=self.cases)
        (self.cases / "c2" / "FAKE_DIAG").write_text("<F>_ERROR")

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_mess(self):
        m = self.cases / "c1" / "x.mess"
        m.write_text("<A> <MESH3_2>\n<F> <SUPERVIS_1>\n--- DIAGNOSTIC JOB : <F>_ERROR\n")
        r = read_mess(m)
        self.assertEqual(r["status"], "FAILED")
        self.assertEqual(r["alarms"], "MESH3_2")
        self.assertEqual(r["errors"], "SUPERVIS_1")

    def test_run_case_ok_and_failed(self):
        self.assertEqual(run_case(self.cases / "c1", self.fake)["status"], "OK (alarms)")
        self.assertEqual(run_case(self.cases / "c2", self.fake)["status"], "FAILED")

    @unittest.skipUnless(sys.platform.startswith("linux"), "peak memory is measured on Linux")
    def test_peak_memory_of_child_process(self):
        (self.cases / "c1" / "FAKE_MEM").write_text("120")
        res = run_case(self.cases / "c1", self.fake)
        self.assertEqual(res["status"], "OK (alarms)")
        self.assertGreater(res["peak_mem_mb"], 110)     # the grandchild's memory is seen
        res3 = run_case(self.cases / "c3", self.fake)
        self.assertLess(res3["peak_mem_mb"], 100)

    def test_timeout(self):
        (self.cases / "c1" / "FAKE_SLEEP").write_text("30")
        res = run_case(self.cases / "c1", self.fake, timeout=1)
        self.assertEqual(res["status"], "TIMEOUT")
        self.assertLess(res["elapsed_s"], 10)

    def test_only_patterns_keep_other_rows(self):
        base = [sys.executable, str(HERE / "run_cases.py"), "--cases", str(self.cases),
                "--aster", str(self.fake)]
        subprocess.run(base, capture_output=True, text=True, check=True)
        out = subprocess.run(base + ["--only", "c[13]", "--rerun"], capture_output=True,
                             text=True, check=True).stdout
        self.assertIn("2 case(s) to run", out)
        with open(self.cases / "run_summary.csv", newline="") as f:
            rows = {r["case"]: r for r in csv.DictReader(f)}
        self.assertEqual(sorted(rows), ["c1", "c2", "c3"])   # c2 kept although not selected
        self.assertEqual(rows["c2"]["status"], "FAILED")
        bad = subprocess.run(base + ["--only", "zz*"], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("no case folder", bad.stderr)

    def test_index_order_interleaves_meshes(self):
        for name, cells in (("big_train0002", 1600), ("big_train0001", 1600), ("small_test0001", 100),
                            ("small_train0001", 100), ("small_train0002", 100)):
            d = self.cases / name
            d.mkdir()
            (d / "case.json").write_text(json.dumps({"mesh": {"n_cells": str(cells)}}))
        names = sorted((self.cases / n for n in ("small_train0002", "big_train0002", "small_test0001",
                                                 "big_train0001", "small_train0001")), key=index_key)
        self.assertEqual([d.name for d in names], ["big_train0001", "small_test0001", "small_train0001",
                                                   "big_train0002", "small_train0002"])

    def test_case_running_elsewhere_is_left_alone(self):
        import socket
        c1 = self.cases / "c1"
        (c1 / "c1.mess").write_text("--- DIAGNOSTIC JOB : OK\n")
        (c1 / "RUNNING").write_text(f"{os.getpid()} {socket.gethostname()} now\n")   # a live owner
        self.assertEqual(run_case(c1, self.fake)["status"], "RUNNING ELSEWHERE")
        self.assertTrue((c1 / "c1.mess").exists())                       # untouched
        out = subprocess.run([sys.executable, str(HERE / "run_cases.py"), "--cases", str(self.cases),
                              "--aster", str(self.fake)], capture_output=True, text=True, check=True).stdout
        self.assertIn("1 case(s) are being run by another runner", out)
        self.assertIn("2 case(s) to run", out)
        # a marker left by a process that no longer exists is ignored and cleaned up
        dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                              capture_output=True, text=True).stdout.strip()
        (c1 / "RUNNING").write_text(f"{dead} {socket.gethostname()} then\n")
        self.assertEqual(run_case(c1, self.fake)["status"], "OK (alarms)")
        self.assertFalse((c1 / "RUNNING").exists())

    def test_batch_then_resume(self):
        cmd = [sys.executable, str(HERE / "run_cases.py"), "--cases", str(self.cases),
               "--aster", str(self.fake), "--jobs", "2"]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        self.assertIn("2/3 case(s) OK", out)
        with open(self.cases / "run_summary.csv", newline="") as f:
            status = {r["case"]: r["status"] for r in csv.DictReader(f)}
        self.assertEqual(status, {"c1": "OK (alarms)", "c2": "FAILED", "c3": "OK (alarms)"})
        # second run: only the failed case is run again
        out2 = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        self.assertIn("1 case(s) to run, 2 already done", out2)


if __name__ == "__main__":
    unittest.main()
