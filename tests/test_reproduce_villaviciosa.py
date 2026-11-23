# -*- coding: utf-8 -*-
"""End to end: the offline replication kit reproduces the published Villaviciosa product.

Runs ``scripts/reproduce_villaviciosa.py --from-kit replication --fast`` in its own process: the kit's files are
checked against their sha256, steps 6-11 of MAREA are rerun through ``pyintertidal.marea.reconstruct``, and the run is
compared with ``products_marea/`` array by array. ``--fast`` skips the 30 bootstrap fits of the delay intervals
(those 4 checks are reported as skipped), so the test takes about 1.5 minutes and 0.4 GB. The full check, with the
intervals, is the same command without ``--fast``. The run writes into a temporary folder, never into the repository.
"""
import os  # paths and the environment of the run
import subprocess  # the script runs in its own process, as a user would run it
import sys  # the same Python as the tests

import pytest  # the test runner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repository root


@pytest.mark.slow
def test_offline_kit_reproduces_the_product(tmp_path):
    """Route A with --fast exits 0 and prints a PASS verdict; its marea.npz is written to tmp_path."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg",
               GIT_OPTIONAL_LOCKS="0")  # the provenance step reads git; a read must not touch the index
    cmd = [sys.executable, "-X", "utf8", os.path.join("scripts", "reproduce_villaviciosa.py"),
           "--from-kit", "replication", "--fast", "--out", str(tmp_path / "villaviciosa_repro")]
    run = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=1800)
    tail = (run.stdout + "\n" + run.stderr)[-4000:]  # what to show if it fails
    assert run.returncode == 0, tail
    verdict = [line for line in run.stdout.splitlines() if line.startswith(("PASS", "FAIL"))]
    assert verdict and verdict[-1].startswith("PASS:"), tail  # e.g. "PASS: 37 of 37 checks passed, 4 skipped"
    assert (tmp_path / "villaviciosa_repro" / "marea.npz").exists()
