"""R1 gate: nothing but V3 may open the reserved survey. Static + dynamic.

Fails loudly (hard rule R1 of plan v4) if:

  * any file under ``pyintertidal/`` or ``experiments/`` other than
    ``experiments/v3_open_reserved.py`` contains ``allow_reserved=True``;
  * the dynamic guard in :func:`pyintertidal.rtk.load_rtk` fails to raise
    when an unauthorised caller passes the flag;
  * the default call leaks reserved rows.

Run:  python -m tests.test_r1_guard   (also picked up by pytest if present)
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_static_no_allow_reserved():
    offenders = []
    for base in ("pyintertidal", "experiments", "data_v4/prototypes"):
        d = os.path.join(ROOT, base)
        if not os.path.isdir(d):
            continue
        for dirpath, _, files in os.walk(d):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, ROOT).replace("\\", "/")
                if rel == "experiments/v3_open_reserved.py":
                    continue
                if rel == "pyintertidal/rtk.py":
                    continue          # defines the flag; may not set it True
                src = io.open(p, encoding="utf-8", errors="replace").read()
                if "allow_reserved=True" in src.replace(" ", ""):
                    offenders.append(rel)
    assert not offenders, f"R1 VIOLATED: {offenders} enable allow_reserved"


def test_dynamic_guard_blocks_unauthorised_caller():
    from pyintertidal.rtk import load_rtk
    try:
        load_rtk(allow_reserved=True)
    except PermissionError:
        return
    raise AssertionError("R1 VIOLATED: load_rtk(allow_reserved=True) did "
                         "not reject an unauthorised caller")


def test_default_hides_reserved():
    from pyintertidal.rtk import load_rtk
    dev = load_rtk()
    assert isinstance(dev, dict)
    assert set(dev["split"]) == {"dev"}
    assert dev["n_reserved_hidden"] > 0
    # sanity: the canonical partition over the 361 raw FIX points keeps
    # ~65 % as dev (270 measured at adoption; the historical 135/57 figures
    # were on the product-matched subset — see docs/PHASE_LOG.md)
    assert 200 <= len(dev["elev"]) <= 320
    assert 50 <= dev["n_reserved_hidden"] <= 150


if __name__ == "__main__":
    test_static_no_allow_reserved()
    print("OK  static: nobody enables allow_reserved")
    test_dynamic_guard_blocks_unauthorised_caller()
    print("OK  dynamic: the guard rejects unauthorised callers")
    test_default_hides_reserved()
    print("OK  default: dev only, reserved hidden")
