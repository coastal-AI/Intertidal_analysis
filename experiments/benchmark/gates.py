# -*- coding: utf-8 -*-
"""Validation v2 — reproduction and identity gates (spec §10.1; v13 §4.1).

Question
--------
Does the v2 harness reproduce production exactly where it claims to
(G-record, G-keep, G-marea on the legacy netCDF inputs), and do the V1
checks of the inputs say what the printing rule needs (G-zarr, G-offset,
G-level, G-ocm)?

Rules
-----
* Every gate writes ``<root>/gates/<site>/<name>.json`` =
  ``{"format": "v2-gate-1", gate, site, verdict ("PASS", "FAIL" or "NOT
  RUN"), reason, package_key, compared {...}, first_difference, utc,
  git}``. A gate whose inputs are absent is NOT RUN (never PASS); any
  difference is FAIL with the first difference; a gate that raises is NOT
  RUN with the error (the traceback tail in ``compared``).
* The legacy package (:func:`legacy_site`) is a v2 package of the site's
  production netCDF cube (source "netcdf", boundary "production":
  ``levels.make_provider(cfg)`` over 2023-2025, catalogue = label instants)
  under ``<root>/gates/_legacy``; each gate runs the ``package.prepare_v2``
  steps it needs (:data:`GATE_STEPS`; done steps are skipped).
* G-record: the legacy record's W3 dates, t_ns and h0 equal, bitwise, the
  production record built independently here: the cube dates in
  2023-2025 with an overpass label (``package.read_overpass``, naive UTC)
  and h_B = ONE ``provider.levels`` call at those instants, finite only
  (``package.build_record``'s rule). Check 'zarr_record': the record the
  lag rows actually use (the site's Zarr package, ``package.v2_site``, on
  the W3 years it holds) against the same production record: t_ns bitwise
  on the common dates (FAIL on any difference); every date in only one of
  them must be explained (a store-only or legacy-only date of G-zarr's
  ``verify.json``, a production date without a label or with a non-finite
  level, a v2 date with a non-finite union h_B), else FAIL; NOT RUN while
  the Zarr record does not exist or an unexplained difference falls in a
  year without ``verify.json``; the h0 differences are descriptive (the
  union boundary of §1.5 shifts them by design).
* Every gate document records ``code_sha256``, the sources its verdict rests
  on (:data:`GATE_CODE`); the printing rule treats a verdict made with other
  sources as stale (:func:`gate_code_stale`).
* G-keep: ``domain.for_setting(legacy, 'C1.raw.ndwi.D1', 'W3')`` (the
  production extraction rule: the SCL clear mask, raw NDWI, NDWI > 0, over
  the whole window) equals the production extraction (``cfg.extract``):
  keep, sea, inter exactly, and s_km (the extraction's stored float32 s_km,
  else the production geometry recomputed from its masks as
  ``package.check_extraction`` does). The P0 (floor0) keep differences are
  counted, descriptively.
* G-marea: the E0 adapter (``methods.elev_marea_pair.compute``) with
  ``production_compat`` on the legacy package at 'C2.raw.ndwi.D1', W3,
  with the production record (result.json ``record_dates``, asserted equal
  to ``pkg.rule_subset('C2', 'W3')``) and the production extraction domain
  (G-keep's domain when it equals the extraction, else the extraction's own
  masks: G-marea tests the adapter, G-keep the domain rule), with the
  production n_boot, against ``marea.npz`` and
  ``result.json``: keep, s_km, band, tau_min, tau_usado_min, tau_lo/hi,
  tau_px_min, z, sigma (and tau_boot_min when stored), bitwise in the
  stored dtypes. The production configuration (n_boot, min_scenes,
  quality_min_bracket, band_km, min_px_band) must equal FROZEN['marea'] or
  the gate refuses (NOT RUN). The no-delay map is compared with the
  production comparison folder (``products.npz`` 'z_plain', float64) when
  one exists, else recorded as 'not compared'.
* G-zarr (``<year>.verify.json``, ``inputs.read_verify``): B03, B08 and SCL
  equal on every common date of each present 2023-2025 store.
* G-offset (``common/inputs_report.json``): written with the current offset
  rule and no 'unclear' scene; the anchor step is reported when present.
* G-level (``common/<site>/sources.json``): every computable source's check
  — LG reproduces each judge exactly at its s_g (computed here); LT (<= 1 mm
  check points) and LM (bitwise subset) are read from the sources' meta and
  are NOT RUN while unrecorded.
* G-ocm (``years/<Y>/ocm/ocm.json``): the real OmniCloudMask ran (not an
  injected stand-in), with one provenance (version, weights, card) over the
  years, equal to the current card and installed version.
* 'lagtruth' is the scorer's gate (``experiments.benchmark.scoring:
  gate_lagtruth``), resolved lazily.

No import-time side effects; heavy modules are imported inside the gates.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import traceback
import types

import numpy as np
import pandas as pd

from experiments.benchmark import config, product

GATE_FORMAT = "v2-gate-1"
VERDICTS = ("PASS", "FAIL", "NOT RUN")
V2_ROOT = "results/validation_v2"
LEGACY_SUBDIR = "gates/_legacy"
#: the legacy package steps each gate needs (``package.prepare_v2``)
GATE_STEPS = {"record": ("record",),
              "keep": ("record", "pass1", "geometry", "support", "dom"),
              "marea": ("record", "pass1", "geometry", "support", "dom", "pass2", "subsets",
                        "levels")}
#: the production extraction rule as a v2 setting, and P0
KEEP_SETTING = "C1.raw.ndwi.D1"
P0_SETTING = "C1.floor0.ndwi.D1"
#: production MAREA's setting (transition-zone record, raw NDWI) and window
MAREA_SETTING = "C2.raw.ndwi.D1"
GATE_WINDOW = "W3"
#: the production configuration G-marea requires (FROZEN['marea'], domain)
GATE_N_BOOT = int(config.FROZEN["marea"]["bootstrap"]["n_boot"])
GATE_MIN_SCENES = int(config.FROZEN["marea"]["floor"])
GATE_MIN_BRACKET = int(config.FROZEN["marea"]["min_bracket"])
GATE_BAND_KM = float(config.FROZEN["domain"]["bands"]["length_km"])
GATE_MIN_PX_BAND = int(config.FROZEN["domain"]["bands"]["min_px"])
#: G-zarr compares these bands (spec §1.11 item 4)
ZARR_BANDS = ("B03", "B08", "SCL")
#: G-level: an LT check point within this many metres passes (spec §10.1)
LT_TOL_M = 1e-3
#: the sources each gate's verdict rests on (recorded in the gate document;
#: the printing rule refuses a verdict made with other sources)
_B = "experiments/benchmark"
GATE_CODE = {
    "record": (f"{_B}/gates.py", f"{_B}/package.py", f"{_B}/inputs.py"),
    "keep": (f"{_B}/gates.py", f"{_B}/package.py", f"{_B}/domain.py", f"{_B}/obs.py"),
    "marea": (f"{_B}/gates.py", f"{_B}/package.py", f"{_B}/domain.py", f"{_B}/obs.py",
              f"{_B}/lags/marea_family.py", f"{_B}/methods/elev_marea_pair.py",
              "pyintertidal/tide_estimators.py", "pyintertidal/marea.py"),
}
del _B


def log(msg):
    print(msg, flush=True)


def gate_code_shas(name):
    """``{repository path: sha256 of the source}`` of the files gate
    ``name``'s verdict rests on (:data:`GATE_CODE`; gates.py alone for the
    others)."""
    files = GATE_CODE.get(name, ("experiments/benchmark/gates.py",))
    out = {}
    for rel in files:
        p = os.path.join(config.ROOT, rel)
        try:
            out[rel] = config.sha256_source(p)
        except OSError:
            out[rel] = "missing"
    return out


def gate_code_stale(doc, name):
    """None when the gate document's ``code_sha256`` equals the current
    sources, else the reason (the files that changed, or no hashes)."""
    if not isinstance(doc, dict):
        return "no gate document"
    have = doc.get("code_sha256")
    if not isinstance(have, dict):
        return "the gate document records no code hashes"
    now = gate_code_shas(name)
    changed = sorted(k for k in set(have) | set(now) if have.get(k) != now.get(k))
    return None if not changed else "code changed since the verdict: " + ", ".join(changed)


# ─────────────────────────────────────────────────────────────────────────────
#  verdict files
# ─────────────────────────────────────────────────────────────────────────────

def gate_path(root, site, name):
    return os.path.join(root, "gates", site, f"{name}.json")


def read_gate(root, site, name):
    """The verdict document of a gate, or None."""
    p = gate_path(root, site, name)
    if not os.path.exists(p):
        return None
    try:
        return product.read_json(p)
    except (OSError, ValueError):
        return None


def gate_passed(root, site, name, package_key=None):
    """True iff the gate's verdict is PASS and, when ``package_key`` is
    given, the verdict was made on that package (a verdict not bound to a
    package, package_key None, passes for any)."""
    doc = read_gate(root, site, name)
    if not doc or doc.get("verdict") != "PASS":
        return False
    if package_key is None or doc.get("package_key") is None:
        return True
    return doc.get("package_key") == package_key


def _result(verdict, reason=None, package_key=None, compared=None, first_difference=None):
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    return {"verdict": verdict, "reason": reason, "package_key": package_key,
            "compared": dict(compared or {}), "first_difference": first_difference}


def _not_run(reason, **compared):
    return _result("NOT RUN", reason, compared=compared)


def _resolve(name):
    fn = GATES.get(name)
    if fn is None:
        raise KeyError(f"unknown gate {name!r}; known: {sorted(GATES)}")
    if isinstance(fn, str):
        module, _, func = fn.partition(":")
        fn = getattr(importlib.import_module(module), func)
    return fn


def run_gate(name, site, root=V2_ROOT, cubes_root=None, raise_errors=False, **kw):
    """Run one gate and write its verdict file; returns the verdict
    document. An exception inside the gate gives NOT RUN with the error
    (``raise_errors=True`` re-raises it after writing)."""
    fn = _resolve(name)
    err = None
    try:
        res = fn(site, root=root, cubes_root=cubes_root, **kw)
        if not isinstance(res, dict) or res.get("verdict") not in VERDICTS:
            raise ValueError(f"gate {name} returned no verdict: {res!r}")
    except Exception as e:                      # recorded, never a PASS
        err = e
        tb = traceback.format_exc().strip().splitlines()[-12:]
        res = _result("NOT RUN", f"error: {type(e).__name__}: {e}", compared={"traceback": tb})
    doc = {"format": GATE_FORMAT, "gate": name, "site": site, "verdict": res["verdict"],
           "reason": res.get("reason"), "package_key": res.get("package_key"),
           "compared": res.get("compared") or {}, "first_difference": res.get("first_difference"),
           "code_sha256": gate_code_shas(name),
           "utc": product.utc_stamp(), "git": product.git_state()}
    product.write_json(gate_path(root, site, name), doc)
    log(f"[gate] {site} {name}: {doc['verdict']}" + (f" ({doc['reason']})" if doc["reason"]
                                                      else ""))
    if err is not None and raise_errors:
        raise err
    return doc


# ─────────────────────────────────────────────────────────────────────────────
#  comparisons
# ─────────────────────────────────────────────────────────────────────────────

def _int_view(a):
    a = np.ascontiguousarray(a)
    if a.dtype.kind == "f":
        return a.view({2: np.int16, 4: np.int32, 8: np.int64}[a.dtype.itemsize])
    return a


def first_difference(name, got, want, dtype=None):
    """None when ``got`` equals ``want`` bitwise in ``dtype`` (``want``'s
    dtype by default; NaN equals NaN whatever its payload), else a JSON-ready
    description of the first difference."""
    want = np.asarray(want)
    dtype = np.dtype(dtype or want.dtype)
    got = np.asarray(got)
    try:
        g = got.astype(dtype)
    except (TypeError, ValueError) as e:
        return {"check": name, "what": f"cannot cast {got.dtype} to {dtype}: {e}"}
    w = want.astype(dtype)
    if g.shape != w.shape:
        return {"check": name, "what": "shape", "got": list(g.shape), "expected": list(w.shape)}
    if dtype.kind == "f":
        gn, wn = np.isnan(g), np.isnan(w)
        bad = (gn != wn) | (~gn & ~wn & (_int_view(g) != _int_view(w)))
    else:
        bad = g != w
    if not bad.any():
        return None
    i = np.unravel_index(int(np.flatnonzero(bad.ravel())[0]), bad.shape)
    i = tuple(int(x) for x in i)
    gv, wv = g[i], w[i]
    return {"check": name, "what": "value", "index": list(i) if len(i) > 1 else i[0],
            "got": gv.item() if hasattr(gv, "item") else gv,
            "expected": wv.item() if hasattr(wv, "item") else wv,
            "n_differ": int(bad.sum()), "n": int(bad.size), "dtype": str(dtype)}


class _Checks:
    """Ordered checks: ``ok`` True, False (a difference) or None (not
    evaluated); the first False is the gate's first difference."""

    def __init__(self):
        self.items = []
        self.first = None

    def add(self, name, ok, **info):
        e = {"check": name, "ok": ok, **info}
        self.items.append(e)
        if ok is False and self.first is None:
            self.first = info.get("difference") or {"check": name, **info}
        return ok

    def arrays(self, name, got, want, dtype=None):
        d = first_difference(name, got, want, dtype)
        return self.add(name, d is None, **({"difference": d} if d else
                                             {"n": int(np.asarray(want).size)}))

    def verdict(self, required=None):
        oks = [e["ok"] for e in self.items if required is None or e["check"] in required]
        if any(o is False for o in oks):
            return "FAIL"
        if any(o is None for o in oks):
            return "NOT RUN"
        return "PASS"

    def failed(self):
        return [e["check"] for e in self.items if e["ok"] is False]

    def pending(self):
        return [e["check"] for e in self.items if e["ok"] is None]


def _json_list(v):
    return [None if (isinstance(x, float) and np.isnan(x)) else x
            for x in np.asarray(v, float).tolist()]


# ─────────────────────────────────────────────────────────────────────────────
#  the legacy package and its instance
# ─────────────────────────────────────────────────────────────────────────────

def legacy_site(site, root=V2_ROOT):
    """The legacy V2Site: the production 10 m netCDF cube with the
    production boundary provider, under ``<root>/gates/_legacy``."""
    from experiments.benchmark import package
    return package.v2_site(site, root=f"{root}/{LEGACY_SUBDIR}", res=10, source="netcdf",
                           boundary="production")


def _missing_legacy_inputs(s2, extract=False):
    miss = []
    cube = config.resolve(str(s2.legacy_cube)) if s2.legacy_cube else None
    if not cube or not os.path.exists(cube):
        miss.append(f"cube {s2.legacy_cube}")
    lab = s2.legacy_labels
    if not lab or not os.path.exists(lab):
        miss.append(f"overpass labels {lab}")
    if extract:
        ex = config.resolve(s2.cfg.extract) if s2.cfg.extract else None
        if not ex or not os.path.exists(ex):
            miss.append(f"production extraction {s2.cfg.extract}")
    return miss


def _legacy_package(s2, steps, provider, build):
    """The legacy V2Package after the gate's steps (built when ``build``).
    ``provider``: the boundary provider, or None (prepare_v2 and the package
    build the production one only when a step or a live level needs it)."""
    from experiments.benchmark import package
    if build:
        package.prepare_v2(s2, steps=list(steps), provider=provider)
    if not os.path.exists(os.path.join(s2.dir, "record.json")):
        raise FileNotFoundError(f"{s2.dir}/record.json is missing (run the gate with build)")
    pkg = package.V2Package(s2)
    if provider is not None:
        pkg._provider = provider
    return pkg


class _GateLevels:
    """The LevelView of a gate instance: exact table rows on the record
    positions (a live per-shift call beyond the table)."""

    def __init__(self, table, sel):
        self.table = table
        self.sel = np.asarray(sel, np.int64)

    def at(self, tau):
        t = float(tau)
        lo = int(np.floor(t))
        if t == lo:
            return np.asarray(self.table.at(lo, self.sel), np.float64)
        w = t - lo
        return ((1.0 - w) * np.asarray(self.table.at(lo, self.sel), np.float64)
                + w * np.asarray(self.table.at(lo + 1, self.sel), np.float64))

    def stack(self, taus):
        return np.vstack([self.at(t) for t in taus])

    def bank(self, taus=None):
        return self.table.bank(self.sel, taus)


class _GateObs:
    """The ObsView of a gate instance: ``pkg.observe(setting, sel, pos[cols])``."""

    def __init__(self, pkg, setting, sel, pos):
        self.pkg, self.setting = pkg, setting
        self.sel = np.asarray(sel, np.int64)
        self.pos = np.asarray(pos, np.int64)
        self.n, self.T = int(len(self.pos)), int(len(self.sel))

    def read(self, cols=None):
        cols = np.arange(self.n) if cols is None else np.asarray(cols, np.int64)
        return self.pkg.observe(self.setting, self.sel, self.pos[cols])

    def iter_blocks(self, block=100_000):
        for a in range(0, self.n, block):
            cols = np.arange(a, min(a + block, self.n), dtype=np.int64)
            yield cols, self.read(cols)


def gate_instance(pkg, setting, window, sel, dom):
    """A minimal instance (the LagInstance attributes E0 reads) of the
    legacy package: record positions ``sel``, domain ``dom``."""
    sel = np.asarray(sel, np.int64)
    lev = _GateLevels(pkg.levels, sel)
    return types.SimpleNamespace(
        site=pkg.site, window=window, setting=setting, family="rule", mode="F", N="all", rep=0,
        instance_variant="gate", sel=sel, t=pkg.t[sel], dates=np.asarray(pkg.dates)[sel],
        year=np.asarray(pkg.year_of)[sel], h0=lev.at(0), rising=np.asarray(pkg.rising)[sel],
        orbit=np.full(len(sel), -1, np.int64), orbit_source=None, platform=None,
        levels=lev, domain=dom, obs=_GateObs(pkg, setting, sel, dom.pos), pkg=pkg,
        judges=[], boundary_members=[], paths={"root": pkg.site2.root,
                                               "truth_dir": pkg.site2.truth_dir},
        installed_models=(), seed=0, cache={},
        stage=lambda name: contextlib.nullcontext(), progress=lambda n=1: None)


def _package_key(pkg):
    try:
        return pkg.keys["package_key"]
    except Exception:                          # an incomplete package has no key
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  G-record
# ─────────────────────────────────────────────────────────────────────────────

def production_record(s2, provider, window=GATE_WINDOW):
    """The production record (v13 FROZEN['record']): the legacy cube's
    dates inside the window with an overpass label (naive UTC), and h_B
    from ONE provider call at those instants, finite only."""
    from experiments.benchmark import package, settings
    ds, tdim = package.open_cube(config.resolve(str(s2.legacy_cube)))
    try:
        dates = [str(d)[:10] for d in pd.DatetimeIndex(ds[tdim].values).strftime("%Y-%m-%d")]
    finally:
        ds.close()
    over = package.read_overpass(s2.legacy_labels)
    w = settings.window(window)
    in_w = [d for d in dates if w.start <= d < w.end]
    cand = [d for d in in_w if d in over]
    t = pd.DatetimeIndex([over[d] for d in cand])
    t = (t.tz_convert("UTC").tz_localize(None) if t.tz is not None else t).as_unit("ns")
    h = np.asarray(provider.levels(t), np.float64)
    if h.shape != (len(t),):
        raise AssertionError(f"provider.levels returned {h.shape} for {len(t)} instants")
    fin = np.isfinite(h)
    return {"dates": [d for d, f in zip(cand, fin) if f], "t_ns": t.asi8[fin].astype(np.int64),
            "h0": h[fin], "n_cube_dates": len(dates), "n_window": len(in_w),
            "without_label": [d for d in in_w if d not in over],
            "nonfinite_level": [d for d, f in zip(cand, fin) if not f]}


def zarr_record_check(ck, zsite2, ref, cubes_root=None, window=GATE_WINDOW):
    """The 'zarr_record' check of G-record (module docstring): the record of
    the site's Zarr package ``zsite2`` against the production record ``ref``
    (:func:`production_record`) on the W3 years the Zarr record holds.
    Adds the check to ``ck`` and returns the compared details."""
    from experiments.benchmark import inputs, settings
    rp = os.path.join(zsite2.dir, "record.json") if zsite2 is not None else None
    if rp is None or not os.path.exists(rp):
        ck.add("zarr_record", None, note="no Zarr package record yet (prepare the site)")
        return {"status": "not checked: no Zarr package record", "record": rp}
    rec = product.read_json(rp)
    w = settings.window(window)
    years = sorted(int(y) for y in rec.get("years") or [])
    z = {d: int(t) for d, t in zip(rec["dates"], rec["t_ns"]) if w.start <= d < w.end}
    zh = {d: float(h) for d, h in zip(rec["dates"], rec["h0"]) if d in z}
    p = {d: int(t) for d, t in zip(ref["dates"], np.asarray(ref["t_ns"], np.int64).tolist())
         if int(d[:4]) in years}
    ph = {d: float(h) for d, h in zip(ref["dates"], np.asarray(ref["h0"], float).tolist())
          if d in p}
    common = sorted(set(z) & set(p))
    t_diff = [d for d in common if z[d] != p[d]]
    verify = {}
    for y in years:
        v = inputs.read_verify(zsite2.site, y, cubes_root)
        dd = (v or {}).get("dates") or {}
        verify[y] = None if v is None else {"only_store": set(dd.get("only_store") or []),
                                            "only_reference": set(dd.get("only_reference") or [])}
    ref_drop = {"without_label": set(ref.get("without_label") or []),
                "nonfinite_level": set(ref.get("nonfinite_level") or [])}
    v2_nonfinite = set(rec.get("nonfinite_level") or [])
    unexplained, no_verify, why = [], [], {}
    for d in sorted(set(z) - set(p)):
        vy = verify.get(int(d[:4]))
        if vy is not None and d in vy["only_store"]:
            why[d] = "store-only date (G-zarr)"
        elif d in ref_drop["without_label"]:
            why[d] = "production: no overpass label"
        elif d in ref_drop["nonfinite_level"]:
            why[d] = "production: non-finite h_B"
        elif vy is None:
            no_verify.append(d)
        else:
            unexplained.append(d)
    for d in sorted(set(p) - set(z)):
        vy = verify.get(int(d[:4]))
        if vy is not None and d in vy["only_reference"]:
            why[d] = "legacy-only date (G-zarr)"
        elif d in v2_nonfinite:
            why[d] = "v2: non-finite union h_B"
        elif vy is None:
            no_verify.append(d)
        else:
            unexplained.append(d)
    dh = np.array([zh[d] - ph[d] for d in common], float)
    compared = {"record": rp.replace("\\", "/"), "years": years, "n_v2": len(z),
                "n_production": len(p), "n_common": len(common),
                "t_ns_differ": t_diff[:20], "n_t_ns_differ": len(t_diff),
                "only_v2": sorted(set(z) - set(p))[:20], "only_production":
                    sorted(set(p) - set(z))[:20],
                "explained": dict(list(sorted(why.items()))[:40]),
                "unexplained": unexplained[:20], "no_verify_json": no_verify[:20],
                "h0_max_abs_diff_m": float(np.max(np.abs(dh))) if len(dh) else None,
                "h0_role": "descriptive (the union boundary shifts h0 by design, §1.5)",
                "instants": rec.get("instants")}
    if t_diff:
        d0 = t_diff[0]
        ck.add("zarr_record", False, difference={
            "check": "zarr_record", "what": "t_ns", "date": d0, "got": z[d0], "expected": p[d0],
            "n_differ": len(t_diff), "n": len(common)})
    elif unexplained:
        ck.add("zarr_record", False, difference={
            "check": "zarr_record", "what": "dates", "unexplained": unexplained[:10],
            "n": len(unexplained)})
    elif no_verify:
        ck.add("zarr_record", None, note=f"{len(no_verify)} date difference(s) in a year "
                                          "without verify.json (G-zarr)")
    else:
        ck.add("zarr_record", True, n=len(common))
    return compared


def gate_record(site, root=V2_ROOT, cubes_root=None, *, site2=None, provider=None, build=True,
                reference_provider=None, zarr_site2=None, check_zarr=True):
    """G-record (see the module docstring). ``zarr_site2``: the site's Zarr
    package site (default ``package.v2_site(site, root, cubes_root)``);
    ``check_zarr`` False leaves the 'zarr_record' check out (tests of the
    legacy part only)."""
    from experiments.benchmark import package, settings
    s2 = site2 or legacy_site(site, root)
    miss = _missing_legacy_inputs(s2)
    if miss:
        return _not_run("legacy inputs missing: " + "; ".join(miss), legacy=s2.dir)
    if build:
        package.prepare_v2(s2, steps=list(GATE_STEPS["record"]), provider=provider)
    rp = os.path.join(s2.dir, "record.json")
    if not os.path.exists(rp):
        return _not_run(f"{rp} is missing (run the gate with build)")
    rec = product.read_json(rp)
    w = settings.window(GATE_WINDOW)
    idx = [i for i, d in enumerate(rec["dates"]) if w.start <= d < w.end]
    mine = {"dates": [rec["dates"][i] for i in idx],
            "t_ns": np.asarray(rec["t_ns"], np.int64)[idx],
            "h0": np.asarray(rec["h0"], np.float64)[idx]}
    if reference_provider is None and provider is None:
        from experiments.benchmark import levels
        reference_provider = levels.make_provider(s2.cfg)       # production's call
    ref = production_record(s2, reference_provider or provider)
    ck = _Checks()
    if mine["dates"] != ref["dates"]:
        a, b = mine["dates"], ref["dates"]
        k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        ck.add("dates", False, difference={
            "check": "dates", "what": "date", "index": k,
            "got": a[k] if k < len(a) else None, "expected": b[k] if k < len(b) else None,
            "n_got": len(a), "n_expected": len(b),
            "only_v2": sorted(set(a) - set(b))[:10],
            "only_production": sorted(set(b) - set(a))[:10]})
    else:
        ck.add("dates", True, n=len(mine["dates"]))
        ck.arrays("t_ns", mine["t_ns"], ref["t_ns"], np.int64)
        ck.arrays("h0", mine["h0"], ref["h0"], np.float64)
    zc_cmp = None
    if check_zarr:
        zs = zarr_site2
        if zs is None:
            try:
                zs = package.v2_site(site, root=root, cubes_root=cubes_root)
            except Exception as e:              # recorded: the check is not evaluated
                zs = None
                ck.add("zarr_record", None, note=f"the Zarr site: {type(e).__name__}: {e}")
        if zs is not None or zarr_site2 is not None:
            zc_cmp = zarr_record_check(ck, zs, ref, cubes_root)
    pk = None
    with contextlib.suppress(Exception):
        pk = config.sha256_json({"record": product.sha256_file(rp)})
    compared = {"legacy": s2.dir.replace("\\", "/"), "window": GATE_WINDOW,
                "T_v2_window": len(mine["dates"]), "T_production": len(ref["dates"]),
                "n_cube_dates": ref["n_cube_dates"], "without_label": ref["without_label"][:20],
                "nonfinite_level": ref["nonfinite_level"][:20],
                "boundary": (rec.get("boundary") or {}), "instants": rec.get("instants"),
                "zarr_record": zc_cmp, "checks": ck.items}
    v = ck.verdict()
    reason = (None if v == "PASS" else f"differs: {ck.failed()}" if v == "FAIL"
              else "not evaluated: " + "; ".join(
                  f"{e['check']} ({e.get('note') or 'pending'})" for e in ck.items
                  if e["ok"] is None))
    return _result(v, reason, pk, compared, ck.first)


# ─────────────────────────────────────────────────────────────────────────────
#  G-keep
# ─────────────────────────────────────────────────────────────────────────────

def _extraction(cfg):
    """keep, sea, inter, shape, dates and s_km (None when not stored) of the
    production extraction, read member by member (no pickles)."""
    p = config.resolve(cfg.extract)
    with np.load(p, allow_pickle=False) as z:
        out = {"keep": np.asarray(z["keep"], np.int64), "sea": np.asarray(z["sea"], bool),
               "inter": np.asarray(z["inter"], bool),
               "shape": tuple(int(v) for v in z["shape"]),
               "dates": [str(d)[:10] for d in z["dates"]],
               "s_km": np.asarray(z["s_km"]) if "s_km" in z.files else None}
    out["path"] = str(p).replace("\\", "/")
    return out


def gate_keep(site, root=V2_ROOT, cubes_root=None, *, site2=None, provider=None, build=True):
    """G-keep (see the module docstring)."""
    from experiments.benchmark import domain
    s2 = site2 or legacy_site(site, root)
    miss = _missing_legacy_inputs(s2, extract=True)
    if miss:
        return _not_run("legacy inputs missing: " + "; ".join(miss), legacy=s2.dir)
    pkg = _legacy_package(s2, GATE_STEPS["keep"], provider, build)
    try:
        ex = _extraction(s2.cfg)
        ck = _Checks()
        compared = {"legacy": s2.dir.replace("\\", "/"), "setting": KEEP_SETTING,
                    "window": GATE_WINDOW, "extraction": ex["path"]}
        if tuple(ex["shape"]) != tuple(pkg.shape):
            ck.add("shape", False, difference={"check": "shape", "got": list(pkg.shape),
                                               "expected": list(ex["shape"])})
            return _result("FAIL", "the frame differs", _package_key(pkg),
                           dict(compared, checks=ck.items), ck.first)
        rec_dates = [str(x) for x in np.asarray(pkg.dates)[pkg.window(GATE_WINDOW)]]
        compared["dates"] = {"n_extraction": len(ex["dates"]), "n_record_window": len(rec_dates),
                             "only_extraction": sorted(set(ex["dates"]) - set(rec_dates))[:20],
                             "only_record": sorted(set(rec_dates) - set(ex["dates"]))[:20]}
        d = domain.for_setting(pkg, KEEP_SETTING, GATE_WINDOW)
        compared["domain"] = {"status": d.status, "n_keep": d.n_keep, "K": d.K,
                              "notes": getattr(d, "notes", {})}
        ck.arrays("keep", d.keep, ex["keep"], np.int64)
        ck.arrays("sea", d.sea, ex["sea"], bool)
        ck.arrays("inter", d.inter, ex["inter"], bool)
        if ck.failed():
            compared["s_km"] = "not compared (the domains differ)"
        elif ex["s_km"] is not None:
            ck.arrays("s_km", d.s_km.astype(np.float32), ex["s_km"], np.float32)
            compared["s_km"] = "the extraction's stored s_km (float32)"
        else:
            prod = domain.assemble("P", ex["keep"], ex["sea"], ex["inter"], pkg.pixel_m,
                                   pkg.cfg.mouth_side, dom=None)
            ck.arrays("s_km", d.s_km, prod.s_km, np.float64)
            compared["s_km"] = "the production geometry recomputed from the extraction's masks"
        d0 = domain.for_setting(pkg, P0_SETTING, GATE_WINDOW)
        compared["p0_keep_differences"] = {
            "setting": P0_SETTING, "n_keep_p0": d0.n_keep,
            "only_p0": int(len(np.setdiff1d(d0.keep, ex["keep"]))),
            "only_production": int(len(np.setdiff1d(ex["keep"], d0.keep))),
            "role": "descriptive (floor0 against production's raw preprocessing)"}
        compared["checks"] = ck.items
        v = ck.verdict()
        reason = None
        if v == "FAIL":
            reason = f"differs: {ck.failed()}"
            if compared["dates"]["only_extraction"] or compared["dates"]["only_record"]:
                reason += " (the extraction counts other dates than the W3 record)"
        return _result(v, reason, _package_key(pkg), compared, ck.first)
    finally:
        pkg.close()


# ─────────────────────────────────────────────────────────────────────────────
#  G-marea
# ─────────────────────────────────────────────────────────────────────────────

def production_config_differences(result):
    """``{key: (production, frozen)}`` where result.json's configuration
    differs from FROZEN['marea'] (a missing key is a difference)."""
    want = {"n_boot": GATE_N_BOOT, "min_scenes": GATE_MIN_SCENES,
            "quality_min_bracket": GATE_MIN_BRACKET, "band_km": GATE_BAND_KM,
            "min_px_band": GATE_MIN_PX_BAND}
    out = {}
    for k, v in want.items():
        have = result.get(k)
        try:
            same = have is not None and float(have) == float(v)
        except (TypeError, ValueError):
            same = False
        if not same:
            out[k] = [have, v]
    return out


def _compare_nodelay(ck, comp_npz, res, rec_dates, keep):
    """The no-delay map against the comparison folder's 'z_plain' (float64)."""
    with np.load(comp_npz, allow_pickle=False) as z:
        files = set(z.files)
        if "z_plain" not in files:
            return "not compared (no z_plain in the comparison folder)"
        zp = np.asarray(z["z_plain"], np.float64)
        ck_keep = np.asarray(z["keep"], np.int64) if "keep" in files else None
        cdates = [str(x) for x in z["record_dates"]] if "record_dates" in files else None
    if cdates is not None and cdates != list(rec_dates):
        return "not compared (the comparison folder holds another record)"
    if ck_keep is not None and not np.array_equal(ck_keep, keep):
        return "not compared (the comparison folder holds another keep)"
    mine = np.asarray(res["z_a"], np.float64)
    d = first_difference("nodelay_z", mine, zp, np.float64)
    if d is None:
        ck.add("nodelay_z", True, n=int(zp.size))
        return "compared (bitwise float64)"
    fm, fp = np.isfinite(mine), np.isfinite(zp)
    both = fm & fp
    if (not (fm & ~fp).any() and (fp & ~fm).any()
            and np.array_equal(mine[both].view(np.int64), zp[both].view(np.int64))):
        # equal wherever both are defined, production defined beyond: a map
        # made before the quality rule (min_bracket 0)
        ck.add("nodelay_z", None, note="equal where both are defined; the comparison map has "
                                       "more pixels (made without the quality rule)")
        return "not compared (the comparison map predates the quality rule)"
    ck.add("nodelay_z", False, difference=d)
    return "compared (bitwise float64)"


def _marea_domain(pkg, cfg):
    """The domain G-marea runs the adapter on: the v2 extraction-rule domain
    (``KEEP_SETTING`` over the window, G-keep's) when it equals the
    production extraction, else the production extraction's own keep, sea
    and inter (mode 'P'; G-keep reports the rule's difference), so this gate
    tests the adapter, not the domain rule. Returns (domain, description)."""
    from experiments.benchmark import domain
    d = domain.for_setting(pkg, KEEP_SETTING, GATE_WINDOW)
    ex = _extraction(cfg)
    same = (tuple(ex["shape"]) == tuple(pkg.shape) and np.array_equal(d.keep, ex["keep"])
            and np.array_equal(d.sea, ex["sea"]) and np.array_equal(d.inter, ex["inter"]))
    if same:
        return d, f"{KEEP_SETTING} at {GATE_WINDOW} (equal to the production extraction)"
    try:
        p = domain.assemble("P", ex["keep"], ex["sea"], ex["inter"], pkg.pixel_m,
                            pkg.cfg.mouth_side, dom=pkg.dom)
    except KeyError as e:                       # keep_prod partly outside dom
        return None, f"the production keep lies partly outside the legacy dom ({e})"
    return p, (f"the production extraction's keep, sea and inter: the {KEEP_SETTING} domain "
               f"differs from it at {int(len(np.setxor1d(d.keep, ex['keep'])))} keep pixels "
               "(see G-keep)")


def gate_marea(site, root=V2_ROOT, cubes_root=None, *, site2=None, provider=None, build=True,
               production_dir=None, comparison_dir=None, boot_threads=1):
    """G-marea (see the module docstring)."""
    from experiments.benchmark.methods import elev_marea_pair as emp
    s2 = site2 or legacy_site(site, root)
    prod = production_dir or (s2.cfg.published or {}).get("marea")
    prod = config.resolve(prod) if prod else None
    rp = os.path.join(prod, "result.json") if prod else None
    zp = os.path.join(prod, "marea.npz") if prod else None
    if not prod or not os.path.exists(rp) or not os.path.exists(zp):
        return _not_run(f"production MAREA products missing ({prod})")
    rj = product.read_json(rp)
    if rj.get("estado") != "ok":
        return _not_run(f"the production result is {rj.get('estado')!r}, not 'ok'")
    diffs = production_config_differences(rj)
    if diffs:
        return _not_run("the production configuration differs from FROZEN['marea']: "
                        + ", ".join(f"{k} {v[0]} (frozen {v[1]})" for k, v in diffs.items()),
                        production=prod, differences=diffs)
    miss = _missing_legacy_inputs(s2, extract=True)
    if miss:
        return _not_run("legacy inputs missing: " + "; ".join(miss), legacy=s2.dir)
    pkg = _legacy_package(s2, GATE_STEPS["marea"], provider, build)
    try:
        ck = _Checks()
        rec_dates = [str(d) for d in rj["record_dates"]]
        where = {str(d): i for i, d in enumerate(pkg.dates)}
        absent = [d for d in rec_dates if d not in where]
        compared = {"legacy": s2.dir.replace("\\", "/"),
                    "production": str(prod).replace("\\", "/"),
                    "setting": MAREA_SETTING, "window": GATE_WINDOW, "n_boot": int(rj["n_boot"]),
                    "domain": "production ran MAREA on the full-record extraction domain"}
        if absent:
            ck.add("record", False, difference={"check": "record", "what": "production record "
                                                "dates absent from the legacy record",
                                                "dates": absent[:10], "n": len(absent)})
            return _result("FAIL", "the production record is not in the legacy record",
                           _package_key(pkg), dict(compared, checks=ck.items), ck.first)
        sel = np.array([where[d] for d in rec_dates], np.int64)
        if len(sel) > 1 and np.any(np.diff(sel) <= 0):
            ck.add("record", False, difference={"check": "record",
                                                "what": "record dates not in time order"})
            return _result("FAIL", "the production record is not ordered", _package_key(pkg),
                           dict(compared, checks=ck.items), ck.first)
        ck.add("record", True, n=int(len(sel)))
        rule = pkg.rule_subset("C2", GATE_WINDOW)
        ck.arrays("c2_rule", rule, sel, np.int64)
        dom, compared["domain_source"] = _marea_domain(pkg, s2.cfg)
        if dom is None:
            ck.add("domain", False, difference={"check": "domain",
                                                "what": compared["domain_source"]})
            return _result("FAIL", compared["domain_source"], _package_key(pkg),
                           dict(compared, checks=ck.items), ck.first)
        inst = gate_instance(pkg, MAREA_SETTING, GATE_WINDOW, sel, dom)
        n_boot = int(rj["n_boot"])
        comp = comparison_dir or (s2.cfg.published or {}).get("comparison")
        comp = config.resolve(comp) if comp else None
        cnpz = os.path.join(comp, "products.npz") if comp else None
        has_comp = bool(cnpz and os.path.exists(cnpz))
        res = emp.compute(inst, n_boot=n_boot, boot_threads=boot_threads, seed_offset=0,
                          production_compat=True,
                          products=("E0a", "E0b", "L01") if has_comp else ("E0b", "L01"))
        st = res["lag"]
        with np.load(zp, allow_pickle=False) as z:
            npz = {k: np.asarray(z[k]) for k in z.files}
        if not ck.arrays("keep", dom.keep, npz["keep"], np.int64):
            compared["arrays"] = "not compared (another keep)"
        else:
            ck.arrays("s_km", dom.s_km.astype(np.float32), npz["s_km"], np.float32)
            ck.arrays("band", np.asarray(dom.band_of).astype(np.int16), npz["band"], np.int16)
            ck.arrays("tau_min", st["tau_raw"], npz["tau_min"], np.float64)
            ck.arrays("tau_usado_min", st["tau_used"], npz["tau_usado_min"], np.float64)
            lo = st["lo"] if st["lo"] is not None else np.full(st["K"], np.nan)
            hi = st["hi"] if st["hi"] is not None else np.full(st["K"], np.nan)
            ck.arrays("tau_lo_min", lo, npz["tau_lo_min"], np.float64)
            ck.arrays("tau_hi_min", hi, npz["tau_hi_min"], np.float64)
            if "tau_boot_min" in npz:
                ck.arrays("tau_boot_min", st["boot"] if st["boot"] is not None
                          else np.zeros((0, st["K"])), npz["tau_boot_min"], np.float64)
            ck.arrays("tau_px_min", st["tau_px"], npz["tau_px_min"], np.float32)
            ck.arrays("z", res["z_b"].astype(np.float32), npz["z"], np.float32)
            ck.arrays("sigma", res["sigma_b"].astype(np.float32), npz["sigma"], np.float32)
            tau_px64 = np.asarray(st["tau_px"], np.float64)
            for key, got in (("n_escenas", len(sel)), ("n_px", dom.n_keep),
                             ("n_px_cota", int(np.isfinite(res["z_b"]).sum()))):
                ck.add(f"result.{key}", int(rj.get(key, -1)) == int(got),
                       **({} if int(rj.get(key, -1)) == int(got) else {"difference": {
                           "check": f"result.{key}", "got": int(got), "expected": rj.get(key)}}))
            for key, got in (("centros_km", st["centers"]), ("tau_min", st["tau_raw"]),
                             ("tau_usado_min", st["tau_used"]), ("tau_lo_min", lo),
                             ("tau_hi_min", hi), ("rango_marea_m", [res["lo"], res["hi"]]),
                             ("tau_px_median_min", [float(np.median(tau_px64))]),
                             ("tau_px_max_min", [float(tau_px64.max())])):
                want = rj.get(key)
                want = [want] if not isinstance(want, list) else want
                want = np.asarray([np.nan if v is None else v for v in want], np.float64)
                ck.arrays(f"result.{key}", np.asarray(got, np.float64), want, np.float64)
        if has_comp and not ck.failed():
            compared["nodelay"] = _compare_nodelay(ck, cnpz, res, rec_dates, dom.keep)
        else:
            compared["nodelay"] = "not compared" + ("" if not ck.failed()
                                                    else " (earlier differences)")
        compared["checks"] = ck.items
        compared["lag"] = {"tau_raw": _json_list(st["tau_raw"]),
                           "tau_used": _json_list(st["tau_used"]), "flags": st["flags"],
                           "K": st["K"], "T": st["T"]}
        required = [e["check"] for e in ck.items if e["check"] != "nodelay_z"]
        v = ck.verdict(required) if not ck.failed() else "FAIL"
        reason = None if v == "PASS" else (f"differs: {ck.failed()}" if v == "FAIL"
                                           else f"not evaluated: {ck.pending()}")
        return _result(v, reason, _package_key(pkg), compared, ck.first)
    finally:
        pkg.close()


# ─────────────────────────────────────────────────────────────────────────────
#  thin gates over V1 outputs
# ─────────────────────────────────────────────────────────────────────────────

def _site_package_key(site, root, cubes_root=None):
    """The package key of the site's v2 package, when it exists."""
    try:
        from experiments.benchmark import package
        s2 = package.v2_site(site, root=root, cubes_root=cubes_root)
        if not os.path.exists(os.path.join(s2.dir, "record.json")):
            return None
        pkg = package.V2Package(s2)
        try:
            return _package_key(pkg)
        finally:
            pkg.close()
    except Exception:
        return None


def gate_zarr(site, root=V2_ROOT, cubes_root=None, *, years=None):
    """G-zarr: each present 2023-2025 store's ``<year>.verify.json``."""
    from experiments import zarr_cubes as zc
    from experiments.benchmark import inputs, settings
    w3 = settings.window("W3").years
    present = [y for y in zc.list_years(site, cubes_root) if y in w3]
    ys = list(years) if years is not None else present
    if not ys:
        return _not_run(f"no 2023-2025 store of {site} in the cubes root")
    ck = _Checks()
    per = {}
    for y in ys:
        v = inputs.read_verify(site, y, cubes_root)
        if v is None:
            per[str(y)] = "no verify.json"
            ck.add(str(y), None, note="no verify.json (run the downloader's verification)")
            continue
        verdict = v.get("verdict")
        per[str(y)] = verdict
        if verdict in (None, "unreadable verify.json", "not comparable"):
            ck.add(str(y), None, note=f"{verdict}: {v.get('reason') or v.get('error')}")
            continue
        bands = v.get("bands") or {}
        bad = {}
        for b in ZARR_BANDS:
            s = bands.get(b)
            if not isinstance(s, dict):
                bad[b] = "not compared"
                continue
            n = {k: inputs._count(s.get(k)) for k in ("equal", "value_differs",
                                                      "nodata_only_store",
                                                      "nodata_only_reference")}
            if not n["equal"] or any(n[k] for k in ("value_differs", "nodata_only_store",
                                                    "nodata_only_reference")):
                bad[b] = n
        if bad:
            listed = v.get("dates_with_differences_listed") or {}
            first = next(iter(listed.items()), None)
            ck.add(str(y), False, difference={"check": f"G-zarr {y}", "bands": bad,
                                              "verdict": verdict,
                                              "first_date": None if first is None else first[0],
                                              "dates_with_differences":
                                                  v.get("dates_with_differences")})
        else:
            ck.add(str(y), True, verdict=verdict,
                   dates_only_store=len((v.get("dates") or {}).get("only_store") or []),
                   dates_only_reference=len((v.get("dates") or {}).get("only_reference") or []))
    compared = {"years": [int(y) for y in ys], "per_year": per, "bands": list(ZARR_BANDS),
                "checks": ck.items}
    v = ck.verdict()
    reason = None
    if v == "FAIL":
        reason = (f"Zarr differs from the legacy cube in {ck.failed()}: pre-registered response "
                  "(spec §1.11 item 4): production MAREA rerun on the Zarr inputs becomes the "
                  "as-published reference, G-marea stays on the legacy inputs, declared")
    elif v == "NOT RUN":
        reason = f"not evaluated: {ck.pending()}"
    return _result(v, reason, _site_package_key(site, root, cubes_root), compared, ck.first)


def gate_offset(site, root=V2_ROOT, cubes_root=None, *, report=None):
    """G-offset: the inputs report's per-scene offsets of the site."""
    from experiments.benchmark import inputs
    path = report or os.path.join(root, "common", "inputs_report.json")
    if not os.path.exists(path):
        return _not_run(f"{path} is missing (run inputs)")
    doc = product.read_json(path)
    rep = (doc.get("sites") or {}).get(site) or (doc if doc.get("site") == site else None)
    if not rep:
        return _not_run(f"{site} is not in {path}")
    kind = (rep.get("source") or {}).get("kind", "zarr")
    ck = _Checks()
    if kind != "netcdf":
        stale = inputs.offset_rule_status(rep)
        if stale:
            return _not_run(f"the inputs report was written with {stale}: rerun inputs",
                            report=path)
    offs = rep.get("offsets")
    if offs is None:
        return _not_run("the inputs report holds no per-scene offsets", report=path)
    unclear = sorted(d for d, v in offs.items() if v is None)
    dec = rep.get("offset_decisions") or {}
    unclear = sorted(set(unclear) | {d for d, e in dec.items()
                                     if isinstance(e, dict) and e.get("state") == "unclear"})
    if unclear:
        ck.add("unclear", False, difference={"check": "unclear", "dates": unclear[:10],
                                             "n": len(unclear)})
    else:
        ck.add("unclear", True, n=len(offs))
    step = None
    for k in ("anchor_step", "anchors", "anchor_steps"):
        if k in rep:
            step = rep[k]
            break
    compared = {"report": str(path).replace("\\", "/"), "source": kind,
                "offset_counts": rep.get("offset_counts"),
                "offset_rule_sha256": rep.get("offset_rule_sha256"),
                "anchor_step": step if step is not None else "not in the report (reported, not "
                                                             "a criterion)",
                "checks": ck.items}
    v = ck.verdict()
    return _result(v, None if v == "PASS" else "BOA offset 'unclear' scenes",
                   _site_package_key(site, root, cubes_root), compared, ck.first)


def _lg_reproduces(src):
    """None when LG's pixel interpolation gives each knot's own row at its
    s (the judge at s_g, spec §10.1), else the first difference."""
    from experiments.benchmark import levels
    s = np.asarray(src.knots_s, np.float64)
    codes = [str(c) for c in src.groups.get("code", [])]
    if len(s) < 2:
        return {"check": "LG", "what": "no judge knot"}
    pos = np.arange(src.T)
    knots = [{"code": codes[k] if k < len(codes) else f"k{k}", "s_km": float(s[k]),
              "values": src.H[k]} for k in range(1, len(s))]
    tmp = levels.level_lg(src.site, src.t, src.ref, knots, np.arange(len(s) - 1, dtype=np.int64),
                          s[1:], demean=src.demean)
    got = tmp.pixel_levels(None, pos, pos, allow_nan=True)
    want = src.rows(pos, pos, allow_nan=True)[1:]
    return first_difference("LG knots", got, want, np.float64)


def gate_level(site, root=V2_ROOT, cubes_root=None, *, sources_json=None):
    """G-level: the level sources step's files and checks."""
    from experiments.benchmark import levels
    sdir = os.path.join(root, "common", site)
    path = sources_json or os.path.join(sdir, "sources.json")
    if not os.path.exists(path):
        return _not_run(f"{path} is missing (prepare --step sources)")
    doc = product.read_json(path)
    base = os.path.dirname(path)
    ck = _Checks()
    statuses = {}
    for name, e in sorted((doc.get("files") or {}).items()):
        src_name = e.get("source")
        statuses[name] = {"source": src_name, "status": e.get("status"),
                          "unit_status": e.get("unit_status")}
        fp = os.path.join(base, name)
        if not os.path.exists(fp):
            ck.add(name, False, difference={"check": name, "what": "file listed but missing"})
            continue
        if product.sha256_file(fp) != e.get("sha256"):
            ck.add(name, False, difference={"check": name, "what": "sha256 changed since the "
                                                                    "sources step"})
            continue
        if e.get("status") != levels.STATUS_OK:
            ck.add(name, True, note=f"not computable here ({e.get('status')}): nothing to check")
            continue
        src = levels.SourceLevels.load(fp)
        if src_name == "LG":
            d = _lg_reproduces(src)
            ck.add(name, d is None, **({"difference": d} if d else
                                       {"note": "each judge reproduced at its s_g"}))
        elif src_name == "LT":
            cp = src.meta.get("check_points") or src.flags.get("check_points")
            if not cp:
                ck.add(name, None, note="LT check points (eo-tides vs direct pyTMD, <= 1 mm) not "
                                        "recorded by the sources step")
            else:
                err = max(abs(float(x.get("diff_m", x.get("error_m", np.inf)))) for x in cp)
                ck.add(name, err <= LT_TOL_M, max_abs_m=err,
                       **({} if err <= LT_TOL_M else {"difference": {
                           "check": name, "what": "LT check point", "max_abs_m": err}}))
        elif src_name == "LM":
            sc = src.meta.get("subset_check") or src.flags.get("subset_check")
            if not sc:
                ck.add(name, None, note="LM subset identity with the CMEMS toolbox output not "
                                        "recorded by the sources step")
            else:
                ok = bool(sc.get("bitwise", sc.get("ok", False)))
                ck.add(name, ok, **({} if ok else {"difference": {"check": name,
                                                                  "what": "LM subset", **sc}}))
        else:
            ck.add(name, True, note="LB: the level table's tau = 0 row")
    if not ck.items:
        return _not_run(f"{path} lists no source file")
    compared = {"sources_json": str(path).replace("\\", "/"), "files": statuses,
                "checks": ck.items}
    v = ck.verdict()
    reason = None if v == "PASS" else (f"differs: {ck.failed()}" if v == "FAIL"
                                       else f"not evaluated: {ck.pending()}")
    return _result(v, reason, _site_package_key(site, root, cubes_root), compared, ck.first)


def gate_ocm(site, root=V2_ROOT, cubes_root=None, *, years=None):
    """G-ocm: the OmniCloudMask provenance of every OCM year."""
    import importlib.metadata as md

    from experiments.benchmark import package, settings
    s2 = package.v2_site(site, root=root, cubes_root=cubes_root)
    rp = os.path.join(s2.dir, "record.json")
    ys = list(years) if years is not None else (
        [int(y) for y in product.read_json(rp)["years"]] if os.path.exists(rp)
        else list(package.W3_YEARS))
    ck = _Checks()
    provs = {}
    for y in ys:
        p = os.path.join(s2.dir, "years", str(y), "ocm", "ocm.json")
        if not os.path.exists(p):
            ck.add(str(y), None, note="no ocm.json (the OCM step has not run)")
            continue
        prov = (product.read_json(p).get("provenance") or {})
        provs[str(y)] = prov
        if prov.get("injected"):
            ck.add(str(y), False, difference={"check": f"G-ocm {y}", "what": "an injected "
                                              "predictor (a stand-in), not OmniCloudMask",
                                              "predictor": prov.get("predictor")})
        elif dict(prov.get("card") or {}) != json.loads(json.dumps(
                dict(settings.CLOUD["C11"].params), default=product._json_default)):
            ck.add(str(y), False, difference={"check": f"G-ocm {y}", "what": "the C11 card "
                                              "changed since the OCM step"})
        else:
            ck.add(str(y), True, version=prov.get("version"))
    if provs:
        ident = {y: (p.get("version"), json.dumps(p.get("weights"), sort_keys=True))
                 for y, p in provs.items()}
        if len(set(ident.values())) > 1:
            ck.add("consistency", False, difference={"check": "consistency", "what":
                                                     "the years were classified with different "
                                                     "versions or weights",
                                                     "per_year": {y: v[0] for y, v in
                                                                  ident.items()}})
        else:
            ck.add("consistency", True)
        try:
            installed = md.version("omnicloudmask")
        except md.PackageNotFoundError:
            installed = None
        rec_v = next(iter(provs.values())).get("version")
        if installed is not None and rec_v is not None and installed != rec_v:
            ck.add("installed", False, difference={"check": "installed", "what": "the installed "
                                                   "OmniCloudMask differs from the one that ran",
                                                   "installed": installed, "recorded": rec_v})
        else:
            ck.add("installed", True, installed=installed, recorded=rec_v)
    compared = {"years": [int(y) for y in ys], "provenance": provs, "checks": ck.items,
                "not_checked": "the one-scene bitwise identity with OmniCloudMask's own "
                               "prediction (spec §10.1) is not recorded by V1"}
    v = ck.verdict()
    reason = None if v == "PASS" else (f"differs: {ck.failed()}" if v == "FAIL"
                                       else f"not evaluated: {ck.pending()}")
    return _result(v, reason, _site_package_key(site, root, cubes_root), compared, ck.first)


GATES = {"record": gate_record, "keep": gate_keep, "marea": gate_marea, "zarr": gate_zarr,
         "offset": gate_offset, "level": gate_level, "ocm": gate_ocm,
         "lagtruth": "experiments.benchmark.scoring:gate_lagtruth"}
