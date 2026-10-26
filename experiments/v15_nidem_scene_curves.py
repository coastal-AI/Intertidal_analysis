# -*- coding: utf-8 -*-
"""V15 - NIDEM elevation error against the number of scenes used, on exactly
MAREA's subsets, and the MAREA-vs-NIDEM figure (10 m).

Question (co-author meeting, 2026-10-20)
----------------------------------------
How does each method respond to adding images? Per site ONE MAREA curve and
ONE NIDEM curve: elevation error against the number N of Sentinel-2 scenes
used, the same random subsets for both methods, scored on the same units.

MAREA curve
-----------
experiments/v14_scene_count_curves.py run with ``--lags refit``: pool ``all``,
quality rule mb 0 (the rule of the published products), read from
<site>_refit.csv as it is. v14 is not changed; its sha256 is a gate.

NIDEM curve on the same subsets
-------------------------------
Pool ``all`` = every cube scene with an overpass time and a finite boundary
level (v7.build_record; Villaviciosa 2023-2025): 465 / 464 / 462 / 462 scenes,
which is also NIDEM's production record. The N grid, the replicates and the
job order are v14's plan as recorded in <site>_refit_settings.json; every
subset is ``v14.draw(seed, site, 'all', N, rep, pool_size)`` and its
draw_sha1 must equal the one in v14's CSV row. The method is NIDEM's
production run (experiments/v7_waterline_contour.py on
pyintertidal.waterline_contour): ITEM pixel mask, 9 tide intervals of the
subset's own observed tidal range with h = the boundary level at the scene
time (no lag), median NDWI composites, R, gap fill, contours, TIN on the
full frame, filtered = R in 1..8 and conf <= 0.25. As in production a subset
is NOT COMPUTABLE when an interval is empty (EmptyIntervalError) or when fewer
than wc.MIN_VALID_SHARE (1 %) of the keep pixels have a valid observation in
all 9 intervals (the pre-flight rule); it gets a CSV row with coverage 0 and
its reason, never dropped. Scored exactly as v14 scored NIDEM (v14.score:
RTK ols_against_truth / Vaklodingen v7.vscore) on v14's units: the 194 RTK
points at Villaviciosa (pyintertidal.rtk.load_rtk() with default arguments,
through v14.rtk_units) and v14's seeded 100,000-px subsample of F at the
Dutch sites. Coverage = share of the units with a NIDEM value.

Fast and exact
--------------
Running v7 per subset would take 6-22 min each. Instead:

phase 1  ONE pass over the cube in chunk-aligned blocks of rows; every pool
         scene is read as v7 reads it (masked NDWI, float32). For every
         subset at once, per block: the relative-extents code (R 0..9, 255 =
         invalid) WITHOUT sorting, exactly: per interval m = valid
         observations and p = positive ones; water iff 2p > m, or (m even and
         2p = m) and float32(0.5 (f64(largest non-positive) + f64(smallest
         positive))) > 0, which is the library's median > 0 test; the
         pre-flight counts on keep; the masked NDWI at the scored pixels;
         and, for the two gate subsets, the library composites at every keep
         pixel.
phase 2  per subset: the pre-flight rule; conf at the scored pixels from
         wc.composites_from_arrays on their NDWI (the library's R there is
         asserted equal to the fast R, every job); synthetic composites that
         carry R and that conf; then wc.nidem itself (gap fill, contours, TIN,
         filters) on the full frame. The filtered value at the units is the
         production value. conf outside the scored pixels is NaN, so the
         full-frame 'filtered' of a curve job is not a product and is never
         saved.

Gates (before any curve row; a failed gate stops the run)
---------------------------------------------------------
* inputs: v14's sha256; the v14 refit CSV and settings equal the copies
  checked on the laptop (sha256);
* units, stored MAREA z, pool, levels and plan: equal to v14's compat and
  settings, the pool levels equal the stored NIDEM h bitwise, every
  draw_sha1 equals v14's CSV;
* products: the full pool reproduces the stored primary nidem.npz (R,
  valid, unfiltered and uncertainty on the full frame; filtered, conf,
  count and median at keep; Z, U, sizes, edges, interval; contour vertices),
  its pre-flight numbers and v14's reference scores; MAREA's record
  reproduces sensitivity_marea_record/nidem.npz. At Ems the primary is NOT
  COMPUTABLE: the full pool must stop at the pre-flight with the stored
  numbers, and MAREA's 54-scene record is the product gate. Bitwise is
  expected; otherwise the gate reports R flips, filtered validity flips and
  max |dz| at the units and the score differences, and passes only with R
  bitwise, 0 validity flips at the units, |dz| <= 1e-6 m and scores within
  1e-9;
* fast R of curve draws: on every job the library's R at the scored pixels
  equals the fast R (asserted), and in phase 1 two curve draws chosen from
  the plan alone (with an interval assignment, N closest to 40 and to 80,
  and the fullest sparsest interval among those) are checked against the
  library's R on the WHOLE frame (``fast_r_full_frame_audit``: 0 differing
  pixels).

Speed and memory (server)
-------------------------
One process per site; phase 1 holds one block of rows of every pool scene
(6 bytes per scene-pixel), the R codes of every subset (1 byte per pixel and
subset) and the NDWI at the scored pixels; phase 2 runs ``--workers`` forked
processes (Linux; concurrent.futures, so a worker killed by the OOM killer
stops the run with BrokenProcessPool instead of hanging) that share them
copy-on-write. ``--io-lock`` serialises the block reads of several processes
on one disk; ``--cache-dir`` keeps the phase-1 arrays (keyed by the inputs,
the subsets and the source of the phase-1 code) so that a stopped or crashed
phase 2 resumes without reading the cube again. The CSV is resumable, one
row per (N, replicate); a lock file next to it (Linux) stops a second run of
the same site at once.

Outputs (new paths only)
------------------------
results/scene_count_curves/<site>_nidem_all.csv            one row per (N, replicate)
results/scene_count_curves/<site>_nidem_all_settings.json  gates, inputs, plan, timings,
                                                           not-computable counts per N
results/scene_count_curves/<site>_nidem_all.run.lock       empty lock file (Linux)
plot: docs/informe_fer/fig_marea_nidem_escenas.{pdf,png}    Spanish (co-author note)
      results/scene_count_curves/marea_vs_nidem_all_sites.{png,pdf}  English (poster)

Run from the repository root:
    python -X utf8 -m experiments.v15_nidem_scene_curves run villaviciosa --workers 2
    python -X utf8 -m experiments.v15_nidem_scene_curves plot
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    # one BLAS thread per process (phase 2 forks workers); must be set before
    # numpy loads its BLAS. A value already in the environment wins.
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import v14_scene_count_curves as v14  # noqa: E402
from pyintertidal import waterline_contour as wc  # noqa: E402

OUT = v14.OUT
POOL = "all"
V14_PATH = "experiments/v14_scene_count_curves.py"
V14_SHA256 = "ef0f9a2e7b4f9b02f78d19fb22bed07a3b91ed5e8a4a59615e8333cfd4de5caf"
#: sha256 of v14's <site>_refit.csv and <site>_refit_settings.json as copied
#: from the server on 2026-10-20 (results/scene_count_curves/server_2026-10-20)
MAREA_SHA256 = {
    "villaviciosa": ("776adc941d05932453e1fbba08a37ce1e620cca22762676b891c00ea978b54a8",
                     "5b7f39575b08d7b20403b7ed69f2b6ddbc4d58b25e6695562b4cae7c69d51327"),
    "escalda": ("84eb41c002a06460fe6da67e6e3728713634b706e79160f682c7287b882fa546",
                "e07b79aebd6b048d332471fe2b62c9e9d85693329b559142dfb9cfd0af6687ea"),
    "wadden": ("ab01391256789b96aea3655e3b6555f47442494b51b33a17d2ff5115eb604f91",
               "067089ecf41339be4d3250afbcbea50f7e2770903cc734004a90f1bf1a625aa3"),
    "ems": ("440b69a16ba9e0710d8af46bc0289e681f36f460ac0a634689112abd87680369",
            "0039c955bd7c9c99640e9c377a648801cc85ba214f510ba829074dbe293c5ab3"),
}
#: where v14's refit outputs are looked for (first found): the server's own
#: folder, then the laptop copy
MAREA_DIRS = (OUT, os.path.join(OUT, "server_2026-10-20"))
NIDEM_DIRS = (OUT, os.path.join(OUT, "server_2026-10-20"))
CSV_COLUMNS = ("site", "pool", "pool_size", "N", "replicate", "status", "reason", "rmse", "bias",
               "slope", "r", "coverage", "n", "n_set", "median_valid_obs", "keep_valid_share",
               "sizes", "n_vertices", "seconds", "draw_sha1")
STATUS_OK = "ok"
STATUS_EMPTY = "not_computable_empty_interval"
STATUS_PREFLIGHT = "not_computable_preflight"
STATUS_NO_VALID = "not_computable_no_valid_pixel"
NOT_COMPUTABLE = (STATUS_EMPTY, STATUS_PREFLIGHT, STATUS_NO_VALID)
TOL_DZ = 1e-6
TOL_SCORE = 1e-9
INVALID = 255
NI = wc.N_INTERVALS
SENS = "sensitivity_marea_record"
FIG_ES = "docs/informe_fer/fig_marea_nidem_escenas"
FIG_EN = os.path.join(OUT, "marea_vs_nidem_all_sites")

_T0 = time.time()
_CTX = {}                    # phase-2 context, inherited by forked workers


def log(msg):
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now} +{(time.time() - _T0) / 60:6.1f} min] {msg}", flush=True)


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _eq(a, b):
    """Bitwise-style equality: same shape and values, NaN equal to NaN (for
    float arrays)."""
    a, b = np.asarray(a), np.asarray(b)
    nan_ok = np.issubdtype(a.dtype, np.inexact) and np.issubdtype(b.dtype, np.inexact)
    return bool(a.shape == b.shape and np.array_equal(a, b, equal_nan=nan_ok))


def _sizes_text(sizes):
    return ";".join(str(int(v)) for v in sizes)


def peak_rss_mb(children=False):
    """Peak resident memory (MB) of this process, or with ``children`` of
    its largest finished (reaped) child process, e.g. a phase-2 worker after
    the pool has closed; None when unknown (``children`` off Linux/macOS)."""
    try:
        import resource
        who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
        r = resource.getrusage(who).ru_maxrss
        return float(r) / 1024.0 if sys.platform != "darwin" else float(r) / 2 ** 20
    except Exception:  # noqa: BLE001
        pass
    if children:
        return None
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return float(getattr(mi, "peak_wset", mi.rss)) / 2 ** 20
    except Exception:  # noqa: BLE001
        return None


def versions():
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    for mod in ("scipy", "skimage", "xarray", "netCDF4", "pandas"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  v14's plan and draws
# ─────────────────────────────────────────────────────────────────────────────

def find_inputs(site, dirs, suffixes):
    """The first directory of ``dirs`` holding <site><suffix> for every suffix."""
    for d in dirs:
        if all(os.path.exists(os.path.join(d, f"{site}{s}")) for s in suffixes):
            return d
    return None


def marea_rows(csv_path):
    """v14's MAREA rows of pool 'all' with quality rule mb 0 (the curve), as a
    DataFrame."""
    import pandas as pd
    d = pd.read_csv(csv_path)
    return d[(d["pool"] == POOL) & (d["mb"] == 0)].copy()


def plan_from_settings(msettings):
    """(grid, reps, reps_large, pool size, jobs) of v14's pool 'all', jobs in
    v14's replicate-major order."""
    plan = msettings["plan"]
    grid = [int(n) for n in plan["grid"][POOL]]
    reps, reps_large = int(plan["reps"]), int(plan["reps_large"])
    size = int(msettings["pools"][POOL]["size"])
    jobs = v14.make_jobs({POOL: grid}, {POOL: size}, reps, reps_large)
    return grid, reps, reps_large, size, jobs


def interval_sizes(h):
    """|S_j| of the 9 intervals of a scene set, zeros included: the binning of
    :func:`pyintertidal.waterline_contour.interval_assignment` (diagnostic
    only, for the rows of subsets with an empty interval; the tests assert it
    equals the library's sizes where the assignment exists)."""
    h = np.asarray(h, float).ravel()
    lot, hot = float(h.min()), float(h.max())
    edges = lot + 0.1 * np.arange(wc.N_UNIFORM + 1) * (hot - lot)
    raw = np.clip(np.searchsorted(edges, h, side="left"), 1, wc.N_UNIFORM)
    j = np.minimum(raw, NI).astype(int)
    return np.bincount(j, minlength=NI + 1)[1:]


def make_subset(key, N, rep, idx, h_pool, gate=False):
    """One subset (sorted pool indices ``idx``) with its interval assignment,
    or its not-computable status when an interval is empty."""
    idx = np.asarray(idx, np.int64)
    s = dict(key=key, N=int(N), rep=int(rep), idx=idx, sha=v14.draw_sha1(idx), gate=bool(gate),
             status="assigned", reason="", sizes=interval_sizes(h_pool[idx]))
    try:
        j, edges, Z, U, sizes = wc.interval_assignment(h_pool[idx])
    except wc.EmptyIntervalError as e:
        s.update(status=STATUS_EMPTY, reason=f"empty tide interval: {e}")
        return s
    s.update(j=j, edges=edges, Z=Z, U=U, sizes=sizes,
             members=[idx[j == q] for q in range(1, NI + 1)])
    return s


def build_subsets(seed, site, size, jobs, h_pool, extra=()):
    """The curve subsets (v14's draws of pool 'all', in job order) and the
    extra gate subsets ``extra`` = [(key, idx)]."""
    subs = []
    for pool, N, rep in jobs:
        idx = v14.draw(seed, site, pool, N, rep, size)
        subs.append(make_subset(f"N{int(len(idx))}_rep{int(rep)}", len(idx), rep, idx, h_pool,
                                gate=len(idx) == size))
    for key, idx in extra:
        subs.append(make_subset(key, len(idx), -1, idx, h_pool, gate=True))
    return subs


#: the N whose curve draws have their fast R audited on the whole frame:
#: few scenes per interval (partly valid frames, many ties) and more
AUDIT_N = (40, 80)


def audit_choice(subsets, targets=AUDIT_N):
    """The curve draws whose fast R phase 1 checks against the library's R
    on the WHOLE frame, fixed from the plan alone (scene tide levels, before
    any pixel is read): for each N of ``targets``, among the curve draws with
    an interval assignment and N closest to it, the one whose sparsest
    interval holds the most scenes (so that its frame is likely to be partly
    valid rather than all invalid), then the lowest replicate; never a gate
    subset, never the same draw twice. Returns their keys."""
    cand = sorted((s for s in subsets if s["status"] == "assigned" and not s["gate"] and s["rep"] >= 0),
                  key=lambda s: (s["N"], s["rep"]))
    out = []
    for t in targets:
        rest = [s for s in cand if s["key"] not in out]
        if rest:
            out.append(min(rest, key=lambda s: (abs(s["N"] - int(t)), s["N"], -int(np.min(s["sizes"])),
                                                s["rep"]))["key"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  phase 1: one pass over the cube, every subset at once
# ─────────────────────────────────────────────────────────────────────────────

def fast_codes(FIN, POS, Df, members, kloc=None):
    """The relative-extents code of one block for one subset, without sorting.

    ``FIN``/``POS`` (T, P) uint8 0/1: finite / > 0 masked NDWI ``Df`` (T, P)
    float32 of the block's pixels; ``members`` the 9 arrays of rows (scenes)
    of the intervals. Per interval, m = valid observations, p = positive ones:
    the library's median (mean of the two middle values for an even count, in
    float64, cast to float32) is > 0 iff 2p > m, or 2p = m (m even) and
    float32(0.5 (f64(largest non-positive) + f64(smallest positive))) > 0;
    a median of exactly 0 is land, as :func:`wc.land_flags`. A pixel is valid
    iff m >= 1 in every interval (the ITEM rule).

    Returns (code (P,) uint8 = number of land intervals, 255 where invalid;
    pixels of ``kloc`` with >= 1 observation per interval (9,); valid pixels
    of ``kloc``; tie pixel-intervals resolved by values)."""
    P = FIN.shape[1]
    nwater = np.zeros(P, np.uint8)
    ok = np.ones(P, bool)
    has_k = np.zeros(NI, np.int64)
    n_tie = 0
    for q, mem in enumerate(members):
        dt = np.uint8 if len(mem) <= 255 else np.uint16
        m = np.zeros(P, dt)
        p = np.zeros(P, dt)
        for t in mem:
            m += FIN[t]
            p += POS[t]
        p2 = p.astype(np.uint16)
        p2 *= 2
        water = p2 > m
        tie = p2 == m
        tie &= m > 0
        ti = np.flatnonzero(tie)
        if len(ti):
            n_tie += len(ti)
            X = Df[np.ix_(np.asarray(mem), ti)]
            with np.errstate(invalid="ignore"):
                a = np.where(X <= 0, X, -np.inf).max(axis=0)
                b = np.where(X > 0, X, np.inf).min(axis=0)
            med = (0.5 * (a.astype(np.float64) + b.astype(np.float64))).astype(np.float32)
            water[ti] = med > 0
        nwater += water
        has = m > 0
        ok &= has
        if kloc is not None and len(kloc):
            has_k[q] = int(np.count_nonzero(has[kloc]))
    code = (NI - nwater).astype(np.uint8)
    code[~ok] = INVALID
    valid_k = int(np.count_nonzero(ok[kloc])) if kloc is not None and len(kloc) else 0
    return code, has_k, valid_k, n_tie


def preflight_numbers(n_has, n_valid, P):
    """The pre-flight of :func:`wc.interval_coverage` from the counts on keep:
    shares of keep with >= 1 valid observation per interval, intervals below
    the floor, expected valid pixels and share, passed."""
    share = [float(v) / P if P else 0.0 for v in np.asarray(n_has, np.int64)]
    vs = float(n_valid) / P if P else 0.0
    return {"px": int(P), "share_px_with_ge1_clear_obs": share,
            "intervals_with_zero_clear_obs": [q + 1 for q, v in enumerate(n_has) if int(v) == 0],
            "intervals_below_floor": [q + 1 for q, v in enumerate(share) if v < wc.MIN_VALID_SHARE],
            "expected_valid_px": int(n_valid), "expected_valid_share": vs,
            "min_valid_share": float(wc.MIN_VALID_SHARE), "passed": bool(vs >= wc.MIN_VALID_SHARE)}


def phase1(read_block, shape, subsets, keep, unit_px, row_block, label="", audit=()):
    """One pass over the frame by blocks of rows. ``read_block(r0, r1)``
    returns the masked NDWI (T, r1 - r0, W) float32 of every pool scene.

    Returns a dict: ``codes`` (S, H, W) uint8 for the subsets with an
    interval assignment (``slot`` of each subset, -1 otherwise), ``n_has``
    (S, 9) and ``n_valid`` (S,) on keep, ``Du`` (T, len(unit_px)) the masked
    NDWI at the scored pixels, ``gate`` {key: library composites at keep}
    for the gate subsets, ``audit`` {key: pixels compared, pixels whose fast
    R differs from the library's R} on the whole frame for the subsets
    ``audit`` (keys), ``timing``."""
    T, H, W = (int(v) for v in shape)
    keep = np.asarray(keep, np.int64)
    unit_px = np.asarray(unit_px, np.int64)
    P = len(keep)
    slots = []
    for s in subsets:
        if s["status"] == "assigned":
            s["slot"] = len(slots)
            slots.append(s)
        else:
            s["slot"] = -1
    S = len(slots)
    codes = np.full((S, H, W), INVALID, np.uint8)
    n_has = np.zeros((S, NI), np.int64)
    n_valid = np.zeros(S, np.int64)
    n_tie = np.zeros(S, np.int64)
    Du = np.full((T, len(unit_px)), np.nan, np.float32)
    gate = {s["key"]: dict(median=np.full((NI, P), np.nan, np.float32), count=np.zeros((NI, P), np.uint16),
                           sd_mean=np.full(P, np.nan, np.float32),
                           sd_mean_bounded=np.full(P, np.nan, np.float32))
            for s in slots if s["gate"]}
    audit = [str(a) for a in audit]
    aud = {s["key"]: {"N": int(s["N"]), "rep": int(s["rep"]), "px": 0, "differing_px": 0, "valid_px_library": 0,
                      "valid_px_fast": 0, "seconds": 0.0}
           for s in slots if s["key"] in audit}
    lost = sorted(set(audit) - set(aud))
    if lost:
        raise ValueError(f"audit subsets without an interval assignment: {lost}")
    tim = {"read_s": 0.0, "codes_s": 0.0, "gate_composites_s": 0.0, "audit_s": 0.0, "blocks": []}
    t_start = time.time()
    nblk = int(math.ceil(H / row_block))
    for bi, r0 in enumerate(range(0, H, row_block)):
        r1 = min(H, r0 + row_block)
        t0 = time.time()
        D = read_block(r0, r1)
        if D.shape != (T, r1 - r0, W) or D.dtype != np.float32:
            raise RuntimeError(f"read_block gave {D.shape} {D.dtype}, expected {(T, r1 - r0, W)} float32")
        t1 = time.time()
        Df = D.reshape(T, -1)
        lo, hi = r0 * W, r1 * W
        ua, ub = np.searchsorted(unit_px, [lo, hi])
        if ub > ua:
            Du[:, ua:ub] = Df[:, unit_px[ua:ub] - lo]
        ka, kb = np.searchsorted(keep, [lo, hi])
        kloc = keep[ka:kb] - lo
        for s in slots:
            if s["gate"] and kb > ka:
                X = Df[np.ix_(s["idx"], kloc)]
                cmp = wc.composites_from_arrays(X[:, None, :], np.ones((X.shape[0], 1, X.shape[1]), bool),
                                                s["j"])
                g = gate[s["key"]]
                g["median"][:, ka:kb] = cmp["median"][:, 0, :]
                g["count"][:, ka:kb] = cmp["count"][:, 0, :]
                g["sd_mean"][ka:kb] = cmp["sd_mean"][0]
                g["sd_mean_bounded"][ka:kb] = cmp["sd_mean_bounded"][0]
                del X, cmp
        t2 = time.time()
        FIN = np.isfinite(Df).view(np.uint8)
        with np.errstate(invalid="ignore"):
            POS = (Df > 0).view(np.uint8)
        t_aud = 0.0
        for k, s in enumerate(slots):
            code, hk, vk, nt = fast_codes(FIN, POS, Df, s["members"], kloc)
            codes[k].reshape(-1)[lo:hi] = code
            n_has[k] += hk
            n_valid[k] += vk
            n_tie[k] += nt
            if s["key"] in aud:
                # the library's R of this draw on every pixel of the block
                ta = time.time()
                X = Df[s["idx"]]
                cmp = wc.composites_from_arrays(X[:, None, :], np.ones((X.shape[0], 1, X.shape[1]), bool),
                                                s["j"])
                Rl = wc.relative_extents(cmp["median"], cmp["count"])[0].ravel()
                Rf = np.where(code != INVALID, code, np.nan).astype(np.float32)
                a = aud[s["key"]]
                a["px"] += int(Rf.size)
                a["differing_px"] += int(np.count_nonzero(~((Rl == Rf) | (np.isnan(Rl) & np.isnan(Rf)))))
                a["valid_px_library"] += int(np.count_nonzero(np.isfinite(Rl)))
                a["valid_px_fast"] += int(np.count_nonzero(np.isfinite(Rf)))
                del X, cmp, Rl, Rf
                a["seconds"] += time.time() - ta
                t_aud += time.time() - ta
        del D, Df, FIN, POS
        t3 = time.time()
        tim["read_s"] += t1 - t0
        tim["gate_composites_s"] += t2 - t1
        tim["codes_s"] += t3 - t2 - t_aud
        tim["audit_s"] += t_aud
        tim["blocks"].append({"rows": [r0, r1], "read_s": round(t1 - t0, 2), "gate_s": round(t2 - t1, 2),
                              "codes_s": round(t3 - t2 - t_aud, 2), "audit_s": round(t_aud, 2)})
        el = time.time() - t_start
        log(f"{label}phase 1: rows {r0}-{r1} of {H} ({bi + 1}/{nblk}): read {t1 - t0:.1f} s, gate composites "
            f"{t2 - t1:.1f} s, R codes of {S} subsets {t3 - t2 - t_aud:.1f} s, full-frame audit {t_aud:.1f} s | "
            f"ETA {el / (bi + 1) * (nblk - bi - 1) / 60:.1f} min")
    tim["total_s"] = time.time() - t_start
    for a in aud.values():
        a["seconds"] = round(a["seconds"], 2)
    return dict(codes=codes, n_has=n_has, n_valid=n_valid, n_tie=n_tie, Du=Du, gate=gate, audit=aud, timing=tim,
                n_slots=S)


class IoLock:
    """An exclusive advisory lock (fcntl.flock) on a file shared by several
    processes: each block read of the cube is done holding it, so processes
    reading cubes on one disk take turns instead of interleaving seeks. A
    no-op without a path (or off Linux/macOS)."""

    def __init__(self, path):
        self.path, self.fd, self.wait_s, self.n = path, None, 0.0, 0
        if path:
            try:
                import fcntl  # noqa: F401
                self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
            except ImportError:
                log(f"--io-lock {path}: no fcntl on this platform; reading without a lock")

    def __enter__(self):
        if self.fd is not None:
            import fcntl
            t0 = time.time()
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            self.wait_s += time.time() - t0
            self.n += 1
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            import fcntl
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        return False


def acquire_run_lock(path):
    """An exclusive, non-blocking lock (fcntl.flock) on ``path``, held while
    this process lives: a second run of the same site (a launcher started
    twice, a resume while the first run still goes) stops at once instead of
    appending rows to the same CSV. Returns the open descriptor (keep it), or
    None without fcntl (Windows laptop)."""
    try:
        import fcntl
    except ImportError:
        return None
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise SystemExit(f"{path} is locked: another run of this site is writing its CSV; stop it or wait for it")
    return fd


class CubeReader:
    """Masked NDWI of the pool scenes for a block of rows, exactly as
    :func:`wc.stream_composites` reads it for v7 (isel of B03/B08/SCL,
    :func:`wc.scene_ndwi` with the ITEM rule, NaN where not valid, float32),
    ``scene_batch`` scenes per isel (1 = v7's call, one scene at a time)."""

    def __init__(self, path, time_index, scene_batch=32, lock=None):
        import xarray as xr
        self.path = path
        self.ti = np.asarray(time_index, np.int64)
        self.batch = max(1, int(scene_batch))
        self.lock = lock or IoLock(None)
        self.ds = xr.open_dataset(path)
        self.t_dim = [k for k in self.ds["B03"].dims if k not in ("x", "y")][0]
        for v in ("B03", "B08", "SCL"):
            if self.ds[v].dims != (self.t_dim, "y", "x"):
                raise ValueError(f"{v} dims {self.ds[v].dims} are not (t, y, x)")
        self.H, self.W = int(self.ds.sizes["y"]), int(self.ds.sizes["x"])
        ch = self.ds["B03"].encoding.get("chunksizes")
        self.chunk_rows = int(ch[1]) if ch else None
        self.bands = {v: self.ds[v] for v in ("B03", "B08", "SCL")}

    def close(self):
        self.ds.close()

    def __call__(self, r0, r1):
        T = len(self.ti)
        D = np.empty((T, r1 - r0, self.W), np.float32)
        with self.lock:
            for s0 in range(0, T, self.batch):
                tb = self.ti[s0:s0 + self.batch]
                if self.batch == 1:
                    sl = {self.t_dim: int(tb[0]), "y": slice(r0, r1)}
                else:
                    sl = {self.t_dim: tb, "y": slice(r0, r1)}
                scl = self.bands["SCL"].isel(sl).values
                ndwi, valid = wc.scene_ndwi(self.bands["B03"].isel(sl).values,
                                            self.bands["B08"].isel(sl).values, scl, rule="item")
                D[s0:s0 + len(tb)] = np.where(valid, ndwi, np.nan).astype(np.float32).reshape(
                    len(tb), r1 - r0, self.W)
        return D


# ─────────────────────────────────────────────────────────────────────────────
#  phase 2: one NIDEM per subset
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_composites(code, sd, sdb):
    """Composites that give :func:`wc.nidem` the relative extents ``code``
    (R = number of land intervals, 255 = invalid) and the confidence ``sd`` /
    ``sdb`` (NaN where not supplied): median -1 in the first R intervals and
    +1 above, count 1 where valid. R, valid, conf, gap fill, contours, TIN
    and filters are then those of the real composites; only non_monotone
    (unused) differs."""
    H, W = code.shape
    valid = code != INVALID
    Rz = np.where(valid, code, 0)
    med = np.ones((NI, H, W), np.float32)
    for q in range(NI):
        med[q][Rz > q] = -1.0
    cnt = np.broadcast_to(valid.astype(np.uint16)[None], (NI, H, W))
    return {"median": med, "count": cnt, "sd_mean": np.asarray(sd, np.float32).reshape(H, W),
            "sd_mean_bounded": np.asarray(sdb, np.float32).reshape(H, W)}


def _row(ctx, s, **kw):
    """A CSV row of subset ``s`` (defaults: not scored)."""
    r = dict(site=ctx["site"], pool=POOL, pool_size=int(ctx["size"]), N=int(s["N"]), replicate=int(s["rep"]),
             status=STATUS_OK, reason="", rmse=np.nan, bias=np.nan, slope=np.nan, r=np.nan, coverage=0.0, n=0,
             n_set=int(len(ctx["truth"])), median_valid_obs=np.nan, keep_valid_share=np.nan,
             sizes=_sizes_text(s["sizes"]), n_vertices=np.nan, seconds=0.0, draw_sha1=s["sha"])
    r.update(kw)
    return r


def median_valid_obs(ctx, s):
    """Median over the scored units of the subset's valid observations."""
    nobs = np.isfinite(ctx["p1"]["Du"][s["idx"]]).sum(axis=0)
    return float(np.median(nobs[ctx["unit_col"]])) if len(ctx["unit_col"]) else np.nan


def stage2(ctx, k, keep_conf=None):
    """NIDEM of subset ``k`` from the phase-1 arrays, scored on the units.
    Returns (CSV row, the wc.nidem result or None). ``keep_conf`` (the
    phase-1 library composites at keep) puts the confidence at every keep
    pixel (gate subsets) instead of the scored pixels only."""
    s = ctx["subsets"][k]
    p1 = ctx["p1"]
    t0 = time.time()
    mvo = median_valid_obs(ctx, s)
    if s["status"] == STATUS_EMPTY:
        return _row(ctx, s, status=STATUS_EMPTY, reason=s["reason"], median_valid_obs=mvo,
                    keep_valid_share=0.0), None
    slot = s["slot"]
    P = len(ctx["keep"])
    pre = preflight_numbers(p1["n_has"][slot], p1["n_valid"][slot], P)
    share = pre["expected_valid_share"]
    if not pre["passed"]:
        reason = (f"pre-flight (ITEM rule): {pre['expected_valid_px']:,} of {P:,} keep px "
                  f"({share:.3g}) have a valid observation in all 9 intervals < {wc.MIN_VALID_SHARE}; "
                  f"intervals below the floor {pre['intervals_below_floor']}; shares "
                  f"{[round(v, 6) for v in pre['share_px_with_ge1_clear_obs']]}")
        return _row(ctx, s, status=STATUS_PREFLIGHT, reason=reason, median_valid_obs=mvo,
                    keep_valid_share=share, seconds=round(time.time() - t0, 3)), None
    H, W = ctx["H"], ctx["W"]
    code = p1["codes"][slot]
    upx = ctx["unit_px"]
    # the confidence at the scored pixels through the library composites; the
    # library's R there must be the fast R
    Xu = p1["Du"][s["idx"]]
    ck = wc.composites_from_arrays(Xu[:, None, :], np.ones((Xu.shape[0], 1, Xu.shape[1]), bool), s["j"])
    R_lib = wc.relative_extents(ck["median"], ck["count"])[0].ravel()
    cu = code.reshape(-1)[upx]
    R_fast = np.where(cu != INVALID, cu, np.nan).astype(np.float32)
    if not _eq(R_lib, R_fast):
        raise AssertionError(f"[{ctx['site']}] {s['key']}: fast R differs from the library's R at "
                             f"{int(np.sum(~((R_lib == R_fast) | (np.isnan(R_lib) & np.isnan(R_fast)))))} "
                             "scored pixels")
    sd = np.full(H * W, np.nan, np.float32)
    sdb = np.full(H * W, np.nan, np.float32)
    sd[upx] = ck["sd_mean"].ravel()
    sdb[upx] = ck["sd_mean_bounded"].ravel()
    if keep_conf is not None:
        sd_k, sdb_k = keep_conf["sd_mean"], keep_conf["sd_mean_bounded"]
        pos = np.searchsorted(ctx["keep"], upx)
        if not (_eq(sd_k[pos], sd[upx]) and _eq(sdb_k[pos], sdb[upx])):
            raise AssertionError(f"{s['key']}: keep composites differ from the unit composites")
        sd[ctx["keep"]] = sd_k
        sdb[ctx["keep"]] = sdb_k
    comp = synthetic_composites(code, sd, sdb)
    del sd, sdb, ck, Xu
    try:
        res = wc.nidem(comp, s["Z"], s["U"])
    except wc.NoClearObservationsInIntervalError as e:
        return _row(ctx, s, status=STATUS_NO_VALID, reason=str(e), median_valid_obs=mvo,
                    keep_valid_share=share, seconds=round(time.time() - t0, 3)), None
    del comp
    p = res["filtered"].reshape(-1)[ctx["units_flat"]]
    sc = v14.score(ctx["kind"], ctx["truth"], p)
    row = _row(ctx, s, status=STATUS_OK, rmse=sc["rmse"], bias=sc["bias"], slope=sc["slope"], r=sc["r"],
               coverage=sc["coverage"], n=int(sc["n"]), median_valid_obs=mvo, keep_valid_share=share,
               n_vertices=int(res["n_vertices"]), seconds=round(time.time() - t0, 3))
    return row, res


def _pool_job(k):
    """Worker entry (forked): the CSV row of subset k."""
    row, _ = stage2(_CTX, k)
    return k, row


# ─────────────────────────────────────────────────────────────────────────────
#  resumable CSV
# ─────────────────────────────────────────────────────────────────────────────

class RowLog:
    """The per-site CSV, one row per (N, replicate), appended and fsynced job
    by job. A partial last line (a job interrupted mid-write) is dropped on
    load, so that job is redone."""

    def __init__(self, path):
        self.path = path
        self.done = {}
        if os.path.exists(path) and os.path.getsize(path) > 0:
            self._load()
        else:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_COLUMNS)

    def _load(self):
        with open(self.path, "rb") as f:
            data = f.read()
        if not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            with open(self.path, "r+b") as f:
                f.truncate(cut)
            log(f"{self.path}: dropped a partial last line ({len(data) - cut} bytes)")
        with open(self.path, newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            header = next(rd, None)
            if tuple(header or ()) != CSV_COLUMNS:
                raise SystemExit(f"{self.path}: unexpected header; refusing to append (use another --out)")
            for r in rd:
                if len(r) == len(CSV_COLUMNS):
                    d = dict(zip(CSV_COLUMNS, r))
                    self.done.setdefault((int(d["N"]), int(d["replicate"])), d)

    def is_done(self, N, rep):
        return (int(N), int(rep)) in self.done

    def append(self, row):
        key = (int(row["N"]), int(row["replicate"]))
        if key in self.done:
            return False
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([v14._fmt(row[c]) for c in CSV_COLUMNS])
            f.flush()
            os.fsync(f.fileno())
        self.done[key] = {c: v14._fmt(row[c]) for c in CSV_COLUMNS}
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  inputs (v14's units without streaming the extraction's Y/C)
# ─────────────────────────────────────────────────────────────────────────────

def site_config(site, args=None):
    """v14's inputs of a site plus NIDEM's cube, with the ``run`` overrides."""
    from experiments import v7_waterline_contour as v7
    cfg = v14.site_config(site)
    cfg["cube"] = v7.SITES[site]["cube"]
    for k in ("extract", "product", "comparison", "nidem", "cube"):
        v = getattr(args, f"{k}_override", None) if args is not None else None
        if v:
            cfg[k] = v
    return cfg


def load_site_light(site, cfg, seed, max_px):
    """What :func:`v14.load_site` gives the curves, without streaming the
    extraction's Y/C (NIDEM reads the cube): keep and the grid, the stored
    production MAREA z (F's filter), the 'all' pool (v7.build_record),
    MAREA's record inside it and the scored units with their truth
    (:func:`v14.rtk_units` / :func:`v14.dutch_truth` and load_site's seeded
    subsample). Gated against v14's compat hashes by the caller."""
    from experiments import v7_waterline_contour as v7

    t0 = time.time()
    with np.load(cfg["extract"], allow_pickle=False) as ex:
        keep = np.asarray(ex["keep"], np.int64)
        ext_dates = np.asarray(ex["dates"]).astype(str)
        H, W = (int(v) for v in ex["shape"])
    pdir = cfg["product"]
    res = json.load(open(f"{pdir}/result.json", encoding="utf-8"))
    mp, comp = None, None
    mp_path = f"{pdir}/marea.npz"
    if os.path.exists(mp_path):
        with np.load(mp_path, allow_pickle=False) as z:
            mp = {k: z[k] for k in ("keep", "z") if k in z.files}
        if not np.array_equal(mp["keep"], keep):
            raise RuntimeError(f"{mp_path}: keep differs from the extraction")
    if cfg.get("comparison"):
        cp = f"{cfg['comparison']}/products.npz"
        if os.path.exists(cp):
            with np.load(cp, allow_pickle=False) as P:
                comp = {k: P[k] for k in ("keep", "z_marea", "z_plain", "record_dates") if k in P.files}
            if not np.array_equal(comp["keep"], keep):
                raise RuntimeError(f"{cp}: keep differs from the extraction")
            if "record_dates" in comp and [str(d) for d in comp["record_dates"]] != \
                    [str(d)[:10] for d in res["record_dates"]]:
                raise RuntimeError(f"{cp}: record_dates differ from {pdir}/result.json")
    if comp is not None:
        z_stored, z_src = comp["z_marea"].astype(float), f"{cfg['comparison']}/products.npz z_marea"
    elif mp is not None:
        z_stored, z_src = mp["z"].astype(float), f"{mp_path} z"
    else:
        raise SystemExit(f"[{site}] neither {mp_path} nor a comparison products.npz: no stored product")
    gauges = None
    if site != "villaviciosa":
        gauges, missing = v14.gauge_cache(site)
        if missing:
            raise SystemExit(f"[{site}] gauge cache missing for {missing} in data_v4/gauges: refusing to "
                             "download (production used the cache)")
    t1 = time.time()
    rec = v7.build_record(site, v7.SITES[site], ext_dates)
    log(f"[{site}] pool '{POOL}': {len(rec['time_index'])} scenes, boundary {rec['boundary']} "
        f"[{time.time() - t1:.1f} s]")
    dates_all = np.asarray(rec["dates"]).astype(str)
    rec_dates = [str(d)[:10] for d in res["record_dates"]]
    pos = {d: i for i, d in enumerate(dates_all)}
    lost = [d for d in rec_dates if d not in pos]
    if lost:
        raise RuntimeError(f"{len(lost)} record date(s) not in the 'all' pool: {lost[:5]}")
    idx_scr = np.array([pos[d] for d in rec_dates], np.int64)
    if np.any(np.diff(idx_scr) <= 0):
        raise RuntimeError("record_dates are not ascending")
    t1 = time.time()
    if cfg["truth"] == "rtk":
        unit_keep, unit_truth, tinfo = v14.rtk_units(keep, W, z_stored)
        kind = "rtk"
    else:
        if comp is None:
            raise SystemExit(f"[{site}] the range filter needs the comparison products.npz z_plain")
        truth_px, tinfo = v14.dutch_truth(site, cfg, keep, H, W, comp["z_plain"])
        F = np.flatnonzero(np.isfinite(truth_px) & np.isfinite(z_stored))
        unit_keep, unit_truth = F, truth_px[F]
        kind = "vaklodingen"
    nF = len(unit_keep)
    if max_px and nF > max_px:
        perm = np.random.default_rng(np.random.SeedSequence([int(seed), v14.site_id(site), 7])).permutation(nF)
        sel = np.sort(perm[:int(max_px)])
    else:
        sel = np.arange(nF)
    units_keep, truth = unit_keep[sel], unit_truth[sel]
    log(f"[{site}] scored units: {len(units_keep):,} of F = {nF:,} ({kind}) [{time.time() - t1:.1f} s]")
    return dict(keep=keep, H=H, W=W, ext_dates=ext_dates, res=res, z_stored=z_stored, z_src=z_src, rec=rec,
                dates_all=dates_all, idx_scr=idx_scr, rec_dates=rec_dates, units_keep=units_keep, truth=truth,
                kind=kind, n_units_F=int(nF), truth_info=tinfo, gauges=gauges,
                load_seconds=time.time() - t0)


def input_gates(site, cfg, L, msettings):
    """Units, stored z, pool and levels against v14's settings and the stored
    NIDEM products. Returns {name: dict(..., ok)}."""
    g = {}
    c14 = msettings["compat"]
    mine = {"units_sha1": v14.array_sha1(L["units_keep"]), "z_stored_sha1": v14.array_sha1(L["z_stored"]),
            "n_units_F": L["n_units_F"], "n_units_scored": int(len(L["units_keep"])),
            "pool_all": int(len(L["rec"]["time_index"])), "pool_screened": int(len(L["idx_scr"]))}
    want = {"units_sha1": c14["units_sha1"], "z_stored_sha1": c14["z_stored_sha1"], "n_units_F": c14["n_units_F"],
            "n_units_scored": c14["n_units_scored"], "pool_all": c14["pools"]["all"],
            "pool_screened": c14["pools"]["screened"]}
    g["units_vs_v14_compat"] = {"mine": mine, "v14": want, "ok": mine == want}
    same_dates = L["dates_all"].tolist() == list(msettings["pools"][POOL]["dates"])
    g["pool_dates_vs_v14_settings"] = {"n": int(len(L["dates_all"])), "ok": bool(same_dates)}
    h = np.asarray(L["rec"]["h"], float)
    ti = np.asarray(L["rec"]["time_index"], np.int64)
    for key, d, rows in (("primary", cfg["nidem"], np.arange(len(h))),
                         ("sensitivity", os.path.join(cfg["nidem"], SENS), L["idx_scr"])):
        f = os.path.join(d, "nidem.npz")
        pj = os.path.join(d, "nidem_preflight.json")
        out = {"path": f, "checked": False}
        if os.path.exists(f):
            with np.load(f, allow_pickle=False) as z:
                out.update(checked=True, dates=_eq(np.asarray(z["dates"]).astype(str), L["dates_all"][rows]),
                           time_index=_eq(z["time_index"], ti[rows]), h_bitwise=_eq(z["h"], h[rows]))
            out["ok"] = out["dates"] and out["time_index"] and out["h_bitwise"]
        elif os.path.exists(pj):
            # no product (NOT COMPUTABLE): the record's levels against the pre-flight
            pre = json.load(open(pj, encoding="utf-8"))
            j, edges, Z, U, sizes = wc.interval_assignment(h[rows])
            ek = [k for k in pre if k.startswith("edges_m")][0]
            Uj = [k for k in pre if k.startswith("U_j_m")][0]
            Ustored = np.array([np.nan if v is None else v for v in pre[Uj]], float)
            out.update(checked=True, path=pj, n_scenes=int(pre["n_scenes"]) == len(rows),
                       LOT=float(pre["LOT_m"]) == float(h[rows].min()),
                       HOT=float(pre["HOT_m"]) == float(h[rows].max()),
                       edges=_eq(np.asarray(pre[ek], float), edges), Z=_eq(np.asarray(pre["Z_j_m"], float), Z),
                       U=_eq(Ustored, U), sizes=[int(v) for v in pre["interval_sizes |S_j|"]] == sizes.tolist())
            out["ok"] = all(out[k] for k in ("n_scenes", "LOT", "HOT", "edges", "Z", "U", "sizes"))
        else:
            out["ok"] = False
            out["reason"] = "neither nidem.npz nor nidem_preflight.json"
        g[f"levels_vs_nidem_{key}"] = out
    diag = os.path.join(cfg["nidem"], "nidem_diagnostics.json")
    st = json.load(open(diag, encoding="utf-8")).get("status") if os.path.exists(diag) else None
    g["nidem_primary_status"] = {"path": diag, "status": st, "computable": st != "NOT COMPUTABLE",
                                 "ok": os.path.exists(diag)}
    return g


def gate_ok(gates):
    return all(v.get("ok", True) for v in gates.values() if isinstance(v, dict))


# ─────────────────────────────────────────────────────────────────────────────
#  product gates
# ─────────────────────────────────────────────────────────────────────────────

def preflight_gate(ctx, k, pre_path, expect_stop):
    """The subset's pre-flight numbers against a stored nidem_preflight.json
    (and, with ``expect_stop``, that it stops as the stored run did)."""
    from experiments import v7_waterline_contour as v7

    s = ctx["subsets"][k]
    p1 = ctx["p1"]
    out = {"path": pre_path, "subset": s["key"], "N": s["N"]}
    if s["status"] != "assigned":
        out.update(ok=False, reason=f"no interval assignment: {s['reason']}")
        return out
    mine = preflight_numbers(p1["n_has"][s["slot"]], p1["n_valid"][s["slot"]], len(ctx["keep"]))
    out["mine"] = mine
    if not os.path.exists(pre_path):
        out.update(checked=False, ok=not expect_stop)
        return out
    pre = json.load(open(pre_path, encoding="utf-8"))
    cov = pre[v7.COV_KEY]
    ek = [kk for kk in pre if kk.startswith("edges_m")][0]
    chk = {"px": mine["px"] == int(cov["px"]),
           "shares": mine["share_px_with_ge1_clear_obs"] == [float(v) for v in cov["share_px_with_ge1_clear_obs"]],
           "intervals_below_floor": mine["intervals_below_floor"] == list(cov["intervals_below_floor"]),
           "expected_valid_px": mine["expected_valid_px"] == int(cov["expected_valid_px"]),
           "passed": mine["passed"] == bool(cov["passed"]),
           "n_scenes": int(pre["n_scenes"]) == s["N"],
           "interval_sizes": [int(v) for v in pre["interval_sizes |S_j|"]] == s["sizes"].tolist(),
           "edges": _eq(np.asarray(pre[ek], float), s["edges"]),
           "Z_j": _eq(np.asarray(pre["Z_j_m"], float), s["Z"])}
    if expect_stop:
        chk["stops_at_preflight"] = not mine["passed"]
    out.update(checked=True, stored={"expected_valid_px": cov["expected_valid_px"], "passed": cov["passed"],
                                     "intervals_below_floor": cov["intervals_below_floor"]},
               checks=chk, ok=all(chk.values()))
    return out


def product_gate(ctx, k, npz_path, ref):
    """The fast NIDEM of gate subset ``k`` (confidence at every keep pixel)
    against a stored nidem.npz and v14's reference scores ``ref``
    (on_scored_units). Returns (gate dict, CSV row)."""
    s = ctx["subsets"][k]
    t0 = time.time()
    row, res = stage2(ctx, k, keep_conf=ctx["p1"]["gate"].get(s["key"]))
    out = {"path": npz_path, "subset": s["key"], "N": s["N"], "status": row["status"],
           "stage2_seconds": round(time.time() - t0, 2)}
    if res is None:
        out.update(ok=False, reason=f"the subset is not computable: {row['reason']}")
        return out, row
    keep, uk = ctx["keep"], ctx["units_keep"]
    g = ctx["p1"]["gate"][s["key"]]
    chk = {}
    with np.load(npz_path, allow_pickle=False) as st:
        R0 = st["R"]
        chk["R_full_frame"] = _eq(res["R"], R0)
        chk["valid_full_frame"] = _eq(res["valid"], st["valid"])
        chk["unfiltered_full_frame"] = _eq(res["unfiltered"], st["unfiltered"])
        chk["uncertainty_full_frame"] = _eq(res["uncertainty"], st["uncertainty"])
        fk_m = res["filtered"].reshape(-1)[keep]
        fk_s = np.asarray(st["filtered_keep"])
        chk["filtered_keep"] = _eq(fk_m, fk_s)
        chk["filtered_bounded_keep"] = _eq(res["filtered_bounded"].reshape(-1)[keep], st["filtered_bounded_keep"])
        chk["conf_keep"] = _eq(res["conf"].reshape(-1)[keep], st["conf_keep"])
        chk["conf_bounded_keep"] = _eq(res["conf_bounded"].reshape(-1)[keep], st["conf_bounded_keep"])
        chk["R_keep"] = _eq(res["R"].reshape(-1)[keep], st["R_keep"])
        chk["count_keep"] = _eq(g["count"], st["count_keep"])
        chk["median_keep"] = _eq(g["median"], st["median_keep"])
        for a, b in (("Z", "Z"), ("U", "U"), ("sizes", "sizes"), ("edges", "edges"), ("j", "interval")):
            chk[b] = _eq(s[a], st[b])
        chk["time_index"] = _eq(ctx["time_index"][s["idx"]], st["time_index"])
        rc, lv = wc.contour_vertices(res["contours"])
        chk["contour_rc"] = _eq(rc.astype(np.float32), st["contour_rc"])
        chk["contour_level"] = _eq(lv.astype(np.float32), st["contour_level"])
        fin_m, fin_s = np.isfinite(res["R"]), np.isfinite(R0)
        tol = {"R_validity_flips": int(np.sum(fin_m != fin_s)),
               "R_values_differing": int(np.sum(fin_m & fin_s & (res["R"] != R0))),
               "n_vertices": [int(res["n_vertices"]), int(len(st["contour_rc"]))]}
    for name, a, b in (("keep", fk_m, fk_s), ("units", fk_m[uk], fk_s[uk])):
        fa, fb = np.isfinite(a), np.isfinite(b)
        both = fa & fb
        tol[f"filtered_validity_flips_{name}"] = int(np.sum(fa != fb))
        tol[f"filtered_max_abs_dz_{name}_m"] = (float(np.max(np.abs(a[both].astype(float) - b[both].astype(float))))
                                                 if both.any() else 0.0)
    sc = {kk: row[kk] for kk in ("rmse", "bias", "slope", "r", "n", "coverage")}
    out.update(checks=chk, bitwise=all(chk.values()), tolerance=tol, scores_mine=sc)
    if ref:
        d = max(abs(float(sc[kk]) - float(ref[kk])) for kk in ("rmse", "bias", "slope", "r", "coverage"))
        out.update(scores_v14_reference=ref, max_abs_score_diff=d)
        score_ok = int(sc["n"]) == int(ref["n"]) and d <= TOL_SCORE
    else:
        out["scores_v14_reference"] = None
        score_ok = False
    out["ok"] = bool(chk["R_full_frame"] and score_ok and (out["bitwise"] or (
        tol["filtered_validity_flips_units"] == 0 and tol["filtered_max_abs_dz_units_m"] <= TOL_DZ)))
    out["policy"] = (f"pass iff R bitwise on the full frame, scores within {TOL_SCORE:g} of v14's reference "
                     f"(same n), and either every check bitwise or 0 filtered validity flips at the units with "
                     f"max |dz| <= {TOL_DZ:g} m")
    return out, row


# ─────────────────────────────────────────────────────────────────────────────
#  phase-1 cache (outside the repository)
# ─────────────────────────────────────────────────────────────────────────────

def phase1_code_digest():
    """sha256 of the source of the phase-1 code path (reading, the sort-free
    codes, the audit choice, phase 1 itself and the cache layout) and of the
    library module: a phase-1 cache is reused only by the code that wrote
    it. Edits elsewhere (phase 2, the plot, the docs) keep the cache valid."""
    import inspect
    h = hashlib.sha256()
    for f in (CubeReader, fast_codes, audit_choice, phase1, save_phase1, load_phase1):
        h.update(inspect.getsource(f).encode("utf-8"))
    with open(wc.__file__, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def phase1_key(site, cube_info, time_index, h_pool, keep, unit_px, subsets, audit=()):
    """What the phase-1 arrays depend on: the cube file, the pool scenes and
    levels, keep, the scored pixels, every subset, the audited subsets and
    the phase-1 code (:func:`phase1_code_digest`)."""
    h = hashlib.sha1()
    h.update(f"v15-phase1-2|{site}|{cube_info.get('path')}|{cube_info.get('bytes')}|{cube_info.get('mtime')}".encode())
    h.update(phase1_code_digest().encode())
    for a in (time_index, keep, unit_px):
        h.update(np.ascontiguousarray(a, np.int64).tobytes())
    h.update(np.ascontiguousarray(h_pool, np.float64).tobytes())
    for s in subsets:
        h.update(f"|{s['key']}|{s['status']}|{int(s['gate'])}|".encode())
        h.update(np.ascontiguousarray(s["idx"], np.int64).tobytes())
    h.update(("|audit|" + "|".join(str(a) for a in audit)).encode())
    return h.hexdigest()[:20]


def save_phase1(path, p1, key):
    tmp = path + ".tmp.npz"
    arrays = {"key": np.array(key), "codes": p1["codes"], "n_has": p1["n_has"], "n_valid": p1["n_valid"],
              "n_tie": p1["n_tie"], "Du": p1["Du"], "timing": np.array(json.dumps(p1["timing"])),
              "audit": np.array(json.dumps(p1.get("audit", {})))}
    for gk, g in p1["gate"].items():
        for a, v in g.items():
            arrays[f"gate__{gk}__{a}"] = v
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_phase1(path, key, subsets):
    with np.load(path, allow_pickle=False) as z:
        if str(z["key"]) != key or "audit" not in z.files:
            return None
        p1 = {"codes": z["codes"], "n_has": z["n_has"], "n_valid": z["n_valid"], "n_tie": z["n_tie"],
              "Du": z["Du"], "timing": json.loads(str(z["timing"])), "gate": {},
              "audit": json.loads(str(z["audit"]))}
        for name in z.files:
            if name.startswith("gate__"):
                _, gk, a = name.split("__", 2)
                p1["gate"].setdefault(gk, {})[a] = z[name]
    slot = 0
    for s in subsets:
        if s["status"] == "assigned":
            s["slot"] = slot
            slot += 1
        else:
            s["slot"] = -1
    if slot != p1["codes"].shape[0]:
        return None
    p1["n_slots"] = slot
    p1["timing"]["from_cache"] = path
    return p1


# ─────────────────────────────────────────────────────────────────────────────
#  run
# ─────────────────────────────────────────────────────────────────────────────

def nc_counts(rows, grid):
    """Per N: draws, not computable (by reason), computable, with a score."""
    out = {}
    for N in grid:
        rs = [r for r in rows if int(r["N"]) == int(N)]
        st = [r["status"] for r in rs]
        out[str(N)] = {"draws": len(rs), "not_computable_empty_interval": st.count(STATUS_EMPTY),
                       "not_computable_preflight": st.count(STATUS_PREFLIGHT),
                       "not_computable_no_valid_pixel": st.count(STATUS_NO_VALID),
                       "computable": st.count(STATUS_OK),
                       "with_score": sum(1 for r in rs if r["status"] == STATUS_OK
                                         and np.isfinite(float(r["rmse"])))}
    return out


def preflight_stop_units(ctx):
    """For every curve draw that stops at the pre-flight: how many scored
    units have a valid R, and an R in 1..8, under its (exact) codes. R in
    1..8 is necessary for a NIDEM value, so ``max_coverage_without_stop`` is
    an upper bound on the coverage the draw could have had without the 1 %
    stop. Returns {key: dict}."""
    out = {}
    uf = ctx["units_flat"]
    for s in ctx["subsets"]:
        if s["status"] != "assigned" or s["rep"] < 0:
            continue
        pre = preflight_numbers(ctx["p1"]["n_has"][s["slot"]], ctx["p1"]["n_valid"][s["slot"]], len(ctx["keep"]))
        if pre["passed"]:
            continue
        cu = ctx["p1"]["codes"][s["slot"]].reshape(-1)[uf]
        n18 = int(np.count_nonzero((cu >= 1) & (cu <= NI - 1)))
        out[s["key"]] = {"N": int(s["N"]), "rep": int(s["rep"]),
                         "keep_valid_share": float(pre["expected_valid_share"]),
                         "units_valid_R": int(np.count_nonzero(cu != INVALID)), "units_R_1_8": n18,
                         "n_units": int(len(cu)), "max_coverage_without_stop": float(n18) / max(1, len(cu))}
    return out


def fork_pool_factory(n):
    """The default phase-2 pool (Linux): ``n`` forked processes that inherit
    the phase-1 arrays copy-on-write (ProcessPoolExecutor forks every worker
    at the first submit, before its manager thread starts). A worker that
    dies abruptly (OOM killer, a crash in Qhull or skimage) makes the
    pending results raise BrokenProcessPool: the run stops loudly and is
    resumable, instead of waiting forever as multiprocessing.Pool would."""
    import concurrent.futures as cf
    import multiprocessing as mp
    return cf.ProcessPoolExecutor(max_workers=int(n), mp_context=mp.get_context("fork"))


def run_jobs(ctx, todo, rowlog, workers, deadline, t_pred, pool_factory=None):
    """Phase 2 of the curve subsets ``todo`` (indices); rows go to
    ``rowlog`` as each finishes (in completion order; the CSV is keyed by
    (N, replicate)). The subsets whose row needs no NIDEM (empty interval,
    pre-flight stop) are written at once. Stops submitting before a job
    predicted to pass ``deadline``. With ``workers`` > 1 the NIDEMs run in
    ``pool_factory(workers)``, a concurrent.futures executor (default
    :func:`fork_pool_factory`, Linux); a new job is submitted as soon as any
    running one finishes. Returns (jobs done, stop info)."""
    global _CTX
    times = []
    done = [0]
    n_all = len(todo)
    stop = None

    def put(k, row):
        rowlog.append(row)
        done[0] += 1
        if row["status"] == STATUS_OK:
            times.append(float(row["seconds"]))
        s = ctx["subsets"][k]
        msg = (f"RMSE {row['rmse']:.4f} n {row['n']:,} cov {row['coverage']:.3f} | keep valid "
               f"{row['keep_valid_share']:.3f} | {int(row['n_vertices']):,} vertices | {row['seconds']:.1f} s"
               if row["status"] == STATUS_OK else row["status"])
        log(f"[{ctx['site']}] N={s['N']:4d} rep {s['rep']:2d}: {msg} | {done[0]}/{n_all}")

    heavy = []
    for k in todo:
        s = ctx["subsets"][k]
        if s["status"] == STATUS_EMPTY:
            put(k, stage2(ctx, k)[0])
            continue
        pre = preflight_numbers(ctx["p1"]["n_has"][s["slot"]], ctx["p1"]["n_valid"][s["slot"]],
                                len(ctx["keep"]))
        if not pre["passed"]:
            put(k, stage2(ctx, k)[0])
            continue
        heavy.append(k)

    def pred():
        return float(np.median(times)) if times else float(t_pred)

    if pool_factory is None and workers > 1 and hasattr(os, "fork"):
        pool_factory = fork_pool_factory

    if pool_factory is not None and workers > 1 and heavy:
        import concurrent.futures as cf
        _CTX = ctx
        pending = set()
        with pool_factory(workers) as pool:
            for i, k in enumerate(heavy):
                while len(pending) >= workers:
                    fin, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                    for f in fin:
                        put(*f.result())      # a dead worker raises BrokenProcessPool here
                if time.time() + pred() > deadline:
                    stop = {"at": ctx["subsets"][k]["key"], "jobs_left": len(heavy) - i,
                            "reason": "predicted to pass --max-minutes"}
                    break
                pending.add(pool.submit(_pool_job, k))
            for f in cf.as_completed(pending):
                put(*f.result())
    else:
        for i, k in enumerate(heavy):
            if time.time() + pred() > deadline:
                stop = {"at": ctx["subsets"][k]["key"], "jobs_left": len(heavy) - i,
                        "reason": "predicted to pass --max-minutes"}
                break
            put(k, stage2(ctx, k)[0])
    if stop:
        log(f"[{ctx['site']}] STOP before {stop['at']}: {stop['jobs_left']} job(s) left ({stop['reason']})")
    return done[0], stop


def run(args):
    site = args.site
    out = args.out
    v14.guard_out(out)
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f"{site}_nidem_all")
    csv_path, json_path = stem + ".csv", stem + "_settings.json"
    run_lock = acquire_run_lock(stem + ".run.lock")          # noqa: F841 (held until the process exits)
    deadline = _T0 + float(args.max_minutes) * 60.0
    script_sha = sha256_file(os.path.abspath(__file__))
    log(f"[{site}] v15 NIDEM scene-count curve (script sha256 {script_sha[:12]}): workers {args.workers}, row block "
        f"{args.row_block}, scene batch {args.scene_batch}, io lock {args.io_lock}, cache {args.cache_dir}, max "
        f"{args.max_minutes} min; threads {v14.blas_info()}")
    gates = {}

    # 0. inputs: v14 and its refit outputs
    sha = sha256_file(V14_PATH)
    gates["v14_script_sha256"] = {"path": V14_PATH, "expected": V14_SHA256, "found": sha, "ok": sha == V14_SHA256}
    mdir = find_inputs(site, [args.marea_dir] if args.marea_dir else MAREA_DIRS,
                       ("_refit.csv", "_refit_settings.json"))
    if mdir is None:
        raise SystemExit(f"[{site}] no {site}_refit.csv / _refit_settings.json in {args.marea_dir or MAREA_DIRS}")
    mcsv, mjson = os.path.join(mdir, f"{site}_refit.csv"), os.path.join(mdir, f"{site}_refit_settings.json")
    hc, hs = sha256_file(mcsv), sha256_file(mjson)
    ec, es = MAREA_SHA256[site]
    gates["marea_inputs_sha256"] = {"csv": {"path": mcsv, "expected": ec, "found": hc},
                                    "settings": {"path": mjson, "expected": es, "found": hs},
                                    "ok": hc == ec and hs == es}
    msettings = json.load(open(mjson, encoding="utf-8"))
    mdf = marea_rows(mcsv)
    seed, max_px = int(msettings["seed"]), int(msettings["max_px"])
    grid, reps, reps_large, size, jobs = plan_from_settings(msettings)
    log(f"[{site}] v14 plan (pool all): grid {grid}, reps {reps} (N > {v14.LARGE_N}: {reps_large}), "
        f"{len(jobs)} jobs, seed {seed}, max px {max_px:,}; from {mdir}")

    # 1. units, truth, pool
    cfg = site_config(site, args)
    L = load_site_light(site, cfg, seed, max_px)
    gates.update(input_gates(site, cfg, L, msettings))
    rec = L["rec"]
    h_pool = np.asarray(rec["h"], float)
    time_index = np.asarray(rec["time_index"], np.int64)
    if len(h_pool) != size:
        raise SystemExit(f"[{site}] pool has {len(h_pool)} scenes, v14's plan {size}")

    # 2. subsets = v14's draws (+ MAREA's record)
    subsets = build_subsets(seed, site, size, jobs, h_pool, extra=[("marea_record", L["idx_scr"])])
    want = {(int(r.N), int(r.replicate)): str(r.draw_sha1) for r in mdf.itertuples()}
    have = {(s["N"], s["rep"]): s["sha"] for s in subsets if s["rep"] >= 0}
    mism = sorted(k for k in have if want.get(k) != have[k])
    gates["jobs_vs_marea_csv"] = {"jobs": len(have), "csv_rows": len(want), "same_set": set(have) == set(want),
                                  "ok": set(have) == set(want)}
    gates["draws_vs_marea_csv"] = {"checked": len(have), "mismatches": len(mism), "first": mism[:5],
                                   "ok": not mism}
    for k_, v_ in gates.items():
        log(f"[{site}] gate {k_}: {'OK' if v_.get('ok', True) else 'FAILED'}")
    n_empty = sum(1 for s in subsets if s["status"] == STATUS_EMPTY and s["rep"] >= 0)
    log(f"[{site}] {len(subsets)} subsets ({len(jobs)} curve jobs + MAREA's record); {n_empty} with an empty "
        "tide interval (not computable)")

    # settings, compat, resumable CSV
    from experiments import v7_waterline_contour as v7
    cube_info = v14.file_info(cfg["cube"])
    compat = {"version": 1, "site": site, "pool": POOL, "seed": seed, "pool_size": size,
              "pool_dates_sha1": hashlib.sha1("|".join(L["dates_all"]).encode()).hexdigest()[:16],
              "grid": grid, "reps": reps, "reps_large": reps_large, "jobs": len(jobs),
              "units_sha1": v14.array_sha1(L["units_keep"]), "z_stored_sha1": v14.array_sha1(L["z_stored"]),
              "marea_csv_sha256": hc, "pixel_rule": "item"}
    prev = None
    if os.path.exists(json_path):
        prev = json.load(open(json_path, encoding="utf-8"))
        if prev.get("compat") != v14._clean(compat):
            raise SystemExit(f"{json_path} was written with other settings:\n  {prev.get('compat')}\nnow:\n"
                             f"  {compat}\nrefusing to mix them: use another --out")
    rowlog = RowLog(csv_path)
    if rowlog.done:
        log(f"[{site}] resuming: {len(rowlog.done)} row(s) already in {csv_path}")
    settings = prev or {}
    settings.update({
        "site": site, "site_name": v14.SITE_NAMES.get(site, site), "compat": compat, "status": "running",
        "git_head": v14.git_head(), "script": v14.file_info(os.path.relpath(os.path.abspath(__file__), ROOT)),
        "versions": versions(), "blas": v14.blas_info(),
        "question": "NIDEM elevation error against the number of scenes used, on exactly MAREA's 'all'-pool "
                    "subsets of v14 (refit run), scored on the same units",
        "marea_curve": {"source": mcsv, "rows": "pool 'all', mb 0 (the rule of the published products)",
                        "lags": "refit (v14 --lags refit)"},
        "method": {"code": "pyintertidal.waterline_contour via experiments/v7_waterline_contour.py (production "
                           "NIDEM): ITEM pixel mask, 9 tide intervals of the subset's own OTR, h = boundary level "
                           "at the scene time (no lag), median composites, R, gap fill, contours, TIN on the full "
                           "frame, filtered = R in 1..8 and conf <= 0.25",
                   "not_computable": "empty interval (EmptyIntervalError) or pre-flight share of keep with a "
                                     f"valid observation in all 9 intervals < {wc.MIN_VALID_SHARE} (v7's "
                                     "pre-flight guard: a stop of this study, not a rule of published NIDEM); "
                                     "recorded with coverage 0",
                   "fast_path": "phase 1: one pass over the cube, exact sort-free R codes of every subset "
                                "(two curve draws audited against the library's R on the whole frame); "
                                "phase 2: conf at the scored pixels by wc.composites_from_arrays (library R "
                                "asserted equal there), synthetic composites, wc.nidem on the full frame",
                   "scoring": "v14.score on v14's units: RTK ols_against_truth (>= 20 points) / Vaklodingen "
                              "v7.vscore (>= 30 px, median-centred); coverage = share of the units with a value"},
        "inputs": {"extract": v14.file_info(cfg["extract"]), "cube": cube_info,
                   "overpass": v14.file_info(v7.SITES[site]["overpass"]),
                   "nidem_primary": v14.file_info(os.path.join(cfg["nidem"], "nidem.npz")),
                   "nidem_sensitivity": v14.file_info(os.path.join(cfg["nidem"], SENS, "nidem.npz")),
                   "z_stored": L["z_src"], "v14": V14_PATH, "marea_csv": mcsv, "marea_settings": mjson,
                   "input_overrides": {k: getattr(args, f"{k}_override") for k in
                                       ("extract", "product", "comparison", "nidem", "cube")
                                       if getattr(args, f"{k}_override", None)}},
        "pool": {"size": size, "boundary": rec["boundary"], "rule": rec["rule"],
                 "first_last": [str(L["dates_all"][0]), str(L["dates_all"][-1])],
                 "marea_record_scenes": int(len(L["idx_scr"]))},
        "units": {"kind": L["kind"], "n_scored": int(len(L["units_keep"])), "n_F": L["n_units_F"],
                  "pixels": int(len(np.unique(L["units_keep"]))), **{k: v for k, v in L["truth_info"].items()
                                                                       if not isinstance(v, np.ndarray)}},
        "plan": {"grid": grid, "reps": reps, "reps_large": reps_large, "large_n": v14.LARGE_N,
                 "jobs": len(jobs), "order": "v14's replicate-major order (whole pool, then replicate 0 of "
                                             "every N, ...)", "seed": seed},
        "reading": {"row_block": int(args.row_block), "scene_batch": int(args.scene_batch),
                    "io_lock": args.io_lock, "cache_dir": args.cache_dir, "workers": int(args.workers)},
        "max_minutes": float(args.max_minutes),
    })
    settings.setdefault("runs", [])
    this_run = {"started": _dt.datetime.now().isoformat(timespec="seconds"), "argv": sys.argv[1:],
                "script_sha256": script_sha}
    settings["runs"].append(this_run)
    settings.pop("error", None)
    settings["gates"] = gates
    settings["timing"] = {"load_s": round(L["load_seconds"], 1)}

    def stop_on_gates(where):
        if gate_ok(gates):
            return
        failed = [k for k, v in gates.items() if isinstance(v, dict) and not v.get("ok", True)]
        if args.soft_gates:
            settings["gates_overridden"] = failed
            log(f"[{site}] WARNING: gate(s) {failed} failed and --soft-gates is set: continuing")
            return
        settings["status"] = "gate_failed"
        this_run["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
        v14.write_json(json_path, settings)
        raise SystemExit(f"[{site}] gate(s) {failed} failed ({where}); see {json_path}")

    stop_on_gates("inputs")
    v14.write_json(json_path, settings)

    # 3. phase 1
    keep = L["keep"]
    units_flat = keep[L["units_keep"]]
    unit_px = np.unique(units_flat)
    audit = audit_choice(subsets)
    key = phase1_key(site, cube_info, time_index, h_pool, keep, unit_px, subsets, audit)
    cache = os.path.join(args.cache_dir, f"v15_phase1_{site}_{key}.npz") if args.cache_dir else None
    if cache:
        v14.guard_out(args.cache_dir)
        os.makedirs(args.cache_dir, exist_ok=True)
    p1 = load_phase1(cache, key, subsets) if cache and os.path.exists(cache) else None
    lock = IoLock(args.io_lock)
    if p1 is not None:
        log(f"[{site}] phase 1 from the cache {cache}")
    else:
        reader = CubeReader(cfg["cube"], time_index, args.scene_batch, lock)
        if (reader.H, reader.W) != (L["H"], L["W"]):
            raise SystemExit(f"[{site}] cube {reader.H}x{reader.W} != extraction {L['H']}x{L['W']}")
        grid_dates = v7.cube_grid(cfg["cube"])["dates"]
        if not np.array_equal(grid_dates, L["ext_dates"]):
            raise SystemExit(f"[{site}] cube dates differ from the extraction dates")
        if reader.chunk_rows and int(args.row_block) % reader.chunk_rows:
            log(f"[{site}] WARNING: --row-block {args.row_block} is not a multiple of the cube's {reader.chunk_rows} "
                "chunk rows: chunks are decompressed more than once")
        log(f"[{site}] phase 1: {len(time_index)} scenes x {L['H']}x{L['W']} px, {sum(1 for s in subsets if s['status'] == 'assigned')} "
            f"subsets with an interval assignment, {len(unit_px):,} scored px, {len(keep):,} keep px; full-frame "
            f"audit of {audit}")
        p1 = phase1(reader, (len(time_index), L["H"], L["W"]), subsets, keep, unit_px, int(args.row_block),
                    label=f"[{site}] ", audit=audit)
        reader.close()
        p1["timing"].update(io_lock_wait_s=round(lock.wait_s, 1), io_lock_acquired=lock.n,
                            chunk_rows=reader.chunk_rows)
        if cache:
            t0 = time.time()
            try:
                with lock:
                    save_phase1(cache, p1, key)
                log(f"[{site}] phase-1 cache written: {cache} [{time.time() - t0:.1f} s]")
            except OSError as e:          # the cache is optional: never lose phase 1 over it
                log(f"[{site}] WARNING: phase-1 cache not written ({e}); continuing without it")
    settings["timing"]["phase1"] = {k: v for k, v in p1["timing"].items() if k != "blocks"}
    settings["timing"]["phase1_blocks"] = p1["timing"].get("blocks")
    settings["timing"]["peak_rss_mb_after_phase1"] = peak_rss_mb()

    ctx = dict(site=site, size=size, subsets=subsets, p1=p1, keep=keep, units_keep=L["units_keep"],
               units_flat=units_flat, unit_px=unit_px, unit_col=np.searchsorted(unit_px, units_flat),
               truth=L["truth"], kind=L["kind"], H=L["H"], W=L["W"], time_index=time_index)

    # the full-frame audit of the fast R of two curve draws (phase 1)
    aud = p1.get("audit", {})
    gates["fast_r_full_frame_audit"] = {
        "subsets": aud, "chosen": audit, "frame_px": int(L["H"] * L["W"]),
        "rule": "fast R equal to wc.relative_extents(wc.composites_from_arrays(...)) on every pixel of the frame",
        "ok": bool(sorted(aud) == sorted(audit) and all(a["differing_px"] == 0 and a["px"] == L["H"] * L["W"]
                                                        for a in aud.values()))}

    # 4. product gates (before any curve row)
    refs = {p.get("key"): p.get("on_scored_units") for p in msettings.get("reference_points", [])}
    k_full = next(i for i, s in enumerate(subsets) if s["rep"] >= 0 and s["N"] == size)
    k_rec = next(i for i, s in enumerate(subsets) if s["key"] == "marea_record")
    prim_dir, sens_dir = cfg["nidem"], os.path.join(cfg["nidem"], SENS)
    primary_computable = gates["nidem_primary_status"]["computable"]
    t0 = time.time()
    full_row = None
    gates["full_pool_preflight_vs_stored"] = preflight_gate(ctx, k_full, os.path.join(prim_dir, "nidem_preflight.json"),
                                                            expect_stop=not primary_computable)
    if primary_computable:
        gates["full_pool_vs_nidem_primary"], full_row = product_gate(
            ctx, k_full, os.path.join(prim_dir, "nidem.npz"), refs.get("nidem_primary"))
    else:
        full_row = stage2(ctx, k_full)[0]
        gates["full_pool_not_computable_as_primary"] = {
            "status": full_row["status"], "reason": full_row["reason"],
            "ok": full_row["status"] == STATUS_PREFLIGHT}
    gates["marea_record_preflight_vs_stored"] = preflight_gate(ctx, k_rec, os.path.join(sens_dir, "nidem_preflight.json"),
                                                               expect_stop=False)
    gates["marea_record_vs_nidem_sensitivity"], rec_row = product_gate(
        ctx, k_rec, os.path.join(sens_dir, "nidem.npz"), refs.get("nidem_sensitivity"))
    t_gate = time.time() - t0
    settings["timing"]["product_gates_s"] = round(t_gate, 1)
    for k_ in ("fast_r_full_frame_audit", "full_pool_preflight_vs_stored", "full_pool_vs_nidem_primary",
               "full_pool_not_computable_as_primary", "marea_record_preflight_vs_stored",
               "marea_record_vs_nidem_sensitivity"):
        if k_ in gates:
            v_ = gates[k_]
            extra = {kk: v_[kk] for kk in ("bitwise", "max_abs_score_diff", "status") if kk in v_}
            if k_ == "fast_r_full_frame_audit":
                extra = {kk: (a["px"], a["differing_px"]) for kk, a in v_["subsets"].items()}
            log(f"[{site}] gate {k_}: {'OK' if v_.get('ok') else 'FAILED'} {extra}")
            if not v_.get("ok") and "checks" in v_:
                log(f"[{site}]   checks {v_['checks']}; tolerance {v_.get('tolerance')}")
    settings["reference_rows"] = {"full_pool": full_row, "marea_record": rec_row}
    # draws stopped by the pre-flight: the coverage they could have had at most without the stop
    stops = preflight_stop_units(ctx)
    settings["preflight_stops"] = {
        "rule": "units with R in 1..8 under the draw's exact codes (necessary for a NIDEM value): an upper bound "
                "on the coverage of a draw stopped by the 1 % pre-flight",
        "max_coverage_without_stop": max((v["max_coverage_without_stop"] for v in stops.values()), default=None),
        "draws": stops}
    if stops:
        worst = max(stops.values(), key=lambda v: v["max_coverage_without_stop"])
        log(f"[{site}] {len(stops)} draw(s) stop at the pre-flight; without the stop at most "
            f"{worst['units_R_1_8']:,} of {worst['n_units']:,} units could have had a value "
            f"(coverage <= {worst['max_coverage_without_stop']:.4f}, N={worst['N']} rep {worst['rep']})")
    stop_on_gates("products")
    v14.write_json(json_path, settings)

    def n_curve_rows():
        return sum(1 for (N, rep) in rowlog.done if any(s["N"] == N and s["rep"] == rep for s in subsets))

    if args.only_gates:
        # a gates-only rerun of a finished site leaves it complete
        settings["status"] = "complete" if n_curve_rows() == len(jobs) else "gates_only"
        this_run["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
        this_run["status"] = f"only_gates ({settings['status']})"
        v14.write_json(json_path, settings)
        log(f"[{site}] --only-gates: gates passed; stopping before the curve (status {settings['status']})")
        return
    rowlog.append(full_row)

    # 5. the curve
    todo = [i for i, s in enumerate(subsets) if s["rep"] >= 0 and not rowlog.is_done(s["N"], s["rep"])]
    if args.max_jobs is not None:
        todo = todo[:int(args.max_jobs)]
    t_pred = max(1.0, float(gates.get("marea_record_vs_nidem_sensitivity", {}).get("stage2_seconds", 5.0)))
    log(f"[{site}] phase 2: {len(todo)} job(s) to do, workers {args.workers}, about {t_pred:.0f} s each")
    t0 = time.time()
    try:
        n_done, stop = run_jobs(ctx, todo, rowlog, int(args.workers), deadline, t_pred)
    except BaseException as e:            # a crash (a dead worker, an assertion): say so, then stop loudly
        settings["status"] = "crashed"
        settings["error"] = f"{type(e).__name__}: {e}"
        settings["timing"]["phase2_s"] = round(time.time() - t0, 1)
        settings["not_computable_per_N"] = nc_counts(list(rowlog.done.values()), grid)
        settings["progress"] = {"jobs": len(jobs), "rows": n_curve_rows(), "stopped": "crashed"}
        this_run.update(finished=_dt.datetime.now().isoformat(timespec="seconds"), status="crashed",
                        error=settings["error"])
        v14.write_json(json_path, settings)
        log(f"[{site}] CRASHED in phase 2 ({settings['error']}); {n_curve_rows()}/{len(jobs)} rows kept in "
            f"{csv_path}: rerun the same command to resume")
        raise
    settings["timing"]["phase2_s"] = round(time.time() - t0, 1)
    settings["timing"]["total_s"] = round(time.time() - _T0, 1)
    settings["timing"]["peak_rss_mb"] = peak_rss_mb()
    settings["timing"]["peak_rss_mb_largest_worker"] = peak_rss_mb(children=True)

    # finish
    rows = list(rowlog.done.values())
    n_jobs_done = n_curve_rows()
    settings["not_computable_per_N"] = nc_counts(rows, grid)
    for N_, v_ in settings["not_computable_per_N"].items():
        cov_ = [d["max_coverage_without_stop"] for d in stops.values() if str(d["N"]) == N_]
        v_["preflight_stop_max_coverage_without_stop"] = max(cov_) if cov_ else None
    settings["status"] = ("complete" if n_jobs_done == len(jobs) else "stopped_by_time" if stop else "incomplete")
    settings["progress"] = {"jobs": len(jobs), "rows": n_jobs_done, "done_this_run": n_done, "stopped": stop}
    this_run["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    this_run["status"] = settings["status"]
    this_run["timing"] = {k: v for k, v in settings["timing"].items() if k != "phase1_blocks"}
    v14.write_json(json_path, settings)
    tot = {k: sum(v[k] for v in settings["not_computable_per_N"].values())
           for k in ("draws", "not_computable_empty_interval", "not_computable_preflight", "with_score")}
    log(f"[{site}] {settings['status']}: {n_jobs_done}/{len(jobs)} rows -> {csv_path}, {json_path}; draws "
        f"{tot['draws']}, empty interval {tot['not_computable_empty_interval']}, pre-flight stop "
        f"{tot['not_computable_preflight']}, with a score {tot['with_score']}; peak RSS {peak_rss_mb()} MB "
        f"(largest worker {peak_rss_mb(children=True)} MB)")


# ─────────────────────────────────────────────────────────────────────────────
#  the hook: NIDEM on GIVEN scene subsets (added 2026-10-26 for the
#  fixed-configuration tables of v19 and the windows of v16; nothing above
#  changes)
# ─────────────────────────────────────────────────────────────────────────────

#: the key of the hook's gate subset: MAREA's screened record (result.json
#: record_dates: 97 / 68 / 71 / 54 scenes), whose NIDEM is the stored
#: sensitivity_marea_record product
HOOK_GATE = "marea_record"
#: the hook's gates already passed in this process (key: the site's inputs,
#: the units and the truth): v16 calls the hook once per window, the gate
#: subset is read and checked once
_HOOK_GATES = {}
#: the per-pixel masks the hook applies on request (``pixel_rule``, per call
#: or per subset): 'item' = ITEM's pixel-quality mask, v15's and v7's
#: production NIDEM (valid iff the SCL is finite and not in (0, 1, 3, 8, 9,
#: 10), i.e. SCL in {2, 4, 5, 6, 7, 11}); 'common' = the fixed configuration's
#: mask of every method (SCL in wc.CLEAR = {4, 5, 6, 7}, the extraction's C).
#: Both mask the same raw float32 NDWI; one read of the cube gives both. The
#: gate (MAREA's record against the stored sensitivity product) is ITEM's.
HOOK_PIXEL_RULES = ("item", "common")
PIXEL_RULE_TEXT = {
    "item": "item (wc.scene_ndwi rule='item', v7's reading of the cube)",
    "common": ("common (SCL in wc.CLEAR = {4, 5, 6, 7} = the extraction's C, on the same raw NDWI; asserted equal to "
               "marea_extract's C and Y at every keep pixel of every scene)"),
}


class HookGateError(RuntimeError):
    """A gate of the hook failed (its product gate, or the common mask not
    being the extraction's C and Y): a site-level failure, the same for every
    subset of the call."""


class MaskedCubeReader(CubeReader):
    """:class:`CubeReader`'s read (the same isel calls of B03, B08 and SCL,
    the same float32 NDWI of wc.scene_ndwi) giving the masked NDWI under every
    pixel rule of ``rules``, stacked along time: rows [k T, (k + 1) T) hold
    rules[k]. The 'item' rows are bitwise CubeReader's; 'common' masks the
    same NDWI with SCL in wc.CLEAR. ``check(t, r0, r1, ndwi, valid, scl)``
    (v7.ExtractionCheck: the SCL gives exactly the extraction's C and the
    NDWI its Y at every keep pixel) sees every scene of every block, with the
    common mask as ``valid``."""

    def __init__(self, path, time_index, scene_batch=32, lock=None, rules=("common",), check=None):
        super().__init__(path, time_index, scene_batch, lock)
        rules = tuple(rules)
        if not rules or len(set(rules)) != len(rules) or any(r not in HOOK_PIXEL_RULES for r in rules):
            self.close()
            raise ValueError(f"rules must be distinct values of {HOOK_PIXEL_RULES}: {rules}")
        self.rules, self.check = rules, check

    def __call__(self, r0, r1):
        T, R = len(self.ti), r1 - r0
        D = np.empty((len(self.rules) * T, R, self.W), np.float32)
        with self.lock:
            for s0 in range(0, T, self.batch):
                tb = self.ti[s0:s0 + self.batch]
                if self.batch == 1:
                    sl = {self.t_dim: int(tb[0]), "y": slice(r0, r1)}
                else:
                    sl = {self.t_dim: tb, "y": slice(r0, r1)}
                shp = (len(tb), R, self.W)
                scl = self.bands["SCL"].isel(sl).values.reshape(shp)
                ndwi, item = wc.scene_ndwi(self.bands["B03"].isel(sl).values.reshape(shp),
                                           self.bands["B08"].isel(sl).values.reshape(shp), scl, rule="item")
                common = np.isin(scl, wc.CLEAR)
                if self.check is not None:
                    for k, t in enumerate(tb):
                        self.check(int(t), r0, r1, ndwi[k], common[k], scl[k])
                for m, rule in enumerate(self.rules):
                    D[m * T + s0:m * T + s0 + len(tb)] = np.where(item if rule == "item" else common, ndwi,
                                                                  np.nan).astype(np.float32)
        return D


def _hook_site(site, inputs=None):
    """What :func:`nidem_on_scenes` reads of a site: v15's inputs
    (:func:`site_config`, with the non-empty ``inputs`` of extract / product /
    comparison / nidem / cube), keep and the grid, the extraction dates (the
    cube's time axis), the overpass times, v7.build_record's 'all' pool with
    its boundary object, MAREA's record inside the pool (result.json
    record_dates) and the folder of the stored sensitivity product."""
    import pandas as pd
    from experiments import v7_waterline_contour as v7

    cfg = site_config(site)
    for k in ("extract", "product", "comparison", "nidem", "cube"):
        v = (inputs or {}).get(k)
        if v:
            cfg[k] = v
    with np.load(cfg["extract"], allow_pickle=False) as ex:
        keep = np.asarray(ex["keep"], np.int64)
        ext_dates = np.asarray(ex["dates"]).astype(str)
        H, W = (int(v) for v in ex["shape"])
    with open(os.path.join(cfg["product"], "result.json"), encoding="utf-8") as f:
        res = json.load(f)
    if site != "villaviciosa":
        _, missing = v14.gauge_cache(site)
        if missing:
            raise SystemExit(f"[{site}] gauge cache missing for {missing} in data_v4/gauges: refusing to download "
                             "(production used the cache)")
    rec = v7.build_record(site, v7.SITES[site], ext_dates)
    pool_dates = np.asarray(rec["dates"]).astype(str)
    pos = {d: i for i, d in enumerate(pool_dates)}
    rec_dates = [str(d)[:10] for d in res["record_dates"]]
    lost = [d for d in rec_dates if d not in pos]
    if lost:
        raise RuntimeError(f"[{site}] {len(lost)} record date(s) not in the 'all' pool: {lost[:5]}")
    with open(v7.SITES[site]["overpass"], encoding="utf-8") as f:
        overpass = {k: pd.Timestamp(v).to_pydatetime() for k, v in json.load(f).items()}
    return dict(site=site, cfg=cfg, keep=keep, H=H, W=W, ext_dates=ext_dates, overpass=overpass,
                pool_dates=pool_dates, pool_ti=np.asarray(rec["time_index"], np.int64),
                pool_h=np.asarray(rec["h"], float), boundary=rec["boundary_obj"], boundary_name=rec["boundary"],
                record=np.array([pos[d] for d in rec_dates], np.int64), sens_dir=os.path.join(cfg["nidem"], SENS),
                cube=cfg["cube"], cube_info=v14.file_info(cfg["cube"]), extract=cfg["extract"])


def _hook_open_cube(S, time_index, lock, scene_batch):
    """The cube reader of the hook: :class:`CubeReader` (v7's reading, the
    ITEM rule) on the scenes ``time_index``, with v15's checks (the cube grid
    and dates are the extraction's)."""
    reader = CubeReader(S["cube"], time_index, scene_batch, lock)
    try:
        if (reader.H, reader.W) != (int(S["H"]), int(S["W"])):
            raise SystemExit(f"[{S['site']}] cube {reader.H}x{reader.W} != extraction {S['H']}x{S['W']}")
        cube_dates = np.array([str(np.datetime64(v, "D")) for v in reader.ds[reader.t_dim].values])
        if not np.array_equal(cube_dates, np.asarray(S["ext_dates"]).astype(str)):
            raise SystemExit(f"[{S['site']}] cube dates differ from the extraction dates")
    except BaseException:
        reader.close()
        raise
    return reader


def _hook_open_cube_masked(S, time_index, lock, scene_batch, rules=("common",), check=None):
    """The cube reader of the hook when a subset asks for the common mask:
    :class:`MaskedCubeReader` (one read, the masked NDWI under every rule of
    ``rules``; ``check`` = the extraction check) with :func:`_hook_open_cube`'s
    checks (the cube grid and dates are the extraction's)."""
    reader = MaskedCubeReader(S["cube"], time_index, scene_batch, lock, rules=rules, check=check)
    try:
        if (reader.H, reader.W) != (int(S["H"]), int(S["W"])):
            raise SystemExit(f"[{S['site']}] cube {reader.H}x{reader.W} != extraction {S['H']}x{S['W']}")
        cube_dates = np.array([str(np.datetime64(v, "D")) for v in reader.ds[reader.t_dim].values])
        if not np.array_equal(cube_dates, np.asarray(S["ext_dates"]).astype(str)):
            raise SystemExit(f"[{S['site']}] cube dates differ from the extraction dates")
    except BaseException:
        reader.close()
        raise
    return reader


def _hook_rule_of(specs, pixel_rule):
    """{subset key: pixel rule}: the spec's 'pixel_rule' (a dict spec) or the
    call's ``pixel_rule``; every value one of :data:`HOOK_PIXEL_RULES`."""
    if pixel_rule not in HOOK_PIXEL_RULES:
        raise ValueError(f"pixel_rule {pixel_rule!r} is not one of {HOOK_PIXEL_RULES}")
    out = {}
    for k, spec in specs:
        r = (spec.get("pixel_rule") if isinstance(spec, dict) else None) or pixel_rule
        if r not in HOOK_PIXEL_RULES:
            raise ValueError(f"subset {k!r}: pixel_rule {r!r} is not one of {HOOK_PIXEL_RULES}")
        out[str(k)] = r
    return out


def _hook_resolve(S, key, spec):
    """(dates, cube time indices, the caller's levels or None) of one given
    subset: ``spec`` holds 'dates', or 'rows' (extraction rows = cube time
    indices), or 'pool_idx' (indices into v7.build_record's 'all' pool), and
    optionally 'h' (the caller's zero-lag levels, checked later); a bare
    sequence is a list of dates."""
    if not isinstance(spec, dict):
        spec = {"dates": spec}
    ext = np.asarray(S["ext_dates"]).astype(str)
    rows = spec.get("rows")
    if spec.get("pool_idx") is not None:
        dates = np.asarray(S["pool_dates"])[np.asarray(spec["pool_idx"], np.int64)]
    elif spec.get("dates") is not None:
        dates = np.array([str(d)[:10] for d in spec["dates"]])
    elif rows is not None:
        dates = ext[np.asarray(rows, np.int64)]
    else:
        raise ValueError(f"subset {key!r}: give 'dates', 'rows' (extraction rows) or 'pool_idx'")
    dates = np.asarray(dates).astype(str)
    if not len(dates) or (len(dates) > 1 and np.any(dates[1:] <= dates[:-1])):
        raise ValueError(f"subset {key!r}: the scenes must be non-empty, sorted and unique")
    epos = {d: i for i, d in enumerate(ext)}
    miss = [d for d in dates if d not in epos]
    if miss:
        raise ValueError(f"subset {key!r}: {len(miss)} date(s) are not extraction (cube) dates: {miss[:5]}")
    ti = np.array([epos[d] for d in dates], np.int64)
    if rows is not None and not np.array_equal(ti, np.asarray(rows, np.int64)):
        raise ValueError(f"subset {key!r}: the rows given are not the extraction rows of its dates")
    h = spec.get("h")
    return dates, ti, (None if h is None else np.asarray(h, float))


def _hook_levels(S, dates):
    """h_B without lag at the scenes ``dates`` (sorted extraction dates):
    v7.build_record's level where the scene is in its pool (bitwise), else
    the same boundary object read at the overpass label instant (one
    chronological call, as build_record reads it). Returns (levels, in
    pool)."""
    import pandas as pd

    pos = {d: i for i, d in enumerate(np.asarray(S["pool_dates"]).astype(str))}
    h = np.full(len(dates), np.nan)
    in_pool = np.array([d in pos for d in dates], bool)
    for q in np.flatnonzero(in_pool):
        h[q] = float(np.asarray(S["pool_h"])[pos[dates[q]]])
    other = np.flatnonzero(~in_pool)
    if len(other):
        miss = [dates[q] for q in other if dates[q] not in S["overpass"]]
        if miss:
            raise ValueError(f"[{S['site']}] {len(miss)} scene(s) without an overpass time: {miss[:5]}")
        t = pd.DatetimeIndex(pd.to_datetime([S["overpass"][dates[q]] for q in other]))
        if t.tz is not None:
            t = t.tz_localize(None)
        if len(t) > 1 and not t.is_monotonic_increasing:
            raise RuntimeError("overpass times are not chronological")
        lev = np.asarray(S["boundary"].levels(t), float)
        if lev.shape != (len(other),):
            raise RuntimeError(f"the boundary returned {lev.shape} levels for {len(other)} instants")
        h[other] = lev
    bad = [dates[q] for q in np.flatnonzero(~np.isfinite(h))]
    if bad:
        raise ValueError(f"[{S['site']}] no finite boundary level at {len(bad)} scene(s): {bad[:5]}")
    return h, in_pool


def _hook_gate_without_scores(g):
    """v15's product gate when the caller gave no truth: the scores cannot be
    compared, the product is (R bitwise on the full frame and either every
    check bitwise or 0 validity flips at the units with max |dz| <= TOL_DZ)."""
    g = dict(g)
    tol = g.get("tolerance") or {}
    chk = g.get("checks") or {}
    g["scores_v14_reference"] = None
    g["note"] = "no truth given: the product is compared with the stored one, the scores are not"
    g["ok"] = bool(chk.get("R_full_frame") and (g.get("bitwise") or (
        tol.get("filtered_validity_flips_units") == 0 and tol.get("filtered_max_abs_dz_units_m", 1.0) <= TOL_DZ)))
    return g


def _hook_run(S, specs, units_keep, truth=None, kind=None, open_reader=None, io_lock=None, cache_dir=None,
              row_block=256, scene_batch=32, gate=True, soft_gates=False, pixel_rule="item"):
    """v15's NIDEM path on the given subsets ``specs`` = [(key, spec)] of the
    site ``S`` (:func:`_hook_site`): ONE phase-1 read of the cube for every
    subset (and MAREA's record, the gate), then phase 2 per subset, scored on
    the caller's units. Returns {key: result} (see :func:`nidem_on_scenes`).

    ``pixel_rule`` ('item', the default and v15's rule, or 'common'; a dict
    spec may set its own 'pixel_rule'): the per-pixel mask of the subset. One
    read serves both rules (:class:`MaskedCubeReader`: the scenes are stacked
    once per rule needed, a subset reads its rule's rows); the gate subset is
    always ITEM's (the stored sensitivity product is ITEM's). With only ITEM
    subsets the path, the phase-1 key and the values are those of the
    ITEM-only hook. With a common subset every scene is checked, at every
    keep pixel, against the extraction's C and Y (v7.ExtractionCheck, never
    soft: a mismatch raises :class:`HookGateError`).

    ``open_reader(S, time_index, lock, scene_batch)`` gives the block reader
    (default :func:`_hook_open_cube`); when a common subset is asked for it
    is called with ``rules=`` and ``check=`` too (default
    :func:`_hook_open_cube_masked`)."""
    import inspect
    import shutil
    import tempfile

    t_start = time.perf_counter()
    site = S["site"]
    keep = np.asarray(S["keep"], np.int64)
    H, W = int(S["H"]), int(S["W"])
    units_keep = np.ascontiguousarray(units_keep, np.int64)
    if units_keep.ndim != 1 or not len(units_keep) or units_keep.min() < 0 or units_keep.max() >= len(keep):
        raise ValueError("units_keep must be a non-empty 1-D array of keep indices")
    truth_given = truth is not None
    truth = (np.full(len(units_keep), np.nan) if not truth_given
             else np.ascontiguousarray(truth, float).reshape(-1))
    if truth.shape != units_keep.shape:
        raise ValueError(f"truth has {truth.shape} values for {len(units_keep)} units")
    kind = str(kind or "vaklodingen")
    keys = [str(k) for k, _ in specs]
    if len(set(keys)) != len(keys) or HOOK_GATE in keys:
        raise ValueError(f"subset keys must be unique and not {HOOK_GATE!r}: {keys}")
    rule_of = _hook_rule_of(specs, pixel_rule)
    units_flat = keep[units_keep]
    unit_px = np.unique(units_flat)
    ci = S.get("cube_info") or {}
    gkey = hashlib.sha1((f"{site}|{S.get('cube')}|{ci.get('bytes')}|{ci.get('mtime')}|{S.get('extract')}|"
                         f"{S.get('sens_dir')}|{kind}|").encode() + units_keep.tobytes() + truth.tobytes()).hexdigest()
    prev_gate = _HOOK_GATES.get(gkey) if gate else None
    run_gate = bool(gate) and prev_gate is None

    # the subsets and the union of their scenes (+ MAREA's record for the gate)
    resolved = [(k, *_hook_resolve(S, k, spec)) for k, spec in specs]
    rec_dates = np.asarray(S["pool_dates"]).astype(str)[np.asarray(S["record"], np.int64)]
    union = set()
    for _, d, _, _ in resolved:
        union |= set(d.tolist())
    if run_gate:
        union |= set(rec_dates.tolist())
    dates_u = np.array(sorted(union))
    epos = {d: i for i, d in enumerate(np.asarray(S["ext_dates"]).astype(str))}
    ti_u = np.array([epos[d] for d in dates_u], np.int64)
    h_u, in_pool = _hook_levels(S, dates_u)
    # the rules read (ITEM first: its rows, offset 0, are the ITEM-only hook's), the scenes stacked per rule
    rules = tuple(r for r in HOOK_PIXEL_RULES if r in set(rule_of.values()) or (r == "item" and run_gate))
    off = {r: q * len(ti_u) for q, r in enumerate(rules)}
    ti_all, h_all, pool_all = (np.tile(a, len(rules)) for a in (ti_u, h_u, in_pool))
    upos = {d: i for i, d in enumerate(dates_u)}
    subsets = []
    for key, d, ti, hg in resolved:
        idx = np.array([upos[x] for x in d], np.int64)
        if hg is not None:
            if hg.shape != idx.shape:
                raise ValueError(f"subset {key!r}: {hg.shape} levels for {len(idx)} scenes")
            dh = float(np.max(np.abs(hg - h_u[idx])))
            if not dh <= v14.TOL_LEVEL:
                raise ValueError(f"subset {key!r}: the caller's levels differ from NIDEM's (v7.build_record's "
                                 f"boundary at the overpass times) by up to {dh:.3g} m")
        subsets.append(make_subset(str(key), len(idx), -1, idx + off[rule_of[str(key)]], h_all))
    k_gate = None
    if run_gate:
        subsets.append(make_subset(HOOK_GATE, len(rec_dates), -1,
                                   np.array([upos[x] for x in rec_dates], np.int64) + off["item"], h_all,
                                   gate=True))
        k_gate = len(subsets) - 1
    # the full-frame audit of the fast R: the given subset with the fewest scenes
    cand = [s for s in subsets if s["status"] == "assigned" and not s["gate"]]
    audit = [min(cand, key=lambda s: (s["N"], s["key"]))["key"]] if cand else []

    # phase 1: one read of the cube for every subset (and every rule)
    pkey = phase1_key(site, ci, ti_all, h_all, keep, unit_px, subsets, audit)
    if rules != ("item",):
        # another reader and the extraction check: never the key of an ITEM-only read
        ext = v14.file_info(S["extract"]) if S.get("extract") and os.path.exists(str(S["extract"])) else {}
        pkey = hashlib.sha1((f"{pkey}|rules={','.join(rules)}|{S.get('extract')}|{ext.get('bytes')}|"
                             f"{ext.get('mtime')}|").encode()
                            + inspect.getsource(MaskedCubeReader).encode()).hexdigest()[:20]
    cache = os.path.join(cache_dir, f"v15_hook_phase1_{site}_{pkey}.npz") if cache_dir else None
    lock = IoLock(io_lock)
    p1 = None
    if cache:
        v14.guard_out(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.exists(cache):
            p1 = load_phase1(cache, pkey, subsets)
    if p1 is None:
        chk, tmp, ext_rows = None, None, ()
        try:
            if "common" in rules:
                from experiments import v7_waterline_contour as v7
                # the extraction's Y and C of the scenes read (memory-mapped when large; with a cache_dir outside
                # the repository), for the check at every keep pixel of every scene
                tmp = tempfile.mkdtemp(prefix="v15_hook_ext_", dir=cache_dir or None)
                ext_rows = (v7.npz_rows(S["extract"], "Y", ti_u, tmp), v7.npz_rows(S["extract"], "C", ti_u, tmp))
                chk = v7.ExtractionCheck(ext_rows[0], ext_rows[1], ti_u, keep, W)
                reader = (open_reader or _hook_open_cube_masked)(S, ti_u, lock, scene_batch, rules=rules, check=chk)
            else:
                reader = (open_reader or _hook_open_cube)(S, ti_u, lock, scene_batch)
            try:
                if getattr(reader, "chunk_rows", None) and int(row_block) % int(reader.chunk_rows):
                    log(f"[{site}] hook: row block {row_block} is not a multiple of the cube's {reader.chunk_rows} "
                        "chunk rows")
                log(f"[{site}] NIDEM hook phase 1: {len(ti_u)} scenes ({int(in_pool.sum())} in the 'all' pool), "
                    f"pixel rules {list(rules)}, {len(subsets)} subsets{' (with the gate)' if run_gate else ''}, "
                    f"{len(unit_px):,} scored px")
                try:
                    p1 = phase1(reader, (len(ti_all), H, W), subsets, keep, unit_px, int(row_block),
                                label=f"[{site}] hook ", audit=audit)
                except AssertionError as e:
                    if chk is None:
                        raise
                    raise HookGateError(f"[{site}] the common mask is not the extraction's C and Y: {e}") from e
            finally:
                reader.close()
            if chk is not None:
                if chk.n_c != len(ti_u) * len(keep):
                    raise HookGateError(f"[{site}] the extraction check compared {chk.n_c:,} scene-pixels, not "
                                        f"{len(ti_u)} x {len(keep):,}")
                p1["timing"]["extraction_check"] = chk.summary()
        finally:
            chk = ext_rows = None
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
        p1["timing"].update(io_lock_wait_s=round(lock.wait_s, 1), io_lock_acquired=lock.n)
        if cache:
            try:
                with lock:
                    save_phase1(cache, p1, pkey)
            except OSError as e:          # the cache is optional
                log(f"[{site}] hook: phase-1 cache not written ({e})")
    ctx = dict(site=site, size=len(ti_u), subsets=subsets, p1=p1, keep=keep, units_keep=units_keep,
               units_flat=units_flat, unit_px=unit_px, unit_col=np.searchsorted(unit_px, units_flat),
               truth=truth, kind=kind, H=H, W=W, time_index=ti_all)

    # the gates: the fast R on a whole frame, the common mask against the extraction, and MAREA's record
    # against the stored product (ITEM's)
    gates = {}
    aud = p1.get("audit", {})
    if audit:
        gates["fast_r_full_frame_audit"] = {
            "subsets": aud, "chosen": audit, "frame_px": H * W,
            "ok": bool(sorted(aud) == sorted(audit) and all(a["differing_px"] == 0 and a["px"] == H * W
                                                            for a in aud.values()))}
    if "common" in rules:
        ec = (p1.get("timing") or {}).get("extraction_check")
        gates["common_mask_equals_extraction_C_and_Y_at_keep"] = {
            "scenes": int(len(ti_u)), "summary": ec, "ok": bool(ec),
            "rule": "SCL in wc.CLEAR gives the extraction's C and the raw NDWI its Y at every keep pixel (v7)"}
    stored_npz = os.path.join(S["sens_dir"], "nidem.npz")
    gate_row = None
    gate_bitwise = False
    if run_gate:
        if not os.path.exists(stored_npz):
            gates["marea_record_vs_nidem_sensitivity"] = {"path": stored_npz, "ok": False,
                                                         "reason": "no stored sensitivity_marea_record product"}
        else:
            gates["marea_record_preflight_vs_stored"] = preflight_gate(
                ctx, k_gate, os.path.join(S["sens_dir"], "nidem_preflight.json"), expect_stop=False)
            ref = None
            if truth_given:
                with np.load(stored_npz, allow_pickle=False) as st:
                    ref = v14.score(kind, truth, np.asarray(st["filtered_keep"])[units_keep])
            g, gate_row = product_gate(ctx, k_gate, stored_npz, ref)
            gates["marea_record_vs_nidem_sensitivity"] = g if truth_given else _hook_gate_without_scores(g)
            gate_bitwise = bool(gates["marea_record_vs_nidem_sensitivity"].get("ok") and g.get("bitwise"))
    ok = all(v.get("ok", True) for v in gates.values())
    failed = [k for k, v in gates.items() if not v.get("ok", True)]
    summary = {"ok": bool(ok), "failed": failed, "reused": not run_gate and prev_gate is not None,
               "checked": bool(gate), "subset": HOOK_GATE, "n_record": int(len(rec_dates)), "stored": stored_npz,
               "pixel_rule": "item (the stored sensitivity product is ITEM's)",
               "bitwise": gate_bitwise if run_gate else (prev_gate or {}).get("bitwise"),
               "max_abs_score_diff": gates.get("marea_record_vs_nidem_sensitivity", {}).get("max_abs_score_diff",
                                                                                           (prev_gate or {}).get(
                                                                                               "max_abs_score_diff")),
               "audit_differing_px": {k: a["differing_px"] for k, a in aud.items()}}
    if "common" in rules:
        summary["common_mask_check"] = gates["common_mask_equals_extraction_C_and_Y_at_keep"]["ok"]
    for k, v in gates.items():
        log(f"[{site}] hook gate {k}: {'OK' if v.get('ok', True) else 'FAILED'}"
            + (f" (bitwise {v.get('bitwise')}, max |score diff| {v.get('max_abs_score_diff')})"
               if "bitwise" in v else ""))
    if not ok:
        if not soft_gates:
            detail = {k: {kk: vv for kk, vv in gates[k].items() if kk in ("checks", "tolerance", "reason", "mine")}
                      for k in failed}
            raise HookGateError(f"[{site}] NIDEM hook gate(s) {failed} failed: {detail}")
        log(f"[{site}] WARNING: hook gate(s) {failed} failed and soft gates are set: continuing")
    if run_gate and ok:
        _HOOK_GATES[gkey] = summary

    # phase 2: NIDEM of every given subset; the phase-1 R codes are shared out by the subsets' scenes
    tim = p1["timing"]
    n_coded = sum(int(x["N"]) for x in subsets if int(x.get("slot", -1)) >= 0)
    gate_ti = ti_all[subsets[k_gate]["idx"]] if k_gate is not None else None
    out = {}
    for k, (key, d, ti, _) in enumerate(resolved):
        s = subsets[k]
        rule = rule_of[str(key)]
        t0 = time.perf_counter()
        # MAREA's record under ITEM's rule: the gate reproduced the stored product bitwise (never a common subset)
        same = rule == "item" and gate_bitwise and gate_ti is not None and np.array_equal(ti, gate_ti)
        if same:
            # MAREA's record: the gate reproduced the stored product bitwise; its values are this subset's
            with np.load(stored_npz, allow_pickle=False) as st:
                z = np.asarray(st["filtered_keep"])[units_keep].astype(float)
            row = gate_row
            t_est = float(gates["marea_record_vs_nidem_sensitivity"].get("stage2_seconds", 0.0))
            values = "the stored sensitivity_marea_record product, reproduced bitwise by the gate"
        else:
            row, res = stage2(ctx, k)
            t_est = time.perf_counter() - t0
            z = None if res is None else res["filtered"].reshape(-1)[units_flat].astype(float)
            values = "computed (phase 2 on the full frame)"
            del res
        # no R codes without an assignment; else the subset's share by its number of scenes
        share = (float(tim.get("codes_s", 0.0)) * int(s["N"]) / max(1, n_coded)
                 if int(s.get("slot", -1)) >= 0 else 0.0)
        reason = "" if z is not None else f"{row['status']}: {row['reason']}"
        if rule == "common":
            reason = reason.replace("(ITEM rule)", "(common mask)")
        r = {"status": "ok" if z is not None else "NOT COMPUTABLE",
             "reason": reason,
             "nidem_status": row["status"], "n_scenes": int(s["N"]), "first_date": str(d[0]),
             "last_date": str(d[-1]), "scenes_sha1": v14.draw_sha1(ti), "sizes": [int(v) for v in s["sizes"]],
             "levels_from_pool": int(pool_all[s["idx"]].sum()),
             "levels_from_boundary": int((~pool_all[s["idx"]]).sum()),
             "keep_valid_share": row["keep_valid_share"], "median_valid_obs": row["median_valid_obs"],
             "n_vertices": row["n_vertices"], "values": values,
             "score": ({kk: row[kk] for kk in ("rmse", "bias", "slope", "r", "n", "coverage")} if truth_given
                       else None),
             "seconds_stage2": round(float(t_est), 3), "seconds_codes_share": round(share, 3),
             "seconds_estimator": round(float(t_est) + share, 3),
             "phase1": {kk: tim.get(kk) for kk in ("read_s", "codes_s", "gate_composites_s", "audit_s", "total_s",
                                                   "io_lock_wait_s", "from_cache")},
             "phase1_scenes": int(len(ti_u)), "phase1_subsets": int(len(subsets)), "phase1_rules": list(rules),
             "gate": summary, "mask": rule, "pixel_rule": PIXEL_RULE_TEXT[rule],
             "product": "filtered: R in 1..8 and conf <= 0.25 (wc.nidem on the full frame)",
             "function": "experiments.v15_nidem_scene_curves.nidem_on_scenes"}
        if z is not None:
            r["z_units"] = z
        out[key] = r
        log(f"[{site}] hook {key} ({rule}): {r['status']}" + (f" ({r['reason'][:160]})" if z is None else
                                                             f", {int(np.isfinite(z).sum()):,} of {len(z):,} units")
            + f" | {r['seconds_estimator']:.1f} s")
    log(f"[{site}] NIDEM hook: {len(out)} subset(s) in {time.perf_counter() - t_start:.1f} s")
    return out


def nidem_on_scenes(site, dates=None, rows=None, h=None, units_keep=None, subsets=None, truth=None, kind=None,
                    inputs=None, io_lock=None, cache_dir=None, row_block=256, scene_batch=32, gate=True,
                    soft_gates=False, pixel_rule="item"):
    """NIDEM (v15's own production path) on GIVEN scene subsets of a site,
    scored on the caller's units: the hook of v19 (fixed-configuration
    tables) and v16 (window curves).

    The path is :func:`run`'s, unchanged: phase 1 reads the cube ONCE for
    every subset in chunk-aligned blocks of rows (``io_lock``: the lock file
    shared by processes reading cubes on one disk), with v7's reading and
    the ITEM pixel rule (:class:`CubeReader`), the exact sort-free R codes,
    the pre-flight counts and the masked NDWI at the scored pixels; phase 2
    per subset applies the pre-flight rule (wc.MIN_VALID_SHARE), puts the
    confidence at the scored pixels, and runs wc.nidem on the full frame;
    the value is the 'filtered' product (R in 1..8, conf <= 0.25). Levels:
    h = the site's boundary level at the scene's overpass time, no lag
    (v7.build_record's level for a scene of its pool, the same boundary
    object otherwise); ``h`` given by the caller must equal it (1e-9 m).

    Pixel rule (``pixel_rule``, per call; a dict spec may set its own
    'pixel_rule'): 'item' (the default: ITEM's pixel-quality mask, SCL in {2,
    4, 5, 6, 7, 11}, v15's and v7's production rule; the path and values above,
    unchanged) or 'common' (the fixed configuration's mask of every method,
    SCL in {4, 5, 6, 7} = the extraction's C, on the same raw NDWI, the whole
    frame included). ONE read of the cube serves both rules
    (:class:`MaskedCubeReader`); with a common subset every scene is checked
    at every keep pixel against the extraction's C and Y (never soft). The
    gate is always ITEM's, and only an ITEM subset of MAREA's record reuses
    the stored product.

    Gate (``gate``, default on): MAREA's screened record (97 / 68 / 71 / 54
    scenes) is added to the phase-1 read and must reproduce the stored
    sensitivity_marea_record product as :func:`run`'s gate does
    (:func:`preflight_gate`, :func:`product_gate`; the reference scores are
    the stored product's on the same units with the caller's ``truth``; with
    no truth, the product alone is compared); the fast R of the smallest
    given subset is audited on the whole frame. A failed gate raises
    :class:`HookGateError`, a RuntimeError (``soft_gates``: recorded, the run
    goes on). A passed gate is not repeated in this process for the same
    inputs and units.

    One subset: ``dates`` (YYYY-MM-DD) or ``rows`` (extraction rows, i.e.
    cube time indices), optionally ``h``. Several subsets, one cube read:
    ``subsets`` = {key: spec} (or [(key, spec)]), spec = dict with 'dates',
    'rows' or 'pool_idx' (indices into v7.build_record's 'all' pool) and
    optionally 'h' and 'pixel_rule', or a bare list of dates. ``units_keep``:
    the keep index of every scored unit; ``truth``/``kind`` (optional): the
    truth per unit and v14.score's kind, for the scores and the gate.
    ``inputs``: overrides of extract / product / comparison / nidem / cube;
    ``cache_dir`` (outside the repository): the phase-1 arrays, keyed by
    every input (and the extraction check's temporary rows).

    Returns, per subset, a dict: 'status' ('ok' or 'NOT COMPUTABLE' with a
    'reason': an empty tide interval or the pre-flight stop), 'z_units'
    (float, one value per unit, NaN where NIDEM has none; absent when not
    computable), 'mask' (the rule applied: 'item' or 'common') and
    'pixel_rule' (its text), the scenes, interval sizes, keep valid share,
    the timing ('seconds_estimator' = phase 2 of the subset + its share of
    the phase-1 R codes by number of scenes; the cube read is not counted),
    the score on the units when a truth is given, and the gate summary. One
    subset: that dict; several: {key: dict}."""
    if subsets is None:
        if dates is None and rows is None:
            raise ValueError("give one subset (dates or rows) or several (subsets)")
        specs = [("subset", {"dates": dates, "rows": rows, "h": h, "pixel_rule": pixel_rule})]
    else:
        if dates is not None or rows is not None or h is not None:
            raise ValueError("give either one subset (dates / rows / h) or subsets, not both")
        items = list(subsets.items()) if isinstance(subsets, dict) else list(subsets)
        if not items:
            raise ValueError("subsets is empty")
        specs = [(str(k), v) for k, v in items]
    if units_keep is None:
        raise ValueError("units_keep (the keep index of every scored unit) is needed")
    _hook_rule_of(specs, pixel_rule)                          # refuse an unknown rule before any read
    S = _hook_site(site, inputs)
    out = _hook_run(S, specs, units_keep, truth=truth, kind=kind, io_lock=io_lock, cache_dir=cache_dir,
                    row_block=row_block, scene_batch=scene_batch, gate=gate, soft_gates=soft_gates,
                    pixel_rule=pixel_rule)
    return out["subset"] if subsets is None else out


# ─────────────────────────────────────────────────────────────────────────────
#  plot
# ─────────────────────────────────────────────────────────────────────────────

PAL = v14.PAL
STYLE = {"marea": dict(color=PAL["teal"], marker="s"), "nidem": dict(color=PAL["sand"], marker="o")}
TXT = {
    "es": dict(
        names={"villaviciosa": "Villaviciosa", "escalda": "Westerschelde", "wadden": "Wadden (Vlie)",
               "ems": "Ems-Dollard"},
        x="Escenas usadas, N (escala logarítmica)", y="Error de la cota, RMSE (m)", cov="Cobertura (%)",
        rtk="RTK, {n} puntos", vak="Vaklodingen, {n} píxeles", pool="archivo de {n} escenas",
        band="mediana y banda 10–90 % de los sorteos",
        nc_legend="sorteos de NIDEM no calculables en esa N",
        no_nidem="NIDEM: sin resultados todavía",
        nidem_partial="NIDEM: {st}; {k} de {K} sorteos (MAREA en los mismos)",
        status={"running": "en curso o interrumpido", "crashed": "interrumpido", "stopped_by_time":
                "parado por tiempo", "incomplete": "incompleto", "gate_failed": "parado en una comprobación",
                "gates_only": "solo comprobaciones", None: "sin fichero de ajustes"},
        foot=["Mismos sorteos de N escenas de todo el archivo para los dos métodos, puntuados en el mismo "
              "conjunto: los puntos RTK (Villaviciosa) o los píxeles Vaklodingen (submuestra fija de {n} en los "
              "sitios holandeses) con verdad y con cota en el MAREA publicado; por eso la cobertura de MAREA tiende "
              "al 100 % por construcción.",
              "Error: RMSE centrado en la mediana, sobre la parte del conjunto con cota; un sorteo tiene RMSE con al "
              "menos 20 puntos RTK o 30 píxeles con cota. La mediana y la banda son de esos sorteos y no se dibujan "
              "si son menos de la mitad de los sorteos o si la mediana de la cobertura (franja inferior) no llega "
              "al 50 %.",
              "MAREA: retardos re-estimados en cada sorteo, sin regla de calidad y con su máscara de nubes "
              "(SCL 4–7), como lo publicado. NIDEM: el producto filtrado publicado (R de 1 a 8 y desviación media "
              "del NDWI ≤ 0.25) con la máscara de nubes de ITEM. NIDEM no es calculable si un intervalo de marea "
              "queda vacío o si menos del 1 % de los píxeles tiene observación válida en los 9 intervalos (paradas "
              "de este estudio, no reglas del NIDEM publicado); esos sorteos cuentan con cobertura 0."],
        reps="Sorteos por N: ", reps_one="{r} ({rl} por encima de N = 100) en {sites}",
        reps_more="{r} ({rl}) en {sites}", conj=" y ",
        ems="Ems-Dollard: con todas las escenas NIDEM no es calculable (la única escena del intervalo más bajo "
            "está nublada); los sorteos sin esa escena pueden serlo.",
        thousands=" ", decimal="."),
    "en": dict(
        names={"villaviciosa": "Villaviciosa", "escalda": "Westerschelde", "wadden": "Wadden Sea (Vlie)",
               "ems": "Ems-Dollard"},
        x="Scenes used, N (log scale)", y="Elevation error, RMSE (m)", cov="Coverage (%)",
        rtk="RTK, {n} points", vak="Vaklodingen, {n} pixels", pool="archive of {n} scenes",
        band="median and 10–90 % band of the draws",
        nc_legend="NIDEM draws not computable at that N",
        no_nidem="NIDEM: no results yet",
        nidem_partial="NIDEM: {st}; {k} of {K} draws (MAREA on the same)",
        status={"running": "running or interrupted", "crashed": "interrupted", "stopped_by_time": "stopped by time",
                "incomplete": "incomplete", "gate_failed": "stopped at a gate", "gates_only": "gates only",
                None: "no settings file"},
        foot=["The same random draws of N scenes from the whole archive for both methods, scored on the same set: "
              "the RTK points (Villaviciosa) or the Vaklodingen pixels (a fixed {n}-pixel subsample at the Dutch "
              "sites) with truth and a value in the published MAREA product, so MAREA's coverage tends to 100 % by "
              "construction.",
              "Error: median-centred RMSE over the part of the set with an elevation; a draw has an RMSE with at "
              "least 20 RTK points or 30 pixels with an elevation. The median and band are over those draws and "
              "are not drawn when they are under half of the draws or when the median coverage (strip below) is "
              "under 50 %.",
              "MAREA: lags re-fitted on every draw, no quality rule, its own cloud mask (SCL 4–7), as published. "
              "NIDEM: the published filtered product (R 1–8 and mean NDWI SD ≤ 0.25) with ITEM's cloud mask. "
              "NIDEM is not computable when a tide interval is empty or under 1 % of the pixels has a valid "
              "observation in all 9 intervals (this study's stops, not rules of published NIDEM); such draws count "
              "as coverage 0."],
        reps="Draws per N: ", reps_one="{r} ({rl} above N = 100) at {sites}", reps_more="{r} ({rl}) at {sites}",
        conj=" and ",
        ems="Ems-Dollard: NIDEM is not computable with all scenes (the only scene of the lowest interval is "
            "cloudy); draws without that scene can be.",
        thousands=",", decimal="."),
}


def _fmt_int(n, lang):
    return f"{int(n):,}".replace(",", TXT[lang]["thousands"])


def _pick_marea_csv(site, marea_dirs, want):
    """The MAREA refit CSV of ``site`` among ``marea_dirs``: the one whose
    sha256 is ``want`` (the CSV the NIDEM run was gated against); without
    ``want``, the server copy of 2026-10-20 if present (MAREA_SHA256), else
    the first found. Returns (path or None, note)."""
    cands = [os.path.join(d, f"{site}_refit.csv") for d in marea_dirs
             if os.path.exists(os.path.join(d, f"{site}_refit.csv"))]
    if not cands:
        return None, "no MAREA refit CSV"
    shas = {p: sha256_file(p) for p in cands}
    if want:
        hit = next((p for p in cands if shas[p] == want), None)
        if hit is None:
            raise SystemExit(f"[{site}] none of {cands} is the MAREA CSV the NIDEM run was gated against "
                             f"(sha256 {want[:16]}...): refusing to plot the two curves together")
        return hit, ""
    known = MAREA_SHA256.get(site, ("",))[0]
    hit = next((p for p in cands if shas[p] == known), None)
    if hit is not None:
        return hit, ""
    return cands[0], f"{cands[0]} is not the server copy of 2026-10-20 (sha256 differs)"


def plot_inputs(site, marea_dirs, nidem_dirs):
    """MAREA's and NIDEM's curves of one site (v14.summarize) on the same
    draws, and the counts of not-computable NIDEM draws per N.

    The MAREA CSV must be the one the NIDEM run was gated against (its
    sha256 in the NIDEM settings); every NIDEM draw must be MAREA's draw
    (draw_sha1); MAREA is summarised on the draws NIDEM has (all of them
    once the NIDEM run is complete), so both medians rest on the same
    draws."""
    import pandas as pd

    D = {"site": site, "marea": None, "nidem": None, "nc": {}, "msettings": {}, "nsettings": {}, "notes": []}
    nd = find_inputs(site, nidem_dirs, ("_nidem_all.csv",))
    if nd is not None:
        js = os.path.join(nd, f"{site}_nidem_all_settings.json")
        if os.path.exists(js):
            D["nsettings"] = json.load(open(js, encoding="utf-8"))
    want = (D["nsettings"].get("compat") or {}).get("marea_csv_sha256")
    mpath, note = _pick_marea_csv(site, marea_dirs, want)
    if note:
        D["notes"].append(note)
    if mpath is None:
        return D
    m = marea_rows(mpath)
    D["marea_src"] = mpath
    D["pool_size"] = int(m["pool_size"].max())
    D["n_draws_marea"] = int(len(m))
    js = os.path.join(os.path.dirname(mpath), f"{site}_refit_settings.json")
    if os.path.exists(js):
        D["msettings"] = json.load(open(js, encoding="utf-8"))
    if nd is not None:
        n = pd.read_csv(os.path.join(nd, f"{site}_nidem_all.csv")).drop_duplicates(["N", "replicate"])
        if len(n):
            # the two curves must rest on the same draws
            ms = {(int(r.N), int(r.replicate)): str(r.draw_sha1) for r in m.itertuples()}
            if "draw_sha1" not in n.columns:
                raise SystemExit(f"[{site}] the NIDEM CSV has no draw_sha1: cannot check its draws against MAREA's")
            bad = [(int(r.N), int(r.replicate)) for r in n.itertuples()
                   if ms.get((int(r.N), int(r.replicate))) != str(r.draw_sha1)]
            if bad:
                raise SystemExit(f"[{site}] {len(bad)} NIDEM row(s) whose draw differs from MAREA's "
                                 f"(first {bad[:3]}): refusing to plot the two curves together")
            keys = set(zip(n["N"].astype(int), n["replicate"].astype(int)))
            m = m[[(int(a), int(b)) in keys for a, b in zip(m["N"], m["replicate"])]]
            n["median_clear_obs"] = n["median_valid_obs"]
            D["nidem"] = v14.summarize(n, POOL)
            D["nidem_src"] = os.path.join(nd, f"{site}_nidem_all.csv")
            D["n_draws_nidem"] = int(len(n))
            for N, g in n.groupby("N"):
                D["nc"][int(N)] = (int(g["status"].isin(NOT_COMPUTABLE).sum()), int(len(g)))
    D["marea"] = v14.summarize(m, POOL)
    return D


def _draw_site(ax, axc, D, lang, fs):
    """One site: RMSE against N for both methods (median line and 10-90 %
    band, only at the N where the median exists and the median coverage
    reaches v14.LOW_COVERAGE, i.e. at least half of the set has an
    elevation) and the coverage strip (every N) with the number of
    not-computable NIDEM draws per N."""
    ends = []
    for meth in ("marea", "nidem"):
        cv = D.get(meth)
        if cv is None or cv.empty:
            continue
        st = STYLE[meth]
        xs = cv["N"].to_numpy(float)
        cov = cv["cov"].to_numpy(float)
        # no point (and no line or band through it) where under half of the
        # set has an elevation: such an RMSE rests on a minority of the set
        rm = np.where(cov >= v14.LOW_COVERAGE, cv["rmse"].to_numpy(float), np.nan)
        lo, hi = cv["rmse_lo"].to_numpy(float), cv["rmse_hi"].to_numpy(float)
        okb = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(rm)
        if okb.any():
            ax.fill_between(xs, np.where(okb, lo, 0.0), np.where(okb, hi, 0.0), where=okb, color=st["color"],
                            alpha=0.14, lw=0, zorder=1)
        fin = np.isfinite(rm)
        ax.plot(xs, rm, color=st["color"], lw=fs["lw"], solid_capstyle="round", zorder=3)
        ax.plot(xs[fin], rm[fin], ls="none", marker=st["marker"], ms=fs["ms"], color=st["color"], mec="white",
                mew=fs["mew"], zorder=4)
        okc = np.isfinite(cv["cov_lo"].to_numpy(float)) & np.isfinite(cv["cov_hi"].to_numpy(float))
        if okc.any():
            axc.fill_between(xs[okc], 100 * cv["cov_lo"].to_numpy(float)[okc], 100 * cv["cov_hi"].to_numpy(float)[okc],
                             color=st["color"], alpha=0.14, lw=0)
        axc.plot(xs, 100 * cov, color=st["color"], lw=fs["lw"] * 0.8, marker=st["marker"], ms=fs["ms"] * 0.62,
                 mec="white", mew=0.6, zorder=3)
        if len(xs) and int(xs[-1]) == D.get("pool_size") and np.isfinite(rm[-1]):
            ends.append((meth, xs[-1], rm[-1]))
    # end values (text in ink, the colour stays on the marks)
    if ends:
        ends.sort(key=lambda e: e[2])
        for i, (meth, x, y) in enumerate(ends):
            below = (i == 0 and len(ends) > 1) or len(ends) == 1
            ax.annotate(f"{y:.2f}".replace(".", TXT[lang]["decimal"]), (x, y),
                        xytext=(fs["ms"] * 0.9, -fs["ms"] * 1.1 if below else fs["ms"] * 1.1),
                        textcoords="offset points", fontsize=fs["small"], color=PAL["ink"],
                        ha="left", va="top" if below else "bottom")
    # the number of not-computable NIDEM draws at each N, just above the strip
    for N, (k_nc, k) in sorted(D["nc"].items()):
        if k_nc:
            axc.text(N, 1.04, f"{k_nc}", transform=axc.get_xaxis_transform(), ha="center", va="bottom",
                     fontsize=fs["tiny"], color=PAL["ink"], clip_on=False)


def figure(Ds, lang, fs, figsize, legend_y, foot_y, margins):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    T = TXT[lang]
    by = {D["site"]: D for D in Ds}
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(2, 2, **margins)
    for i, site in enumerate(v14.SITE_ORDER):
        sg = outer[i // 2, i % 2].subgridspec(2, 1, height_ratios=[3.1, 1.0], hspace=fs["strip_gap"])
        ax = fig.add_subplot(sg[0])
        axc = fig.add_subplot(sg[1], sharex=ax)
        for a in (ax, axc):
            v14._style_axes(a)
            a.tick_params(labelsize=fs["tick"])
        D = by.get(site, {"site": site, "marea": None, "nidem": None, "nc": {}, "msettings": {}, "nsettings": {}})
        _draw_site(ax, axc, D, lang, fs)
        nmax = D.get("pool_size") or 465
        v14._log_n_axis(ax, nmax)
        ax.set_xlim(ax.get_xlim()[0], nmax * fs["xpad"])          # room for the end values inside the axes
        plt.setp(ax.get_xticklabels(), visible=False)
        axc.set_ylim(0, 105)
        axc.set_yticks([0, 50, 100])
        ax.set_title(T["names"][site], loc="left", fontsize=fs["title"], fontweight="bold", color=PAL["deep"],
                     pad=fs["title_pad"])
        fx = D["msettings"].get("fixed_set", {})
        n_u = fx.get("n_units_scored")
        sub = (T["rtk"] if site == "villaviciosa" else T["vak"]).format(n=_fmt_int(n_u, lang) if n_u else "?")
        if D.get("pool_size"):
            sub += "; " + T["pool"].format(n=D["pool_size"])
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=fs["small"], color=PAL["slate"], ha="left",
                va="bottom")
        st = D["nsettings"].get("status")
        if D.get("nidem") is None:
            ax.text(0.98, 0.96, T["no_nidem"], transform=ax.transAxes, ha="right", va="top", fontsize=fs["small"],
                    color=PAL["slate"])
        elif st != "complete":
            ax.text(0.98, 0.96, T["nidem_partial"].format(st=T["status"].get(st, st), k=D.get("n_draws_nidem", "?"),
                                                          K=D.get("n_draws_marea", "?")),
                    transform=ax.transAxes, ha="right", va="top", fontsize=fs["small"], color=PAL["slate"])
        if i % 2 == 0:
            ax.set_ylabel(T["y"], fontsize=fs["label"], color=PAL["ink"])
            axc.set_ylabel(T["cov"], fontsize=fs["label"], color=PAL["ink"])
        if i // 2 == 1:
            axc.set_xlabel(T["x"], fontsize=fs["label"], color=PAL["ink"])
        lo_y, hi_y = ax.get_ylim()
        ax.set_ylim(lo_y, hi_y + 0.04 * (hi_y - lo_y))
    h = [Line2D([], [], color=STYLE["marea"]["color"], lw=fs["lw"], marker="s", ms=fs["ms"], mec="white",
                mew=fs["mew"], label="MAREA"),
         Line2D([], [], color=STYLE["nidem"]["color"], lw=fs["lw"], marker="o", ms=fs["ms"], mec="white",
                mew=fs["mew"], label="NIDEM"),
         Patch(facecolor=PAL["slate"], alpha=0.18, lw=0, label=T["band"])]
    if any(v[0] for D in Ds for v in D["nc"].values()):
        h.append(Line2D([], [], ls="none", marker="$3$", ms=fs["ms"] * 1.15, color=PAL["ink"], mew=0,
                        label=T["nc_legend"]))
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, legend_y), ncol=2, frameon=False,
               fontsize=fs["legend"], handlelength=2.2, columnspacing=2.0)
    fig.align_ylabels(fig.axes[0::4] + fig.axes[1::4])
    import textwrap

    n_sub = next((D["msettings"].get("fixed_set", {}).get("n_units_scored") for D in Ds
                  if D["site"] != "villaviciosa" and D["msettings"].get("fixed_set", {}).get("n_units_scored")),
                 100000)
    foot = [t.format(n=_fmt_int(n_sub, lang)) if "{n}" in t else t for t in T["foot"]]
    reps = {}
    for D in Ds:
        p = D["msettings"].get("plan", {})
        if p.get("reps"):
            reps.setdefault((p["reps"], p["reps_large"]), []).append(T["names"][D["site"]])
    if reps:
        def _join(s):
            return s[0] if len(s) == 1 else ", ".join(s[:-1]) + T["conj"] + s[-1]

        foot.append(T["reps"] + "; ".join(T["reps_one" if i == 0 else "reps_more"].format(r=r, rl=rl, sites=_join(s))
                                          for i, ((r, rl), s) in enumerate(reps.items())) + ".")
    if any(D["site"] == "ems" and D.get("nidem") is not None for D in Ds):
        foot.append(T["ems"])
    width = int(figsize[0] * 72 * (margins["right"] - margins["left"]) / (0.53 * fs["foot"]))
    lines = []
    for t in foot:
        lines += textwrap.wrap(t, width)
    fig.text(margins["left"], foot_y, "\n".join(lines), fontsize=fs["foot"], color=PAL["slate"], ha="left",
             va="top", linespacing=1.3)
    return fig


LAYOUT = {
    # co-author note (A4 text width)
    "es": dict(figsize=(7.2, 9.3), legend_y=0.168, foot_y=0.160,
               margins=dict(left=0.095, right=0.985, top=0.955, bottom=0.276, hspace=0.36, wspace=0.2),
               fs=dict(lw=1.7, ms=4.6, mew=0.9, tick=8, label=8.6, title=10, title_pad=13, small=7.4, tiny=6.6,
                       legend=8.2, foot=6.9, strip_gap=0.18, xpad=1.6)),
    # poster card
    "en": dict(figsize=(11.0, 13.6), legend_y=0.160, foot_y=0.152,
               margins=dict(left=0.085, right=0.985, top=0.958, bottom=0.264, hspace=0.34, wspace=0.17),
               fs=dict(lw=2.4, ms=7.0, mew=1.2, tick=12, label=13, title=15.5, title_pad=19, small=11, tiny=10,
                       legend=12.5, foot=10.2, strip_gap=0.18, xpad=1.7)),
}


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    marea_dirs = [args.marea_dir] if args.marea_dir else list(MAREA_DIRS)
    nidem_dirs = [args.nidem_dir] if args.nidem_dir else list(NIDEM_DIRS)
    Ds = [plot_inputs(s, marea_dirs, nidem_dirs) for s in v14.SITE_ORDER]
    for D in Ds:
        log(f"{D['site']}: MAREA {D.get('marea_src', 'missing')}; NIDEM {D.get('nidem_src', 'no results yet')}"
            + (f" (run {D['nsettings'].get('status')}, {D.get('n_draws_nidem')} of {D.get('n_draws_marea')} draws)"
               if D.get("nidem") is not None else "")
            + (f"; not computable per N {D['nc']}" if D["nc"] else "")
            + (f"; NOTE {D['notes']}" if D["notes"] else ""))
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.titlepad": 6})
    for lang in [x.strip() for x in args.lang.split(",") if x.strip()]:
        stem = {"es": args.out_es, "en": args.out_en}[lang]
        d = os.path.dirname(stem)
        v14.guard_out(d or ".")
        os.makedirs(d, exist_ok=True)
        lay = LAYOUT[lang]
        fig = figure(Ds, lang, lay["fs"], lay["figsize"], lay["legend_y"], lay["foot_y"], lay["margins"])
        fig.savefig(stem + ".png", dpi=args.dpi)
        fig.savefig(stem + ".pdf")
        plt.close(fig)
        log(f"-> {stem}.png/.pdf")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    ap = argparse.ArgumentParser(prog="python -m experiments.v15_nidem_scene_curves",
                                 description="NIDEM elevation error against the number of scenes used, on MAREA's "
                                             "subsets (v14), and the MAREA-vs-NIDEM figure.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the NIDEM curve of one site (resumable)")
    r.add_argument("site", choices=list(v14.SITE_NAMES))
    r.add_argument("--max-minutes", type=float, default=90.0, help="wall-time cap of this process (phase 2 stops "
                                                                   "before a job predicted to pass it)")
    r.add_argument("--workers", type=int, default=1, help="phase-2 processes (fork; Linux only)")
    r.add_argument("--row-block", type=int, default=256, help="rows per block (a multiple of the cube's chunk rows, "
                                                              "256, reads every chunk once)")
    r.add_argument("--scene-batch", type=int, default=32, help="scenes per isel call (1 = v7's call)")
    r.add_argument("--io-lock", default=None, help="lock file shared by processes reading cubes on one disk")
    r.add_argument("--cache-dir", default=None, help="folder OUTSIDE the repository for the phase-1 arrays")
    r.add_argument("--marea-dir", default=None, help=f"folder of v14's <site>_refit.csv/_settings.json "
                                                     f"(default: first of {list(MAREA_DIRS)})")
    r.add_argument("--out", default=OUT)
    r.add_argument("--max-jobs", type=int, default=None, help="only the first K curve jobs (smoke runs)")
    r.add_argument("--only-gates", action="store_true", help="stop after the product gates")
    r.add_argument("--soft-gates", action="store_true",
                   help="a failed gate is recorded instead of stopping the run (only after reading why)")
    r.add_argument("--product-dir", dest="product_override", default=None,
                   help="folder of the production MAREA result.json/marea.npz (e.g. a laptop mirror)")
    r.add_argument("--extract", dest="extract_override", default=None)
    r.add_argument("--comparison-dir", dest="comparison_override", default=None)
    r.add_argument("--nidem-dir", dest="nidem_override", default=None)
    r.add_argument("--cube", dest="cube_override", default=None)
    p = sub.add_parser("plot", help="the two figure versions (Spanish note, English poster)")
    p.add_argument("--marea-dir", default=None, help=f"default: first of {list(MAREA_DIRS)} holding the site")
    p.add_argument("--nidem-dir", default=None, help=f"default: first of {list(NIDEM_DIRS)} holding the site")
    p.add_argument("--lang", default="es,en")
    p.add_argument("--out-es", default=FIG_ES)
    p.add_argument("--out-en", default=FIG_EN)
    p.add_argument("--dpi", type=int, default=220)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        run(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
