# -*- coding: utf-8 -*-
"""Validation v2, Stage L — MAREA's lag family: L01 (m2a band lags with its
bootstrap), L01v (m2a per relative orbit), L02 (m2b concordance) and L03
(m2c flooded-area rank correlation).

Question
--------
Which band lag does MAREA's own estimator, and each of its in-house
siblings, give on the instance every lag method of the benchmark receives
(the same record, domain, bands and exact boundary levels), and does the
benchmark's re-sequenced driver give exactly what the production code gives
on the same matrices?

Rules (docs/validation_v2_spec.md §3.3, §3.3.1, §4.2, §6.6; v13 §3.2;
FROZEN['marea'])
-------------------------------------------------------------------------
* The estimators are called unchanged (spec §0.5: no change to any
  estimator in ``pyintertidal/``); this module only re-sequences the reads:
  it replays each estimator's own column draw and reads ONLY the drawn
  columns, then calls the estimator on those columns, where the draw is a
  no-op. The tests prove equality with the full-matrix calls, bit for bit.
* L01 = ``te.m2a_rasch`` exactly as ``marea.reconstruct`` calls it: the
  configs/m2.yaml values (sigma0 0.20, sigma atoms 0.06..0.45, 60 z points,
  1,200 columns per band), ``marea.TAU_BOUNDS`` (-45, 180), ``monotone=True``,
  ``fit_alpha=False``; columns drawn per band, in band order, with
  ``default_rng(20260819 + 31 + 1000 * seed_offset)`` (the 'seed<k>' rows of
  the base design change only this draw); bank = the exact
  ``marea.BANK_TAUS`` rows of the level table (``inst.levels.bank()``),
  which are what reconstruct reads from its boundary provider.
* Record floor 40 scenes (v13 D4, FROZEN['marea']['floor']): fewer raise
  ``NotComputable('record_floor')``; a domain with fewer than 3,000 keep
  pixels raises ``NotComputable('no_intertidal')``; a domain without a
  mouth anchor (status 'no_anchor') has one band with tau = 0
  (reconstruct's no-anchor path). A domain with one band and an anchor
  gets tau = 0 too: m2a pins the mouth band and has no free parameter
  (``scipy.optimize.minimize`` refuses the empty problem, so reconstruct
  would stop; declared in the card).
* Causality clip (reconstruct): ``tau_used = where(|tau| > TAU_DETECT_MIN,
  tau, 0)`` then negative bands -> 0. The applied lag of a pixel is
  ``te.smooth_lag_profile(centers, tau_used, s_km, 3 km)`` rounded to the
  minute, 0 where s is NaN (``tau_px``).
* Bootstrap (spec §3.3.1: 30 draws, Stage L at P0, its nominal 95 %
  interval feeds the interval score): the draws of ``te.bootstrap_lags``
  (``default_rng(20260819).integers(0, T, T)`` per draw; refit with
  ``default_rng(20260819 + 100 + b)`` and 600 columns per band), run on an
  explicit pool of ``boot_threads`` threads instead of joblib's
  ``cpu_count - 2`` (the draws are independent: the result does not depend
  on the pool); lo/hi = ``np.nanpercentile`` 2.5 / 97.5 per band.
* L01v (spec §3.3.1, §6.4 'truth-free reproducibility'): m2a on each
  relative orbit's scenes (``inst.orbit``; -1 unknown, left out), on the
  instance's domain and bands, with its own column draw
  ``default_rng([column_seed, SITE_ID, orbit])``; a group with fewer than
  40 scenes is 'not_computable' inside ``groups``. The groups are named
  ``orbit<k>`` with the scene metadata sidecar and ``phase5-<k>`` with the
  date-phase fallback (``instance.scene_orbits``); the sidecar's sha256 and
  the orbit source enter L01v's inputs key (:func:`external_inputs`).
* L02 = ``te.m2b_concordance`` with the common scan window TAU_SCAN
  (-45..180 step 5, spec §3.3) on the exact shift rows
  ``te.ShiftBank(TAU_SCAN, inst.levels.stack(TAU_SCAN))``, min_obs 10, the
  estimator's own default column rng (``default_rng(2)``, 1,500 columns per
  band; bands under 50 pixels are skipped before any draw).
* L03 = ``te.m2c_waterline`` with TAU_SCAN, min_cover 0.3, min_scenes 60,
  computed band by band (m2c treats bands independently): one band's
  columns are read at a time.
* Every band method also returns its field (``common.broadcast_band``) and
  per-band edge flags; nothing is tuned and no truth is read here.

``production_compat=True`` is for the reproduction gates only (G-marea):
the clear mask is the pixel cloud rule alone (production's
``marea_extract`` 'C', so a clear pixel whose NDWI is NaN enters as a dry
observation with Y = 0) and wet = C & (Y > 0) with Y = nan_to_num(NDWI),
exactly reconstruct's ``wet = (C > 0) & (Y > 0)``.

Imports numpy, the harness core and pyintertidal; no import-time side
effects (configs/m2.yaml is read and checked against FROZEN at the first
fit).
"""
from __future__ import annotations

import concurrent.futures as _cf
import contextlib
import os
from collections.abc import Mapping

import numpy as np

from experiments.benchmark import config, obs, settings
from experiments.benchmark.lags import common
from experiments.benchmark.product import NotComputable
from pyintertidal import marea
from pyintertidal import tide_estimators as te

_M = config.FROZEN["marea"]
#: record floor of MAREA (v13 D4), also the per-orbit floor of L01v
FLOOR = int(_M["floor"])
#: m2a's column seed in reconstruct: configs/m2.yaml seed + 31
COLUMN_SEED = int(_M["column_seed"])
#: the seed offset of a 'seed<k>' row multiplies this (plan INTERFACES §8)
SEED_OFFSET_STEP = 1000
#: smoothing bandwidth of the applied lag = the band length (reconstruct)
SMOOTH_KM = float(_M["smooth_km"])
#: m2a's frozen fit (configs/m2.yaml, checked at the first fit)
SIGMA0 = float(_M["m2_yaml"]["sigma0_m"])
SG_GRID = tuple(float(x) for x in _M["m2_yaml"]["sigma_perfil_m"])
Z_POINTS = int(_M["m2_yaml"]["z_puntos"])
MAX_PX_BAND = int(_M["m2_yaml"]["max_px_banda_m2a"])
TAU_BOUNDS = tuple(float(x) for x in _M["tau_bounds"])
MONOTONE = bool(_M["monotone"])
#: the bootstrap of te.bootstrap_lags as reconstruct calls it
N_BOOT_FROZEN = int(_M["bootstrap"]["n_boot"])
BOOT_MAX_PX = int(_M["bootstrap"]["max_px_band"])
BOOT_SEED = int(_M["bootstrap"]["seed"])
#: per-band identification diagnostics of v13 §3.2 (descriptive)
USABLE_MIN_CLEAR = int(_M["usable_min_clear"])
IDENTIFIED_MIN_USABLE = int(_M["identified_min_usable"])
#: reconstruct's detection threshold (0: every band lag applied as measured)
TAU_DETECT_MIN = float(marea.TAU_DETECT_MIN)
#: v13 §3.2 'bound_hit': a band lag or an increment at the 3-h bound
EDGE_BOUND_MIN = 179.5
#: m2b (te.m2b_concordance defaults, configs/m2.yaml max_px_banda.m2b)
M2B_MIN_OBS = 10
M2B_MAX_PX = 1500
M2B_RNG_SEED = 2                 # te.m2b_concordance: rng or default_rng(2)
M2B_MIN_BAND_PX = 50             # te.m2b_concordance skips smaller bands before drawing
#: m2c (te.m2c_waterline defaults; configs/m2.yaml m2c_min_cobertura)
M2C_MIN_COVER = 0.3
M2C_MIN_SCENES = 60
M2C_MIN_BAND_PX = 50
#: the published (as-built) grid of the phase estimators (configs/m2.yaml
#: tau_malla_min), an unscheduled 'asbuilt' variant
ASBUILT_GRID = (-20.0, 100.0, 10.0)
#: observation cells read per block (memory only; never changes a result)
READ_CELLS = 20_000_000

if TAU_BOUNDS != tuple(float(x) for x in marea.TAU_BOUNDS):
    raise AssertionError(f"FROZEN marea tau_bounds {TAU_BOUNDS} differ from marea.TAU_BOUNDS "
                         f"{marea.TAU_BOUNDS}")
del _M

CODE_DEPS = ("pyintertidal/tide_estimators.py", "pyintertidal/marea.py",
             "pyintertidal/elevation.py", "configs/m2.yaml")
M2_YAML = os.path.join(config.ROOT, "configs", "m2.yaml")


# ─────────────────────────────────────────────────────────────────────────────
#  small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(obj, key, default=None):
    """``obj[key]`` for a mapping, ``obj.key`` otherwise (units and opts
    may be dicts or namespaces)."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _stage(inst, name):
    fn = getattr(inst, "stage", None)
    return fn(name) if callable(fn) else contextlib.nullcontext()


def _progress(inst, n=1):
    fn = getattr(inst, "progress", None)
    if callable(fn):
        fn(n)


def _cache(inst):
    c = getattr(inst, "cache", None)
    if c is None:
        c = {}
        try:
            inst.cache = c
        except AttributeError:
            pass
    return c


def fit_kw(max_px_band=None):
    """The keywords of ``te.m2a_rasch`` as ``marea.reconstruct`` passes
    them (``fit_alpha`` False, its default, passed explicitly)."""
    return dict(sigma0=SIGMA0, sg_grid=tuple(SG_GRID), tau_bounds=tuple(TAU_BOUNDS),
                z_points=int(Z_POINTS),
                max_px_band=int(MAX_PX_BAND if max_px_band is None else max_px_band),
                monotone=bool(MONOTONE), fit_alpha=False)


_YAML_OK = {}


def check_m2_yaml(path=None):
    """configs/m2.yaml must hold the FROZEN values reconstruct reads (sigma0,
    sigma atoms, z points, m2a columns, seed); raises RuntimeError
    otherwise. Read once per process."""
    path = path or M2_YAML
    if path in _YAML_OK:
        return _YAML_OK[path]
    import yaml
    with open(path, encoding="utf-8") as f:
        y = yaml.safe_load(f)
    want = config.FROZEN["marea"]["m2_yaml"]
    have = {"sigma0_m": float(y["sigma0_m"]),
            "sigma_perfil_m": [float(v) for v in y["sigma_perfil_m"]],
            "z_puntos": int(y["z_puntos"]), "max_px_banda_m2a": int(y["max_px_banda"]["m2a"]),
            "seed": int(y["seed"])}
    exp = {"sigma0_m": float(want["sigma0_m"]),
           "sigma_perfil_m": [float(v) for v in want["sigma_perfil_m"]],
           "z_puntos": int(want["z_puntos"]), "max_px_banda_m2a": int(want["max_px_banda_m2a"]),
           "seed": int(want["seed"])}
    if have != exp:
        raise RuntimeError(f"configs/m2.yaml {have} differs from FROZEN['marea']['m2_yaml'] {exp}: "
                           "the frozen MAREA fit would not be production's")
    if int(y["seed"]) + 31 != COLUMN_SEED:
        raise RuntimeError("FROZEN column_seed is not configs/m2.yaml seed + 31")
    _YAML_OK[path] = have
    return have


def draw_columns(band_of, K, max_px, rng, skip_below=0):
    """Replay of an estimator's per-band column draw: for k = 0..K-1, the
    band's columns (ascending) and, above ``max_px``, ``np.sort(rng.choice(
    cols, max_px, replace=False))`` — m2a's and m2b's loops line for line
    (``skip_below``: m2b skips bands under 50 pixels BEFORE drawing, m2a
    draws for every band). Returns a list of K int64 arrays."""
    band_of = np.asarray(band_of)
    out = []
    for k in range(int(K)):
        cols = np.where(band_of == k)[0]
        if len(cols) < skip_below:
            out.append(np.zeros(0, np.int64))
            continue
        if len(cols) > max_px:
            cols = np.sort(rng.choice(cols, max_px, replace=False))
        out.append(np.asarray(cols, np.int64))
    return out


def _union(*lists):
    parts = [np.asarray(c, np.int64) for lst in lists for c in lst if len(c)]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)


def pixel_cloud(inst, cols):
    """The pixel cloud rule alone of the instance's setting at keep
    positions ``cols`` (production's clear mask; gates only): the stored SCL
    through ``obs.pixel_valid``. Rules that need CLD or the OmniCloudMask
    bits are refused (no production run used them)."""
    s = settings.Setting.parse(inst.setting)
    rule = s.pixel_rule
    if rule in ("C3", "C5", "C11"):
        raise ValueError(f"production_compat reads the SCL rule only, not {rule}")
    if s.prep != "raw" or s.index != "ndwi":
        raise ValueError(f"production_compat reproduces production's raw NDWI; the setting "
                         f"{s.id} has prep {s.prep!r} and index {s.index!r}")
    pkg = getattr(inst, "pkg", None)
    if pkg is None:
        raise ValueError("production_compat needs inst.pkg (the package's stored SCL)")
    pos = np.asarray(inst.domain.pos, np.int64)[np.asarray(cols, np.int64)]
    scl = pkg.read("SCL", np.asarray(inst.sel, np.int64), pos)["SCL"]
    return obs.pixel_valid(rule, scl)


def read_columns(inst, cols, what=("wet", "clear"), production_compat=False,
                 read_cells=None):
    """Observation arrays of keep positions ``cols`` (sorted int64) over the
    instance's record, read in column blocks so only the requested arrays
    stay in memory.

    ``what`` among "wet", "clear" (bool (T, m)) and "Y" (float32, NaN -> 0,
    the inversion's input). Normal mode: wet = W, clear = V (the setting's
    observation, spec §2.5). ``production_compat``: clear = the pixel cloud
    rule alone, Y = nan_to_num(Y), wet = clear & (Y > 0) (reconstruct).
    """
    cols = np.asarray(cols, np.int64)
    T = int(len(inst.sel))
    m = int(len(cols))
    out = {}
    if "wet" in what:
        out["wet"] = np.zeros((T, m), bool)
    if "clear" in what:
        out["clear"] = np.zeros((T, m), bool)
    if "Y" in what:
        out["Y"] = np.zeros((T, m), np.float32)
    if m == 0:
        return out
    cells = int(read_cells or READ_CELLS)
    step = max(1, cells // max(T, 1))
    for a in range(0, m, step):
        b = min(a + step, m)
        o = inst.obs.read(cols[a:b])
        if production_compat:
            C = np.asarray(pixel_cloud(inst, cols[a:b]), bool)
            Y = np.nan_to_num(np.asarray(o.Y, np.float32), nan=0.0)
            if "wet" in out:
                out["wet"][:, a:b] = C & (Y > 0)
            if "clear" in out:
                out["clear"][:, a:b] = C
            if "Y" in out:
                out["Y"][:, a:b] = Y
        else:
            if "wet" in out:
                out["wet"][:, a:b] = np.asarray(o.W, bool)
            if "clear" in out:
                out["clear"][:, a:b] = np.asarray(o.V, bool)
            if "Y" in out:
                out["Y"][:, a:b] = np.nan_to_num(np.asarray(o.Y, np.float32), nan=0.0)
        del o
    return out


def _domain(inst):
    """The instance's domain; 'no_intertidal' raises NotComputable."""
    d = inst.domain
    if d.status == "no_intertidal":
        raise NotComputable("no_intertidal",
                            f"{len(d.keep)} keep pixels: fewer than the domain's minimum",
                            {"n_keep": int(len(d.keep))})
    return d


def _floor(inst, floor=None, what="MAREA"):
    T = int(len(inst.sel))
    f = FLOOR if floor is None else int(floor)
    if T < f:
        raise NotComputable("record_floor", f"{T} scenes < {what}'s floor of {f}",
                            {"T": T, "floor": f})
    return T


def band_counts(band_of, K):
    """Keep pixels per band (band -1 left out)."""
    b = np.asarray(band_of, np.int64)
    return np.bincount(b[b >= 0], minlength=int(K))[:int(K)].astype(np.int64)


def band_mean_s(s_km, band_of, K):
    """The mean s of each band's pixels (finite s only; NaN for an empty
    band): the band's representative s of E0g (levels.GAUGE_LAG_RULE)."""
    s = np.asarray(s_km, np.float64)
    b = np.asarray(band_of, np.int64)
    out = np.full(int(K), np.nan)
    for k in range(int(K)):
        m = (b == k) & np.isfinite(s)
        if m.any():
            out[k] = float(np.mean(s[m]))
    return out


def scan_edge(tau):
    """Edge flag of a scan estimate: on the common window's bound."""
    t = np.asarray(tau, np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(t) & ((np.abs(t - common.SCAN_LO) <= common.EDGE_TOL_MIN)
                                 | (np.abs(t - common.SCAN_HI) <= common.EDGE_TOL_MIN))


def m2a_edge(tau_raw):
    """v13 §3.2 'bound_hit' per band: the band lag or its increment over the
    previous band at the 3-h bound of m2a's monotone increments."""
    t = np.asarray(tau_raw, np.float64)
    edge = np.zeros(len(t), bool)
    if len(t) > 1:
        inc = np.diff(t)
        edge[1:] = (t[1:] >= EDGE_BOUND_MIN) | (inc >= EDGE_BOUND_MIN)
    return edge


def causality(tau):
    """reconstruct's applied band lags: |tau| > TAU_DETECT_MIN, then
    negative bands clipped to 0 (returns tau_used, veto)."""
    tau = np.asarray(tau, float)
    tau_used = np.where(np.abs(tau) > TAU_DETECT_MIN, tau, 0.0)
    neg = tau_used < 0
    return np.where(neg, 0.0, tau_used), bool(neg.any())


def applied_lag(centers, tau_used, s_km):
    """reconstruct's per-pixel lag: the smoothed profile (more than one band)
    rounded to the minute, 0 where s is NaN; float64."""
    centers = np.asarray(centers, float)
    if len(centers) > 1:
        tau_px = te.smooth_lag_profile(centers, np.asarray(tau_used, float),
                                       np.asarray(s_km, float), bandwidth_km=SMOOTH_KM)
    else:
        tau_px = np.zeros(len(np.asarray(s_km)))
    return np.where(np.isfinite(tau_px), np.round(tau_px), 0.0)


def _sorted_cols(per_band):
    """The drawn columns of every band, ascending (one estimator call)."""
    if not per_band:
        return np.zeros(0, np.int64)
    return np.sort(np.concatenate([np.asarray(c, np.int64) for c in per_band]))


def _boot_one(b, draws, wet_u, clear_u, u, bank, band_of, centers, K, boot_cols, kw):
    """Draw b of te.bootstrap_lags: scenes ``draws[b]``, bank columns of
    those scenes, m2a refit with ``default_rng(seed + 100 + b)`` on the
    replayed (<= 600 per band) columns."""
    idx = draws[b]
    bk = te.ShiftBank(bank.taus, bank.H[:, idx])
    cb = _sorted_cols(boot_cols[b])
    p = np.searchsorted(u, cb)
    rows = np.ix_(idx, p)                      # = wet[idx][:, cols], without the full copy
    r = te.m2a_rasch(wet_u[rows], clear_u[rows], bk, np.asarray(band_of)[cb],
                     centers, K, rng=np.random.default_rng(BOOT_SEED + 100 + b), **kw)
    return np.asarray(r["tau"], float)


# ─────────────────────────────────────────────────────────────────────────────
#  L01: the MAREA lag stage (shared by E0 and the L01 product)
# ─────────────────────────────────────────────────────────────────────────────

def lag_stage(inst, n_boot=0, boot_threads=1, seed_offset=0, production_compat=False):
    """MAREA's band lags on an instance, exactly as ``marea.reconstruct``.

    Memoised in ``inst.cache['marea_family.lag_stage']`` by (n_boot,
    seed_offset, production_compat), so E0's elevation products and the L01
    lag product share one m2a call per instance.

    Returns
    -------
    dict
        ``tau_raw`` (K,) m2a's band lags, ``tau_used`` (K,) after the
        causality clip, ``tau_px`` (n_keep,) float32 the applied lag,
        ``lo``/``hi`` (K,) bootstrap percentiles or None, ``boot`` (n_boot,
        K) or None, ``centers``, ``band_of``, ``nll``, ``converged``,
        ``flags`` (list), ``edge`` (K,) bool, ``n_px`` (K,), ``veto``,
        ``usable``/``identified`` (K,) of the drawn columns, ``n_cols``
        (K,), ``T``, ``K``, ``status`` (the domain's).
    """
    key = (int(n_boot), int(seed_offset), bool(production_compat))
    memo = _cache(inst).setdefault("marea_family.lag_stage", {})
    if key in memo:
        return memo[key]
    check_m2_yaml()
    T = _floor(inst)
    d = _domain(inst)
    K = int(d.K)
    band_of = np.asarray(d.band_of, np.int64)
    centers = np.asarray(d.centers, np.float64)
    s_km = np.asarray(d.s_km, np.float64)
    flags = []
    nll, converged = float("nan"), None
    lo = hi = boot = None
    n_cols = np.zeros(K, np.int64)
    usable = np.zeros(K, np.int64)
    if d.status == "no_anchor" or K < 2:
        # reconstruct's no-anchor path: one band, tau = 0, no interval;
        # one band WITH an anchor has no free parameter (m2a pins band 0)
        tau = np.zeros(K if K >= 1 else 1)
        flags.append("no_anchor" if d.status == "no_anchor" else "single_band")
        if n_boot:
            lo = np.full(len(tau), np.nan)
            hi = np.full(len(tau), np.nan)
    else:
        kw = fit_kw()
        seed = COLUMN_SEED + SEED_OFFSET_STEP * int(seed_offset)
        main = draw_columns(band_of, K, kw["max_px_band"], np.random.default_rng(seed))
        boot_cols, draws = [], []
        if n_boot:
            rng_b = np.random.default_rng(BOOT_SEED)
            draws = [rng_b.integers(0, T, T) for _ in range(int(n_boot))]
            boot_cols = [draw_columns(band_of, K, BOOT_MAX_PX,
                                      np.random.default_rng(BOOT_SEED + 100 + b))
                         for b in range(int(n_boot))]
        u = _union(main, *boot_cols)
        with _stage(inst, "read"):
            arr = read_columns(inst, u, ("wet", "clear"), production_compat)
            _progress(inst)
        wet_u, clear_u = arr["wet"], arr["clear"]
        bank = inst.levels.bank()
        cm = _sorted_cols(main)
        pm = np.searchsorted(u, cm)
        with _stage(inst, "m2a"):
            r = te.m2a_rasch(wet_u[:, pm], clear_u[:, pm], bank, band_of[cm], centers, K,
                             rng=np.random.default_rng(seed), **kw)
            _progress(inst)
        tau = np.asarray(r["tau"], float)
        nll, converged = float(r["nll"]), bool(r["converged"])
        if not converged:
            flags.append("m2a_not_converged")
        for k, c in enumerate(main):
            n_cols[k] = len(c)
            if len(c):
                usable[k] = int((clear_u[:, np.searchsorted(u, c)].sum(axis=0)
                                 >= USABLE_MIN_CLEAR).sum())
        if n_boot:
            bkw = fit_kw(BOOT_MAX_PX)
            with _stage(inst, "bootstrap"):
                args = (draws, wet_u, clear_u, u, bank, band_of, centers, K, boot_cols, bkw)
                nthr = max(1, int(boot_threads or 1))
                if nthr == 1:
                    taus = []
                    for b in range(int(n_boot)):
                        taus.append(_boot_one(b, *args))
                        _progress(inst)
                else:
                    def one(b):
                        # progress per finished draw (the queue's hang detection,
                        # spec §13.5); the draws and their order are unchanged
                        t = _boot_one(b, *args)
                        _progress(inst)
                        return t

                    with _cf.ThreadPoolExecutor(max_workers=nthr) as ex:
                        taus = list(ex.map(one, range(int(n_boot))))
            boot = np.asarray(taus, float)
            with np.errstate(invalid="ignore"):
                lo = np.nanpercentile(boot, 2.5, axis=0)
                hi = np.nanpercentile(boot, 97.5, axis=0)
        del wet_u, clear_u, arr
    tau_used, veto = causality(tau)
    if veto:
        flags.append("causality_clip")
    with _stage(inst, "smooth"):
        tau_px = applied_lag(centers, tau_used, s_km)
    identified = usable >= IDENTIFIED_MIN_USABLE
    out = {"tau_raw": tau, "tau_used": tau_used, "tau_px": tau_px.astype(np.float32),
           "lo": lo, "hi": hi, "boot": boot, "centers": centers, "band_of": band_of,
           "nll": nll, "converged": converged, "flags": flags, "edge": m2a_edge(tau),
           "n_px": band_counts(band_of, len(tau)), "veto": veto, "n_cols": n_cols,
           "usable": usable, "identified": identified, "T": T, "K": int(len(tau)),
           "status": d.status, "seed_offset": int(seed_offset), "n_boot": int(n_boot),
           "production_compat": bool(production_compat)}
    memo[key] = out
    return out


def l01_result(st):
    """The L01 product (kind 'band', linear) of a lag stage: field = the
    applied smoothed lag, band tau = tau_used (tau_raw in diag), interval95
    at band level when the bootstrap ran."""
    K = int(st["K"])
    iv = None
    if st["n_boot"] and st["lo"] is not None:
        iv = {"level": "band", "lo": st["lo"], "hi": st["hi"],
              "nominal": f"bootstrap{int(st['n_boot'])}"}
    diag = {"tau_raw": np.asarray(st["tau_raw"], float), "nll": st["nll"],
            "converged": st["converged"], "flags": list(st["flags"]),
            "centers_km": np.asarray(st["centers"], float), "n_cols": st["n_cols"],
            "usable": st["usable"], "identified": st["identified"], "veto": st["veto"],
            "domain_status": st["status"], "T": int(st["T"]),
            "seed_offset": int(st["seed_offset"]),
            "production_compat": bool(st["production_compat"])}
    if st["boot"] is not None:
        diag["boot"] = np.asarray(st["boot"], float)
    return common.lag_result(
        "L01", "ok", n_keep=len(st["tau_px"]), K=K, kind="band", lag_kind="linear",
        field=st["tau_px"],
        band={"tau": st["tau_used"], "n_px": st["n_px"], "edge": st["edge"],
              "lo": st["lo"], "hi": st["hi"]},
        interval95=iv, diag=diag)


# ─────────────────────────────────────────────────────────────────────────────
#  L01v: m2a per relative orbit
# ─────────────────────────────────────────────────────────────────────────────

def orbit_seed(inst, orbit):
    """``[column_seed, SITE_ID, orbit]`` of L01v's per-group column draw."""
    return [COLUMN_SEED, int(config.SITE_ID[inst.site]), int(orbit)]


def group_name(orbit, source):
    """The group id of L01v: ``orbit<k>`` for a relative orbit of the scene
    metadata sidecar, ``phase5-<k>`` for the date-phase fallback (the groups
    are then phases of the date, not orbits; spec §6.4)."""
    if str(source or "").startswith("phase"):
        return f"{source}-{int(orbit)}"
    return f"orbit{int(orbit)}"


def _group_order(gid):
    tail = str(gid).rsplit("-", 1)[-1] if "-" in str(gid) else str(gid)[5:]
    try:
        return int(tail)
    except ValueError:
        return 0


def external_inputs(inst, unit):
    """L01v's grouping is an input: the sha256 of the scene metadata sidecar
    (``cubes/<site>/scene_meta.json``; None without it) and the instance's
    ``orbit_source`` ('scene_meta' or the 'phase5' fallback), so a sidecar
    fetched later reruns L01v with the real relative orbits. Nothing for the
    other units of this module."""
    method = unit.get("method") if isinstance(unit, Mapping) else getattr(unit, "method", "")
    if str(method or "") != "L01v":
        return {}
    from experiments.benchmark import inputs, product
    site2 = getattr(inst, "site2", None)
    cubes = getattr(site2, "cubes_root", None)
    if cubes is None:
        cubes = (getattr(inst, "paths", None) or {}).get("cubes_root")
    try:
        p = inputs.scene_meta_path(inst.site, cubes)
        sha = product.sha256_file(str(p)) if p.is_file() else None
    except Exception as e:                       # recorded: never fatal
        sha = f"error: {type(e).__name__}: {e}"
    return {"scene_meta": sha, "orbit_source": getattr(inst, "orbit_source", None)}


def run_l01v(inst, unit, opts):
    """L01v: m2a on each relative orbit's scenes (kind 'groups')."""
    check_m2_yaml()
    _floor(inst)
    d = _domain(inst)
    orbit = getattr(inst, "orbit", None)
    orbit = np.full(len(inst.sel), -1, np.int64) if orbit is None else np.asarray(orbit, np.int64)
    if orbit.shape != (len(inst.sel),):
        raise ValueError(f"inst.orbit has shape {orbit.shape}, not ({len(inst.sel)},)")
    known = np.unique(orbit[orbit >= 0])
    src = getattr(inst, "orbit_source", None)
    if not len(known):
        raise NotComputable("no_orbit", "the relative orbit of every scene is unknown",
                            {"orbit_source": src})
    K = int(d.K)
    band_of = np.asarray(d.band_of, np.int64)
    centers = np.asarray(d.centers, np.float64)
    kw = fit_kw()
    groups, diag_groups = {}, {}
    fits = []
    for r in known:
        idx = np.flatnonzero(orbit == r)
        gid = group_name(r, src)
        if len(idx) < FLOOR:
            groups[gid] = {"tau": np.full(K, np.nan), "n_scenes": int(len(idx)),
                           "status": "not_computable",
                           "reason": f"{len(idx)} scenes < the floor of {FLOOR}"}
            continue
        if d.status == "no_anchor" or K < 2:
            groups[gid] = {"tau": np.zeros(K), "n_scenes": int(len(idx)), "status": "ok",
                           "reason": None,
                           "flags": ["no_anchor" if d.status == "no_anchor" else "single_band"]}
            continue
        seed = orbit_seed(inst, r)
        cols = draw_columns(band_of, K, kw["max_px_band"], np.random.default_rng(seed))
        fits.append((gid, idx, seed, cols))
    if fits:
        u = _union(*[f[3] for f in fits])
        with _stage(inst, "read"):
            arr = read_columns(inst, u, ("wet", "clear"))
            _progress(inst)
        bank = inst.levels.bank()
        for gid, idx, seed, cols in fits:
            cm = _sorted_cols(cols)
            p = np.searchsorted(u, cm)
            bk = te.ShiftBank(bank.taus, bank.H[:, idx])
            with _stage(inst, "m2a"):
                rows = np.ix_(idx, p)
                r = te.m2a_rasch(arr["wet"][rows], arr["clear"][rows], bk, band_of[cm],
                                 centers, K, rng=np.random.default_rng(seed), **kw)
                _progress(inst)
            tau = np.asarray(r["tau"], float)
            tau_used, veto = causality(tau)
            groups[gid] = {"tau": tau_used, "n_scenes": int(len(idx)), "status": "ok",
                           "reason": None, "tau_raw": tau, "nll": float(r["nll"]),
                           "converged": bool(r["converged"]), "veto": veto,
                           "edge": m2a_edge(tau)}
            diag_groups[gid] = {"seed": seed}
        del arr
    groups = {g: groups[g] for g in sorted(groups, key=_group_order)}
    diag = {"orbit_source": src, "n_unknown_orbit": int((orbit < 0).sum()),
            "orbits": [int(x) for x in known], "domain_status": d.status,
            "centers_km": centers, "seeds": diag_groups, "floor": FLOOR}
    ok = [g for g, e in groups.items() if e["status"] == "ok"]
    if not ok:
        return [common.lag_result("L01v", "not_computable", K=K, stage="orbit_floor",
                                  reason=f"no relative orbit holds {FLOOR} scenes",
                                  kind="groups", groups=groups, diag=diag)]
    return [common.lag_result("L01v", "ok", K=K, kind="groups", lag_kind="linear",
                              groups=groups, diag=diag)]


# ─────────────────────────────────────────────────────────────────────────────
#  L02 m2b and L03 m2c
# ─────────────────────────────────────────────────────────────────────────────

def scan_bank(inst):
    """The exact shift rows of the common window: ``te.ShiftBank(TAU_SCAN,
    inst.levels.stack(TAU_SCAN))`` (every TAU_SCAN value is an integer
    minute, so ``bank.at(tau)`` returns the exact row)."""
    grid = np.asarray(common.TAU_SCAN, float)
    return te.ShiftBank(grid, np.asarray(inst.levels.stack(grid), np.float64))


def _band_product(pid, tau, d, diag, reason_empty):
    tau = np.asarray(tau, np.float64)
    K = int(d.K)
    band = {"tau": tau, "n_px": band_counts(d.band_of, K), "edge": scan_edge(tau),
            "lo": None, "hi": None}
    field = common.broadcast_band(tau, np.asarray(d.band_of, np.int64))
    status = "ok" if np.isfinite(tau).any() else "empty"
    return common.lag_result(pid, status, n_keep=len(d.keep), K=K, kind="band",
                             lag_kind="linear", field=field, band=band,
                             reason=None if status == "ok" else reason_empty, diag=diag)


def run_l02(inst, unit, opts):
    """L02: ``te.m2b_concordance`` on the common window, exact rows."""
    d = _domain(inst)
    K = int(d.K)
    band_of = np.asarray(d.band_of, np.int64)
    cols = draw_columns(band_of, K, M2B_MAX_PX, np.random.default_rng(M2B_RNG_SEED),
                        skip_below=M2B_MIN_BAND_PX)
    u = _union(cols)
    with _stage(inst, "read"):
        arr = read_columns(inst, u, ("wet", "clear"))
        _progress(inst)
    grid = np.asarray(common.TAU_SCAN, float)
    with _stage(inst, "m2b"):
        tau = te.m2b_concordance(arr["wet"], arr["clear"], scan_bank(inst), band_of[u], K, grid,
                                 min_obs=M2B_MIN_OBS, rng=None, max_px_band=M2B_MAX_PX)
        _progress(inst)
    diag = {"n_cols": np.array([len(c) for c in cols], np.int64), "domain_status": d.status,
            "scan": [common.SCAN_LO, common.SCAN_HI, 5.0], "min_obs": M2B_MIN_OBS,
            "max_px_band": M2B_MAX_PX}
    return [_band_product("L02", tau, d, diag,
                          "every band fails m2b's rules (>= 50 px, >= 10 clear obs, "
                          "0.02 < wet share < 0.98)")]


def run_l03(inst, unit, opts):
    """L03: ``te.m2c_waterline`` on the common window, band by band."""
    d = _domain(inst)
    T = int(len(inst.sel))
    if T < M2C_MIN_SCENES:
        raise NotComputable("min_scenes", f"{T} scenes < m2c's {M2C_MIN_SCENES}",
                            {"T": T, "min_scenes": M2C_MIN_SCENES})
    K = int(d.K)
    band_of = np.asarray(d.band_of, np.int64)
    bank = scan_bank(inst)
    grid = np.asarray(common.TAU_SCAN, float)
    tau = np.full(K, np.nan)
    n_use = np.zeros(K, np.int64)
    for k in range(K):
        cols = np.where(band_of == k)[0]
        if len(cols) < M2C_MIN_BAND_PX:
            continue
        with _stage(inst, "read"):
            arr = read_columns(inst, cols, ("wet", "clear"))
            _progress(inst)
        n_use[k] = int((arr["clear"].mean(axis=1) >= M2C_MIN_COVER).sum())
        with _stage(inst, "m2c"):
            tau[k] = te.m2c_waterline(arr["wet"], arr["clear"], bank,
                                      np.zeros(len(cols), np.int64), 1, grid,
                                      min_cover=M2C_MIN_COVER, min_scenes=M2C_MIN_SCENES)[0]
            _progress(inst)
        del arr
    diag = {"n_scenes_used": n_use, "domain_status": d.status,
            "scan": [common.SCAN_LO, common.SCAN_HI, 5.0], "min_cover": M2C_MIN_COVER,
            "min_scenes": M2C_MIN_SCENES}
    return [_band_product("L03", tau, d, diag,
                          "every band fails m2c's rules (>= 50 px, >= 60 scenes seeing "
                          ">= 30 % of the band)")]


# ─────────────────────────────────────────────────────────────────────────────
#  configuration and cards
# ─────────────────────────────────────────────────────────────────────────────

def _signature(method):
    from experiments.benchmark import designs
    m = designs.METHODS[method]
    return {"reads": sorted(m.reads), "domain": bool(m.domain)}


def _cost(method):
    from experiments.benchmark import designs
    m = designs.METHODS[method]
    return {"wh_cell": float(m.cost_wh), "ram_gb": float(m.ram_gb)}


CONFIG = {
    "module": "lags.marea_family",
    "m2a": {"sigma0": SIGMA0, "sg_grid": list(SG_GRID), "z_points": Z_POINTS,
            "max_px_band": MAX_PX_BAND, "tau_bounds": list(TAU_BOUNDS), "monotone": MONOTONE,
            "fit_alpha": False, "bank": "marea.BANK_TAUS exact level-table rows",
            "column_seed": COLUMN_SEED, "seed_offset_step": SEED_OFFSET_STEP},
    "floor": FLOOR, "smooth_km": SMOOTH_KM, "tau_detect_min": TAU_DETECT_MIN,
    "bootstrap": {"n_boot_frozen": N_BOOT_FROZEN, "max_px_band": BOOT_MAX_PX,
                  "seed": BOOT_SEED, "refit_seed": "seed + 100 + b",
                  "percentiles": [2.5, 97.5]},
    "edge_bound_min": EDGE_BOUND_MIN,
    "l01v": {"group": "relative orbit (inst.orbit; -1 left out)",
             "group_names": "orbit<k> (scene_meta.json), phase5-<k> (date-phase fallback)",
             "rng": "default_rng([column_seed, SITE_ID, orbit])", "floor": FLOOR},
    "m2b": {"min_obs": M2B_MIN_OBS, "max_px_band": M2B_MAX_PX, "rng": "default_rng(2)",
            "min_band_px": M2B_MIN_BAND_PX},
    "m2c": {"min_cover": M2C_MIN_COVER, "min_scenes": M2C_MIN_SCENES,
            "min_band_px": M2C_MIN_BAND_PX},
    "scan_window": [common.SCAN_LO, common.SCAN_HI, 5.0],
    "scan_bank": "te.ShiftBank(TAU_SCAN, exact level-table rows)",
}

_COMMON = ("P0 = C1.floor0.ndwi.D1 at W3 and LB, N = all; the instance's record, harness "
           "domain, 3-km bands and exact boundary rows, the same for every lag method")
_REF_MAREA = [{"cite": "MAREA (this study): pyintertidal.tide_estimators.m2a_rasch called as "
                       "pyintertidal.marea.reconstruct calls it; configs/m2.yaml",
               "doi_or_url": "repository: pyintertidal/tide_estimators.py, docs/METHOD.md"}]
_DEV_V = {"paper": ("marea.reconstruct: clear = the SCL clear mask, so a clear pixel whose NDWI is "
                    "NaN enters as a dry observation (Y = 0)"),
          "ours": "clear = V, the setting's valid observation (cloud rule & finite index)",
          "why": "one observation definition for every method (spec §2.5)",
          "rows": "every row but the G-marea gate (production_compat)",
          "sensitivity": "G-marea reproduces production with clear = the cloud rule alone"}
_DEV_ONEBAND = {"paper": "marea.reconstruct calls m2a with one band and stops (no free parameter)",
                "ours": "one band with an anchor: tau = 0, as m2a pins the mouth band",
                "why": "the anchored band has no free lag; NOT COMPUTABLE would drop a valid map",
                "rows": "domains with K = 1 and status 'ok' (small A-mode subsets)",
                "sensitivity": "flag 'single_band' in diag"}
_DEV_BOOT = {"paper": "te.bootstrap_lags runs its draws on joblib threads (cpu_count - 2)",
             "ours": "the same draws and refit seeds on an explicit pool of boot_threads",
             "why": "a fixed thread count per unit (spec §13.4); draws are independent",
             "rows": "E0 boot30 (L01 interval95)", "sensitivity": "none needed (identical draws)"}

_SCAN_DEV = [{"paper": ("configs/m2.yaml tau_malla_min -20..100 step 10 on marea.BANK_TAUS rows "
                        "(15-min shifts, linear interpolation)"),
              "ours": "the common window -45..180 step 5 on exact integer-minute shift rows",
              "why": "one lag window and exact levels for every scanning method (spec §3.3)",
              "rows": "every L02/L03 row", "sensitivity": "unscheduled 'asbuilt' variant"}]

CARDS = {
    "L01": {
        "id": "L01", "name": "MAREA m2a band lags (profiled Bernoulli likelihood, monotone)",
        "family": "likelihood", "kind": "lag", "tier": "1", "fidelity": "F0",
        "evidence": ("G-marea: bitwise reproduction of marea.reconstruct's band lags, interval, "
                     "applied lag and map on the legacy inputs; unit tests equal te.m2a_rasch "
                     "and te.bootstrap_lags on full matrices; positive control"),
        "references": _REF_MAREA, "output": "band", "lag_kind": "linear",
        "signature": _signature("E0"), "scan_window": None,
        "published_window": {"tau_bounds": list(TAU_BOUNDS), "increments_min": [0.0, 180.0]},
        "as_published": {"setting": "C2.raw.ndwi.D1", "window": "W3", "design": "published",
                         "note": "production settings (transition-zone record, raw NDWI)"},
        "native_cell": {"setting": "C2.raw.ndwi.D1", "window": "W3"},
        "common_conditions": _COMMON,
        "deviations": [_DEV_V, _DEV_ONEBAND, _DEV_BOOT],
        "qc": {"floor_scenes": FLOOR, "min_intertidal_px": 3000, "min_finite_s": 5000,
               "band_min_columns_m2a": 50, "usable_min_clear": USABLE_MIN_CLEAR,
               "identified_min_usable": IDENTIFIED_MIN_USABLE,
               "edge": "band lag or increment >= 179.5 min (v13 bound_hit)"},
        "n_min": FLOOR, "tuning": [], "control_offset_min": 0.0,
        "cost": _cost("E0"),
        "variants": {"boot30": "30-draw scene bootstrap (Stage L at P0): interval95",
                     "seed<k>": "column draw seed + 1000 k (seed-variance floor)",
                     "tau120": ("tau_bounds (-45, 120) and BANK_TAUS <= 120 (the setting before "
                                "2026-09-16); unscheduled")},
    },
    "L01v": {
        "id": "L01v", "name": "MAREA m2a per relative orbit (truth-free consistency)",
        "family": "in-house variants", "kind": "lag", "tier": "1", "fidelity": "F0",
        "evidence": "unit tests: each group equals te.m2a_rasch on that orbit's scenes",
        "references": _REF_MAREA, "output": "groups", "lag_kind": "linear",
        "signature": _signature("L01v"), "scan_window": None,
        "published_window": {"tau_bounds": list(TAU_BOUNDS), "increments_min": [0.0, 180.0]},
        "as_published": None, "native_cell": None, "common_conditions": _COMMON,
        "deviations": [_DEV_V, _DEV_ONEBAND],
        "qc": {"floor_scenes_per_orbit": FLOOR, "unknown_orbit": "left out (-1)",
               "rng": "default_rng([column_seed, SITE_ID, orbit])"},
        "n_min": FLOOR, "tuning": [], "control_offset_min": 0.0, "cost": _cost("L01v"),
        "variants": {"m2a_free": "monotone=False (Tier 2, unscheduled)",
                     "unanchored_scan": "the band NLL scanned without the anchor (Tier 2)",
                     "v11_local": "experiments/v11_local_lag.py local scan (Tier 2)",
                     "hysteresis_oos": "te.hysteresis_oos limb clocks (Tier 2)",
                     "tau120": "tau_bounds (-45, 120) (Tier 2)"},
    },
    "L02": {
        "id": "L02", "name": "m2b per-pixel wet/level concordance, band median",
        "family": "order statistic", "kind": "lag", "tier": "1", "fidelity": "F0",
        "evidence": "unit tests: equal to te.m2b_concordance on full matrices (bitwise)",
        "references": [{"cite": ("pyintertidal.tide_estimators.m2b_concordance (in-house phase-M2 "
                                 "estimator: point-biserial correlation, affine-blind)"),
                        "doi_or_url": "repository: pyintertidal/tide_estimators.py"}],
        "output": "band", "lag_kind": "linear", "signature": _signature("L02"),
        "scan_window": [common.SCAN_LO, common.SCAN_HI, 5.0],
        "published_window": list(ASBUILT_GRID), "as_published": None, "native_cell": None,
        "common_conditions": _COMMON, "deviations": list(_SCAN_DEV) + [_DEV_V],
        "qc": {"min_obs": M2B_MIN_OBS, "wet_share": [0.02, 0.98], "min_band_px": M2B_MIN_BAND_PX,
               "max_px_band": M2B_MAX_PX, "edge": "estimate on the window's bound"},
        "n_min": M2B_MIN_OBS, "tuning": [], "control_offset_min": 0.0, "cost": _cost("L02"),
        "variants": {"asbuilt": ("configs/m2.yaml grid -20..100 step 10 on the BANK_TAUS bank "
                                 "(unscheduled)")},
    },
    "L03": {
        "id": "L03", "name": "m2c flooded-area rank correlation per band",
        "family": "order statistic", "kind": "lag", "tier": "1", "fidelity": "F0",
        "evidence": ("unit tests: band-by-band equal to te.m2c_waterline on full matrices "
                     "(bitwise)"),
        "references": [{"cite": ("pyintertidal.tide_estimators.m2c_waterline (in-house phase-M2 "
                                 "estimator: Spearman rank of the flooded-area share)"),
                        "doi_or_url": "repository: pyintertidal/tide_estimators.py"}],
        "output": "band", "lag_kind": "linear", "signature": _signature("L03"),
        "scan_window": [common.SCAN_LO, common.SCAN_HI, 5.0],
        "published_window": list(ASBUILT_GRID), "as_published": None, "native_cell": None,
        "common_conditions": _COMMON, "deviations": list(_SCAN_DEV) + [_DEV_V],
        "qc": {"min_cover": M2C_MIN_COVER, "min_scenes": M2C_MIN_SCENES,
               "min_band_px": M2C_MIN_BAND_PX, "edge": "estimate on the window's bound"},
        "n_min": M2C_MIN_SCENES, "tuning": [], "control_offset_min": 0.0, "cost": _cost("L03"),
        "variants": {"asbuilt": ("configs/m2.yaml grid -20..100 step 10 on the BANK_TAUS bank "
                                 "(unscheduled)"),
                     "min_scenes20": "min_scenes 20 (curve variant, unscheduled)"},
    },
}
