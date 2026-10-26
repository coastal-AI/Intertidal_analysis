# -*- coding: utf-8 -*-
"""Validation v2 — the lag-benchmark instance: what every method of a cell receives (spec §5.1).

Question
--------
Given one plan row (site, window, setting, subset family, mode, N, rep and
the boundary variant), which scenes, observations, harness domain, level
rows, boundary series and judges does every lag method of that cell get, so
that every method comparison is a paired contrast inside one instance
(common random numbers, spec §5.1)?

Pre-registered rules (spec §2.7, §2.8, §2.10, §5.1; Appendix A ``subsets``)
---------------------------------------------------------------------------
* Scenes (:func:`instance_selection`), global record positions, sorted:
  - N = "all": the window's record (``pkg.window``), or for a scene-rule
    setting its kept record (``pkg.rule_subset``);
  - int N, family ``rand``: ``settings.rand_subsets(site, w, T_w, N, R)[rep]``
    mapped through the window record, with R the curve point's replicate
    count (the curve set of the row's design: disjoint batches iff R <=
    floor(T/N), nested otherwise, never mixed, §2.8); a scene rule then
    acts on the drawn scenes (N scenes are drawn before any cloud handling,
    §0.7);
  - family ``win``: ``settings.win_subset`` (contiguous, Tier 2);
  - family ``half``, rep r: the window record ``[r::2]`` (alternate scenes,
    §7.3.4);
  - family ``matched-<C>``: ``settings.matched_subsets(site, w, C1 record,
    n_rule = len(pkg.rule_subset(C, w)), draws)[rep]`` (§2.2).
* Partial-year guard: ``pkg.window`` silently returns a partial record when
  a store year is missing, so every instance first calls
  ``pkg.window_years`` and raises :class:`InstanceUnavailable` when the
  window's years are not all in the package (e.g. W3 at a site holding only
  2023: its units read 'unavailable' and are re-checked on every pass). A
  missing package step (no dom, no level table, no shard manifest of a
  window year, no subsets file for a scene rule, no 1-min series for the
  methods that read it) also raises :class:`InstanceUnavailable`.
* Domain: ``domain.get_v2(pkg, setting, window, sel)`` (A-mode: the subset's
  own counts; F-mode: the window's full record), built lazily so the queue's
  done check needs no domain.
* Levels (:class:`LevelView`): integer tau -> the exact ``LevelTable`` row
  (a live provider call beyond the table, as ``marea.reconstruct``); a
  non-integer tau -> the linear interpolation of the floor and ceil rows;
  ``bank()`` = ``LevelTable.bank(sel)`` (MAREA's shift bank, exact rows);
  ``series_at`` interpolates the 1-min h_B linearly (NaN outside).
* Observations (:class:`ObsView`): ``pkg.observe(setting, sel,
  pos=domain.pos[cols])`` on sorted keep columns, refused above
  :data:`MAX_CELLS` cells (T x m) per read; ``iter_blocks`` covers every
  keep column once, in order.
* Relative orbits (L01v, spec §6.4): the first relative orbit of each date
  from ``cubes/<site>/scene_meta.json`` (double-overpass dates flagged in
  ``orbit_double``); without the sidecar the 5-day phase
  ``date.toordinal() % 5`` (``orbit_source`` 'phase5', declared): with
  S2A and S2B 5 days apart on each track, one residue is one relative orbit
  of the constellation (the 10-day phase of the V2 build split each orbit
  into its two satellites, about 36 scenes a year per group, below m2a's
  40-scene floor at every one-year window).
* Judges: ``judges.load(truth/<site>/judges.json, grid)`` or [] (no table,
  or a site without judges).
* ``seed``: a per-instance integer, the first 4 bytes (little endian) of the
  sha256 of the instance key; the methods with a published seed use their
  own (plan INTERFACES §8).

Nothing here writes data except :func:`write_instance_npz` (the instance
record the scorer reads, plan INTERFACES §6). RTK is never read.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from experiments.benchmark import config, settings

INSTANCE_FORMAT = "v2-lag-instance-1"
#: one observation read may hold at most this many cells (T x columns)
MAX_CELLS = 250_000_000
#: the instance folder next to the product folders of a cell
INSTANCE_DIR = "_instance"
#: the date phase of the relative-orbit fallback (S2A and S2B 5 days apart on a track)
ORBIT_PHASE_DAYS = 5
#: matched-N draws of a scene rule (designs.MATCHED_DRAWS; frame methods 1)
MATCHED_DRAWS = 3
#: the curve sets of each curve design (R of a curve point, §2.8, §5.10)
DESIGN_CURVES = {"curves-k1": ("K1",), "curves-k3": ("K3",), "curves-k4": ("K4",),
                 "curves-k5": ("K5_W1", "K5_W2", "K5_W5", "K5_W9"),
                 "granadeiro-curves": ("K6",)}
#: where the EMODnet tiles of L18 live under the output root: ``fetch emodnet``
#: writes ``<root>/common/<site>/emodnet`` (``fetch.emodnet.site_dir``)
EMODNET_SUBDIR = ("common", "{site}", "emodnet")


class InstanceUnavailable(Exception):
    """The instance cannot be built yet: a window year is not in the
    package, or a package step it needs is missing. Not final: the queue
    records 'unavailable' and re-checks it on every pass."""

    def __init__(self, reason):
        self.reason = str(reason)
        super().__init__(self.reason)


# ─────────────────────────────────────────────────────────────────────────────
#  plan rows
# ─────────────────────────────────────────────────────────────────────────────

def parse_N(v):
    """'all' | int | 'rule:<C>' from a plan cell."""
    if isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
        return int(v)
    s = str(v).strip()
    if s == "all":
        return "all"
    if s.startswith("rule:"):
        return s
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"N {v!r} is neither 'all', an integer nor 'rule:<C>'") from None


def instance_variant_of(variant):
    """The part of a unit variant that changes the instance itself: the
    boundary variants ``bnd-*`` (plan INTERFACES §5); '' otherwise."""
    parts = [p for p in str(variant or "").split("+") if p]
    return "+".join(p for p in parts if p.startswith("bnd-"))


def lb_setting(setting_id):
    """The setting id without its level suffix (lag instances read no level
    source; plan INTERFACES §4)."""
    return settings.Setting.parse(setting_id).replace(level="LB").id


def norm_row(row):
    """A plan row (CSV strings or Python values) with typed fields:
    ``site, window, setting_id`` (as planned), ``setting`` (LB id),
    ``family, mode, N, rep, variant, instance_variant, method, design,
    unit_id, products`` (tuple), ``est_hours, est_ram_gb``."""
    r = dict(row)
    sid = r.get("setting_id", r.get("setting"))
    out = {"site": str(r["site"]), "window": settings.window(r["window"]).id,
           "setting_id": str(sid), "setting": lb_setting(sid),
           "family": str(r.get("family") or "rand"), "mode": str(r.get("mode") or "A"),
           "N": parse_N(r.get("N", "all")), "rep": int(r.get("rep") or 0),
           "variant": str(r.get("variant") or ""),
           "method": str(r.get("method") or ""), "design": str(r.get("design") or ""),
           "unit_id": str(r.get("unit_id") or ""),
           "status": str(r.get("status") or "planned"), "note": str(r.get("note") or "")}
    prods = r.get("products") or ()
    out["products"] = tuple(prods.split()) if isinstance(prods, str) else tuple(prods)
    for k in ("est_hours", "est_ram_gb", "est_read_gb"):
        try:
            out[k] = float(r.get(k) or 0.0)
        except (TypeError, ValueError):
            out[k] = 0.0
    for k in ("priority", "cut_step"):
        try:
            out[k] = int(r.get(k) or 0)
        except (TypeError, ValueError):
            out[k] = 0
    out["effective_key"] = str(r.get("effective_key") or "")
    out["kind"] = str(r.get("kind") or "all")
    out["tier"] = str(r.get("tier") or "")
    out["instance_variant"] = instance_variant_of(out["variant"])
    if out["mode"] not in ("A", "F"):
        raise ValueError(f"unknown mode {out['mode']!r}")
    return out


def instance_key(row):
    """``(site, window, setting (LB), family, mode, N, rep, instance_variant)``."""
    r = norm_row(row) if "instance_variant" not in row else row
    return (r["site"], r["window"], r["setting"], r["family"], r["mode"], str(r["N"]),
            int(r["rep"]), r["instance_variant"])


def instance_id(key):
    """First 16 hex digits of sha256 of the instance key (claims, logs)."""
    return hashlib.sha256(json.dumps(list(key)).encode("utf-8")).hexdigest()[:16]


def instance_seed(key):
    """The per-instance seed (the first 4 bytes of sha256 of the key)."""
    d = hashlib.sha256(json.dumps(list(key)).encode("utf-8")).digest()
    return int.from_bytes(d[:4], "little")


def leaf_name(N, rep):
    """``Nall_r00``, ``N045_r03`` or ``NruleC2_r01``."""
    if N == "all":
        n = "Nall"
    elif isinstance(N, str) and N.startswith("rule:"):
        n = "Nrule" + N.split(":", 1)[1]
    else:
        n = f"N{int(N):03d}"
    return f"{n}_r{int(rep):02d}"


def curve_R(design, window, N, mode="A"):
    """The replicate count R of the curve point (window, N) of a curve design
    (:data:`DESIGN_CURVES`); for another design, the R every rand curve set
    holding that point agrees on (ValueError if none or ambiguous)."""
    ids = DESIGN_CURVES.get(design)
    pool = ids if ids else tuple(settings.CURVES)
    found = set()
    for cid in pool:
        cs = settings.CURVES[cid]
        if cs.family != "rand" or window not in cs.windows:
            continue
        if not ids and cs.mode != mode:
            continue
        for n, R, _ in cs.grid:
            if n != "all" and int(n) == int(N):
                found.add(int(R))
    if len(found) == 1:
        return found.pop()
    if not found:
        raise ValueError(f"no rand curve set holds N = {N} at {window} (design {design!r})")
    raise ValueError(f"curve sets disagree on R ({sorted(found)}) for N = {N} at {window}; "
                     "the row's design must name its curve set")


# ─────────────────────────────────────────────────────────────────────────────
#  the scenes of an instance
# ─────────────────────────────────────────────────────────────────────────────

def _guard_years(pkg, wid):
    try:
        return list(pkg.window_years(wid))
    except Exception as e:                       # package.PackageError
        have = list(getattr(pkg, "years", []) or [])
        need = list(settings.window(wid).years)
        miss = [y for y in need if y not in have]
        if miss:
            raise InstanceUnavailable(f"store years {miss} of window {wid} are not in the "
                                      f"package (years {have}): {e}") from None
        raise InstanceUnavailable(f"window {wid}: {e}") from None


def _rule_subset(pkg, rule, wid):
    try:
        return np.asarray(pkg.rule_subset(rule, wid), np.int64)
    except Exception as e:                       # subsets_<w>.npz missing
        raise InstanceUnavailable(f"scene rule {rule} at {wid}: {e}") from None


def instance_selection(pkg, row):
    """Global record positions (sorted int64) of a plan row's instance.

    Raises :class:`InstanceUnavailable` when the window's years are not all
    in the package or a subsets file it needs is missing.
    """
    r = norm_row(row) if "instance_variant" not in row else row
    wid = r["window"]
    _guard_years(pkg, wid)
    s = settings.Setting.parse(r["setting"])
    rule = s.scene_rule
    win = np.asarray(pkg.window(wid), np.int64)
    fam, N, rep = r["family"], r["N"], int(r["rep"])
    if fam.startswith("matched-"):
        C = fam.split("-", 1)[1]
        n_rule = len(_rule_subset(pkg, C, wid))
        subs = settings.matched_subsets(pkg.site, wid, win, n_rule,
                                        draws=max(MATCHED_DRAWS, rep + 1))
        return np.asarray(subs[rep], np.int64)
    base = _rule_subset(pkg, rule, wid) if rule else win
    if fam == "half":
        if N != "all":
            raise ValueError("the alternate-scene halves split the whole record (N = all)")
        if rep not in (0, 1):
            raise ValueError(f"half rep must be 0 or 1, not {rep}")
        return np.asarray(base[rep::2], np.int64)
    if N == "all":
        if fam not in ("rand", "rule", "win"):
            raise InstanceUnavailable(f"subset family {fam!r} is not built in V2")
        return np.asarray(base, np.int64)
    if isinstance(N, str):
        raise ValueError(f"N {N!r} needs a matched-<C> family")
    T = len(win)
    if fam == "rand":
        R = curve_R(r["design"], wid, N, r["mode"])
        _, subs = settings.rand_subsets(pkg.site, wid, T, N, R)
        if rep >= len(subs):
            raise ValueError(f"rep {rep} >= R = {R} at N = {N}")
        sel = win[subs[rep]]
    elif fam == "win":
        sel = win[settings.win_subset(pkg.site, wid, T, N, rep)]
    else:
        raise InstanceUnavailable(f"subset family {fam!r} with N = {N} is not built in V2")
    if rule:
        sel = np.intersect1d(sel, base)
    return np.asarray(sel, np.int64)


# ─────────────────────────────────────────────────────────────────────────────
#  levels and observations
# ─────────────────────────────────────────────────────────────────────────────

class LevelView:
    """The boundary levels h_B(t_i - tau) at the instance's scenes.

    Parameters
    ----------
    table : levels.LevelTable
        The package's exact level table.
    sel : (T,) int64
        The instance's record positions.
    series1 : tuple or callable, optional
        ``(t_min, h)`` of the 1-min h_B, or a zero-argument callable giving it.
    """

    def __init__(self, table, sel, series1=None):
        self.table = table
        self.sel = np.asarray(sel, np.int64)
        self._series1 = series1
        self._rows = {}

    @property
    def T(self):
        return int(len(self.sel))

    def _row(self, k):
        k = int(k)
        if k not in self._rows:
            self._rows[k] = np.asarray(self.table.at(k, self.sel), np.float64)
        return self._rows[k]

    def at(self, tau):
        """(T,) float64: the exact row at an integer tau, else the linear
        interpolation of the floor and ceil rows."""
        f = float(tau)
        if not math.isfinite(f):
            raise ValueError(f"tau must be finite, got {tau!r}")
        lo = math.floor(f)
        if f == lo:
            return self._row(lo).copy()
        w = f - lo
        return (1.0 - w) * self._row(lo) + w * self._row(lo + 1)

    def stack(self, taus):
        """(len(taus), T) float64."""
        taus = list(np.atleast_1d(np.asarray(taus, float)))
        if not taus:
            return np.empty((0, self.T), np.float64)
        return np.vstack([self.at(t) for t in taus])

    def bank(self, taus=None):
        """MAREA's shift bank (``te.ShiftBank``) on the instance's scenes:
        ``LevelTable.bank(sel, taus)`` (exact rows; marea.BANK_TAUS by
        default)."""
        return self.table.bank(self.sel, taus)

    @property
    def series1(self):
        s = self._series1
        return s() if callable(s) else s

    def series_at(self, t_min):
        """h_B at minutes since 1970-01-01 by linear interpolation of the
        1-min series (NaN outside it)."""
        s = self.series1
        if s is None:
            raise InstanceUnavailable("no 1-min boundary series for this instance")
        t1, h1 = s
        return np.interp(np.asarray(t_min, np.float64), np.asarray(t1, np.float64),
                         np.asarray(h1, np.float64), left=np.nan, right=np.nan)


class ObsView:
    """The instance's observations X, Y, V, W on its keep pixels.

    ``read(cols)`` returns ``obs.Observation`` of shape (T, m) for sorted
    unique indices ``cols`` into ``domain.keep``; ``iter_blocks`` yields
    every keep column once, in order.
    """

    def __init__(self, pkg, setting, sel, domain, max_cells=MAX_CELLS):
        self.pkg = pkg
        self.setting = setting
        self.sel = np.asarray(sel, np.int64)
        self.domain = domain
        self.max_cells = int(max_cells)

    @property
    def n(self):
        return int(len(self.domain.keep))

    @property
    def T(self):
        return int(len(self.sel))

    def _cols(self, cols):
        if cols is None:
            return np.arange(self.n, dtype=np.int64)
        c = np.asarray(cols)
        if c.dtype == bool:
            if c.shape != (self.n,):
                raise ValueError(f"a boolean column mask must have length {self.n}")
            return np.flatnonzero(c).astype(np.int64)
        if c.ndim != 1 or (c.size and c.dtype.kind not in "iu"):
            raise ValueError("cols must be 1-D integer indices into domain.keep")
        c = c.astype(np.int64)
        if c.size and (c[0] < 0 or c[-1] >= self.n or np.any(np.diff(c) <= 0)):
            raise ValueError("cols must be sorted unique indices in [0, n)")
        return c

    def read(self, cols=None):
        """``obs.Observation(X, Y float32 (T, m); V, W bool (T, m))``."""
        c = self._cols(cols)
        if self.T * len(c) > self.max_cells:
            raise ValueError(f"a read of {self.T} scenes x {len(c)} pixels exceeds "
                             f"MAX_CELLS = {self.max_cells:,}: use iter_blocks")
        if self.domain.pos is None:
            raise InstanceUnavailable("the domain has no package positions (no dom)")
        return self.pkg.observe(self.setting, self.sel, pos=self.domain.pos[c])

    def iter_blocks(self, block=100_000):
        """Yield ``(cols, Observation)`` over every keep column, in order."""
        block = int(block)
        if block < 1:
            raise ValueError("block must be >= 1")
        for a in range(0, self.n, block):
            cols = np.arange(a, min(a + block, self.n), dtype=np.int64)
            yield cols, self.read(cols)


# ─────────────────────────────────────────────────────────────────────────────
#  the instance
# ─────────────────────────────────────────────────────────────────────────────

def _null_stage(name):
    return contextlib.nullcontext()


def _null_progress(n=1):
    return None


@dataclass(eq=False)
class LagInstance:
    """Everything a lag method of one cell receives (plan INTERFACES §4).

    Eager attributes: the identity (site, window, setting, family, mode, N,
    rep, instance_variant), the scenes (sel, t, dates, year, h0, rising,
    orbit, orbit_source, orbit_double, platform), the index (index,
    nir_type), the frame (pixel_m, shape, mouth_side), paths, seed, pkg,
    site2, key, instance_id, subset_key, package_keys, cache and the timing
    hooks ``stage(name)`` / ``progress(n)`` (set per unit by the queue).
    Lazy attributes (built on first use): domain, obs, levels, series1,
    series10, marks_lp, judges, boundary_members, installed_models, grid,
    index_params.
    """

    site: str
    window: str
    setting: str
    family: str
    mode: str
    N: object
    rep: int
    instance_variant: str
    sel: np.ndarray
    t: pd.DatetimeIndex
    dates: np.ndarray
    year: np.ndarray
    h0: np.ndarray
    rising: np.ndarray
    orbit: np.ndarray
    orbit_source: str
    platform: np.ndarray
    index: str
    nir_type: bool
    pixel_m: float
    shape: tuple
    mouth_side: str
    paths: dict
    seed: int
    pkg: object = None
    site2: object = None
    key: tuple = ()
    instance_id: str = ""
    subset_key: str = ""
    package_keys: dict = field(default_factory=dict)
    orbit_double: Optional[np.ndarray] = None
    stage: object = _null_stage
    progress: object = _null_progress
    cache: dict = field(default_factory=dict)
    _lazy: dict = field(default_factory=dict, repr=False)

    @property
    def T(self):
        return int(len(self.sel))

    # -- lazy pieces -----------------------------------------------------------
    def _get(self, name, build):
        if name not in self._lazy:
            self._lazy[name] = build()
        return self._lazy[name]

    @property
    def domain(self):
        def build():
            from experiments.benchmark import domain as dm
            sel = None if self.mode == "F" else self.sel
            return dm.get_v2(self.pkg, self.setting, self.window, sel)
        return self._get("domain", build)

    @property
    def obs(self):
        return self._get("obs", lambda: ObsView(self.pkg, self.setting, self.sel, self.domain))

    @property
    def levels(self):
        def build():
            try:
                table = self.pkg.levels
            except Exception as e:                # package.PackageError
                raise InstanceUnavailable(f"level table: {e}") from None
            return LevelView(table, self.sel, series1=lambda: self.series1)
        return self._get("levels", build)

    @property
    def series1(self):
        def build():
            try:
                return self.pkg.series1
            except Exception as e:
                raise InstanceUnavailable(f"1-min boundary series: {e}") from None
        return self._get("series1", build)

    @property
    def series10(self):
        def build():
            try:
                return self.pkg.series10
            except Exception as e:
                raise InstanceUnavailable(f"10-min boundary series: {e}") from None
        return self._get("series10", build)

    @property
    def marks_lp(self):
        def build():
            try:
                return self.pkg.marks_lp
            except Exception as e:
                raise InstanceUnavailable(f"tide marks (marks_lp): {e}") from None
        return self._get("marks_lp", build)

    @property
    def judges(self):
        def build():
            from experiments.benchmark import judges as jd
            p = os.path.join(self.paths["truth_dir"], "judges.json")
            if not getattr(self.pkg.cfg, "judges", ()) or not os.path.exists(p):
                return []
            return jd.load(p, jd.grid_of(self.pkg))
        return self._get("judges", build)

    @property
    def boundary_members(self):
        return self._get("boundary_members", lambda: boundary_members(self.pkg.cfg))

    @property
    def installed_models(self):
        def build():
            try:
                from experiments.benchmark.fetch import tides
                return tuple(tides.installed_models(self.paths["tide_dir"]))
            except Exception:                     # eo-tides absent or no model folder
                return ()
        return self._get("installed_models", build)

    @property
    def grid(self):
        def build():
            from experiments.benchmark import judges as jd
            return jd.grid_of(self.pkg)
        return self._get("grid", build)

    @property
    def index_params(self):
        def build():
            try:
                return dict(self.pkg.index_params(self.index))
            except Exception as e:                # the settings pass has not run
                raise InstanceUnavailable(f"index {self.index} parameters: {e}") from None
        return self._get("index_params", build)

    def label(self):
        return (f"{self.site} {self.window} {self.setting} {self.family}-{self.mode} "
                f"{leaf_name(self.N, self.rep)}"
                + (f" [{self.instance_variant}]" if self.instance_variant else ""))


def boundary_members(cfg):
    """The members of the site's boundary h_B with their coordinates:
    the IOC gauges of ``make_boundary`` (``levels.boundary_codes``, positions
    from ioc_stations.json, kind 'ioc') or the tide-model point of a pytmd
    boundary (Villaviciosa: EOT20 at the AOI centroid, kind 'model')."""
    b = dict(cfg.boundary)
    kind = b.get("kind")
    if kind == "make_boundary":
        from experiments.benchmark import judges as jd
        return [{"code": m["code"], "lon": float(m["lon"]), "lat": float(m["lat"]),
                 "kind": "ioc"} for m in jd.boundary_members(cfg)]
    if kind == "pytmd":
        from pyintertidal import sites
        lat, lon = sites.get(b["aoi"]).centroid
        return [{"code": f"{b.get('model', 'model')}@{b.get('at', 'aoi.centroid')}",
                 "lon": float(lon), "lat": float(lat), "kind": "model"}]
    return []


def scene_orbits(site2, dates):
    """``(orbit int64, orbit_double bool, platform str, source)`` per date."""
    dates = [str(d) for d in dates]
    meta = None
    try:
        from experiments.benchmark import inputs
        p = inputs.scene_meta_path(site2.site, site2.cubes_root)
        if p.is_file():
            meta = inputs.read_scene_meta(p)
    except Exception:                             # unreadable sidecar: the phase rule
        meta = None
    if meta:
        dm = meta.get("dates") or {}
        orb, dbl, plat = [], [], []
        for d in dates:
            v = dm.get(d) or {}
            ro = list(v.get("relative_orbits") or [])
            orb.append(int(ro[0]) if ro else -1)
            dbl.append(bool(v.get("double_overpass", False)))
            pl = list(v.get("platforms") or [])
            plat.append(str(pl[0]) if pl else "")
        return (np.asarray(orb, np.int64), np.asarray(dbl, bool), np.asarray(plat, str),
                "scene_meta")
    orb = [pd.Timestamp(d).date().toordinal() % ORBIT_PHASE_DAYS for d in dates]
    return (np.asarray(orb, np.int64), np.zeros(len(dates), bool),
            np.asarray([""] * len(dates), str), f"phase{ORBIT_PHASE_DAYS}")


def instance_paths(site2):
    from experiments.benchmark import package
    from experiments import zarr_cubes as zc
    try:
        cubes = str(zc.cubes_root(site2.cubes_root)).replace("\\", "/")
    except Exception:
        cubes = site2.cubes_root
    root = str(site2.root)
    edir = os.path.join(root, *(p.format(site=site2.site) for p in EMODNET_SUBDIR))
    return {"root": root, "common_dir": site2.dir, "truth_dir": site2.truth_dir,
            "cubes_root": cubes, "tide_dir": config.TIDE_DIR,
            "cmems_dir": package._cmems_root(site2),
            "emodnet_dir": edir.replace("\\", "/")}


def package_keys(pkg, window):
    """``{"shards": {year: package_key}, "package_key", "dom_version"}`` of
    the window's year shards (``package.shard_keys``) and the package's own
    key (record, levels, series, marks, geometry). Raises
    :class:`InstanceUnavailable` when a year or its manifest is missing."""
    from experiments.benchmark import package
    years = _guard_years(pkg, window)
    if getattr(pkg, "dom_version", None) is None:
        raise InstanceUnavailable(f"{pkg.site}: no dom yet (prepare --step dom)")
    try:
        sk = package.shard_keys(pkg.site2, pkg.dom_version, years)
    except Exception as e:
        raise InstanceUnavailable(f"shard manifests: {e}") from None
    miss = [y for y in years if y not in sk]
    if miss:
        raise InstanceUnavailable(f"no shard manifest of {miss} under dom {pkg.dom_version} "
                                  f"(prepare {pkg.site} --step pass2,manifest)")
    return {"shards": {str(y): sk[y] for y in sorted(sk) if y in years},
            "package_key": pkg.keys["package_key"], "dom_version": pkg.dom_version}


def build_instance(site2, row, pkg=None, provider=None, stopwatch=None):
    """The :class:`LagInstance` of a plan row (lazy domain, observations and
    levels; plan INTERFACES §4).

    Parameters
    ----------
    site2 : package.V2Site
    row : dict
        A plan row (``designs.CSV_COLUMNS``; CSV strings accepted).
    pkg : package.V2Package, optional
        Default ``package.V2Package(site2)``.
    provider : object with ``levels``, optional
        The live boundary provider for level rows beyond the table (default
        the package's).
    stopwatch : timing.Stopwatch, optional
        Times the selection as ``io.instance``.

    Raises
    ------
    InstanceUnavailable
        Window years missing, or a package step missing (dom, level table,
        shard manifests, subsets).
    """
    from experiments.benchmark import package, product
    r = norm_row(row)
    if r["instance_variant"]:
        raise InstanceUnavailable(f"instance variant {r['instance_variant']!r}: the "
                                  "boundary-variant level rows are not built in V2")
    cm = stopwatch.stage("io.instance") if stopwatch is not None else contextlib.nullcontext()
    with cm:
        if pkg is None:
            try:
                pkg = package.V2Package(site2)
            except Exception as e:
                raise InstanceUnavailable(f"{site2.site}: package not readable: {e}") from None
        if provider is not None and getattr(pkg, "_provider", None) is None:
            pkg._provider = provider
        pk = package_keys(pkg, r["window"])
        sel = instance_selection(pkg, r)
        try:
            pkg.levels                                   # the level table must exist
        except Exception as e:
            raise InstanceUnavailable(f"level table: {e}") from None
        key = instance_key(r)
        dates = np.asarray(pkg.dates[sel], str)
        orbit, dbl, plat, osrc = scene_orbits(site2, dates)
        s = settings.Setting.parse(r["setting"])
        inst = LagInstance(
            site=r["site"], window=r["window"], setting=r["setting"], family=r["family"],
            mode=r["mode"], N=r["N"], rep=r["rep"], instance_variant=r["instance_variant"],
            sel=sel, t=pkg.t[sel], dates=dates,
            year=np.asarray(pkg.year_of[sel], np.int64),
            h0=np.asarray(pkg.h0[sel], np.float64), rising=np.asarray(pkg.rising[sel], bool),
            orbit=orbit, orbit_source=osrc, platform=plat, orbit_double=dbl,
            index=s.index, nir_type=(s.index == "nir"), pixel_m=float(pkg.pixel_m),
            shape=(int(pkg.H), int(pkg.W)), mouth_side=str(pkg.cfg.mouth_side),
            paths=instance_paths(site2), seed=instance_seed(key), pkg=pkg, site2=site2,
            key=key, instance_id=instance_id(key), subset_key=product.subset_key(sel),
            package_keys=pk)
    return inst


# ─────────────────────────────────────────────────────────────────────────────
#  the instance record (plan INTERFACES §6)
# ─────────────────────────────────────────────────────────────────────────────

def instance_dir(root, inst_or_key):
    """``<root>/runs/<site>/<window>/<setting>/_instance/<family>-<mode>/<leaf>[__<variant>]``."""
    if isinstance(inst_or_key, (tuple, list)):
        site, wid, sid, fam, mode, N, rep, iv = inst_or_key
        N = parse_N(N)
    else:
        i = inst_or_key
        site, wid, sid, fam, mode, N, rep, iv = (i.site, i.window, i.setting, i.family,
                                                 i.mode, i.N, i.rep, i.instance_variant)
    leaf = leaf_name(N, rep) + (f"__{iv}" if iv else "")
    return "/".join([str(root).rstrip("/\\").replace("\\", "/"), "runs", site, wid, sid,
                     INSTANCE_DIR, f"{fam}-{mode}", leaf])


def instance_arrays(inst):
    """The arrays of instance.npz."""
    d = inst.domain
    return {"sel": np.asarray(inst.sel, np.int64),
            "t_ns": np.asarray(pd.DatetimeIndex(inst.t).asi8, np.int64),
            "h0": np.asarray(inst.h0, np.float64),
            "keep": np.asarray(d.keep, np.int64),
            "pos": (np.asarray(d.pos, np.int64) if d.pos is not None
                    else np.empty(0, np.int64)),
            "band_of": np.asarray(d.band_of, np.int16),
            "s_km": np.asarray(d.s_km, np.float32),
            "centers": np.asarray(d.centers, np.float64),
            "edges": np.asarray(d.edges, np.float64),
            "K": np.array(int(d.K), np.int64), "status": np.array(str(d.status)),
            "orbit": np.asarray(inst.orbit, np.int64),
            "platform": np.asarray(inst.platform, str)}


def instance_doc(inst, timings=None):
    d = inst.domain
    from experiments.benchmark import product
    return {"format": INSTANCE_FORMAT, "site": inst.site, "window": inst.window,
            "setting": inst.setting, "family": inst.family, "mode": inst.mode,
            "N": inst.N, "rep": int(inst.rep), "instance_variant": inst.instance_variant,
            "instance_id": inst.instance_id, "T": int(len(inst.sel)),
            "n_keep": int(d.n_keep), "K": int(d.K), "domain_status": str(d.status),
            "orbit_source": inst.orbit_source, "package_keys": inst.package_keys,
            "subset_key": inst.subset_key, "timings": dict(timings or {}),
            "saved_utc": product.utc_stamp()}


def write_instance_npz(root, inst, timings=None):
    """Write ``instance.npz`` and ``instance.json`` (atomic files; the
    instance-level stage timings, e.g. ``domain.build``, go to the JSON). An
    existing record with the same arrays is left untouched. Returns
    ``(dir, sha256 of instance.npz)``."""
    from experiments.benchmark import product
    d = instance_dir(root, inst)
    npz = os.path.join(d, "instance.npz")
    arrays = instance_arrays(inst)
    if os.path.exists(npz):
        try:
            old = product.load_npz(npz)
            same = set(old) == set(arrays) and all(
                np.asarray(old[k]).dtype == np.asarray(arrays[k]).dtype
                and np.array_equal(old[k], arrays[k]) for k in arrays)
        except Exception:
            same = False
        if same and os.path.exists(os.path.join(d, "instance.json")):
            return d, product.sha256_file(npz)
    product.save_npz(npz, arrays, compress=True)
    product.write_json(os.path.join(d, "instance.json"), instance_doc(inst, timings))
    return d, product.sha256_file(npz)


def read_instance(path):
    """``{arrays..., "json": instance.json}`` of an instance folder (or of
    its instance.npz)."""
    from experiments.benchmark import product
    d = path[:-len("instance.npz")] if str(path).endswith("instance.npz") else path
    out = product.load_npz(os.path.join(d, "instance.npz"))
    out["status"] = str(out["status"])
    out["K"] = int(out["K"])
    jp = os.path.join(d, "instance.json")
    out["json"] = product.read_json(jp) if os.path.exists(jp) else None
    return out
