# -*- coding: utf-8 -*-
"""v13 benchmark — the pixel domain and the along-estuary geometry (§1.5, §2.6, §3.1).

Question
--------
On which pixels does a method that follows MAREA's extraction rule work, how
far is each of them from the mouth along the water, and in which 3-km band
does it lie? Two answers are needed for every record subset: the
FULL-RECORD domain (F-mode; also the fixed scoring strata, the judges' bands
and neighbourhoods) and the domain the subset's OWN observations give
(A-mode, "the archive has N images", §0.5).

Pre-registered rules (FROZEN['domain'], FROZEN['package']['U_rec'])
-------------------------------------------------------------------
* Counts: V (valid observation) and wet = V & NDWI > 0 under the primary
  mask rule and index mode (``obs.MASK_RULE`` = marea_clear,
  ``obs.INDEX_MODE`` = floor0), over the scenes of the record or subset.
* The extraction rule of ``marea.extract``: ``wf = wet / max(clr, 1)`` where
  ``clr > 30`` (NaN elsewhere); intertidal iff ``0.10 < wf < 0.90``, sea iff
  ``wf >= 0.90``; keep = the flat indices of the intertidal pixels,
  ascending, no polygon clip.
* Fewer than 3,000 keep pixels: status ``no_intertidal`` and no geometry
  (``marea.reconstruct``'s "sin_intermareal": the run stops).
* ``s_km = geometry.along_distance(sea | inter, geometry.mouth_seeds(sea,
  side=mouth_side), pixel_m).ravel()[keep] / 1000`` in float64; a
  ValueError (no sea on the mouth side) gives NaN everywhere, as in
  ``reconstruct``.
* More than 5,000 finite s: ``te.make_bands_fixed(s_km, 3.0, 500)``, NaN s
  in band -1; otherwise ONE band and tau = 0, status ``no_anchor`` (band 0
  for every keep pixel, ``reconstruct``'s no-anchor path).
* F-mode = the full record. A-mode (:func:`for_subset`) = the same rule on
  the subset's own counts: inside the package domain ``dom`` from the stored
  raw values, outside it from the frame validity bits. This is exact: a
  pixel outside dom is outside U_rec (= blocklist + floor0 counts clr >= 31,
  wet >= 4, dry >= 4 over the full record), so in any subset it is never
  intertidal, and it is sea exactly when it is "sea-like" (full-record
  primary counts clr >= 31, dry <= 3) and its subset clear count exceeds 30
  (then wf >= 28/31 > 0.90). ``for_subset(arange(T)) == full(pkg)``
  bitwise (tested).
* Observation sensitivities (``sens`` mask or index other than the primary)
  hold the domain at the full record (§3.9): only the observation changes.
* Strata outside keep (:func:`band_map_frame`): the raw band
  ``floor(s / 3)`` of any frame pixel with finite s, mapped to the merged
  band of the keep pixels in that raw band; a raw band without keep pixels
  takes the merged band of the nearest lower raw band that has some (band
  0 below the first one); s beyond the last raw band is clipped to it, as in
  ``make_bands_fixed``; NaN s -> -1 ("no s").
* Production geometry (:func:`production`, gates only): the same s_km and
  bands from the production extraction's keep, sea and inter.

Validation v2 (docs/validation_v2_spec.md §2.9, §2.10; :func:`for_setting`)
------------------------------------------------------------------------------
* The harness domain of an INSTANCE (setting, window, subset) follows the
  same extraction rule on the counts of the instance's own setting
  (V and W of ``obs.observe_setting``), never on P0's.
* Whole window, pixel setting with full-frame counters (the package's
  uint8 year counters): the rule on the frame counts, exact; keep = its
  intertidal pixels inside dom (pixels outside dom are dropped and
  counted; declared).
* A subset, a scene rule (its kept record), or a setting counted only at
  dom (D2, D3, NIR): counts at dom from the stored values; outside dom the
  sea is the "sea-like" rule on the full-window counters of the setting's
  frame-bit family (P0 for C1 and the scene rules, C11, C3), exact for
  them as in v13, else P0's full-window sea (declared).
* Cache key: (setting id, window, subset_key) under the dom_version, with
  the shards' package keys and this module's source in the bookkeeping.

The constants below are the FROZEN values; tests on tiny synthetic frames
lower MIN_INTERTIDAL, MIN_FINITE_S and MIN_PX_BAND with monkeypatch, which
is why every function reads them at call time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from experiments.benchmark import config, obs
from pyintertidal import geometry
from pyintertidal import tide_estimators as te

_D = config.FROZEN["domain"]
_R = _D["extract_rule"]
#: marea.extract: a pixel needs more than 30 clear observations
MIN_CLEAR = int(_R["min_clear"])
WF_LO = float(_R["wf_lo"])
WF_HI = float(_R["wf_hi"])
SEA_WF_MIN = float(_R["sea_wf_min"])
#: the spec's summary of the rule (marea.extract: clr > 30, 0.10 < wf < 0.90,
#: sea wf >= 0.90)
EXTRACT_RULE = dict(min_clear=MIN_CLEAR, wf_lo=WF_LO, wf_hi=WF_HI)
#: marea.reconstruct: fewer keep pixels -> "sin_intermareal"
MIN_INTERTIDAL = int(_D["min_intertidal"])
#: marea.reconstruct: the interior tide only with more finite s than this
MIN_FINITE_S = int(_D["min_finite_s"])
#: te.make_bands_fixed(s_km, 3.0, 500), as reconstruct calls it
BAND_KM = float(_D["bands"]["length_km"])
MIN_PX_BAND = int(_D["bands"]["min_px"])
STATUSES = ("ok", "no_intertidal", "no_anchor")
#: "A" subset, "F" full record, "P" production extraction (gates only)
MODES = ("A", "F", "P")
DOMAIN_CACHE_FORMAT = "v13-domain-1"

_U = config.FROZEN["package"]["U_rec"]
#: a pixel outside the package domain with at most this many dry
#: observations (and >= MIN_CLEAR clear) over the full record is "sea-like"
SEA_LIKE_MAX_DRY = int(_U["min_dry"]) - 1

# The exactness of the A-mode domain rests on these relations between
# FROZEN['package']['U_rec'] and the extraction rule; a later edit of one
# without the other would silently break it, so it stops the import.
if int(_U["min_clear"]) != MIN_CLEAR:
    raise AssertionError("U_rec min_clear differs from the extraction rule")
if int(np.floor(WF_LO * MIN_CLEAR)) + 1 < int(_U["min_wet"]):
    raise AssertionError("U_rec min_wet exceeds what 0.10 < wf allows at the minimum clear count")
if int(np.floor((1.0 - WF_HI) * MIN_CLEAR)) + 1 < int(_U["min_dry"]):
    raise AssertionError("U_rec min_dry exceeds what wf < 0.90 allows at the minimum clear count")
if (MIN_CLEAR - SEA_LIKE_MAX_DRY) / MIN_CLEAR < SEA_WF_MIN:
    raise AssertionError("a sea-like pixel would not be sea at the minimum clear count")
if not set(obs.CLEAR_SCL).isdisjoint(obs.BLOCK_SCL):
    raise AssertionError("marea_clear observations must be blocklist observations")
if _U["index"] != obs.INDEX_MODE or _U["mask"] != "blocklist":
    raise AssertionError("U_rec must be counted under blocklist and the primary index mode")
del _D, _R, _U


def log(msg):
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  the domain object
# ─────────────────────────────────────────────────────────────────────────────

def _bits_equal(a, b):
    """Bitwise equality of two float arrays (NaN payloads included)."""
    a = np.ascontiguousarray(a, np.float64)
    b = np.ascontiguousarray(b, np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.int64), b.view(np.int64))


@dataclass(eq=False)
class Domain:
    """The pixel domain of one record (sub)set.

    Attributes
    ----------
    mode : str
        "A" (subset), "F" (full record) or "P" (production extraction).
    keep : (n,) int64
        Flat indices (row * W + col) of the intertidal pixels, ascending.
    pos : (n,) int64 or None
        Their positions in the package domain ``pkg.dom``.
    sea, inter : (H, W) bool
        Sea (wf >= 0.90) and intertidal (keep) frame masks.
    s_km : (n,) float64
        Along-water distance from the mouth; NaN = isolated pixel.
    band_of : (n,) int64
        Band of each keep pixel (-1 = NaN s, only with status "ok").
    centers, edges : float64
        Band centres (median s) and the fixed 3-km raw-band edges (empty
        unless status is "ok").
    K : int
        Number of bands.
    status : str
        "ok", "no_intertidal" (fewer than 3,000 keep pixels; no geometry) or
        "no_anchor" (at most 5,000 finite s; one band, tau = 0).
    n_sel : int
        Scenes behind the counts.
    s_frame : (H, W) float64 or None
        s in km on the whole frame (NaN unreachable), when it was asked for.
    """

    mode: str
    keep: np.ndarray
    pos: Optional[np.ndarray]
    sea: np.ndarray
    inter: np.ndarray
    s_km: np.ndarray
    band_of: np.ndarray
    centers: np.ndarray
    edges: np.ndarray
    K: int
    status: str
    n_sel: int = 0
    s_frame: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown domain mode {self.mode!r}; expected one of {MODES}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown domain status {self.status!r}")
        self.keep = np.asarray(self.keep, np.int64)
        if self.pos is not None:
            self.pos = np.asarray(self.pos, np.int64)
        self.band_of = np.asarray(self.band_of, np.int64)
        self.s_km = np.asarray(self.s_km, np.float64)
        self.centers = np.asarray(self.centers, np.float64)
        self.edges = np.asarray(self.edges, np.float64)
        self.K = int(self.K)
        self.n_sel = int(self.n_sel)

    @property
    def shape(self):
        return self.sea.shape

    @property
    def n_keep(self):
        return int(len(self.keep))

    def summary(self):
        """JSON-ready description (counts, bands, status)."""
        fin = np.isfinite(self.s_km)
        return {"mode": self.mode, "status": self.status, "n_sel": self.n_sel,
                "n_keep": self.n_keep, "n_sea": int(np.count_nonzero(self.sea)),
                "K": self.K, "centers_km": [float(c) for c in self.centers],
                "n_isolated": int(np.count_nonzero(~fin)),
                "s_max_km": float(np.max(self.s_km[fin])) if fin.any() else None}

    def equals(self, other, mode=False):
        """Bitwise equality of every array (``mode=True`` also compares the
        mode label; ``s_frame`` only when both carry it)."""
        if not isinstance(other, Domain):
            return False
        if mode and self.mode != other.mode:
            return False
        if (self.status, self.K, self.n_sel) != (other.status, other.K, other.n_sel):
            return False
        if (self.pos is None) != (other.pos is None):
            return False
        if self.pos is not None and not np.array_equal(self.pos, other.pos):
            return False
        for a, b in ((self.keep, other.keep), (self.sea, other.sea),
                     (self.inter, other.inter), (self.band_of, other.band_of)):
            if a.shape != b.shape or not np.array_equal(a, b):
                return False
        for a, b in ((self.s_km, other.s_km), (self.centers, other.centers),
                     (self.edges, other.edges)):
            if not _bits_equal(a, b):
                return False
        if self.s_frame is not None and other.s_frame is not None:
            return _bits_equal(self.s_frame, other.s_frame)
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  the rule, the geometry, the bands
# ─────────────────────────────────────────────────────────────────────────────

def classify(clr, wet):
    """The extraction rule on observation counts: ``(inter, sea)``.

    ``wf = where(clr > 30, wet / max(clr, 1), nan)`` exactly as
    ``marea.extract`` (the counts are taken as int64, so the float64
    division is the same whatever their stored dtype).
    """
    clr = np.asarray(clr).astype(np.int64, copy=False)
    wet = np.asarray(wet).astype(np.int64, copy=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        wf = np.where(clr > MIN_CLEAR - 1, wet / np.maximum(clr, 1), np.nan)
    fin = np.isfinite(wf)
    inter = fin & (wf > WF_LO) & (wf < WF_HI)
    sea = fin & (wf >= SEA_WF_MIN)
    return inter, sea


def s_frame_km(sea, inter, pixel_m, mouth_side):
    """Along-water distance from the mouth on the whole frame, in km.

    ``along_distance(sea | inter, mouth_seeds(sea, side), pixel_m) / 1000``
    (float64, NaN where unreachable), or None when ``mouth_seeds`` finds no
    sea pixel on the mouth side (``reconstruct``'s ValueError branch).
    """
    sea = np.asarray(sea, bool)
    try:
        seeds = geometry.mouth_seeds(sea, side=mouth_side)
        s_m = geometry.along_distance(sea | np.asarray(inter, bool), seeds, pixel_m)
    except ValueError:
        return None
    # element by element the same as reconstruct's s_m.ravel()[keep] / 1000.0
    return s_m / 1000.0


def bands(s_km):
    """Bands along s (reconstruct's rule).

    Returns
    -------
    edges, centers, band_of, status
        ``te.make_bands_fixed(s_km, BAND_KM, MIN_PX_BAND)`` and "ok" when
        more than MIN_FINITE_S values are finite; otherwise no edges, one
        centre at 0, every pixel in band 0, "no_anchor".
    """
    s = np.asarray(s_km, np.float64)
    if int(np.count_nonzero(np.isfinite(s))) > MIN_FINITE_S:
        edges, centers, band_of = te.make_bands_fixed(s, BAND_KM, MIN_PX_BAND)
        return (np.asarray(edges, np.float64), np.asarray(centers, np.float64),
                np.asarray(band_of, np.int64), "ok")
    return np.zeros(0), np.array([0.0]), np.zeros(len(s), np.int64), "no_anchor"


def positions_in(dom, flat):
    """Positions of the flat indices ``flat`` in the sorted array ``dom``;
    KeyError if any is absent."""
    dom = np.asarray(dom, np.int64)
    flat = np.asarray(flat, np.int64)
    if not flat.size:
        return np.zeros(0, np.int64)
    p = np.searchsorted(dom, flat)
    ok = p < len(dom)
    ok[ok] = dom[p[ok]] == flat[ok]
    if not ok.all():
        raise KeyError(f"{int(np.count_nonzero(~ok))} of {flat.size} pixels are outside "
                       "the package domain")
    return p.astype(np.int64)


def assemble(mode, keep, sea, inter, pixel_m, mouth_side, dom=None, n_sel=0,
             with_frame=False):
    """A :class:`Domain` from its frame masks: geometry and bands (§2.6).

    Parameters
    ----------
    mode : str
        "A", "F" or "P".
    keep : (n,) int
        Intertidal flat indices, ascending.
    sea, inter : (H, W) bool
    pixel_m : float
        Grid spacing (m).
    mouth_side : str
        ``geometry.mouth_seeds`` side(s).
    dom : array, optional
        The package domain; ``pos`` is None without it.
    n_sel : int
        Scenes behind the counts.
    with_frame : bool
        Keep the frame-wide s (km) in ``s_frame``.
    """
    keep = np.asarray(keep, np.int64)
    sea = np.asarray(sea, bool)
    inter = np.asarray(inter, bool)
    if keep.size and np.any(np.diff(keep) <= 0):
        raise ValueError("keep must be ascending unique flat indices")
    pos = None if dom is None else positions_in(dom, keep)
    n = len(keep)
    if n < MIN_INTERTIDAL:
        # reconstruct stops here: no geometry, no bands
        return Domain(mode, keep, pos, sea, inter, s_km=np.full(n, np.nan),
                      band_of=np.zeros(n, np.int64), centers=np.array([0.0]),
                      edges=np.zeros(0), K=1, status="no_intertidal", n_sel=n_sel,
                      s_frame=np.full(sea.shape, np.nan) if with_frame else None)
    s_frame = s_frame_km(sea, inter, pixel_m, mouth_side)
    if s_frame is None:
        s_km = np.full(n, np.nan)
    else:
        s_km = s_frame.ravel()[keep]
    edges, centers, band_of, status = bands(s_km)
    if with_frame and s_frame is None:
        s_frame = np.full(sea.shape, np.nan)
    return Domain(mode, keep, pos, sea, inter, s_km=s_km, band_of=band_of,
                  centers=centers, edges=edges, K=len(centers), status=status,
                  n_sel=n_sel, s_frame=s_frame if with_frame else None)


def from_counts(clr, wet, pixel_m, mouth_side, dom=None, mode="F", n_sel=0,
                with_frame=False):
    """The domain of full-frame counts ``clr``, ``wet`` (H, W)."""
    clr = np.asarray(clr)
    inter, sea = classify(clr, wet)
    keep = np.flatnonzero(inter.ravel())
    return assemble(mode, keep, sea, inter, pixel_m, mouth_side, dom=dom,
                    n_sel=n_sel, with_frame=with_frame)


# ─────────────────────────────────────────────────────────────────────────────
#  full record and subsets
# ─────────────────────────────────────────────────────────────────────────────

def full(pkg, with_frame=False):
    """The full-record (F-mode) domain from the package's primary counters.

    ``pkg.full`` holds the copy the package stored at its geometry step; this
    recomputes it (the tests compare the two)."""
    return from_counts(pkg.clr_n, pkg.wet_n, pkg.cfg.pixel_m, pkg.cfg.mouth_side,
                       dom=pkg.dom, mode="F", n_sel=pkg.T, with_frame=with_frame)


def _sel_positions(sel, T):
    sel = np.asarray(sel)
    if sel.dtype == bool:
        if sel.shape != (T,):
            raise ValueError(f"a boolean sel must have length {T}")
        sel = np.flatnonzero(sel)
    sel = sel.astype(np.int64)
    if sel.ndim != 1 or (sel.size and (sel[0] < 0 or sel[-1] >= T
                                       or np.any(np.diff(sel) <= 0))):
        raise ValueError("sel must be sorted unique record positions in [0, T)")
    return sel


def frame_clear_counts(pkg, sel, rows_per_chunk=None):
    """Subset clear counts (primary rule and mode) of every frame pixel,
    from the packed frame validity bits: (H*W,) int32."""
    sel = _sel_positions(sel, pkg.T)
    packed = pkg.frame_valid.packed
    HW = int(pkg.H) * int(pkg.W)
    acc = np.zeros(HW, np.int32)
    if rows_per_chunk is None:
        # unpacked rows are one byte per pixel: about 64 MB per chunk
        rows_per_chunk = max(1, int(64e6 // max(HW, 1)))
    for a in range(0, len(sel), rows_per_chunk):
        rows = sel[a:a + rows_per_chunk]
        bits = np.unpackbits(np.asarray(packed[rows]), axis=1, count=HW)
        acc += bits.sum(axis=0, dtype=np.int32)
    return acc


def for_subset(pkg, sel, with_frame=False):
    """The A-mode domain of a record subset (§3.1).

    The extraction rule on the subset's own observations (primary rule and
    mode): counts at the package domain from the stored raw values; outside
    it, sea from the frame validity bits for the sea-like pixels (exact, see
    the module docstring). Asserts that both sources give the same clear
    counts inside the domain.

    Parameters
    ----------
    pkg : package.CommonPackage
    sel : array of record positions (sorted) or (T,) bool
    with_frame : bool
        Keep the frame-wide s (km) in ``s_frame``.
    """
    sel = _sel_positions(sel, pkg.T)
    dom = pkg.dom
    H, W = int(pkg.H), int(pkg.W)
    clr_d, wet_d = pkg.count(sel)
    clr_frame = frame_clear_counts(pkg, sel)
    if not np.array_equal(clr_frame[dom], clr_d):
        bad = int(np.count_nonzero(clr_frame[dom] != clr_d))
        raise AssertionError(f"the stored values and the frame validity bits disagree on "
                             f"{bad} package pixels: the package is inconsistent")
    inter_d, sea_d = classify(clr_d, wet_d)
    in_dom = np.zeros(H * W, bool)
    in_dom[dom] = True
    clr_n = pkg.clr_n.ravel().astype(np.int64)
    dry_n = clr_n - pkg.wet_n.ravel().astype(np.int64)
    sea_like = (clr_n >= MIN_CLEAR) & (dry_n <= SEA_LIKE_MAX_DRY) & ~in_dom
    del in_dom, dry_n, clr_n
    sea = np.zeros(H * W, bool)
    sea[dom[sea_d]] = True
    sea |= sea_like & (clr_frame >= MIN_CLEAR)
    del sea_like, clr_frame
    keep = dom[inter_d]
    inter = np.zeros(H * W, bool)
    inter[keep] = True
    return assemble("A", keep, sea.reshape(H, W), inter.reshape(H, W),
                    pkg.cfg.pixel_m, pkg.cfg.mouth_side, dom=dom, n_sel=len(sel),
                    with_frame=with_frame)


def held_at_full(sens):
    """True when a unit's ``sens`` changes the observation (mask rule or
    index mode): the domain is then held at the full record (§3.9)."""
    sens = dict(sens or {})
    return (sens.get("mask", obs.MASK_RULE) != obs.MASK_RULE
            or sens.get("index", obs.INDEX_MODE) != obs.INDEX_MODE)


def cache_path(pkg, subset_key):
    """Disk cache of an A-mode domain: ``runs/<site>/_domain/<subset_key>.npz``."""
    return f"{pkg.cfg.runs_dir}/_domain/{subset_key}.npz"


def _cache_meta(pkg, subset_key, n_sel):
    return {"format": DOMAIN_CACHE_FORMAT, "package_key": pkg.keys["package_key"],
            "domain_sha": config.sha256_source(__file__),
            "frozen_domain": config.frozen_sha("domain"),
            "subset_key": subset_key, "n_sel": int(n_sel)}


def save_cached(path, d, meta):
    """Write an A-mode domain to its disk cache (atomic, no s_frame)."""
    import json

    from experiments.benchmark import product

    product.save_npz(path, {
        "meta": np.array(json.dumps(meta, sort_keys=True)),
        "keep": d.keep, "sea": np.packbits(d.sea.ravel()),
        "shape": np.asarray(d.sea.shape, np.int64),
        "s_km": d.s_km, "band_of": d.band_of, "centers": d.centers, "edges": d.edges,
        "K": np.array(d.K), "status": np.array(d.status), "n_sel": np.array(d.n_sel),
    })


def load_cached(path, meta, dom=None, mode="A"):
    """The cached domain at ``path`` if its bookkeeping equals ``meta``,
    else None (``mode``: the Domain's mode label, "A" by default)."""
    import json

    from experiments.benchmark import product

    if not os.path.exists(path):
        return None
    try:
        z = product.load_npz(path)
        if json.loads(str(z["meta"])) != meta:
            return None
        H, W = (int(v) for v in z["shape"])
        sea = np.unpackbits(z["sea"], count=H * W).astype(bool).reshape(H, W)
        keep = z["keep"].astype(np.int64)
        inter = np.zeros(H * W, bool)
        inter[keep] = True
        return Domain(mode, keep, None if dom is None else positions_in(dom, keep), sea,
                      inter.reshape(H, W), s_km=z["s_km"], band_of=z["band_of"],
                      centers=z["centers"], edges=z["edges"], K=int(z["K"]),
                      status=str(z["status"]), n_sel=int(z["n_sel"]))
    except (OSError, ValueError, KeyError):
        return None


def get(pkg, unit):
    """The domain of a unit: A-mode -> :func:`for_subset`, F-mode -> the full
    record (§0.5).

    The full record is also returned for the whole record in A-mode (the two
    coincide, tested) and for observation sensitivities (held at keep_full,
    §3.9). A-mode domains are cached per subset_key in memory and on disk
    (``runs/<site>/_domain/``, keyed by the package key and this module's
    source).
    """
    sel = _sel_positions(unit.sel, pkg.T)
    if unit.mode == "F" or held_at_full(unit.sens) or len(sel) == pkg.T:
        return pkg.full
    if unit.mode != "A":
        raise ValueError(f"unknown mode {unit.mode!r}")
    key = unit.subset_key
    mem = pkg.__dict__.setdefault("_domain_cache", {})
    if key in mem:
        return mem[key]
    meta = _cache_meta(pkg, key, len(sel))
    path = cache_path(pkg, key)
    d = load_cached(path, meta, dom=pkg.dom)
    if d is None:
        d = for_subset(pkg, sel)
        save_cached(path, d, meta)
        log(f"[domain] {pkg.site} {pkg.res_m}m A-mode N={len(sel)}: {d.status}, "
            f"{d.n_keep:,} keep px, K = {d.K} (cached {path})")
    mem[key] = d
    return d


# ─────────────────────────────────────────────────────────────────────────────
#  strata and the production geometry
# ─────────────────────────────────────────────────────────────────────────────

def raw_band_table(d):
    """Merged band of every raw band ``floor(s / BAND_KM)`` of a domain with
    status "ok" (the strata rule of §2.6): the band of the keep pixels in
    that raw band, else that of the nearest lower raw band with keep pixels,
    else 0."""
    if d.status != "ok":
        raise ValueError("raw bands exist only with status 'ok'")
    n_raw = len(d.edges) - 1
    fin = np.isfinite(d.s_km)
    raw = np.clip(np.floor(d.s_km[fin] / BAND_KM).astype(np.int64), -1, n_raw - 1)
    b = d.band_of[fin]
    table = np.full(n_raw, -1, np.int64)
    for r in np.unique(raw):
        vals = np.unique(b[raw == r])
        if len(vals) != 1:
            raise AssertionError(f"raw band {r} holds keep pixels of bands {vals.tolist()}")
        table[r] = vals[0]
    last = -1
    for r in range(n_raw):
        if table[r] >= 0:
            last = table[r]
        else:
            table[r] = last
    table[table < 0] = 0
    return table


def band_map_frame(d, s_frame):
    """Band of every frame pixel with finite s (the strata of §2.6).

    Parameters
    ----------
    d : Domain
        The full-record domain (its s_km, band_of and edges define the merge).
    s_frame : array (any shape), km
        s of the pixels to map; must be the float64 values the domain's s_km
        came from (a float32 copy can move a pixel across a raw-band edge).

    Returns
    -------
    int16 array of ``s_frame``'s shape
        The band; -1 where s is NaN ("no s"). With status "no_anchor" every
        finite s is band 0; with "no_intertidal" there are no bands (all -1
        where s is NaN, 0 elsewhere).
    """
    s = np.asarray(s_frame, np.float64)
    out = np.full(s.shape, -1, np.int16)
    fin = np.isfinite(s)
    if d.status != "ok":
        out[fin] = 0
        return out
    table = raw_band_table(d)
    raw = np.clip(np.floor(s[fin] / BAND_KM).astype(np.int64), 0, len(table) - 1)
    out[fin] = table[raw].astype(np.int16)
    return out


def production(pkg):
    """The production extraction's domain (mode "P", gates only): keep_prod,
    sea_prod, inter_prod with s_km and bands recomputed as reconstruct does
    (G-keep compares this s_km, as float32, with the stored one)."""
    if not len(pkg.keep_prod):
        raise ValueError(f"{pkg.site}: no production extraction (keep_prod is empty)")
    return assemble("P", pkg.keep_prod, pkg.sea_prod, pkg.inter_prod, pkg.cfg.pixel_m,
                    pkg.cfg.mouth_side, dom=pkg.dom, n_sel=0)


# ═════════════════════════════════════════════════════════════════════════════
#  VALIDATION V2: the harness domain of an instance (spec §2.9, §2.10)
# ═════════════════════════════════════════════════════════════════════════════

DOMAIN_V2_FORMAT = "v2-domain-1"
#: the base key of each frame-bit family (its sea-like counts)
BITS_BASE = {"P0": "C1.floor0.ndwi.D1", "C11": "C11.floor0.ndwi.D1"}


def _family_counts(pkg, family, wid):
    """Full-window frame counts of a frame-bit family's base key."""
    from experiments.benchmark import package
    years = pkg.window_years(wid)
    if family.startswith("C3p"):
        return package.window_counters(pkg.site2, years, "C3.floor0.ndwi.D1", int(family[3:]))
    return package.window_counters(pkg.site2, years, BITS_BASE[family])


def instance_selection(pkg, setting, wid, sel=None):
    """``(sel, whole)``: the instance's record positions (global, sorted) and
    whether they are the setting's whole window record (a scene rule's
    whole record is its kept subset, which is not 'whole')."""
    from experiments.benchmark import settings
    s = settings.Setting.parse(setting)
    pos_w = pkg.window(wid)
    if sel is None:
        sel = pkg.rule_subset(s.scene_rule, wid) if s.scene_rule else pos_w
    sel = _sel_positions(np.asarray(sel), pkg.T)
    if not np.isin(sel, pos_w).all():
        raise ValueError(f"the subset holds scenes outside window {wid}")
    return sel, bool(s.scene_rule is None and len(sel) == len(pos_w))


def setting_counts(pkg, setting, wid, sel=None):
    """The counts behind an instance's domain (spec §2.10).

    Returns
    -------
    dict
        ``kind`` "frame" with ``clr``, ``wet`` (H, W), or "dom" with
        ``clr``, ``wet`` at dom (int32), plus ``sel``, ``whole`` and
        ``source`` (how the counts were obtained).
    """
    from experiments.benchmark import package, settings
    s = settings.Setting.parse(setting)
    sel, whole = instance_selection(pkg, s, wid, sel)
    if whole:
        fc = pkg.frame_counts(s, wid)
        if fc is not None:
            return {"kind": "frame", "clr": fc[0], "wet": fc[1], "sel": sel, "whole": True,
                    "source": "full-frame uint8 year counters"}
        key = s.replace(cloud=s.pixel_rule, level="LB").id
        dc = package.dom_counters(pkg.site2, pkg.dom_version, pkg.window_years(wid), key)
        if dc is not None:
            return {"kind": "dom", "clr": dc[0], "wet": dc[1], "sel": sel, "whole": True,
                    "source": "at-dom year counters (settings pass)"}
    clr, wet = pkg.count(s, sel)
    return {"kind": "dom", "clr": clr, "wet": wet, "sel": sel, "whole": whole,
            "source": "package values at dom over the subset"}


def for_setting(pkg, setting, wid, sel=None, with_frame=False):
    """The harness domain of an instance (setting, window, subset) of a v2
    package (spec §2.10): the extraction rule on the setting's own counts;
    keep = its intertidal pixels inside dom.

    Parameters
    ----------
    pkg : package.V2Package
    setting : settings.Setting or str
    wid : str
        Window id.
    sel : array, optional
        Global record positions of a subset of the window (default the
        window's record; a scene rule's kept record for a scene rule).
    with_frame : bool
        Keep the frame-wide s.

    Returns
    -------
    Domain
        Mode "F" for a whole window, "A" for a subset, with the attribute
        ``notes`` (counts source, pixels dropped outside dom, the sea rule
        outside dom).
    """
    from experiments.benchmark import settings
    s = settings.Setting.parse(setting)
    c = setting_counts(pkg, s, wid, sel)
    H, W = pkg.shape
    dom = pkg.dom
    notes = {"counts": c["source"]}
    if c["kind"] == "frame":
        inter, sea = classify(c["clr"], c["wet"])
        keep_frame = np.flatnonzero(inter.ravel())
        keep = np.intersect1d(keep_frame, dom, assume_unique=True).astype(np.int64)
        notes["keep_outside_dom_dropped"] = int(len(keep_frame) - len(keep))
        notes["sea"] = "the setting's own frame counts"
        sea = sea.ravel()
    else:
        inter_d, sea_d = classify(c["clr"], c["wet"])
        keep = dom[inter_d]
        sea = np.zeros(H * W, bool)
        sea[dom[sea_d]] = True
        in_dom = np.zeros(H * W, bool)
        in_dom[dom] = True
        fam = pkg.bits_family(s)
        fc = _family_counts(pkg, fam, wid) if fam else None
        clr_bits = pkg.frame_clear_counts(fam, c["sel"]) if fc is not None else None
        if clr_bits is not None:
            clr_n = fc[0].ravel().astype(np.int64)
            dry_n = clr_n - fc[1].ravel().astype(np.int64)
            sea_like = (clr_n >= MIN_CLEAR) & (dry_n <= SEA_LIKE_MAX_DRY) & ~in_dom
            sea |= sea_like & (clr_bits.ravel() >= MIN_CLEAR)
            base = BITS_BASE.get(fam, "C3.floor0.ndwi.D1")
            exact = s.replace(cloud=s.pixel_rule, level="LB").id == base
            notes["sea"] = (f"outside dom: sea-like rule on the {fam} bits "
                            + ("(exact)" if exact else "(the family's sea; declared)"))
        else:
            p0 = pkg.frame_counts(settings.P0, wid)
            _, sea_p0 = classify(*p0)
            sea |= sea_p0.ravel() & ~in_dom
            notes["sea"] = "outside dom: P0's full-window sea (declared)"
        notes["keep_outside_dom_dropped"] = 0
    inter = np.zeros(H * W, bool)
    inter[keep] = True
    d = assemble("F" if c["whole"] else "A", keep, sea.reshape(H, W), inter.reshape(H, W),
                 pkg.pixel_m, pkg.cfg.mouth_side, dom=dom, n_sel=len(c["sel"]),
                 with_frame=with_frame)
    d.notes = notes
    return d


def cache_key_v2(setting, wid, subset_key):
    """sha256 of (setting id, window, subset_key): the domain cache key."""
    from experiments.benchmark import settings
    return config.sha256_json([settings.Setting.parse(setting).id,
                               settings.window(wid).id, str(subset_key)])


def get_v2(pkg, setting, wid, sel=None):
    """:func:`for_setting` cached in memory and on disk
    (``common/<site>/_domain/<dom_version>/<key>.npz``), keyed by (setting,
    window, subset) and checked against the shards' package keys and this
    module's source."""
    import json

    from experiments.benchmark import package, product, settings
    s = settings.Setting.parse(setting)
    sel2, whole = instance_selection(pkg, s, wid, sel)
    sk = product.subset_key(sel2)
    key = cache_key_v2(s, wid, sk)
    mem = pkg.__dict__.setdefault("_domain_cache", {})
    if key in mem:
        return mem[key]
    meta = {"format": DOMAIN_V2_FORMAT, "setting": s.id, "window": settings.window(wid).id,
            "subset_key": sk, "dom_version": pkg.dom_version,
            "shard_keys": {str(k): v for k, v in package.shard_keys(
                pkg.site2, pkg.dom_version, pkg.window_years(wid)).items()},
            "domain_sha": config.sha256_source(__file__),
            "frozen_domain": config.frozen_sha("domain"), "n_sel": int(len(sel2))}
    path = os.path.join(pkg.dir, "_domain", str(pkg.dom_version), f"{key}.npz")
    mode = "F" if whole else "A"
    d = load_cached(path, meta, dom=pkg.dom, mode=mode)
    if d is None:
        d = for_setting(pkg, s, wid, sel2)
        save_cached(path, d, meta)
        log(f"[domain] {pkg.site} {s.id} @{wid} n={len(sel2)}: {d.status}, {d.n_keep:,} keep "
            f"px, K = {d.K} ({json.dumps(getattr(d, 'notes', {}))})")
    mem[key] = d
    return d
