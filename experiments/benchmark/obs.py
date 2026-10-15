# -*- coding: utf-8 -*-
"""v13 benchmark and validation v2 — THE observation every method receives.

Question
--------
What is one observation of one pixel in one scene? Every method of the
benchmark (no delay, MAREA, NIDEM, Granadeiro, GM20, HD26) must see the same
per-pixel cloud mask and the same band preprocessing, and compute its own
index from them. This module is the single definition; no adapter masks or
preprocesses bands in any other way.

Pre-registered rules (FROZEN['obs']; authors' decisions D1 and D2 of
2026-10-08)
--------------------------------------------------------------------------
* Encoding: B03, B08 and SCL are int16 with ``_FillValue = -32768`` and no
  scale or offset; band values are offset-harmonised L2A (value = DN - 1000,
  so reflectance can be negative; nothing is below -1000).
* Per-pixel cloud mask, a run-time option (the package stores SCL, never a
  mask):
  - ``marea_clear`` (PRIMARY, D1): valid iff SCL in {4, 5, 6, 7}, MAREA's
    production mask (``pyintertidal.marea.CLEAR``);
  - ``blocklist`` (sensitivity): valid iff SCL is present and not in
    {0, 3, 8, 9, 10} (``(0, *pyintertidal.water.BAD_CLASSES)``).
  Every ``marea_clear`` observation is also a ``blocklist`` one, which is
  why the package domain is built under ``blocklist``.
* Band preprocessing, the same for every method:
  - ``floor0`` (PRIMARY, D2): B03 and B08 floored at 0 (non-fill values)
    before any index, so NDWI lies in [-1, 1] and dark water with B08 < 0
    no longer flips to "land";
  - ``raw`` (sensitivity): the cube values unchanged (production).
* NDWI = ``np.where(g + n != 0, (g - n) / (g + n), nan)`` in float32 on the
  decoded (and floored) bands: bit-identical to ``marea.extract`` and
  ``waterline_contour.scene_ndwi`` in raw mode.
* Valid observation V = mask & isfinite(NDWI), for every method; wet =
  V & (NDWI > 0).
* NIR (Granadeiro) = ``granadeiro2021.to_reflectance`` of the floored int16
  B08 (value / 10000, float64, fill -> NaN). In common mode the observations
  outside V are set to the int16 fill first (:func:`mask_raw_bands`).
* ``item_valid`` (ITEM's PQ mask) exists for the NIDEM gates only.

Validation v2 (docs/validation_v2_spec.md §2.2-2.6): the observation of a
SETTING (cloud, prep, index, detect; ``settings.Setting``)
---------------------------------------------------------------------------
* Bands come as a dict ``{band: raw int16 (..., H, W)}`` (SCL and CLD may
  be the uint8 of :func:`scl_u8` / :func:`cld_u8`); ``observe(raw,
  setting)`` (or :func:`observe_setting`) returns ``Observation(X, Y, V,
  W)``: X the oriented index (water high), Y = (X - t_k) / s_k (Y IS X for
  NDWI), V = pixel cloud rule & every needed band present & finite X, W =
  V & the detector says water.
* Preprocessing (:func:`prep_band`): DN - o_s per scene (o_s in {0, 1000},
  ``inputs``), floored at 0 under floor0; normalised differences and WRI are
  computed on that DN scale (scale-free, and bit-identical to production for
  NDWI), AWEI and NIR on rho = DN / 10000 (float32).
* THE NDWI PATH IS UNCHANGED: under (C1, floor0 or raw, ndwi, D1) with
  o_s = 0, X is :func:`ndwi` of :func:`bands`, V is :func:`observe`'s V and
  W is :func:`wet`, bit for bit.
* 20 m rule (:func:`compute_index`): an index with a 20 m band is computed
  on 2 x 2 blocks of the native 20 m lattice (phase ``lattice20`` = (py,
  px): rows r and r + 1 form a block iff r = py mod 2): 10 m bands
  averaged (:func:`block_mean20`, NaN if any in-frame pixel is NaN; edge
  blocks average their in-frame pixels), 20 m bands taken once per block
  (:func:`block_take20`), the index replicated to 10 m (:func:`expand20`);
  ``lattice=None`` (not exact replicas): box means at phase (0, 0) for
  every band (declared).
* Pixel cloud rules ``cloud_<level>`` (C0, C1, C1d, C1s, C3, C4, C5, C11)
  and the per-scene keep decisions of the scene rules ``scene_c2``,
  ``scene_c6``, ``scene_c7``, ``scene_c8``; detectors ``detect_d1``,
  ``detect_threshold`` (D2, D3 with the per-scene thresholds of
  :func:`d2_threshold` / :func:`d3_threshold`), ``detect_d4``,
  ``detect_d5``; OmniCloudMask through :func:`ocm_classes` (the model is
  injected for tests and imported lazily otherwise).

Hashes: :func:`source_sha256` hashes the whole file (the v13 ``obs_sha``);
:func:`core_sha256` hashes only the observation core (decoding, masks of
v13, NDWI, the dispatchers), which is what the v2 package key needs; each
registry entry's own functions are hashed per entry by
``settings.entry_sha``.
"""
from __future__ import annotations

import collections
import hashlib
import inspect
import json
import os
from collections.abc import Mapping

import numpy as np

from experiments.benchmark import config, settings
from pyintertidal import granadeiro2021 as g21
from pyintertidal import marea, water
from pyintertidal import waterline_contour as wc

#: SCL masked by the blocklist rule (no data included)
BLOCK_SCL = (0, 3, 8, 9, 10)
#: SCL accepted by the marea_clear rule
CLEAR_SCL = (4, 5, 6, 7)
#: SCL masked by ITEM's pixel-quality mask (gates only)
ITEM_MASKED_SCL = (0, 1, 3, 8, 9, 10)
#: the Sentinel-2 L2A scene classes
SCL_CODES = tuple(range(12))
#: fill / NaN SCL in the uint8 arrays of the package
SCL_NODATA_U8 = 255
MASK_RULES = ("blocklist", "marea_clear")
INDEX_MODES = ("floor0", "raw")
#: C2 (authors, final): the production MAREA mask; "blocklist" = sensitivity
MASK_RULE = "marea_clear"
#: [D2]; "raw" = production behaviour, sensitivity
INDEX_MODE = "floor0"
#: the cubes' fill value (asserted against each cube's encoding elsewhere)
FILL = -32768
#: smallest valid band value: DN 0 is the fill, so DN - 1000 >= -999; v8
#: asserts >= -1000
BOA_MIN_VALUE = -1000

# the definitions must be the production ones and FROZEN's; a mismatch is a
# bug that would silently change every method, so it stops the import
if BLOCK_SCL != (0, *water.BAD_CLASSES):
    raise AssertionError(f"BLOCK_SCL {BLOCK_SCL} != (0, *water.BAD_CLASSES)")
if CLEAR_SCL != tuple(marea.CLEAR):
    raise AssertionError(f"CLEAR_SCL {CLEAR_SCL} != marea.CLEAR {marea.CLEAR}")
if ITEM_MASKED_SCL != tuple(wc.ITEM_PQ_MASKED_SCL):
    raise AssertionError("ITEM_MASKED_SCL differs from wc.ITEM_PQ_MASKED_SCL")
_F = config.FROZEN["obs"]
if (_F["primary_mask"], _F["primary_index"]) != (MASK_RULE, INDEX_MODE):
    raise AssertionError("obs primary rule/mode differ from FROZEN['obs']")
if (tuple(_F["mask_rules"]["marea_clear"]["valid_scl"]) != CLEAR_SCL
        or tuple(_F["mask_rules"]["blocklist"]["masked_scl"]) != BLOCK_SCL):
    raise AssertionError("obs SCL classes differ from FROZEN['obs']")
if (_F["encoding"]["fill"], _F["encoding"]["boa_min_value"],
        _F["encoding"]["scl_nodata_u8"]) != (FILL, BOA_MIN_VALUE, SCL_NODATA_U8):
    raise AssertionError("obs encoding constants differ from FROZEN['obs']")
if g21.REFLECTANCE_SCALE != 10000.0:
    raise AssertionError("g21.REFLECTANCE_SCALE is not 10000")
del _F


def _lut(valid_codes):
    t = np.zeros(256, bool)
    t[list(valid_codes)] = True
    return t


#: valid flag per uint8 SCL value (index 255 = no data, never valid)
_MASK_LUT = {
    "blocklist": _lut(c for c in SCL_CODES if c not in BLOCK_SCL),
    "marea_clear": _lut(CLEAR_SCL),
}
_ITEM_LUT = _lut(c for c in SCL_CODES if c not in ITEM_MASKED_SCL)
for _t in (*_MASK_LUT.values(), _ITEM_LUT):
    _t.flags.writeable = False
del _t


def check_rule(rule):
    """Raise ValueError unless ``rule`` is a mask rule."""
    if rule not in MASK_RULES:
        raise ValueError(f"unknown mask rule {rule!r}; expected one of {MASK_RULES}")
    return rule


def check_mode(mode):
    """Raise ValueError unless ``mode`` is an index mode."""
    if mode not in INDEX_MODES:
        raise ValueError(f"unknown index mode {mode!r}; expected one of {INDEX_MODES}")
    return mode


def rule_and_mode(sens=None):
    """The (mask rule, index mode) of a unit's sensitivity setting.

    Parameters
    ----------
    sens : dict, optional
        The unit's ``sens`` ({} or None = the controlled setting). Only the
        keys ``mask`` and ``index`` are read; other keys belong to the
        adapters (e.g. ``floor``).

    Returns
    -------
    (rule, mode) : tuple of str
    """
    sens = dict(sens or {})
    return (check_rule(sens.get("mask", MASK_RULE)),
            check_mode(sens.get("index", INDEX_MODE)))


# ─────────────────────────────────────────────────────────────────────────────
#  decoding
# ─────────────────────────────────────────────────────────────────────────────

def decode(raw, fill=FILL):
    """Raw int16 cube values -> float32 with the fill as NaN.

    Equals xarray's default CF decoding of an int16 band with ``_FillValue``
    and no scale (``open_dataset(...)[band].values``), which is what
    ``marea.extract`` and ``waterline_contour.stream_composites`` read.
    """
    raw = np.asarray(raw)
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"decode takes raw integer cube values, not {raw.dtype}")
    out = raw.astype(np.float32)
    if fill is not None:
        out[raw == fill] = np.nan
    return out


def scl_u8(scl_raw, fill=FILL):
    """SCL as uint8 with the fill (or NaN) as :data:`SCL_NODATA_U8`.

    Accepts raw integer SCL with ``fill`` or xarray-decoded float SCL with
    NaN. Raises ValueError if a present value is not an SCL code 0..11 (as
    ``v8.scan_scenes``): a silent wrap would change the mask.
    """
    s = np.asarray(scl_raw)
    if np.issubdtype(s.dtype, np.floating):
        present = np.isfinite(s)
        if fill is not None:
            present &= s != fill
        floating = True
    elif np.issubdtype(s.dtype, np.integer):
        present = (s != fill) if fill is not None else np.ones(s.shape, bool)
        floating = False
    else:
        raise TypeError(f"scl_u8 takes integer or float SCL, not {s.dtype}")
    v = s[present]
    if v.size:
        lo, hi = v.min(), v.max()
        if lo < 0 or hi > SCL_CODES[-1] or (floating and np.any(v != np.floor(v))):
            bad = np.unique(v[(v < 0) | (v > SCL_CODES[-1]) | (v != np.floor(v))])
            raise ValueError(f"SCL values outside the codes 0..11: {bad[:10].tolist()}")
    out = np.full(s.shape, SCL_NODATA_U8, np.uint8)
    out[present] = v.astype(np.uint8)
    return out


def scl_counts(scl):
    """Counts of SCL codes 0..11 and of no data (index 12) in a uint8 SCL."""
    s = _as_u8(scl)
    bc = np.bincount(s.ravel(), minlength=256)
    other = np.delete(bc, list(SCL_CODES) + [SCL_NODATA_U8])
    if other.any():
        raise ValueError("uint8 SCL holds values that are neither codes 0..11 nor 255")
    return np.append(bc[list(SCL_CODES)], bc[SCL_NODATA_U8]).astype(np.int64)


def check_boa(raw, fill=FILL, name="band"):
    """Assert the BOA-offset invariant of a raw band (no non-fill value below
    -1000, as ``v8.load_bands``). Returns the number of non-fill values."""
    raw = np.asarray(raw)
    v = raw[raw != fill] if fill is not None else raw.ravel()
    if v.size and int(v.min()) < BOA_MIN_VALUE:
        raise AssertionError(f"{name} values below {BOA_MIN_VALUE}: "
                             "not DN - 1000 harmonised data")
    return int(v.size)


def _as_u8(scl):
    s = np.asarray(scl)
    if s.dtype != np.uint8:
        raise TypeError(f"expected the uint8 SCL of scl_u8 (no data = "
                        f"{SCL_NODATA_U8}), got {s.dtype}")
    return s


# ─────────────────────────────────────────────────────────────────────────────
#  masks
# ─────────────────────────────────────────────────────────────────────────────

def mask_valid(scl_u8, rule=MASK_RULE):
    """Per-pixel cloud mask under ``rule`` (True = usable).

    ``blocklist``: SCL present and not in {0, 3, 8, 9, 10}; ``marea_clear``:
    SCL in {4, 5, 6, 7}. No data (255) is never valid.
    """
    return _MASK_LUT[check_rule(rule)][_as_u8(scl_u8)]


def item_valid(scl_u8):
    """ITEM's pixel-quality mask (gates only): equals
    ``waterline_contour.pixel_valid(decoded_scl, rule="item")``."""
    return _ITEM_LUT[_as_u8(scl_u8)]


# ─────────────────────────────────────────────────────────────────────────────
#  bands, index, observation
# ─────────────────────────────────────────────────────────────────────────────

def bands(b03_raw, b08_raw, fill=FILL, mode=INDEX_MODE):
    """Decoded float32 B03 and B08 after the common preprocessing.

    ``floor0``: ``np.maximum(x, 0)`` on the decoded values (NaN, i.e. the
    fill, stays NaN); ``raw``: decoded values unchanged.
    """
    check_mode(mode)
    g, n = decode(b03_raw, fill), decode(b08_raw, fill)
    if mode == "floor0":
        zero = np.float32(0.0)
        with np.errstate(invalid="ignore"):
            np.maximum(g, zero, out=g)          # decode's copies: safe in place
            np.maximum(n, zero, out=n)
    return g, n


def ndwi(g, n):
    """NDWI in float32, NaN where B03 + B08 == 0 or a band is NaN.

    Bit-identical to ``np.where(g + n != 0, (g - n) / (g + n), np.nan)`` of
    ``marea.extract`` and ``waterline_contour.scene_ndwi`` (float32
    throughout): the same float32 subtraction, sum and division, with NaN
    written where the sum is 0; computed with two temporaries instead of
    four, because blocks of (scenes x 10^5 pixels) are common here.
    """
    g = np.asarray(g, np.float32)
    n = np.asarray(n, np.float32)
    s = g + n
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.subtract(g, n)
        np.divide(out, s, out=out)
    out[s == 0] = np.nan
    return out


def observe(b03_raw, b08_raw=None, scl_u8=None, fill=FILL, rule=MASK_RULE, mode=INDEX_MODE,
            **kw):
    """The observation of a scene (or block): ``(ndwi, V)``.

    Validation v2 form: ``observe(raw, setting, **kw)`` with ``raw`` a dict
    of raw bands returns :func:`observe_setting`'s ``Observation(X, Y, V,
    W)``; the keywords are those of :func:`observe_setting`.

    Parameters
    ----------
    b03_raw, b08_raw : int16 arrays
        Raw cube values (``mask_and_scale=False``).
    scl_u8 : uint8 array
        SCL from :func:`scl_u8`, same shape.
    fill : int
        The bands' fill value.
    rule, mode : str
        Mask rule and index mode.

    Returns
    -------
    ndwi : float32 array
        NDWI of every pixel, NOT masked (NaN where undefined).
    V : bool array
        ``mask_valid(scl_u8, rule) & isfinite(ndwi)``.
    """
    if isinstance(b03_raw, Mapping):
        return observe_setting(b03_raw, b08_raw, fill=fill, **kw)
    if kw:
        raise TypeError(f"observe: unexpected keywords {sorted(kw)} (v13 form)")
    g, n = bands(b03_raw, b08_raw, fill, mode)
    nd = ndwi(g, n)
    return nd, mask_valid(scl_u8, rule) & np.isfinite(nd)


def observe_rules(b03_raw, b08_raw, scl_u8, fill=FILL, mode=INDEX_MODE, rules=MASK_RULES):
    """NDWI once and V under several mask rules: ``(ndwi, {rule: V})``.

    Each V equals :func:`observe`'s for that rule. The package's first pass
    needs two rules per frame (U_rec under blocklist; counters and frame bits
    under the primary rule), and the frame NDWI is the costly part.
    """
    g, n = bands(b03_raw, b08_raw, fill, mode)
    nd = ndwi(g, n)
    del g, n
    fin = np.isfinite(nd)
    return nd, {rule: mask_valid(scl_u8, rule) & fin for rule in rules}


def wet(ndwi, V):
    """Wet observation: ``V & (ndwi > 0)``. Under floor0 this equals
    ``marea.extract``'s own wet test (B03 > B08) wherever NDWI is defined."""
    with np.errstate(invalid="ignore"):
        return np.asarray(V, bool) & (np.asarray(ndwi) > 0)


# ─────────────────────────────────────────────────────────────────────────────
#  raw-value paths (Granadeiro)
# ─────────────────────────────────────────────────────────────────────────────

def floor_raw(raw, fill=FILL):
    """floor0 on raw integer values: ``max(value, 0)``, the fill kept.

    Granadeiro's int16 path: ``to_reflectance(floor_raw(x))`` equals the
    float path of :func:`bands` divided by 10000.
    """
    raw = np.asarray(raw)
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"floor_raw takes raw integer cube values, not {raw.dtype}")
    floored = np.maximum(raw, raw.dtype.type(0))
    if fill is None:
        return floored
    return np.where(raw == fill, raw, floored).astype(raw.dtype, copy=False)


def reflectance(raw, fill=FILL, mode=INDEX_MODE):
    """L2A reflectance (float64, fill -> NaN) after the common preprocessing:
    ``g21.to_reflectance(floor_raw(raw))`` (floor0) or of ``raw`` (raw)."""
    check_mode(mode)
    raw = np.asarray(raw)
    return g21.to_reflectance(floor_raw(raw, fill) if mode == "floor0" else raw, fill)


def nir(b08_raw, fill=FILL, mode=INDEX_MODE):
    """Granadeiro's index: the B08 reflectance (:func:`reflectance`)."""
    return reflectance(b08_raw, fill, mode)


def mask_raw_bands(g_raw, n_raw, V, fill=FILL):
    """Common mode for raw-value readers: B03, B08 := fill where not V.

    Works IN PLACE on writable integer arrays of V's shape, so every
    Granadeiro step treats a masked observation exactly as no data.

    Returns
    -------
    int
        Observations the mask removed: positions outside V where at least
        one band still held data.
    """
    V = np.asarray(V, bool)
    for name, a in (("B03", g_raw), ("B08", n_raw)):
        if not isinstance(a, np.ndarray) or a.shape != V.shape:
            raise ValueError(f"{name} must be an ndarray of shape {V.shape}")
        if not np.issubdtype(a.dtype, np.integer):
            raise TypeError(f"{name} must hold raw integer values, not {a.dtype}")
        if not a.flags.writeable:
            raise ValueError(f"{name} is read-only; mask_raw_bands works in place")
    bad = ~V
    removed = int(np.count_nonzero(bad & ((g_raw != fill) | (n_raw != fill))))
    g_raw[bad] = fill
    n_raw[bad] = fill
    return removed


def source_sha256():
    """sha256 of this module's source (CRLF normalised): the ``obs_sha`` of
    every package manifest."""
    return config.sha256_source(__file__)


# ═════════════════════════════════════════════════════════════════════════════
#  VALIDATION V2: the observation of a setting (spec §2.2-2.6)
# ═════════════════════════════════════════════════════════════════════════════

#: the spectral bands of the Zarr stores, in store order
SPECTRAL_BANDS = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")
#: the bands delivered at 20 m and replicated onto the 10 m grid
BANDS_20M = ("B8A", "B11", "B12", "SCL", "CLD")
#: CLD (cloud probability, %) fill in the uint8 arrays
CLD_NODATA_U8 = 255
#: SCL water class (D4)
SCL_WATER = 6
#: the reflectance scale of every index (fixed; never data-dependent)
REFLECTANCE_SCALE = settings.REFLECTANCE_SCALE
#: OmniCloudMask classes (package README: 0 clear, 1 thick cloud, 2 thin
#: cloud, 3 cloud shadow); 255 = no input data in :func:`ocm_classes`
OCM_CLASSES = {0: "clear", 1: "thick cloud", 2: "thin cloud", 3: "cloud shadow"}
OCM_NODATA_U8 = 255

#: the four observation layers of an instance (§2.5)
Observation = collections.namedtuple("Observation", "X Y V W")


def cld_u8(cld_raw, fill=FILL):
    """CLD (int16 cloud probability, %) as uint8 with the fill as
    :data:`CLD_NODATA_U8`; a present value outside 0..100 raises ValueError.
    A uint8 array is checked and returned as is."""
    c = np.asarray(cld_raw)
    if c.dtype == np.uint8:
        bad = (c > 100) & (c != CLD_NODATA_U8)
        if bad.any():
            raise ValueError(f"uint8 CLD values outside 0..100: {np.unique(c[bad])[:10].tolist()}")
        return c
    if np.issubdtype(c.dtype, np.floating):
        present = np.isfinite(c)
        if fill is not None:
            present &= c != fill
    elif np.issubdtype(c.dtype, np.integer):
        present = (c != fill) if fill is not None else np.ones(c.shape, bool)
    else:
        raise TypeError(f"cld_u8 takes integer or float CLD, not {c.dtype}")
    v = c[present]
    if v.size and (v.min() < 0 or v.max() > 100 or np.any(v != np.floor(v))):
        bad = np.unique(v[(v < 0) | (v > 100) | (v != np.floor(v))])
        raise ValueError(f"CLD values outside the percentages 0..100: {bad[:10].tolist()}")
    out = np.full(c.shape, CLD_NODATA_U8, np.uint8)
    out[present] = v.astype(np.uint8)
    return out


def _as_scl8(scl, fill=FILL):
    s = np.asarray(scl)
    return s if s.dtype == np.uint8 else scl_u8(s, fill)


# ─────────────────────────────────────────────────────────────────────────────
#  band preprocessing (§2.3)
# ─────────────────────────────────────────────────────────────────────────────

def _offset_array(offset, lead_shape):
    """The per-scene offset as an int (one value) or a float32 array of
    shape ``lead_shape + (1, 1)`` (one value per scene); only 0 and 1000."""
    o = np.asarray(offset)
    if not np.isin(o, settings.OFFSETS).all():
        raise ValueError(f"offsets must be in {settings.OFFSETS}, got {np.unique(o)[:5].tolist()}")
    if o.ndim == 0:
        return int(o)
    if o.shape != tuple(lead_shape):
        raise ValueError(f"one offset per scene: shape {o.shape} != {tuple(lead_shape)}")
    return o.reshape(tuple(lead_shape) + (1, 1)).astype(np.float32)


def prep_band(raw, offset=0, prep="floor0", fill=FILL):
    """One band after the preprocessing ``prep`` of ``settings.PREP``.

    float32 ``value - o_s`` (DN scale, the fill as NaN), floored at 0 under
    floor0. With o_s = 0 this is :func:`bands`'s path exactly (``decode``
    then ``np.maximum(x, 0)``), so the NDWI stays bit-identical to
    production. ``offset``: a scalar or one value per scene (shape
    ``raw.shape[:-2]``), each 0 or 1000.
    """
    if prep not in settings.PREP:
        raise ValueError(f"unknown preprocessing {prep!r}; expected one of {tuple(settings.PREP)}")
    out = decode(raw, fill)
    o = _offset_array(offset, out.shape[:-2])
    if np.any(o):
        out -= o                                   # exact: |values| < 2**24
    if settings.PREP[prep].floor:
        with np.errstate(invalid="ignore"):
            np.maximum(out, np.float32(0.0), out=out)
    return out


def prepare(raw, bands_needed, offset=0, prep="floor0", fill=FILL):
    """``{band: prep_band(raw[band])}`` for the spectral bands needed."""
    missing = [b for b in bands_needed if b not in raw]
    if missing:
        raise KeyError(f"bands {missing} are missing from the input (have {sorted(raw)})")
    return {b: prep_band(raw[b], offset, prep, fill) for b in bands_needed}


# ─────────────────────────────────────────────────────────────────────────────
#  the 20 m rule (§2.4)
# ─────────────────────────────────────────────────────────────────────────────

def _block_index(n, p):
    """Block id of each of ``n`` rows (or columns) on lattice phase ``p``:
    ``(i + p) // 2`` (rows r and r + 1 share a block iff r = p mod 2), and
    the number of blocks."""
    idx = (np.arange(int(n)) + int(p)) // 2
    return idx, (int(idx[-1]) + 1 if len(idx) else 0)


def _phase(lattice):
    """``((py, px), exact)``; None -> box means at phase (0, 0), not exact."""
    if lattice is None:
        return (0, 0), False
    py, px = (int(v) for v in lattice)
    if py not in (0, 1) or px not in (0, 1):
        raise ValueError(f"lattice phase must be in {{0, 1}}^2, got {lattice}")
    return (py, px), True


def block_mean20(a, phase=(0, 0)):
    """2 x 2 block means of ``a`` (..., H, W) on the lattice ``phase``.

    float32 of (sum of the block's in-frame pixels in float64) / (their
    count): exact for the integer-valued bands. A NaN in a block makes the
    block NaN; blocks cut by the frame edge average the pixels they have.
    """
    a = np.asarray(a, np.float32)
    py, px = (int(v) for v in phase)
    H, W = a.shape[-2:]
    pads = [(py, (-(H + py)) % 2), (px, (-(W + px)) % 2)]
    lead = a.shape[:-2]
    ap = np.pad(a.astype(np.float64), [(0, 0)] * len(lead) + pads, constant_values=0.0)
    w = np.pad(np.ones((H, W), np.float64), pads, constant_values=0.0)
    Hb, Wb = ap.shape[-2] // 2, ap.shape[-1] // 2
    s = ap.reshape(lead + (Hb, 2, Wb, 2)).sum(axis=(-3, -1))
    n = w.reshape(Hb, 2, Wb, 2).sum(axis=(1, 3))
    return (s / n).astype(np.float32)


def block_take20(a, phase=(0, 0)):
    """One value per 2 x 2 block (its first in-frame pixel): the inverse of
    the nearest-neighbour replication of a 20 m band on its lattice."""
    a = np.asarray(a)
    py, px = (int(v) for v in phase)
    H, W = a.shape[-2:]
    _, Hb = _block_index(H, py)
    _, Wb = _block_index(W, px)
    rows = np.maximum(2 * np.arange(Hb) - py, 0)
    cols = np.maximum(2 * np.arange(Wb) - px, 0)
    return a[..., rows[:, None], cols[None, :]]


def expand20(b, phase, shape):
    """Replicate block values ``b`` (..., Hb, Wb) to the 10 m frame ``shape``
    (H, W) on the lattice ``phase``."""
    b = np.asarray(b)
    H, W = (int(v) for v in shape)
    ry, Hb = _block_index(H, phase[0])
    rx, Wb = _block_index(W, phase[1])
    if b.shape[-2:] != (Hb, Wb):
        raise ValueError(f"blocks {b.shape[-2:]} do not tile a {H} x {W} frame at phase {phase}")
    return b[..., ry[:, None], rx[None, :]]


# ─────────────────────────────────────────────────────────────────────────────
#  indices (§2.4); every index oriented so that water is high
# ─────────────────────────────────────────────────────────────────────────────

def _rho(a):
    """Reflectance in float32: DN-scale value / 10000."""
    return np.asarray(a, np.float32) / np.float32(REFLECTANCE_SCALE)


def index_ndwi(b):
    """I1 NDWI = :func:`ndwi` (B03, B08): production's arithmetic."""
    return ndwi(b["B03"], b["B08"])


def index_mndwi(b):
    """I2 MNDWI = (B03 - B11) / (B03 + B11), the arithmetic of :func:`ndwi`."""
    return ndwi(b["B03"], b["B11"])


def index_ndwi20(b):
    """I7 NDWI@20 = (B03 - B8A) / (B03 + B8A)."""
    return ndwi(b["B03"], b["B8A"])


def index_nir(b):
    """I3 NIR = -rho8 (water high)."""
    out = _rho(b["B08"])
    np.negative(out, out=out)
    return out


def index_awei_sh(b):
    """I4 AWEI_sh = rho2 + 2.5 rho3 - 1.5 (rho8 + rho11) - 0.25 rho12."""
    r2, r3, r8, r11, r12 = (_rho(b[k]) for k in ("B02", "B03", "B08", "B11", "B12"))
    out = r2 + np.float32(2.5) * r3
    out -= np.float32(1.5) * (r8 + r11)
    out -= np.float32(0.25) * r12
    return out


def index_awei_nsh(b):
    """I5 AWEI_nsh = 4 (rho3 - rho11) - (0.25 rho8 + 2.75 rho12)."""
    r3, r8, r11, r12 = (_rho(b[k]) for k in ("B03", "B08", "B11", "B12"))
    out = np.float32(4.0) * (r3 - r11)
    out -= np.float32(0.25) * r8 + np.float32(2.75) * r12
    return out


def index_wri(b):
    """I6 WRI = (B03 + B04) / (B08 + B11) clipped to [0, 10] (NaN where the
    denominator is 0); scale-free, so computed on the DN scale."""
    num = np.asarray(b["B03"], np.float32) + np.asarray(b["B04"], np.float32)
    den = np.asarray(b["B08"], np.float32) + np.asarray(b["B11"], np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.divide(num, den)
    out[den == 0] = np.nan
    np.clip(out, np.float32(0.0), np.float32(10.0), out=out)
    return out


#: the index functions by registry id
INDEX_FUNCS = {"ndwi": index_ndwi, "mndwi": index_mndwi, "nir": index_nir,
               "awei_sh": index_awei_sh, "awei_nsh": index_awei_nsh, "wri": index_wri,
               "ndwi20": index_ndwi20}


def compute_index(prepared, index, lattice=(0, 0)):
    """The index ``index`` of ``settings.INDEX`` from prepared bands.

    Parameters
    ----------
    prepared : dict
        ``{band: float32 (..., H, W)}`` from :func:`prepare`.
    index : str
    lattice : (py, px) or None
        The native 20 m lattice phase (``inputs.SceneSource.lattice20``);
        None: box means at phase (0, 0) for every band (declared).

    Returns
    -------
    float32 array (..., H, W)
    """
    e = settings.INDEX[index]
    f = INDEX_FUNCS[index]
    if not e.at20:
        return f({k: prepared[k] for k in e.bands10})
    phase, exact = _phase(lattice)
    shape = np.asarray(prepared[e.bands10[0]]).shape[-2:]
    blk = {k: block_mean20(prepared[k], phase) for k in e.bands10}
    take = block_take20 if exact else block_mean20
    blk.update({k: np.asarray(take(prepared[k], phase), np.float32) for k in e.bands20})
    return expand20(f(blk), phase, shape)


# ─────────────────────────────────────────────────────────────────────────────
#  pixel cloud rules (§2.2)
# ─────────────────────────────────────────────────────────────────────────────

_CLOUD_LUT = {
    "C0": _lut(range(1, 12)),
    "C1": _lut(CLEAR_SCL),
    "C1d": _lut((2, 4, 5, 6, 7)),
    "C1s": _lut((4, 5, 6)),
    "C4": _lut(c for c in SCL_CODES if c not in ITEM_MASKED_SCL),
}
for _t in _CLOUD_LUT.values():
    _t.flags.writeable = False
del _t


def cloud_c0(scl8):
    """C0: SCL present and not 0 (bands present is added by the observation)."""
    return _CLOUD_LUT["C0"][_as_u8(scl8)]


def cloud_c1(scl8):
    """C1 (BASE): SCL in {4, 5, 6, 7} = :func:`mask_valid` (marea_clear)."""
    return _CLOUD_LUT["C1"][_as_u8(scl8)]


def cloud_c1d(scl8):
    """C1d: SCL in {2, 4, 5, 6, 7}."""
    return _CLOUD_LUT["C1d"][_as_u8(scl8)]


def cloud_c1s(scl8):
    """C1s (Tier 2): SCL in {4, 5, 6}."""
    return _CLOUD_LUT["C1s"][_as_u8(scl8)]


def cloud_c4(scl8):
    """C4: SCL present and not in {0, 1, 3, 8, 9, 10} = :func:`item_valid`."""
    return _CLOUD_LUT["C4"][_as_u8(scl8)]


def disk(r):
    """The disk structuring element of radius ``r`` px (centres within r)."""
    r = int(r)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def within(mask, r):
    """Pixels whose centre is within ``r`` px of a ``mask`` pixel
    (EDT(not mask) <= r, the binary dilation by :func:`disk` (r)); all False
    when the mask is empty."""
    from scipy import ndimage

    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.zeros(mask.shape, bool)
    return ndimage.distance_transform_edt(~mask) <= float(r)


def binary_open(mask, r):
    """Opening by :func:`disk` (r): erosion (pixels outside the array count
    as set, so a crop edge does not erode a cloud) then dilation."""
    from scipy import ndimage

    mask = np.asarray(mask, bool)
    if not mask.any():
        return mask.copy()
    er = ndimage.binary_erosion(mask, structure=disk(r), border_value=1)
    return within(er, r)


def _per_scene(fn, *arrays):
    """Apply a 2-D scene function over the leading axes of (..., H, W) arrays."""
    a0 = arrays[0]
    if a0.ndim == 2:
        return fn(*arrays)
    out = np.empty(a0.shape, bool)
    for idx in np.ndindex(*a0.shape[:-2]):
        out[idx] = fn(*(a[idx] for a in arrays))
    return out


def cloud_c3(scl8, cld8, p=20, open_px=2, cloud_px=8, shadow_px=4):
    """C3: masked = EDT(not opening((CLD >= p) | SCL = 10, disk open_px)) <=
    cloud_px | EDT(not SCL 3) <= shadow_px | SCL in {0, 1} | SCL or CLD no
    data; V = not masked. Per scene on the arrays given (the package passes
    the bounding box of dom plus 1 km)."""
    s, c = _as_u8(scl8), np.asarray(cld8)
    if c.dtype != np.uint8:
        raise TypeError("cloud_c3 takes the uint8 CLD of cld_u8")
    if c.shape != s.shape:
        raise ValueError(f"CLD {c.shape} and SCL {s.shape} differ in shape")

    def one(s2, c2):
        bad = ((c2 != CLD_NODATA_U8) & (c2 >= int(p))) | (s2 == 10)
        masked = within(binary_open(bad, open_px), cloud_px) | within(s2 == 3, shadow_px)
        masked |= (s2 == 0) | (s2 == 1) | (s2 == SCL_NODATA_U8) | (c2 == CLD_NODATA_U8)
        return ~masked
    return _per_scene(one, s, c)


def cloud_c5(scl8, open_px=2, cloud_px=8, shadow_px=4):
    """C5: C4, plus SCL{8, 9, 10} opened (disk open_px) and dilated cloud_px,
    and SCL 3 dilated shadow_px (DEA load_ard mask_filters)."""
    s = _as_u8(scl8)

    def one(s2):
        cloud = (s2 == 8) | (s2 == 9) | (s2 == 10)
        masked = within(binary_open(cloud, open_px), cloud_px) | within(s2 == 3, shadow_px)
        return cloud_c4(s2) & ~masked
    return _per_scene(one, s)


def cloud_c11(scl8, ocm):
    """C11: SCL present and not 0, and OmniCloudMask class clear (0)."""
    s = _as_u8(scl8)
    o = np.asarray(ocm)
    if o.shape != s.shape:
        raise ValueError(f"OCM classes {o.shape} and SCL {s.shape} differ in shape")
    return (s != SCL_NODATA_U8) & (s != 0) & (o == 0)


#: the pixel cloud functions by registry id
CLOUD_FUNCS = {"C0": cloud_c0, "C1": cloud_c1, "C1d": cloud_c1d, "C1s": cloud_c1s,
               "C3": cloud_c3, "C4": cloud_c4, "C5": cloud_c5, "C11": cloud_c11}


def pixel_valid(level, scl, cld=None, ocm=None, params=None, fill=FILL):
    """The pixel cloud rule ``level`` (True = usable), before the band and
    index checks of the observation.

    Parameters
    ----------
    level : str
        A pixel level of ``settings.CLOUD`` (a scene rule raises: pass its
        ``pixel_rule``).
    scl, cld : arrays
        Raw or uint8 SCL and CLD (CLD only for C3).
    ocm : uint8 array, optional
        :func:`ocm_classes` (C11 only).
    params : dict, optional
        Overrides of the level's frozen params (e.g. C3's ``p`` after the
        cross-tab).
    """
    e = settings.CLOUD[level]
    if e.kind != "pixel":
        raise ValueError(f"{level} is a scene rule; its pixel mask is {e.pixel_rule}")
    s = _as_scl8(scl, fill)
    prm = dict(e.params)
    prm.update(params or {})
    if level == "C3":
        if cld is None:
            raise ValueError("C3 needs CLD")
        return cloud_c3(s, cld_u8(cld, fill), int(prm["p"]), int(prm["open_px"]),
                        float(prm["cloud_px"]), float(prm["shadow_px"]))
    if level == "C5":
        return cloud_c5(s, int(prm["open_px"]), float(prm["cloud_px"]), float(prm["shadow_px"]))
    if level == "C11":
        if ocm is None:
            raise ValueError("C11 needs the OmniCloudMask classes (ocm_classes)")
        return cloud_c11(s, ocm)
    return CLOUD_FUNCS[level](s)


# ─────────────────────────────────────────────────────────────────────────────
#  OmniCloudMask (C11)
# ─────────────────────────────────────────────────────────────────────────────

def ocm_input(raw, offset=0, bands_ocm=None, fill=FILL):
    """``(input (3, H, W) float32, nodata (H, W) bool)`` for OmniCloudMask:
    the card's red, green and NIR bands under floor0 (DN - o_s), no data
    (any band missing) written as 0, the package's ``no_data_value``."""
    p = settings.CLOUD["C11"].params
    names = tuple(bands_ocm or p["bands"])
    arrs = [prep_band(raw[b], offset, "floor0", fill) for b in names]
    for a in arrs:
        if a.ndim != 2:
            raise ValueError("OmniCloudMask runs one scene (H, W) at a time")
    stack = np.stack(arrs).astype(np.float32)
    nodata = ~np.isfinite(stack).all(axis=0)
    stack[:, nodata] = 0.0
    return stack, nodata


def ocm_model_dir():
    """Where the OmniCloudMask weights live: ``$OCM_MODEL_DIR``, else
    ``data_v4/omnicloudmask/model_v<model_version>`` inside the project
    (gitignored with the other heavy inputs; the package's own default is
    the user's data folder). Their sha256 is recorded in each ``ocm.json``."""
    v = settings.CLOUD["C11"].params.get("model_version")
    return os.environ.get("OCM_MODEL_DIR") or os.path.join(
        "data_v4", "omnicloudmask", f"model_v{v}")


def ocm_classes(raw, offset=0, predictor=None, params=None, fill=FILL):
    """OmniCloudMask classes of one scene: uint8 (H, W), 0 clear, 1 thick
    cloud, 2 thin cloud, 3 shadow, :data:`OCM_NODATA_U8` without input data.

    ``predictor``: a callable ``(input (3, H, W) float32) -> (1, H, W) or
    (H, W)`` classes; None imports ``omnicloudmask.predict_from_array``
    with the card's patch settings and pinned model version on the CPU,
    the weights in :func:`ocm_model_dir`, and deterministic torch kernels
    (``torch.use_deterministic_algorithms(True)`` is set for the process).
    Tests pass a stub.
    """
    p = dict(settings.CLOUD["C11"].params)
    p.update(params or {})
    stack, nodata = ocm_input(raw, offset, p["bands"], fill)
    if predictor is None:
        import functools

        import torch
        from omnicloudmask import predict_from_array

        torch.use_deterministic_algorithms(True, warn_only=True)
        md = ocm_model_dir()
        os.makedirs(md, exist_ok=True)
        mv = p.get("model_version")
        predictor = functools.partial(
            predict_from_array, patch_size=int(p["patch_size"]),
            patch_overlap=int(p["patch_overlap"]), batch_size=int(p["batch_size"]),
            inference_device=str(p["inference_device"]), no_data_value=p["no_data_value"],
            destination_model_dir=md, model_version=None if mv is None else float(mv))
    out = np.asarray(predictor(stack))
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    if out.shape != nodata.shape:
        raise ValueError(f"OmniCloudMask returned {out.shape}, expected {nodata.shape}")
    vals = np.unique(out)
    if not np.isin(vals, tuple(OCM_CLASSES)).all():
        raise ValueError(f"OmniCloudMask classes outside {tuple(OCM_CLASSES)}: {vals[:10].tolist()}")
    out = out.astype(np.uint8)
    out[nodata] = OCM_NODATA_U8
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  scene rules (§2.2): keep decisions from per-scene statistics
# ─────────────────────────────────────────────────────────────────────────────

def scene_bad_share(scl, region=None, classes=(3, 8, 9, 10)):
    """Share of the pixels of ``region`` (bool mask; default the frame)
    whose SCL is in ``classes``: count / n, as ``stability`` computes it."""
    s = _as_u8(scl)
    bad = np.isin(s, classes)
    if region is not None:
        bad = bad[np.asarray(region, bool)]
    if bad.size == 0:
        raise ValueError("empty region")
    return float(np.count_nonzero(bad)) / bad.size


def scene_c2(frame_bad, tz_bad, clean_scene_max_bad=0.05, tz_max=0.10):
    """C2 keep flags: the cloud percentage is the frame's when the frame is
    <= 5 % bad, else the transition zone's (rounded to 4 decimals of %),
    and a date is kept iff it is <= 10 %: ``stability.reference_and_clouds``
    then ``filter_dates(., 0.10)``, exactly."""
    fb = np.atleast_1d(np.asarray(frame_bad, float)).ravel()
    tb = np.broadcast_to(np.asarray(tz_bad, float), np.shape(frame_bad)).ravel() \
        if np.ndim(frame_bad) else np.atleast_1d(np.asarray(tz_bad, float))
    keep = np.empty(fb.shape, bool)
    for i, (f, t) in enumerate(zip(fb, tb)):
        pct = round(float(f) * 100, 4) if f <= clean_scene_max_bad else round(float(t) * 100, 4)
        keep[i] = pct / 100.0 <= tz_max
    return keep.reshape(np.shape(frame_bad)) if np.ndim(frame_bad) else bool(keep[0])


def scene_c6(scl):
    """C6 keep flag of one scene: frame SCL{3,8,9,10} share < 0.10 and
    no-data share <= 0.02 (``granadeiro2021.scene_frame_shares`` /
    ``scene_rule``)."""
    s = _as_u8(scl)
    cloud, nodata = g21.scene_frame_shares(s, fill=SCL_NODATA_U8)
    return bool(g21.scene_rule(cloud, nodata, True, True))


def scene_c7(scl):
    """C7 keep flag of one scene: load_ard good fraction >= 0.5 (no data
    counts as good; ``waterline_contour.frame_good_fraction``)."""
    return bool(wc.frame_good_fraction(_as_u8(scl)) >= wc.S2_SCENE_MIN_GOODDATA)


def scene_c8(valid_c1, region, min_share=0.5):
    """C8 keep flag of one scene: the share of ``region`` (keep_full(P0), a
    bool mask or flat indices) valid under C1 is >= ``min_share``."""
    v = np.asarray(valid_c1, bool)
    r = np.asarray(region)
    sel = v[r] if r.dtype == bool else v.ravel()[r]
    if sel.size == 0:
        raise ValueError("C8: empty region")
    return bool(float(np.count_nonzero(sel)) / sel.size >= float(min_share))


# ─────────────────────────────────────────────────────────────────────────────
#  detectors (§2.6)
# ─────────────────────────────────────────────────────────────────────────────

def otsu_hist(counts, centres):
    """Otsu on a histogram: ``(t, eta, minority)``.

    t = the centre of the last bin of the lower class (water = x > t; the
    convention of ``skimage.filters.threshold_otsu``), eta = between-class /
    total variance at t, minority = the smaller class share. ``(nan, 0, 0)``
    for an empty or degenerate histogram.
    """
    c = np.asarray(counts, np.float64)
    m = np.asarray(centres, np.float64)
    tot = c.sum()
    if tot <= 0 or len(c) < 2:
        return float("nan"), 0.0, 0.0
    w0 = np.cumsum(c)
    w1 = np.cumsum(c[::-1])[::-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean0 = np.cumsum(c * m) / w0
        mean1 = (np.cumsum((c * m)[::-1]) / w1[::-1])[::-1]
    var12 = w0[:-1] * w1[1:] * (mean0[:-1] - mean1[1:]) ** 2
    var12 = np.where(np.isfinite(var12), var12, -1.0)
    k = int(np.argmax(var12))
    if var12[k] <= 0:
        return float("nan"), 0.0, 0.0
    mu = (c * m).sum() / tot
    var_t = (c * (m - mu) ** 2).sum() / tot
    eta = float(var12[k] / tot ** 2 / var_t) if var_t > 0 else 0.0
    minority = float(min(w0[k], tot - w0[k]) / tot)
    return float(m[k]), eta, minority


def otsu(values, bins=256):
    """Otsu on ``bins`` bins over the range of the finite ``values``:
    ``(t, eta, minority)`` (:func:`otsu_hist`)."""
    v = np.asarray(values, np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0 or v.min() == v.max():
        return float("nan"), 0.0, 0.0
    counts, edges = np.histogram(v, bins=int(bins), range=(float(v.min()), float(v.max())))
    return otsu_hist(counts, (edges[:-1] + edges[1:]) / 2.0)


def _hist_values(x, log_hist):
    x = np.asarray(x, np.float64)
    if log_hist:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(x[x > 0])
    return x


def d2_threshold(x, V, omega, t_k, params=None, log_hist=False):
    """D2 guarded per-scene Otsu of one scene (§2.6).

    Parameters
    ----------
    x, V : (H, W) arrays
        The scene's index and validity.
    omega : bool (H, W)
        The fixed D2 domain.
    t_k : float
        The index's fixed threshold (the fallback).
    log_hist : bool
        Otsu on log(x) (WRI); the threshold is returned in x units.

    Returns
    -------
    dict
        ``t`` (the threshold used), ``t_otsu``, ``eta``, ``minority``,
        ``n``, ``used`` (True when Otsu's threshold was used).
    """
    p = dict(settings.DETECT["D2"].params)
    p.update(params or {})
    vals = _hist_values(np.asarray(x)[np.asarray(V, bool) & np.asarray(omega, bool)], log_hist)
    vals = vals[np.isfinite(vals)]
    out = {"t": float(t_k), "t_otsu": float("nan"), "eta": 0.0, "minority": 0.0,
           "n": int(vals.size), "used": False}
    if vals.size < int(p["min_px"]):
        return out
    t, eta, m = otsu(vals, int(p["bins"]))
    out.update(t_otsu=float(np.exp(t)) if log_hist else t, eta=eta, minority=m)
    if np.isfinite(t) and eta >= float(p["eta_min"]) and m >= float(p["minority_min"]):
        out.update(t=out["t_otsu"], used=True)
    return out


def d3_threshold(x, V, t_k, params=None, log_hist=False):
    """D3 edge-Otsu of one scene (§2.6), on the arrays given (the package
    passes the bounding box of dom plus 1 km).

    Returns
    -------
    dict
        ``t`` (the threshold used), ``t_otsu``, ``n_buffer``, ``n_edges``,
        ``valid_share``, ``used``, ``skipped`` (valid share below 20 %).
    """
    from scipy import ndimage
    from skimage.feature import canny

    p = dict(settings.DETECT["D3"].params)
    p.update(params or {})
    x = np.asarray(x, np.float64)
    V = np.asarray(V, bool)
    share = float(V.mean()) if V.size else 0.0
    out = {"t": float(t_k), "t_otsu": float("nan"), "n_buffer": 0, "n_edges": 0,
           "valid_share": share, "used": False,
           "skipped": bool(share < float(p["min_valid_share"]))}
    if out["skipped"]:
        return out
    with np.errstate(invalid="ignore"):
        b = (V & (x > float(t_k))).astype(np.float64)
    vm = ndimage.binary_erosion(V, structure=disk(int(p["erode_px"])), border_value=0)
    edges = canny(b, sigma=float(p["sigma"]), low_threshold=float(p["low"]),
                  high_threshold=float(p["high"]), mask=vm)
    lab, n = ndimage.label(edges, structure=np.ones((3, 3), bool))
    if n:
        sizes = np.bincount(lab.ravel())
        big = sizes >= int(p["min_edge_px"])
        big[0] = False
        edges = big[lab]
    out["n_edges"] = int(np.count_nonzero(edges))
    buf = within(edges, float(p["buffer_px"])) & V
    out["n_buffer"] = int(np.count_nonzero(buf))
    if out["n_buffer"] < int(p["min_buffer_px"]):
        return out
    vals = _hist_values(x[buf], log_hist)
    t, _, _ = otsu(vals[np.isfinite(vals)], int(p["bins"]))
    if np.isfinite(t):
        out.update(t_otsu=float(np.exp(t)) if log_hist else t, used=True)
        out["t"] = out["t_otsu"]
    return out


def detect_d1(X, V, t_k):
    """D1: W = V & (x > t_k)."""
    with np.errstate(invalid="ignore"):
        return np.asarray(V, bool) & (np.asarray(X) > np.float32(t_k))


def detect_threshold(X, V, t_s):
    """D2 / D3: W = V & (x > t_s), ``t_s`` one threshold per scene (shape
    ``X.shape[:-2]``) or a scalar, from the settings pass."""
    X = np.asarray(X)
    t = np.asarray(t_s, np.float32)
    if t.ndim:
        if t.shape != X.shape[:-2]:
            raise ValueError(f"one threshold per scene: {t.shape} != {X.shape[:-2]}")
        t = t.reshape(t.shape + (1, 1))
    with np.errstate(invalid="ignore"):
        return np.asarray(V, bool) & (X > t)


def detect_d4(V, scl8):
    """D4: W = V & SCL = 6."""
    return np.asarray(V, bool) & (_as_u8(scl8) == SCL_WATER)


def detect_d5(V, ndwi_x, mndwi_x, awei_nsh_x, votes=2):
    """D5: W = V & [(NDWI > 0) + (MNDWI > 0) + (AWEI_nsh > 0) >= votes]."""
    with np.errstate(invalid="ignore"):
        n = ((np.asarray(ndwi_x) > 0).astype(np.int8) + (np.asarray(mndwi_x) > 0)
             + (np.asarray(awei_nsh_x) > 0))
    return np.asarray(V, bool) & (n >= int(votes))


# ─────────────────────────────────────────────────────────────────────────────
#  the observation of a setting (§2.5)
# ─────────────────────────────────────────────────────────────────────────────

def spectral_bands(setting):
    """The spectral bands the observation of ``setting`` reads (index and
    detector), in store order."""
    s = settings.Setting.parse(setting)
    need = set(settings.INDEX[s.index].bands)
    for i in settings.DETECT[s.detect].indices:
        need |= set(settings.INDEX[i].bands)
    return tuple(b for b in SPECTRAL_BANDS if b in need)


def observe_setting(raw, setting, *, offset=0, lattice=(0, 0), params=None, threshold=None,
                    ocm=None, cloud_valid=None, cloud_params=None, fill=FILL):
    """``Observation(X, Y, V, W)`` of a scene or block under ``setting``.

    Parameters
    ----------
    raw : dict
        ``{band: (..., H, W)}``: raw int16 spectral bands, SCL (raw or the
        uint8 of :func:`scl_u8`) and, for C3, CLD (raw or uint8).
    setting : settings.Setting or str
        The setting (its scene rule, if any, is applied to the record by the
        package; here only its pixel rule acts).
    offset : int or array
        The per-scene BOA offset o_s (0 or 1000; one per scene).
    lattice : (py, px) or None
        ``SceneSource.lattice20`` (used by the indices at 20 m).
    params : dict, optional
        ``{"t_k", "s_k"}`` of the index (``settings.index_params``); default
        the fixed values, which exist only for NDWI.
    threshold : float or array, optional
        D2/D3 per-scene thresholds (shape ``raw.shape[:-2]``) in X units.
    ocm : uint8 array, optional
        :func:`ocm_classes` for C11.
    cloud_valid : bool array, optional
        A precomputed pixel cloud rule (the package's stored bits for C3,
        C5, C11); overrides the computation.
    cloud_params : dict, optional
        Overrides of the cloud level's params (C3's p).
    fill : int
        The bands' fill value.

    Returns
    -------
    Observation
        X float32 (unmasked index), Y float32 (``Y is X`` for NDWI), V bool,
        W bool, all of the bands' shape.
    """
    s = settings.Setting.parse(setting)
    de = settings.DETECT[s.detect]
    p = dict(params) if params is not None else settings.index_params(s.index)
    t_k, s_k = float(p["t_k"]), float(p["s_k"])
    if "SCL" not in raw:
        raise KeyError("the observation needs SCL")
    scl8 = _as_scl8(raw["SCL"], fill)
    if cloud_valid is None:
        cv = pixel_valid(s.pixel_rule, scl8, raw.get("CLD"), ocm, cloud_params, fill)
    else:
        cv = np.asarray(cloud_valid, bool)
        if cv.shape != scl8.shape:
            raise ValueError(f"cloud_valid {cv.shape} != bands {scl8.shape}")
    need = spectral_bands(s)
    prep = prepare(raw, need, offset, s.prep, fill)
    X = compute_index(prep, s.index, lattice)
    if X.shape != scl8.shape:
        raise ValueError(f"bands {X.shape} and SCL {scl8.shape} differ in shape")
    V = cv & np.isfinite(X)
    for b in need:
        V &= ~np.isnan(prep[b])
    if t_k == 0.0 and s_k == 1.0:
        Y = X
    else:
        Y = ((X - np.float32(t_k)) / np.float32(s_k)).astype(np.float32)
    if s.detect == "D1":
        W = detect_d1(X, V, t_k)
    elif de.per_scene:
        if threshold is None:
            raise ValueError(f"{s.detect} needs the per-scene thresholds of the settings pass")
        W = detect_threshold(X, V, threshold)
    elif s.detect == "D4":
        W = detect_d4(V, scl8)
    elif s.detect == "D5":
        xs = [X if i == s.index else compute_index(prep, i, lattice) for i in de.indices]
        W = detect_d5(V, *xs, votes=int(de.params["votes"]))
    else:
        raise ValueError(f"no detector implementation for {s.detect}")
    return Observation(X, Y, V, W)


# ─────────────────────────────────────────────────────────────────────────────
#  hashes
# ─────────────────────────────────────────────────────────────────────────────

#: the observation core: what every observation goes through whatever the
#: setting (the registry entries' own functions are hashed per entry)
CORE_FUNCTIONS = ("decode", "scl_u8", "cld_u8", "_as_scl8", "check_boa", "_as_u8", "_lut",
                  "mask_valid", "bands", "ndwi", "observe", "observe_rules", "wet",
                  "prepare", "compute_index", "_phase", "pixel_valid", "_per_scene",
                  "spectral_bands", "observe_setting")
CORE_CONSTANTS = {"FILL": FILL, "BOA_MIN_VALUE": BOA_MIN_VALUE, "SCL_NODATA_U8": SCL_NODATA_U8,
                  "CLD_NODATA_U8": CLD_NODATA_U8, "SCL_CODES": list(SCL_CODES),
                  "CLEAR_SCL": list(CLEAR_SCL), "BLOCK_SCL": list(BLOCK_SCL),
                  "REFLECTANCE_SCALE": REFLECTANCE_SCALE}


def core_sha256():
    """sha256 of the observation core: the CRLF-normalised source of each
    function of :data:`CORE_FUNCTIONS` and the JSON of
    :data:`CORE_CONSTANTS` (the v2 ``obs core`` of the package key, §13.2).
    Adding or editing a registry level does not change it."""
    h = hashlib.sha256(json.dumps(CORE_CONSTANTS, sort_keys=True).encode("utf-8"))
    g = globals()
    for name in CORE_FUNCTIONS:
        h.update(b"\n--- " + name.encode("utf-8") + b"\n")
        h.update(inspect.getsource(g[name]).replace("\r\n", "\n").encode("utf-8"))
    return h.hexdigest()


def _check_registry_implementations():
    """Every ``obs`` function a registry entry names must exist here."""
    pre = "experiments.benchmark.obs."
    names = globals()
    missing = sorted({n[len(pre):] for reg in settings.REGISTRIES.values()
                      for e in reg.values() for n in e.impl
                      if n.startswith(pre) and n[len(pre):] not in names})
    if missing:
        raise AssertionError(f"settings registries name obs functions that do not exist: {missing}")
    if set(INDEX_FUNCS) != set(settings.INDEX):
        raise AssertionError("obs.INDEX_FUNCS and settings.INDEX disagree")
    if set(CLOUD_FUNCS) != set(settings.PIXEL_CLOUD):
        raise AssertionError("obs.CLOUD_FUNCS and settings' pixel cloud levels disagree")


_check_registry_implementations()
