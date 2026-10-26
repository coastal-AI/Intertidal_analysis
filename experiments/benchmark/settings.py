# -*- coding: utf-8 -*-
"""Validation v2 — the settings factors as registries (spec §2).

Question
--------
Which processing settings does every method of validation v2 receive, and
how is a combination of them named, keyed and hashed, so that within one
combination only the method changes (spec §0.3 item 9, §5.1)?

Pre-registered rules (docs/validation_v2_spec.md §2, Appendix A sections
``cloud``, ``prep``, ``index``, ``detect``, ``level``, ``window``,
``curves``; user decisions of 2026-10-13)
-----------------------------------------------------------------------------
* One registry per factor; each entry is a frozen dataclass with a short id,
  the bands it needs, a reference and the dotted names of its
  implementation (``impl``: functions in ``obs.py`` for pixel rules,
  indices and detectors; production functions for the scene rules; the
  level builders of ``levels.py``). An entry is hashed PER ENTRY
  (:func:`entry_sha`: its JSON fields plus the CRLF-normalised source of
  each ``impl`` function), so adding or editing one level changes only the
  units that use it (spec §2, §13.2).
* Cloud handling C (§2.2): C0 none, C1 SCL allowlist {4, 5, 6, 7} (BASE,
  MAREA's production mask), C1d {2, 4, 5, 6, 7}, C1s {4, 5, 6} (Tier 2), C2
  transition-zone rule, C3 CLD >= p | SCL 10 opened (disk 2 px) and buffered
  8 px, SCL 3 buffered 4 px, C4 ITEM-PQ blocklist, C5 C4 + buffers (Tier
  1B), C6 g21 frame rule, C7 load_ard rule (Tier 1B), C8 domain rule, C11
  OmniCloudMask (user decision 2026-10-13: INCLUDED). Scene rules (C2, C6,
  C7, C8) act on the record after the A-mode draw and always combine with
  the C1 pixel mask. No data never counts as cloud.
* Band preprocessing (§2.3): floor0 (BASE) rho = max(DN - o_s, 0) / 10000;
  raw rho = (DN - o_s) / 10000; o_s in {0, 1000} is the scene's frozen
  offset (``inputs.decide_offsets``); the prep levels hash the whole offset
  rule (:data:`OFFSET_RULE_IMPL`: ``inputs.OFFSET_RULE`` as JSON and every
  function of the decision; amendment of 2026-10-19).
* Spectral index (§2.4): I1 NDWI (BASE, bit-identical to production), I2
  MNDWI, I3 NIR, I4 AWEI_sh, I5 AWEI_nsh, I6 WRI, I7 NDWI@20; every index
  that uses a 20 m band is computed at 20 m on 2 x 2 means of its 10 m bands
  over the native lattice ``lattice20`` (box mean at phase (0, 0) when the
  lattice is unknown, declared) and replicated to 10 m.
* Water detection (§2.6): D1 fixed (BASE) W = V & x > t_k; D2 guarded
  per-scene Otsu on Omega; D3 edge-Otsu; D4 SCL water (naive baseline, Tier
  1B); D5 majority vote of NDWI, MNDWI and AWEI_nsh.
* Water-level source (§1.6, elevation units only): LB (BASE), LT, LM, LG
  (labelled upper bound; never in lag scoring nor in a confirmatory family).
* Windows (§2.7): nested W1, W2, W3 (BASE), W5, W9 ending 2026-01-01
  (exclusive); calendar years Y2017..Y2025; survey-centred V<epoch>L<len>.
* Number of images (§2.8): the K grids; ``rand`` subsets
  ``default_rng([20261008, SITE_ID, WINDOW_NUM, r]).permutation(T)``,
  disjoint batches of perm_0 iff R <= floor(T / N), nested sort(perm_r[:N])
  otherwise, never mixed in one point; ``win`` start
  ``default_rng([20261012, SITE_ID, WINDOW_NUM, N, r])``; matched-N draws
  ``default_rng([20261016, SITE_ID, WINDOW_NUM, N_rule, d])``.
* Setting id ``<cloud>.<prep>.<index>.<detect>[@<level>]`` (the level
  suffix omitted for LB); cell = (setting id, window id); the effective key
  projects a cell onto a method's dependency signature (§2.9); a method that
  consumes the harness domain depends on cloud, prep, index and detect.

This module imports only the standard library and ``config`` (numpy is
imported inside the subset functions); the implementations are resolved
lazily by name, only to hash them. No import-time side effects.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
import re
import types
from dataclasses import dataclass, field
from typing import Mapping, Optional

from experiments.benchmark import config

#: the factors of a cell (§2.1), in setting-id order, then the window
FACTORS = ("cloud", "prep", "index", "detect", "level", "window")
#: factors read through the harness domain (its extraction rule uses V and W)
DOMAIN_FACTORS = ("cloud", "prep", "index", "detect")
#: the union period of the Zarr stores (end exclusive; spec §1.4)
UNION_PERIOD = ("2017-01-01", "2026-01-01")
#: seeds (Appendix A: subsets, win, matched-N draws)
SEED_SUBSETS = int(config.SEEDS["subsets"])          # 20261008
SEED_WINDOWS = int(config.SEEDS["windows"])          # 20261012
SEED_MATCHED = 20261016
#: the reflectance scale of every index (§2.4); the data-dependent scale
#: heuristic of pyintertidal.water.index_block is never used
REFLECTANCE_SCALE = 10000.0
#: BOA offsets a scene can carry (§2.3)
OFFSETS = (0, 1000)

_CLOUD_PARAMS = {"open_px": 2, "cloud_px": 8, "shadow_px": 4}


def _ro(mapping):
    """Deep read-only copy of a JSON-like mapping (lists -> tuples)."""
    def fr(v):
        if isinstance(v, Mapping):
            return types.MappingProxyType({str(k): fr(x) for k, x in v.items()})
        if isinstance(v, (list, tuple)):
            return tuple(fr(x) for x in v)
        return v
    return fr(dict(mapping or {}))


def _plain(v):
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, frozenset, set)):
        items = [_plain(x) for x in v]
        return sorted(items, key=str) if isinstance(v, (frozenset, set)) else items
    return v


# ─────────────────────────────────────────────────────────────────────────────
#  entries
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, kw_only=True)
class Entry:
    """One level of one factor.

    Attributes
    ----------
    factor : str
        One of :data:`FACTORS`.
    id : str
        The short id used in setting ids and paths.
    rule : str
        The rule in words (the spec's text).
    reference : str
        Publication or code the rule follows.
    role : str
        Base, sensitivity, native level, control, ...
    tier : str
        "P0" (the base), "1A'", "1B", "2", or "OFAT".
    bands : tuple of str
        Store bands the level reads.
    params : Mapping
        Every frozen number of the level.
    impl : tuple of str
        Dotted names of the functions that implement the level; their
        source is part of :func:`entry_sha`.
    """

    factor: str
    id: str
    rule: str = ""
    reference: str = ""
    role: str = ""
    tier: str = "1A'"
    bands: tuple = ()
    params: Mapping = field(default_factory=dict)
    impl: tuple = ()

    def __post_init__(self):
        if self.factor not in FACTORS + ("curves", "prep"):
            raise ValueError(f"unknown factor {self.factor!r}")
        if not re.fullmatch(r"[A-Za-z0-9_]+", self.id):
            raise ValueError(f"entry id {self.id!r} must be [A-Za-z0-9_]+ (it enters paths)")
        object.__setattr__(self, "bands", tuple(self.bands))
        object.__setattr__(self, "impl", tuple(self.impl))
        object.__setattr__(self, "params", _ro(self.params))

    def __hash__(self):
        return hash((self.factor, self.id))

    def as_dict(self):
        """JSON-ready fields (``impl`` included as names)."""
        return {f.name: _plain(getattr(self, f.name)) for f in dataclasses.fields(self)}

    @property
    def sha(self):
        """:func:`entry_sha` of this entry."""
        return entry_sha(self)


@dataclass(frozen=True, kw_only=True)
class CloudLevel(Entry):
    """A cloud-handling level (§2.2).

    ``kind``: "pixel" (acts per observation) or "scene" (acts on the record
    after the draw, combined with the ``pixel_rule`` pixel mask).
    ``neighbourhood``: True when the rule uses openings or buffers (computed
    per scene on the bounding box of dom plus 1 km by the package).
    """

    factor: str = "cloud"
    kind: str = "pixel"
    pixel_rule: str = ""
    neighbourhood: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.kind not in ("pixel", "scene"):
            raise ValueError(f"{self.id}: kind must be pixel or scene")
        if not self.pixel_rule:
            object.__setattr__(self, "pixel_rule", self.id if self.kind == "pixel" else "C1")


@dataclass(frozen=True, kw_only=True)
class PrepLevel(Entry):
    """A band-preprocessing level (§2.3): ``floor`` True clips at 0."""

    factor: str = "prep"
    floor: bool = True


@dataclass(frozen=True, kw_only=True)
class IndexLevel(Entry):
    """A spectral index (§2.4).

    Attributes
    ----------
    formula : str
    at20 : bool
        Computed at 20 m on the native lattice (it uses a 20 m band).
    bands10, bands20 : tuple of str
        Its 10 m bands (2 x 2 means when ``at20``) and 20 m bands.
    t_k : float or None
        The fixed threshold of D1 in the index's units; None when it is
        data-dependent (``t_k_rule``; I3 NIR: -t_NIR from the settings pass).
    s_k_rule : str
        How the standardisation scale is fixed (1 for NDWI, by definition).
    hist_range, hist_bins
        The fixed per-year histogram of the settings pass (pooled Otsu and
        IQR); values outside are counted in the end bins.
    log_hist : bool
        Histogram detectors (D2, D3) work on log(x) (WRI).
    native_nd : bool
        A normalised difference (NIDEM's filtered product exists, §2.5).
    """

    factor: str = "index"
    formula: str = ""
    at20: bool = False
    bands10: tuple = ()
    bands20: tuple = ()
    t_k: Optional[float] = 0.0
    t_k_rule: str = "fixed"
    s_k_rule: str = "IQR_k / IQR_NDWI over the clear observations of keep_full(P0, W3)"
    hist_range: tuple = (-1.0, 1.0)
    hist_bins: int = 2000
    log_hist: bool = False
    native_nd: bool = False

    def __post_init__(self):
        object.__setattr__(self, "bands10", tuple(self.bands10))
        object.__setattr__(self, "bands20", tuple(self.bands20))
        object.__setattr__(self, "hist_range", tuple(float(v) for v in self.hist_range))
        if not self.bands:
            object.__setattr__(self, "bands", tuple(self.bands10) + tuple(self.bands20))
        if not self.rule:
            object.__setattr__(self, "rule", self.formula)
        super().__post_init__()
        if self.bands20 and not self.at20:
            raise ValueError(f"{self.id}: an index with a 20 m band is computed at 20 m (§2.4)")


@dataclass(frozen=True, kw_only=True)
class DetectLevel(Entry):
    """A water detector (§2.6).

    ``per_scene``: thresholds from the settings pass, one per scene;
    ``index_free``: W does not depend on the index factor (D4, D5);
    ``indices``: the indices the detector computes itself (D5).
    """

    factor: str = "detect"
    per_scene: bool = False
    index_free: bool = False
    indices: tuple = ()


@dataclass(frozen=True, kw_only=True)
class LevelSource(Entry):
    """A water-level source for elevation units (§1.6)."""

    factor: str = "level"
    sites: tuple = ()
    confirmatory: bool = True
    upper_bound: bool = False


@dataclass(frozen=True, kw_only=True)
class Window(Entry):
    """A temporal window (§2.7): ``[start, end)`` (end exclusive).

    ``kind``: "nested", "year" or "survey"; ``num``: the integer that
    enters the subset seeds (:func:`window_num`); ``centre``: the window
    centre in decimal years (epochs at mid-year).
    """

    factor: str = "window"
    start: str = ""
    end: str = ""
    kind: str = "nested"
    num: int = 0
    sites: tuple = ()

    @property
    def years(self):
        return tuple(range(int(self.start[:4]), int(self.end[:4])))

    @property
    def centre(self):
        y = self.years
        return (y[0] + y[-1] + 1) / 2.0


@dataclass(frozen=True, kw_only=True)
class CurveSet(Entry):
    """A number-of-images curve set (§2.8, §5.10).

    ``grid``: ((N, R, kind), ...) with N an int or "all" and kind the
    expected batch kind ("d" disjoint, "n" nested, "all"); the actual kind
    is decided from T by :func:`subset_kind`. ``mode``: "A" or "F";
    ``family``: "rand" or "win"; ``settings``: setting ids; ``windows``:
    window ids.
    """

    factor: str = "curves"
    grid: tuple = ()
    mode: str = "A"
    family: str = "rand"
    settings: tuple = ()
    windows: tuple = ()
    sites: tuple = ()
    methods: str = ""


# ─────────────────────────────────────────────────────────────────────────────
#  per-entry hashes
# ─────────────────────────────────────────────────────────────────────────────

def impl_source(dotted):
    """CRLF-normalised source of the object ``package.module.name``; for a
    JSON-able module constant (e.g. ``inputs.OFFSET_RULE``, the parameters of
    a rule) its canonical JSON instead.

    A name that cannot be resolved (a level whose builder is not written yet)
    gives ``"<missing dotted.name>"``, so writing it later changes the hash.
    """
    mod, _, name = str(dotted).rpartition(".")
    try:
        obj = getattr(importlib.import_module(mod), name)
    except (ImportError, AttributeError):
        return f"<missing {dotted}>"
    if obj is None or isinstance(obj, (dict, list, tuple, str, int, float, bool)):
        try:
            return "<json>\n" + json.dumps(obj, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return f"<missing {dotted}>"
    try:
        src = inspect.getsource(obj)
    except (OSError, TypeError):
        return f"<missing {dotted}>"
    return src.replace("\r\n", "\n")


def entry_sha(entry):
    """sha256 of an entry: its JSON fields (sorted keys) and the source of
    every ``impl`` function, in order (spec §2, §13.2)."""
    h = hashlib.sha256()
    h.update(json.dumps(entry.as_dict(), sort_keys=True).encode("utf-8"))
    for name in entry.impl:
        h.update(b"\n--- " + name.encode("utf-8") + b"\n")
        h.update(impl_source(name).encode("utf-8"))
    return h.hexdigest()


_OBS = "experiments.benchmark.obs."
_MASK_CORE = (_OBS + "_lut", _OBS + "_as_u8")
_HOOD = (_OBS + "disk", _OBS + "binary_open", _OBS + "within")
_I20 = (_OBS + "block_mean20", _OBS + "block_take20", _OBS + "expand20", _OBS + "_block_index")


# ─────────────────────────────────────────────────────────────────────────────
#  cloud handling (§2.2)
# ─────────────────────────────────────────────────────────────────────────────

CLOUD = types.MappingProxyType({e.id: e for e in (
    CloudLevel(id="C0", rule="V = bands present & SCL != 0", reference="design control",
               role="lower bound; one OFAT cell evaluated on S0 only; never in a grid",
               tier="OFAT", bands=("SCL",), params={"valid_scl": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
               impl=(_OBS + "cloud_c0",) + _MASK_CORE),
    CloudLevel(id="C1", rule="V = SCL in {4, 5, 6, 7}",
               reference="marea.CLEAR; obs.mask_valid('marea_clear'); Sen2Cor ATBD 2.10",
               role="BASE (MAREA production; the user's per-pixel cloud mask)", tier="P0",
               bands=("SCL",), params={"valid_scl": [4, 5, 6, 7]},
               impl=(_OBS + "cloud_c1",) + _MASK_CORE),
    CloudLevel(id="C1d", rule="V = SCL in {2, 4, 5, 6, 7}", reference="this study",
               role="MNAR check: wet mud is dark and often labelled class 2 or 3",
               tier="1A'", bands=("SCL",), params={"valid_scl": [2, 4, 5, 6, 7]},
               impl=(_OBS + "cloud_c1d",) + _MASK_CORE),
    CloudLevel(id="C1s", rule="V = SCL in {4, 5, 6}", reference="this study",
               role="strict allowlist", tier="2", bands=("SCL",),
               params={"valid_scl": [4, 5, 6]}, impl=(_OBS + "cloud_c1s",) + _MASK_CORE),
    CloudLevel(id="C2", kind="scene",
               rule=("keep a date iff its SCL{3,8,9,10} share inside the transition zone is "
                     "<= 10 %, or the whole-frame share when the frame is <= 5 % bad; with C1"),
               reference="v12; stability.reference_and_clouds, filter_dates(., 0.10)",
               role="MAREA's as-published record; the user's transition-zone rule",
               tier="1A'", bands=("SCL",),
               params={"bad_scl": [3, 8, 9, 10], "clean_scene_max_bad": 0.05,
                       "tz_max": 0.10, "stable_threshold": 0.95, "transition_buffer_px": 10,
                       "tz_observation": "NDWI > 0, clear classes {4, 5, 6} (production)"},
               impl=(_OBS + "scene_c2", "pyintertidal.stability.reference_and_clouds",
                     "pyintertidal.stability.filter_dates")),
    CloudLevel(id="C3", neighbourhood=True,
               rule=("bad = (CLD >= p) | SCL = 10; bad_o = opening(bad, disk 2 px); masked = "
                     "EDT(not bad_o) <= 8 px | EDT(not SCL 3) <= 4 px | SCL in {0, 1} | CLD fill"),
               reference="Sen2Cor ATBD 2.10 §3.8-3.9; sen2chain CM001; DEA load_ard mask_filters",
               role="Sen2Cor-based other filtering method; C11 fallback in G1'", tier="1A'",
               bands=("SCL", "CLD"),
               params={"p": 20, "p_raised": 35, "p_rule": ("p = 35 if the median CLD over any of "
                       "the three anchor classes exceeds 20 % in more than 10 % of scenes at any "
                       "site (cross-tab, before lock-A), else 20"), **_CLOUD_PARAMS},
               impl=(_OBS + "cloud_c3",) + _MASK_CORE + _HOOD),
    CloudLevel(id="C4", rule="V = SCL present & SCL not in {0, 1, 3, 8, 9, 10}",
               reference="Sagar et al. 2017; waterline_contour.ITEM_PQ_MASKED_SCL",
               role="NIDEM native (competitor-native level in G1'); keeps SCL 11",
               tier="1A'", bands=("SCL",), params={"masked_scl": [0, 1, 3, 8, 9, 10]},
               impl=(_OBS + "cloud_c4",) + _MASK_CORE),
    CloudLevel(id="C5", neighbourhood=True,
               rule="C4, plus SCL{8,9,10} opened (disk 2 px) and dilated 80 m, SCL 3 dilated 40 m",
               reference="DEA load_ard mask_filters", role="DEA-style mask", tier="1B",
               bands=("SCL",), params={"masked_scl": [0, 1, 3, 8, 9, 10],
                                       "cloud_scl": [8, 9, 10], **_CLOUD_PARAMS},
               impl=(_OBS + "cloud_c5", _OBS + "cloud_c4") + _MASK_CORE + _HOOD),
    CloudLevel(id="C6", kind="scene",
               rule="frame SCL{3,8,9,10} share < 0.10 and no-data share <= 0.02; with C1",
               reference="Granadeiro et al. 2021; granadeiro2021.scene_rule",
               role="Granadeiro native", tier="1A'", bands=("SCL",),
               params={"cloud_scl": [3, 8, 9, 10], "cloud_max_exclusive": 0.10,
                       "nodata_max": 0.02},
               impl=(_OBS + "scene_c6", "pyintertidal.granadeiro2021.scene_frame_shares",
                     "pyintertidal.granadeiro2021.scene_rule")),
    CloudLevel(id="C7", kind="scene",
               rule="load_ard good fraction (SCL not in {1,3,8,9,10}, no data good) >= 0.5; with C1",
               reference="DE Africa/DEA load_ard; waterline_contour.s2_scene_rule",
               role="NIDEM-S2 native", tier="1B", bands=("SCL",),
               params={"bad_scl": [1, 3, 8, 9, 10], "min_good": 0.5},
               impl=(_OBS + "scene_c7", "pyintertidal.waterline_contour.frame_good_fraction",
                     "pyintertidal.waterline_contour.load_ard_good")),
    CloudLevel(id="C8", kind="scene", rule="share of keep_full(P0) valid under C1 >= 0.5",
               reference="CoastSat-style rule on the region of interest (Vos et al. 2019)",
               role="comparable across sites", tier="1A'", bands=("SCL",),
               params={"min_share": 0.5, "region": "keep_full(P0, W3)"},
               impl=(_OBS + "scene_c8",)),
    CloudLevel(id="C11",
               rule=("V = bands present & SCL != 0 & OmniCloudMask class = clear (0); inputs "
                     "red, green, NIR; CPU inference, pinned version, deterministic kernels"),
               reference=("Wright et al. 2025, RSE 322:114694, doi:10.1016/j.rse.2025.114694; "
                          "PyPI omnicloudmask (MIT licence, checked 2026-10-14)"),
               role="independent non-Sen2Cor mask; in G1' (user decision 2026-10-13)",
               tier="1A'", bands=("SCL", "B04", "B03", "B8A"),
               params={"package": "omnicloudmask", "version": "1.7.1", "licence": "MIT",
                       "classes": {"0": "clear", "1": "thick cloud", "2": "thin cloud",
                                   "3": "cloud shadow"},
                       "clear_class": 0,
                       "bands": ["B04", "B03", "B8A"],
                       "band_note": ("red, green, NIR in the order predict_from_array expects; "
                                     "B8A as in the package's Sentinel-2 loader load_s2 "
                                     "(PROVISIONAL: fixed in the card from the package "
                                     "documentation before lock-A)"),
                       "input_scale": "floor0 DN - o_s (10000 x reflectance), float32",
                       "resolution_m": 10, "patch_size": 1000, "patch_overlap": 300,
                       "batch_size": 1, "inference_device": "cpu",
                       "no_data_value": 0, "region": "bounding box of dom plus 1 km",
                       "model_version": 4.0,
                       "model_note": ("the newest model of omnicloudmask 1.7.1 "
                                      "(model_download_links.csv), pinned; weights from Hugging "
                                      "Face into data_v4/omnicloudmask (OCM_MODEL_DIR), sha256 "
                                      "in each ocm.json")},
               impl=(_OBS + "cloud_c11", _OBS + "ocm_classes", _OBS + "ocm_input")
               + _MASK_CORE),
)})

#: the pixel-rule cloud levels (computed per observation by obs)
PIXEL_CLOUD = tuple(k for k, e in CLOUD.items() if e.kind == "pixel")
#: the scene-rule cloud levels (applied by the package to the record)
SCENE_CLOUD = tuple(k for k, e in CLOUD.items() if e.kind == "scene")


# ─────────────────────────────────────────────────────────────────────────────
#  band preprocessing (§2.3)
# ─────────────────────────────────────────────────────────────────────────────

_INPUTS = "experiments.benchmark.inputs."
#: what carries the per-scene offset rule o_s (amended 2026-10-19): its
#: parameters and every function of the decision, so the prep levels' hash
#: (and lock-A) covers the whole rule; ``inputs.offset_rule_sha256`` hashes
#: the same list for the inputs report
OFFSET_RULE_IMPL = tuple(_INPUTS + n for n in (
    "OFFSET_RULE", "HARD_ZERO_TAGS", "IDENTITY_VERDICTS", "IDENTITY_BANDS", "_baseline_version",
    "_baseline_class", "scene_offset_stats", "scl_cloud_counts", "_cloud_share",
    "_water_counts", "_strong_rule", "_strong", "_identity_map", "decide_offsets", "_count",
    "_bands_clean", "_per_date_counts", "_covers_store", "identity_from_verify"))
_PREP_IMPL = (_OBS + "prep_band", _OBS + "decode", _OBS + "_offset_array") + OFFSET_RULE_IMPL

PREP = types.MappingProxyType({e.id: e for e in (
    PrepLevel(id="floor0", floor=True,
              rule=("every band an index uses: rho = max(DN - o_s, 0) / 10000 for non-fill "
                    "values; o_s in {0, 1000} the scene's frozen offset"),
              reference="v13 D2", role="BASE", tier="P0",
              params={"scale": REFLECTANCE_SCALE, "offsets": list(OFFSETS)}, impl=_PREP_IMPL),
    PrepLevel(id="raw", floor=False, rule="rho = (DN - o_s) / 10000, no floor",
              reference="production MAREA", role="OFAT sensitivity; MAREA as published",
              tier="1A'", params={"scale": REFLECTANCE_SCALE, "offsets": list(OFFSETS)},
              impl=_PREP_IMPL),
)})


# ─────────────────────────────────────────────────────────────────────────────
#  spectral index (§2.4)
# ─────────────────────────────────────────────────────────────────────────────

_ND = (_OBS + "ndwi",)

INDEX = types.MappingProxyType({e.id: e for e in (
    IndexLevel(id="ndwi", formula="(r3 - r8) / (r3 + r8), NaN when the denominator is 0",
               bands10=("B03", "B08"), t_k=0.0, native_nd=True,
               s_k_rule="1 (y = x exactly)", reference="McFeeters 1996, doi:10.1080/01431169608948714",
               role="BASE; byte-identical to production (marea.extract)", tier="P0",
               impl=(_OBS + "index_ndwi",) + _ND),
    IndexLevel(id="mndwi", formula="(r3 - r11) / (r3 + r11); r3 the 2 x 2 mean", at20=True,
               bands10=("B03",), bands20=("B11",), t_k=0.0, native_nd=True,
               reference="Xu 2006, doi:10.1080/01431160600589179", tier="1A'",
               impl=(_OBS + "index_mndwi",) + _ND + _I20),
    IndexLevel(id="nir", formula="-r8", bands10=("B08",), t_k=None,
               t_k_rule=("-t_NIR, t_NIR = the pooled Otsu of r8 over the clear observations of "
                         "keep_full(P0, W3) per site, from per-year histograms, frozen"),
               hist_range=(-1.5, 0.2), hist_bins=1700,
               reference="Granadeiro et al. 2021; Bue et al. 2020", tier="1A'",
               impl=(_OBS + "index_nir", _OBS + "_rho")),
    IndexLevel(id="awei_sh", formula="r2 + 2.5 r3 - 1.5 (r8 + r11) - 0.25 r12", at20=True,
               bands10=("B02", "B03", "B08"), bands20=("B11", "B12"), t_k=0.0,
               hist_range=(-3.0, 1.5), hist_bins=2250,
               reference="Feyisa et al. 2014, doi:10.1016/j.rse.2013.08.029", tier="1A'",
               impl=(_OBS + "index_awei_sh", _OBS + "_rho") + _I20),
    IndexLevel(id="awei_nsh", formula="4 (r3 - r11) - (0.25 r8 + 2.75 r12)", at20=True,
               bands10=("B03", "B08"), bands20=("B11", "B12"), t_k=0.0,
               hist_range=(-4.0, 4.0), hist_bins=2000,
               reference="Feyisa et al. 2014", tier="1B",
               impl=(_OBS + "index_awei_nsh", _OBS + "_rho") + _I20),
    IndexLevel(id="wri", formula="(r3 + r4) / (r8 + r11), clipped to [0, 10]", at20=True,
               bands10=("B03", "B04", "B08"), bands20=("B11",), t_k=1.0, log_hist=True,
               hist_range=(-7.0, 2.4), hist_bins=1880,
               reference="Shen & Li 2010, doi:10.1109/GEOINFORMATICS.2010.5567762", tier="1B",
               impl=(_OBS + "index_wri",) + _I20),
    IndexLevel(id="ndwi20", formula="(r3 - r8A) / (r3 + r8A); r3 the 2 x 2 mean", at20=True,
               bands10=("B03",), bands20=("B8A",), t_k=0.0, native_nd=True,
               reference="resolution-matched control (splits MNDWI - NDWI)", tier="1A'",
               impl=(_OBS + "index_ndwi20",) + _ND + _I20),
)})


# ─────────────────────────────────────────────────────────────────────────────
#  water detection (§2.6)
# ─────────────────────────────────────────────────────────────────────────────

DETECT = types.MappingProxyType({e.id: e for e in (
    DetectLevel(id="D1", rule="W = V & (x > t_k)", reference="Sagar et al. 2017; marea.extract",
                role="BASE", tier="P0", impl=(_OBS + "detect_d1",)),
    DetectLevel(id="D2", per_scene=True,
                rule=("Omega = keep_full(P0, W3) U a 300 m ring of P0 permanent water (>= 0.98) "
                      "and stable land (<= 0.02); per scene with >= 1,000 valid px in Omega: "
                      "t_s = Otsu (256 bins over the valid range), eta = between/total variance, "
                      "m = minority share; t_s if eta >= 0.75 and m >= 0.15, else t_k"),
                reference=("Otsu 1979; Cao et al. 2019 (Bmax Otsu), doi:10.3390/w11040786; "
                           "hydrafloods bmax_otsu"), tier="1A'",
                params={"min_px": 1000, "bins": 256, "eta_min": 0.75, "minority_min": 0.15,
                        "ring_m": 300.0, "water_freq": 0.98, "land_freq": 0.02},
                impl=(_OBS + "detect_threshold", _OBS + "d2_threshold", _OBS + "otsu",
                      _OBS + "otsu_hist")),
    DetectLevel(id="D3", per_scene=True,
                rule=("b = x > t_k on V; Canny (sigma 1 px, thresholds 0.05) on b within V eroded "
                      "by 2 px and the bounding box of dom + 1 km; 8-connected edges >= 50 px; "
                      "buffer = edges dilated 5 px within V; t_s = Otsu(x[buffer]) if the buffer "
                      "holds >= 500 px, else t_k; scenes with a valid share < 20 % skipped"),
                reference=("Donchyts et al. 2016, doi:10.3390/rs8050386; Markert et al. 2020, "
                           "doi:10.3390/rs12152469"), tier="1A'",
                params={"sigma": 1.0, "low": 0.05, "high": 0.05, "erode_px": 2,
                        "min_edge_px": 50, "buffer_px": 5, "min_buffer_px": 500,
                        "min_valid_share": 0.2, "bins": 256,
                        "region": "bounding box of dom plus 1 km"},
                impl=(_OBS + "detect_threshold", _OBS + "d3_threshold", _OBS + "otsu",
                      _OBS + "otsu_hist") + _HOOD),
    DetectLevel(id="D4", index_free=True, rule="W = V & SCL = 6",
                reference="Sen2Cor ATBD 2.10 §3.8; water.py 'scl'",
                role="naive baseline; excluded from every headline robustness set", tier="1B",
                bands=("SCL",), params={"water_scl": 6}, impl=(_OBS + "detect_d4",)),
    DetectLevel(id="D5", index_free=True, indices=("ndwi", "mndwi", "awei_nsh"),
                rule="W = V & [(NDWI > 0) + (MNDWI > 0) + (AWEI_nsh > 0) >= 2]",
                reference="in-house (F4); no published rule is claimed", tier="1A'",
                bands=("B03", "B08", "B11", "B12"), params={"votes": 2},
                impl=(_OBS + "detect_d5",)),
)})


# ─────────────────────────────────────────────────────────────────────────────
#  water-level source (§1.6; elevation units only)
# ─────────────────────────────────────────────────────────────────────────────

_LEV = "experiments.benchmark.levels."

LEVEL = types.MappingProxyType({e.id: e for e in (
    LevelSource(id="LB", rule="h_B(t), the frozen boundary series over UNION_PERIOD (§1.5)",
                reference="EOT20 + outer IOC gauge consensus (EOT20 only at Villaviciosa)",
                role="BASE; common conditions; P0; families P1, P2", tier="P0",
                sites=tuple(config.SITES), impl=(_LEV + "make_provider",)),
    LevelSource(id="LT", rule=("per-pixel global tide model through eo_tides (EOT20; FES2022 once "
                               "installed; locally ranked ensemble with >= 3 models), on a 1 km "
                               "node grid, each pixel its nearest node"),
                reference="Bishop-Taylor et al. 2025, doi:10.21105/joss.07786",
                role="DEA, NIDEM and Coast X-Ray practice; family P3", tier="1A'",
                sites=tuple(config.SITES), params={"node_km": 1.0, "models": ["EOT20"]},
                impl=(_LEV + "level_lt",)),
    LevelSource(id="LM", rule=("regional SSH (CMEMS NWS AMM15 1.5 km; IBI 1/36 deg at the rias), "
                               "15-min preferred, hourly with cubic interpolation; per 1-km s bin "
                               "from wet channel cells, demeaned against h_B over the window; "
                               "unresolved if the nearest wet cell is > 1 km away"),
                reference="Mason et al. 2010 practice; copernicusmarine toolbox",
                role="family P3; degenerate at Villaviciosa (flagged)", tier="1A'",
                sites=tuple(config.SITES), params={"bin_km": 1.0, "unresolved_km": 1.0},
                impl=(_LEV + "level_lm",)),
    LevelSource(id="LG", rule=("gauge network: linear in s between the boundary (s = 0) and the "
                               "judges at s_g, constant beyond the last gauge"),
                reference="Costa et al. 2023 practice", tier="1A'",
                role="labelled upper bound; never in lag scoring nor a confirmatory family",
                sites=("escalda", "wadden", "ems"), confirmatory=False, upper_bound=True,
                impl=(_LEV + "level_lg",)),
)})


# ─────────────────────────────────────────────────────────────────────────────
#  temporal windows (§2.7)
# ─────────────────────────────────────────────────────────────────────────────

#: survey-centred windows per site (§2.7)
SURVEY_WINDOWS = types.MappingProxyType({
    "escalda": ("V2019L3", "V2021L3", "V2023L3", "V2021L1", "V2021L5"),
    "wadden": ("V2022L3", "V2022L5", "V2019L3"),
    "ems": ("V2020L3", "V2020L5"),
    "villaviciosa": (),
    "ferrol": (),
})


def _nested(k, campaign, role=""):
    return Window(id=f"W{k}", start=f"{2026 - k}-01-01", end="2026-01-01", kind="nested",
                  num=k, rule=f"the {k} calendar year(s) ending 2025-12-31", tier=campaign,
                  role=role, reference="spec §2.7")


def _year(y, campaign):
    return Window(id=f"Y{y}", start=f"{y}-01-01", end=f"{y + 1}-01-01", kind="year", num=y,
                  rule=f"calendar year {y} (archive replicate)", tier=campaign,
                  reference="spec §2.7")


def survey_window(wid):
    """The survey-centred window ``V<epoch>L<length>`` (odd length, epoch
    at mid-year): calendar years epoch - (L - 1)/2 .. epoch + (L - 1)/2."""
    m = re.fullmatch(r"V(\d{4})L(\d{1,2})", str(wid))
    if not m:
        raise ValueError(f"{wid!r} is not a survey-centred window id V<epoch>L<len>")
    e, n = int(m.group(1)), int(m.group(2))
    if n < 1 or n % 2 == 0:
        raise ValueError(f"{wid}: the length must be odd so the epoch is the centre year")
    h = (n - 1) // 2
    sites = tuple(s for s, ws in SURVEY_WINDOWS.items() if wid in ws)
    return Window(id=wid, start=f"{e - h}-01-01", end=f"{e + h + 1}-01-01", kind="survey",
                  num=e * 10 + n, rule=f"{n} year(s) centred on the {e} survey epoch",
                  tier="1B", sites=sites, reference="spec §2.7, §5.9")


WINDOW = types.MappingProxyType({w.id: w for w in (
    _nested(1, "1A'", "native cells; 1B"), _nested(2, "1B"), _nested(3, "P0", "BASE"),
    _nested(5, "1B"), _nested(9, "1B"),
    *(_year(y, "1A'" if y >= 2023 else "1B") for y in range(2017, 2026)),
    *(survey_window(w) for w in dict.fromkeys(w for ws in SURVEY_WINDOWS.values() for w in ws)),
)})


def window(wid):
    """The :class:`Window` of a window id (registry, or a well-formed
    survey-centred id)."""
    if isinstance(wid, Window):
        return wid
    if wid in WINDOW:
        return WINDOW[wid]
    if str(wid).startswith("V"):
        return survey_window(wid)
    raise KeyError(f"unknown window {wid!r}; known: {tuple(WINDOW)}")


def window_num(wid):
    """The integer of a window in the subset seeds: W<k> -> k, Y<yyyy> ->
    yyyy, V<e>L<l> -> 10 e + l."""
    return int(window(wid).num)


# ─────────────────────────────────────────────────────────────────────────────
#  number-of-images curves (§2.8, §5.10)
# ─────────────────────────────────────────────────────────────────────────────

_K1_GRID = ((20, 5, "d"), (45, 5, "d"), (90, 3, "d"), (180, 2, "d"), (330, 2, "n"),
            ("all", 1, "all"))
_K_NS = tuple(n for n, _, _ in _K1_GRID if n != "all")
_RAND_IMPL = ("experiments.benchmark.settings.rand_subsets",
              "experiments.benchmark.settings.rand_perm",
              "experiments.benchmark.settings.subset_kind")


def _labelled(ns, R, T):
    """``(N, R, kind)`` with the kind expected at a record of T scenes."""
    return tuple((n, R, "d" if R <= T // n else "n") for n in ns) + (("all", 1, "all"),)

CURVES = types.MappingProxyType({e.id: e for e in (
    CurveSet(id="K1", grid=_K1_GRID, settings=("C1.floor0.ndwi.D1",), windows=("W3",),
             rule="K1': P0, W3, A-mode", tier="1A'", impl=_RAND_IMPL,
             methods="all Tier-1 elevation and lag methods except Granadeiro"),
    CurveSet(id="K3", grid=_K1_GRID, settings=("C2.floor0.ndwi.D1", "C11.floor0.ndwi.D1"),
             windows=("W3",), rule="C2 and C11 (C3 if C11 is unavailable), the K1' grid",
             tier="1B", methods="all Tier-1 cheap methods", impl=_RAND_IMPL),
    CurveSet(id="K4", grid=((90, 3, "d"), (180, 3, "d"), (330, 3, "d"), (465, 2, "d"),
                            (900, 2, "n"), ("all", 1, "all")),
             settings=("C1.floor0.ndwi.D1",), windows=("W9",), rule="P0, W9 grid", tier="1B",
             methods="all Tier-1 cheap methods", impl=_RAND_IMPL),
    CurveSet(id="K5_W1", grid=((45, 3, "d"), (86, 2, "d")), settings=("C1.floor0.ndwi.D1",),
             windows=("W1",), rule="cross-window equal N", tier="1B", impl=_RAND_IMPL),
    CurveSet(id="K5_W2", grid=((45, 5, "d"), (90, 3, "d"), (158, 2, "d")),
             settings=("C1.floor0.ndwi.D1",), windows=("W2",), rule="cross-window equal N",
             tier="1B", impl=_RAND_IMPL),
    CurveSet(id="K5_W5", grid=((45, 5, "d"), (90, 5, "d"), (172, 4, "d"), (465, 2, "n")),
             settings=("C1.floor0.ndwi.D1",), windows=("W5",), rule="cross-window equal N",
             tier="1B", impl=_RAND_IMPL),
    CurveSet(id="K5_W9", grid=((45, 5, "d"), (172, 5, "d")), settings=("C1.floor0.ndwi.D1",),
             windows=("W9",), rule="cross-window equal N (W9)", tier="1B", impl=_RAND_IMPL),
    CurveSet(id="K6", grid=((45, 3, "d"), (90, 3, "d"), (180, 2, "d")),
             settings=("C1.floor0.ndwi.D1",), windows=("W3",),
             sites=("villaviciosa", "escalda"), rule="Granadeiro W3", tier="1B",
             methods="E3 (+ L06)", impl=_RAND_IMPL),
    CurveSet(id="K2", grid=_labelled(_K_NS, 3, 465),
             mode="F", settings=("C1.floor0.ndwi.D1",), windows=("W3",), rule="P0, F-mode",
             tier="2", methods="methods on the harness domain", impl=_RAND_IMPL),
    CurveSet(id="K7", grid=_labelled(_K_NS, 5, 465),
             settings=("C1.floor0.mndwi.D1", "C1.floor0.ndwi.D3", "C1.floor0.mndwi.D3"),
             windows=("W3",), rule="G3 corners", tier="2", impl=_RAND_IMPL),
    CurveSet(id="K9", grid=tuple((n, 5, "win") for n in _K_NS),
             family="win", settings=("C1.floor0.ndwi.D1",), windows=("W3",),
             rule="contiguous sub-windows", tier="2",
             impl=("experiments.benchmark.settings.win_subset",)),
)})


def subset_kind(T, N, R):
    """The batch kind of a curve point: "all" (N = all), "disjoint" when R
    <= floor(T / N), else "nested" (§2.8). N > T raises ValueError."""
    if N == "all":
        return "all"
    N, T, R = int(N), int(T), int(R)
    if N < 1 or R < 1:
        raise ValueError("N and R must be >= 1")
    if N > T:
        raise ValueError(f"N = {N} exceeds the record (T = {T})")
    return "disjoint" if R <= T // N else "nested"


def rand_perm(site, wid, r, T):
    """``default_rng([20261008, SITE_ID, WINDOW_NUM, r]).permutation(T)``."""
    import numpy as np

    return np.random.default_rng([SEED_SUBSETS, config.SITE_ID[site], window_num(wid),
                                  int(r)]).permutation(int(T))


def rand_subsets(site, wid, T, N, R):
    """The ``rand`` subsets of one curve point: ``(kind, [sel_0, ...])``.

    Disjoint batches ``sort(perm_0[r N:(r + 1) N])`` when R <= floor(T/N),
    else nested ``sort(perm_r[:N])``; never mixed (§2.8). N = "all": one
    ``arange(T)``. Positions are record positions of the window (int64).
    """
    import numpy as np

    kind = subset_kind(T, N, R)
    if kind == "all":
        return kind, [np.arange(int(T), dtype=np.int64)]
    N = int(N)
    if kind == "disjoint":
        p0 = rand_perm(site, wid, 0, T)
        return kind, [np.sort(p0[r * N:(r + 1) * N]).astype(np.int64) for r in range(int(R))]
    return kind, [np.sort(rand_perm(site, wid, r, T)[:N]).astype(np.int64)
                  for r in range(int(R))]


def win_subset(site, wid, T, N, r):
    """The ``win`` family: ``arange(s, s + N)`` with s =
    ``default_rng([20261012, SITE_ID, WINDOW_NUM, N, r]).integers(0, T - N + 1)``."""
    import numpy as np

    T, N = int(T), int(N)
    if N > T:
        raise ValueError(f"N = {N} exceeds the record (T = {T})")
    s = int(np.random.default_rng([SEED_WINDOWS, config.SITE_ID[site], window_num(wid), N,
                                   int(r)]).integers(0, T - N + 1))
    return np.arange(s, s + N, dtype=np.int64)


def matched_subsets(site, wid, positions, n_rule, draws=3):
    """The ``matched-<C>`` family (§2.2): ``draws`` subsets of size
    ``n_rule`` of the C1-record ``positions``, each
    ``sort(positions[default_rng([20261016, SITE_ID, WINDOW_NUM, n_rule, d])
    .permutation(len(positions))[:n_rule]])``."""
    import numpy as np

    pos = np.asarray(positions, np.int64)
    n_rule = int(n_rule)
    if n_rule > len(pos):
        raise ValueError(f"n_rule = {n_rule} exceeds the C1 record ({len(pos)})")
    out = []
    for d in range(int(draws)):
        rng = np.random.default_rng([SEED_MATCHED, config.SITE_ID[site], window_num(wid),
                                     n_rule, d])
        out.append(np.sort(pos[rng.permutation(len(pos))[:n_rule]]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  the Setting, ids, cells and effective keys (§2.9)
# ─────────────────────────────────────────────────────────────────────────────

REGISTRIES = types.MappingProxyType({"cloud": CLOUD, "prep": PREP, "index": INDEX,
                                     "detect": DETECT, "level": LEVEL, "window": WINDOW,
                                     "curves": CURVES})
_SETTING_RE = re.compile(r"(?P<cloud>[A-Za-z0-9_]+)\.(?P<prep>[A-Za-z0-9_]+)\."
                         r"(?P<index>[A-Za-z0-9_]+)\.(?P<detect>[A-Za-z0-9_]+)"
                         r"(?:@(?P<level>[A-Za-z0-9_]+))?")


@dataclass(frozen=True)
class Setting:
    """One level of each factor: (cloud, prep, index, detect) plus the
    water-level source of elevation units (LB by default).

    >>> Setting().id
    'C1.floor0.ndwi.D1'
    >>> Setting.parse("C11.floor0.mndwi.D3@LM").level
    'LM'
    """

    cloud: str = "C1"
    prep: str = "floor0"
    index: str = "ndwi"
    detect: str = "D1"
    level: str = "LB"

    def __post_init__(self):
        for f in ("cloud", "prep", "index", "detect", "level"):
            v = getattr(self, f)
            if v not in REGISTRIES[f]:
                raise ValueError(f"unknown {f} level {v!r}; known: {tuple(REGISTRIES[f])}")

    @classmethod
    def parse(cls, sid):
        """The Setting of a setting id ``<cloud>.<prep>.<index>.<detect>[@<level>]``."""
        if isinstance(sid, Setting):
            return sid
        m = _SETTING_RE.fullmatch(str(sid))
        if not m:
            raise ValueError(f"{sid!r} is not a setting id <cloud>.<prep>.<index>.<detect>"
                             "[@<level>]")
        d = m.groupdict()
        if d["level"] == "LB":
            raise ValueError(f"{sid!r}: the level suffix is omitted for LB")
        return cls(cloud=d["cloud"], prep=d["prep"], index=d["index"], detect=d["detect"],
                   level=d["level"] or "LB")

    @property
    def id(self):
        base = f"{self.cloud}.{self.prep}.{self.index}.{self.detect}"
        return base if self.level == "LB" else f"{base}@{self.level}"

    def __str__(self):
        return self.id

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)

    def entries(self):
        """``{factor: entry}`` of the four observation factors and the level."""
        return {f: REGISTRIES[f][getattr(self, f)]
                for f in ("cloud", "prep", "index", "detect", "level")}

    @property
    def pixel_rule(self):
        """The pixel cloud level applied per observation (C1 for the scene rules)."""
        return CLOUD[self.cloud].pixel_rule

    @property
    def scene_rule(self):
        """The scene-rule level, or None for a pixel level."""
        return self.cloud if CLOUD[self.cloud].kind == "scene" else None

    @property
    def bands(self):
        """Store bands the observation of this setting reads (sorted as the
        stores order them)."""
        need = set(CLOUD[self.pixel_rule].bands) | set(INDEX[self.index].bands)
        d = DETECT[self.detect]
        need |= set(d.bands)
        for i in d.indices:
            need |= set(INDEX[i].bands)
        order = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12", "SCL", "CLD")
        return tuple(b for b in order if b in need) + tuple(sorted(need - set(order)))


#: the base configuration P0 (§5.2)
P0 = Setting()


def setting_id(cloud="C1", prep="floor0", index="ndwi", detect="D1", level="LB"):
    """The id of a setting given its levels (validated)."""
    return Setting(cloud, prep, index, detect, level).id


def parse_setting_id(sid):
    """:meth:`Setting.parse`."""
    return Setting.parse(sid)


def cell_id(setting, wid):
    """``(setting id, window id)``, validated."""
    return Setting.parse(setting).id, window(wid).id


@dataclass(frozen=True)
class Signature:
    """A method's dependency signature (§2.9): the factors it reads, and
    whether it consumes the harness domain (which reads cloud, prep, index
    and detect through V and W)."""

    reads: frozenset = frozenset(FACTORS)
    domain: bool = False

    def __post_init__(self):
        r = frozenset(self.reads)
        bad = r - set(FACTORS)
        if bad:
            raise ValueError(f"unknown factors {sorted(bad)}; known: {FACTORS}")
        object.__setattr__(self, "reads", r)

    @property
    def factors(self):
        """The factors the effective key keeps, in :data:`FACTORS` order."""
        r = set(self.reads) | (set(DOMAIN_FACTORS) if self.domain else set())
        return tuple(f for f in FACTORS if f in r)


def effective_cell(setting, wid, signature):
    """``{factor: level or "*"}``: the cell projected onto ``signature``."""
    s = Setting.parse(setting)
    w = window(wid)
    keep = set(signature.factors)
    vals = {"cloud": s.cloud, "prep": s.prep, "index": s.index, "detect": s.detect,
            "level": s.level, "window": w.id}
    return {f: (vals[f] if f in keep else "*") for f in FACTORS}


def effective_key(setting, wid, signature):
    """The effective key of a cell for a method:
    ``<cloud>.<prep>.<index>.<detect>@<level>|<window>`` with ``*`` for the
    factors the signature does not read (the level always written, so the
    key is unambiguous). One unit runs per effective key; it enters
    ``inputs_key`` (§2.9, §13.2)."""
    c = effective_cell(setting, wid, signature)
    return f"{c['cloud']}.{c['prep']}.{c['index']}.{c['detect']}@{c['level']}|{c['window']}"


def code_shas(setting, wid, signature):
    """``{"<factor>:<level>": entry_sha}`` of every registry entry the
    effective key uses (the per-entry part of a unit's ``code_key``)."""
    c = effective_cell(setting, wid, signature)
    out = {}
    for f in FACTORS:
        v = c[f]
        if v == "*":
            continue
        e = window(v) if f == "window" else REGISTRIES[f][v]
        out[f"{f}:{v}"] = entry_sha(e)
        if f == "cloud" and CLOUD[v].kind == "scene":
            pr = CLOUD[v].pixel_rule
            out[f"cloud:{pr}"] = entry_sha(CLOUD[pr])
        if f == "detect":
            for i in DETECT[v].indices:
                out[f"index:{i}"] = entry_sha(INDEX[i])
    return dict(sorted(out.items()))


def registry_shas(factor):
    """``{level: entry_sha}`` of one registry."""
    return {k: entry_sha(e) for k, e in REGISTRIES[factor].items()}


def frozen_sections():
    """The FROZEN sections of the settings (Appendix A): ``{section:
    {"levels": {id: fields}, "sha256": {id: entry_sha}}}`` for cloud, prep,
    index, detect, level, window and curves (the per-entry hashes are what a
    unit's key reads, never the section as a whole)."""
    return {f: {"levels": {k: e.as_dict() for k, e in REGISTRIES[f].items()},
                "sha256": registry_shas(f)}
            for f in ("cloud", "prep", "index", "detect", "level", "window", "curves")}


def index_params(index, t_nir=None, s_k=None):
    """``{"t_k": float, "s_k": float}`` of an index for the observation.

    NDWI: (0, 1) by definition. I3 NIR needs ``t_nir`` (t_k = -t_NIR); every
    other index needs ``s_k`` from the settings pass (§2.5). Missing values
    raise ValueError: no default may silently change an index's scale.
    """
    e = INDEX[index]
    if e.t_k is None:
        if t_nir is None:
            raise ValueError(f"{index}: t_k is data-dependent ({e.t_k_rule}); pass t_nir")
        t_k = -float(t_nir)
    else:
        t_k = float(e.t_k)
    if index == "ndwi":
        if s_k not in (None, 1, 1.0):
            raise ValueError("ndwi: s_k is 1 by definition")
        return {"t_k": t_k, "s_k": 1.0}
    if s_k is None:
        raise ValueError(f"{index}: s_k comes from the settings pass ({e.s_k_rule}); pass s_k")
    if not float(s_k) > 0:
        raise ValueError(f"{index}: s_k must be > 0")
    return {"t_k": t_k, "s_k": float(s_k)}
