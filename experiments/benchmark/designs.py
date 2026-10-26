# -*- coding: utf-8 -*-
"""Validation v2 — the design tables, the plan CSV and the ten-day cap (spec §5, §13.3, §14).

Question
--------
Which units does each design of validation v2 run (site x setting x window x
subset family x mode x N x replicate x method), what does each unit cost in
worker-hours, RAM and read volume, and does the Tier 1A' campaign fit the
ten-day cap of the server once the pre-ordered cuts are applied?

Pre-registered rules (docs/validation_v2_spec.md §5, §13.2, §13.3, §14;
FROZEN['design'] is the sha256 of every written ``plans/<design>.csv``)
---------------------------------------------------------------------------
* The method catalogue (:data:`METHODS`) holds one row per UNIT a design
  schedules, with the products it writes (``E0`` = the marea_pair unit:
  no delay E0a, MAREA E0b and the L01 lag product), its dependency
  signature (the factors it reads and whether it consumes the harness
  domain, §2.9), the level sources it accepts (§4.3), whether it is a frame
  method (one matched-N draw, §2.2), and the cost model of §14.1: worker
  hours of one cell over the five sites at W3 and N = all, the worst-case
  RAM at the Ems (§14.3). The cards of V2/V3 replace these numbers by the
  pilot's measurements (``pilot units``).
* A design (:data:`DESIGNS`) enumerates units; a unit whose effective key
  (site, effective cell, method, variant, family, mode, N, rep) equals an
  earlier unit's in the same design is written as ``shared`` (cost 0): one
  unit runs per effective key and the scorer maps its products to every
  cell that shares it (§2.9). Units a method cannot run are written as
  ``not_applicable`` rows, never left out (§4.4).
* Cost (§14.1): ``cost_wh / 5`` per site, weighted by the site's frame size
  (``FRAME_PX``), times the window scale (W1 and calendar years 0.4, W2
  0.7, W3 1, W5 1.7, W9 3; survey windows by length), times N / T for a
  curve point (T the window's expected record), plus a per-instance
  constant (harness: 0.3 wh per cell over five sites at N = all; 1 min per
  site-instance on a curve). Granadeiro (E3) is costed in hours of its own
  pane (G-base 26-41 h over five sites, by frame size).
* Projection (§5.14, §14.2): days = fixed Stage 0/T/final days (3.9) +
  cheap wh / (workers x 24 h x utilisation) + half of Granadeiro's wall
  time (two of four workers run beside it). Units are counted once across
  designs (the first design in queue order owns a shared effective key).
  The cut list is applied in its pre-registered order until the projection
  fits the cap; never-cut items are never cut, and the ones plan rows can
  show are CHECKED on the kept units (:func:`never_cut_check`): the
  campaign fits only when they hold.
* Regional models (LM rows, the L16 lag unit): a unit is planned only where
  a CMEMS dataset covers its whole window (``fetch.cmems.choose_dataset``,
  imported lazily); elsewhere it is a not_applicable row whose note says
  "NOT AVAILABLE" and why. The AMM15 archive starts 2024-08-04, so at the
  Dutch sites LM and L16 run at W1 (= Y2025) only; LM rows run at the
  longest window of :data:`LM_WINDOWS` the archive covers (the AMM15
  archive depth is a decision for the user before lock-A: W1 only, or the
  AMM7 7 km reanalysis at W3).
* The E0 'nodelay' unit under LT, LM or LG writes no delay (E0a) only
  (``Unit.products``): MAREA and its lag are LB by design.
* Seeds and subsets are the settings module's (``rand_subsets``,
  ``matched_subsets``); N = all is one unit with rep 0.
* Scene instants (user decision of 2026-10-14): every unit uses the
  catalogue / product-name time (the datatake start), as production does.
  The solar-angle sensing-time diagnostic (``validation_v2 diag time``) is
  no factor, setting or row of any design.
* Error attribution (the user's question if the errors are large: noise,
  lag or elevation estimation?) is read from rows that already exist on
  the same scenes, pixels and settings: the E0 unit writes no delay (E0a,
  tau = 0) and MAREA (E0b, band lags) from ONE run; the levels design adds
  no delay under the measured gauge level (E0 'nodelay' at LG, a labelled
  upper bound, at the sites with judges) and, by the user decision of
  2026-10-15, MAREA's inversion with the judges' measured band lags (E0
  'gaugelag' -> E0g, same sites), which separates the cost of estimating
  the lag from the cost of the one-lag-per-band model; the base design's
  seed rows and alternate-scene halves and the calendar-year windows give
  the noise floors (:func:`attribution_rows`).

This module imports only the standard library and ``config``/``settings``
at import time (``fetch.cmems`` lazily, for the regional archives); no
import-time side effects. ``write_plan`` writes ``plans/<design>.csv`` (LF,
deterministic) under the validation v2 root.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from dataclasses import dataclass, field, replace
from typing import Optional

from experiments.benchmark import config, settings

#: the validation v2 output root (spec §13.1)
V2_ROOT = "results/validation_v2"
PLANS_DIR = f"{V2_ROOT}/plans"
PLAN_FORMAT = "validation-v2-plan/1"

#: elevation sites (Ferrol: lag only, plus P0 repeatability) and lag sites
ELEV_SITES = ("villaviciosa", "escalda", "wadden", "ems")
LAG_SITES = tuple(config.SITES)
DUTCH_SITES = ("escalda", "wadden", "ems")
#: frame sizes (10 m pixels, spec §1.1) for the per-site cost weights
FRAME_PX = {"villaviciosa": 915 * 915, "ferrol": 905 * 1707, "escalda": 1467 * 2101,
            "wadden": 3425 * 2776, "ems": 3206 * 3413}
_MEAN_PX = sum(FRAME_PX.values()) / len(FRAME_PX)
#: share of the frame in dom (spec §14.4: 25-30 %) for the read estimates
DOM_SHARE = 0.27
#: expected record lengths (spec §2.7)
T_EXPECTED = {"W1": 172, "W2": 317, "W3": 463, "W5": 750, "W9": 1340}
T_YEAR = 155
#: window cost scale relative to W3 (spec §14.1)
WINDOW_SCALE = {"W1": 0.4, "W2": 0.7, "W3": 1.0, "W5": 1.7, "W9": 3.0}
#: harness cost (observations, domain, levels, fills) of one N = all cell
#: over five sites, and the constant of one curve site-instance (h)
INSTANCE_WH = 0.3
CURVE_INSTANCE_H = 1.0 / 60.0
#: Granadeiro: G-base hours over five sites at W3 (spec §14.2: 26-41 h)
GRAN_BASE_H = 33.5
#: projection (spec §14.1-14.2)
WORKERS = 4
UTILISATION = 0.78
CAP_DAYS = 10.0
GRAN_BESIDE = 2
#: Stage 0a + 0b + 0c, Stage T and the final scoring and report (days)
FIXED_DAYS = {"0a fetch, inputs, check-env": 0.5,
              "0b prepare, truth, pilots": 1.0,
              "0c gates, control, lock-A": 1.0,
              "T timing": 0.4,
              "final scoring and report": 1.0}
CONTINGENCY = 0.20
#: bytes per stored observation (int16 B02 B03 B04 B08 B8A B11 B12, uint8
#: SCL, CLD and the cloud bits) for the read estimates
BYTES_PER_BAND = {"B02": 2, "B03": 2, "B04": 2, "B08": 2, "B8A": 2, "B11": 2, "B12": 2,
                  "SCL": 1, "CLD": 1}
#: expected share of the record a scene rule keeps (matched-N cost only)
RULE_SHARE = {"C2": 0.6, "C6": 0.3, "C7": 0.6, "C8": 0.5}
#: the queue order of §13.7 (Tier 1A'): earlier designs own shared units
QUEUE_1A = ("lag", "base", "published", "levels", "native", "years", "tuning", "g1",
            "ofat", "curves-k1", "granadeiro-base", "granadeiro-nir")
QUEUE_1B = ("windows", "g2", "centred", "years-1b", "published-1b", "ofat-1b",
            "ndwi20-slice", "ixt", "curves-k3", "curves-k4", "curves-k5",
            "granadeiro-nir-1b", "granadeiro-l9", "granadeiro-curves", "granadeiro-tune")

CSV_COLUMNS = ("design", "tier", "priority", "unit_id", "site", "window", "setting_id",
               "method", "variant", "family", "mode", "N", "rep", "kind", "effective_key",
               "status", "est_hours", "est_ram_gb", "est_read_gb", "cut_step", "products",
               "note")
STATUSES = ("planned", "shared", "not_applicable", "reuse")


def log(msg):
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  the method catalogue (units)
# ─────────────────────────────────────────────────────────────────────────────

_ALL = frozenset(settings.FACTORS)
_NO_D = frozenset(settings.FACTORS) - {"detect"}


@dataclass(frozen=True)
class Method:
    """One unit kind a design schedules (spec §3, §4, §14).

    Attributes
    ----------
    id, name : str
    kind : str
        "elev", "lag" or "both" (E0: elevation and the L01 lag product).
    products : tuple of str
        The method rows the unit writes.
    reads : frozenset
        Factors the unit reads (settings.FACTORS).
    domain : bool
        Consumes the harness domain (adds cloud, prep, index, detect).
    levels : tuple
        Level sources it accepts (elevation units; lag units read none).
    cost_wh : float
        Worker-hours of one cell over the five sites at W3, N = all.
    ram_gb : float
        Worst-case RAM at the Ems, W3.
    frame : bool
        Frame method (reads the package-backed frame view; one matched draw).
    expensive : bool
        Granadeiro (own pane; hours, not worker-hours).
    stochastic : bool
        Has a seeded stage (the seed-variance rows of §5.2).
    tuning : int
        Configurations of its §5.12 tuning grid (0: none).
    n_min : int
        Smallest computable N expected (spec §4.1; information only).
    tier : str
        "1" or "2".
    """

    id: str
    name: str
    kind: str
    products: tuple
    reads: frozenset = _ALL
    domain: bool = False
    levels: tuple = ("LB", "LT", "LM", "LG")
    cost_wh: float = 0.1
    ram_gb: float = 2.0
    frame: bool = False
    expensive: bool = False
    stochastic: bool = False
    tuning: int = 0
    n_min: int = 0
    tier: str = "1"

    @property
    def signature(self):
        return settings.Signature(reads=self.reads, domain=self.domain)

    @property
    def elevation(self):
        return self.kind in ("elev", "both")

    @property
    def lag(self):
        return self.kind in ("lag", "both")


METHODS = {m.id: m for m in (
    Method(id="E0", name="marea_pair (no delay, MAREA; L01 lag)", kind="both",
           products=("E0a", "E0b", "L01"), domain=True, levels=("LB",), cost_wh=0.8,
           ram_gb=1.0, stochastic=True, n_min=100),
    Method(id="E1", name="NIDEM/ITEM", kind="elev", products=("E1", "E1u"), cost_wh=0.8,
           ram_gb=2.5, frame=True, tuning=4, n_min=90),
    Method(id="E2", name="DEA Intertidal", kind="elev", products=("E2",), cost_wh=1.5,
           ram_gb=6.0, tuning=6, n_min=45),
    Method(id="E3", name="Granadeiro et al. 2021 (+ L06)", kind="both",
           products=("E3", "L06"), reads=_NO_D, levels=("LB",), cost_wh=0.0, ram_gb=6.0,
           frame=True, expensive=True, tuning=4, n_min=8),
    Method(id="E4", name="LogiTide2DEM", kind="elev", products=("E4",), cost_wh=0.75,
           ram_gb=2.0, tuning=6, n_min=27),
    Method(id="E5", name="frequency (Flood2Topo, Tseng, sampled)", kind="elev",
           products=("E5a", "E5b", "E5c"), cost_wh=0.08, ram_gb=1.0),
    Method(id="E6a", name="Costa et al. 2023 waterline-SDT", kind="elev", products=("E6a",),
           cost_wh=0.5, ram_gb=2.0, frame=True, tuning=4, n_min=5),
    Method(id="E6b", name="Optical S2 waterline (Salameh; E6b-fixed variant)", kind="elev",
           products=("E6b", "E6b-fixed"), cost_wh=0.5, ram_gb=2.0, frame=True, n_min=5),
    Method(id="E7", name="Coast X-Ray", kind="elev", products=("E7",), cost_wh=0.3,
           ram_gb=2.5, frame=True, tuning=6),
    Method(id="E8", name="Bue et al. 2020", kind="elev", products=("E8",), reads=_NO_D,
           cost_wh=0.5, ram_gb=2.0, frame=True, tuning=6, n_min=10),
    Method(id="L01v", name="MAREA per relative orbit", kind="lag", products=("L01v",),
           domain=True, levels=(), cost_wh=0.1, ram_gb=1.0, stochastic=True),
    Method(id="L02", name="m2b concordance", kind="lag", products=("L02",), domain=True,
           levels=(), cost_wh=0.06, ram_gb=1.0),
    Method(id="L03", name="m2c flooded-area rank", kind="lag", products=("L03",), domain=True,
           levels=(), cost_wh=0.06, ram_gb=1.0, n_min=60),
    Method(id="L04", name="XC (wet, index)", kind="lag", products=("L04w", "L04i"),
           domain=True, levels=(), cost_wh=0.06, ram_gb=1.0, n_min=30),
    Method(id="L07", name="LOOP", kind="lag", products=("L07",), domain=True, levels=(),
           cost_wh=0.06, ram_gb=1.0),
    Method(id="L08", name="GM20", kind="lag", products=("L08",), domain=True, levels=(),
           cost_wh=0.2, ram_gb=1.5, stochastic=True),
    Method(id="L09", name="HD26 (H, R2)", kind="lag", products=("L09", "L10"), domain=True,
           levels=(), cost_wh=0.2, ram_gb=1.0, n_min=30),
    Method(id="L11", name="PPL", kind="lag", products=("L11",), domain=True, levels=(),
           cost_wh=0.06, ram_gb=1.0),
    Method(id="L12", name="BT26 shift", kind="lag", products=("L12",), domain=True,
           levels=(), cost_wh=0.06, ram_gb=1.0),
    Method(id="L14", name="global tide models at the gauges (+ L15 ensemble)", kind="lag",
           products=("L14", "L15"), reads=frozenset({"window"}), levels=(), cost_wh=0.02,
           ram_gb=1.0),
    Method(id="L16", name="regional models (AMM15, IBI)", kind="lag", products=("L16",),
           reads=frozenset({"window"}), levels=(), cost_wh=0.05, ram_gb=2.0),
    Method(id="L17", name="zero lag", kind="lag", products=("L17",),
           reads=frozenset({"window"}), levels=(), cost_wh=0.0, ram_gb=0.5),
    Method(id="L18", name="shallow-water celerity", kind="lag", products=("L18",),
           domain=True, reads=frozenset({"window"}), levels=(), cost_wh=0.02, ram_gb=1.0),
    Method(id="L20", name="REF gauge leave-one-out", kind="lag", products=("L20",),
           reads=frozenset({"window"}), levels=(), cost_wh=0.01, ram_gb=0.5),
)}

#: cheap Tier-1 elevation units (every Tier-1 elevation unit but Granadeiro)
CHEAP_ELEV = ("E0", "E1", "E2", "E4", "E5", "E6a", "E6b", "E7", "E8")
#: Tier-1 lag units (the L01 product comes with E0)
LAG_UNITS = ("L02", "L03", "L04", "L07", "L08", "L09", "L11", "L12", "L14", "L16", "L17",
             "L18", "L20")
#: imagery lag units (read the scenes; run in the settings cells)
IMAGE_LAG = ("L02", "L03", "L04", "L07", "L08", "L09", "L11", "L12")
#: elevation units under LT, LM, LG (§5.4: every Tier-1 elevation method but
#: MAREA, which runs under LB by design, and Granadeiro, Tier 2)
LEVEL_UNITS = ("E0", "E1", "E2", "E4", "E5", "E6a", "E6b", "E7", "E8")
#: the tuning grids of §5.12 (configs at Escalda; tuned rows elsewhere)
TUNING_SITE = "escalda"
TUNED_SITES = ("villaviciosa", "wadden", "ems")
#: Granadeiro's cut order keeps G-nir at Escalda; tuned rows at two sites
#: keep the Dutch confirmatory sites
TUNED_KEEP_AFTER_CUT = ("wadden", "ems")


# ─────────────────────────────────────────────────────────────────────────────
#  units
# ─────────────────────────────────────────────────────────────────────────────

def window_T(wid):
    """Expected record length of a window (spec §2.7)."""
    w = settings.window(wid)
    if w.id in T_EXPECTED:
        return T_EXPECTED[w.id]
    if w.kind == "year":
        return 172 if w.num == 2025 else T_YEAR
    return int(round(T_YEAR * len(w.years)))


def window_scale(wid):
    """Cost scale of a window relative to W3 (spec §14.1)."""
    w = settings.window(wid)
    if w.id in WINDOW_SCALE:
        return WINDOW_SCALE[w.id]
    n = len(w.years)
    return {1: 0.4, 2: 0.7, 3: 1.0, 5: 1.7}.get(n, n / 3.0)


@dataclass(frozen=True)
class Unit:
    """One row of a plan: site x cell x subset x method (spec §13.2)."""

    design: str
    site: str
    window: str
    setting: str
    method: str
    variant: str = ""
    family: str = "rand"
    mode: str = "A"
    N: object = "all"
    rep: int = 0
    kind: str = "all"
    status: str = "planned"
    note: str = ""
    cut_step: int = 0
    tier: str = "1A'"
    #: a subset of the method's products (L09 at a window that cannot hold
    #: HD26-H's 365-day span writes L10 only; its L09 is a not_applicable row)
    only_products: tuple = ()

    @property
    def m(self):
        return METHODS[self.method]

    @property
    def effective_key(self):
        """The cell projected onto the method's signature (§2.9)."""
        return settings.effective_key(self.setting, self.window, self.m.signature)

    @property
    def dedupe_key(self):
        """What makes two plan rows the same run."""
        return (self.site, self.effective_key, self.method, self.variant, self.family,
                self.mode, str(self.N), int(self.rep))

    @property
    def unit_id(self):
        """sha256 of the dedupe key (first 16 hex digits)."""
        return hashlib.sha256(json.dumps(list(self.dedupe_key)).encode("utf-8")).hexdigest()[:16]

    def _n_frac(self):
        if self.N == "all":
            return 1.0
        if isinstance(self.N, str) and self.N.startswith("rule:"):
            return RULE_SHARE.get(self.N[5:], 0.5)
        return min(1.0, int(self.N) / window_T(self.window))

    @property
    def est_hours(self):
        """Estimated hours of this unit (worker-hours; Granadeiro: pane hours)."""
        if self.status in ("shared", "not_applicable", "reuse"):
            return 0.0
        w = FRAME_PX[self.site] / _MEAN_PX
        if self.m.expensive:
            base = GRAN_BASE_H / len(FRAME_PX) * w
        else:
            base = self.m.cost_wh / len(FRAME_PX) * w
        return round(base * window_scale(self.window) * self._n_frac(), 4)

    @property
    def est_ram_gb(self):
        w = max(0.25, FRAME_PX[self.site] / FRAME_PX["ems"])
        return round(self.m.ram_gb * w * window_scale(self.window) ** 0.5, 2)

    @property
    def est_read_gb(self):
        """Package bytes read: dom share of the frame x the window's scenes
        x the bytes of the bands the setting reads (estimate)."""
        if self.status in ("shared", "not_applicable", "reuse") or "cloud" not in \
                self.m.signature.factors:
            return 0.0
        bands = settings.Setting.parse(self.setting).bands
        b = sum(BYTES_PER_BAND.get(x, 1) for x in bands)
        T = window_T(self.window) * self._n_frac()
        return round(FRAME_PX[self.site] * DOM_SHARE * T * b / 1e9, 3)

    @property
    def products(self):
        """The method rows this unit writes: the E0 'nodelay' unit under a
        non-boundary level writes no delay (E0a) only: MAREA (E0b) and its
        lag (L01) are LB by design (§1.6). The E0 'gaugelag' unit writes
        E0g only: MAREA's inversion with the judges' measured band lags
        (``levels.gauge_band_lags``; user decision 2026-10-15)."""
        if self.only_products:
            return tuple(self.only_products)
        parts = self.variant.split("+")
        if self.method == "E0" and "nodelay" in parts:
            return ("E0a",)
        if self.method == "E0" and "gaugelag" in parts:
            return ("E0g",)
        return tuple(self.m.products)

    def row(self, priority):
        return {"design": self.design, "tier": self.tier, "priority": priority,
                "unit_id": self.unit_id, "site": self.site, "window": self.window,
                "setting_id": self.setting, "method": self.method, "variant": self.variant,
                "family": self.family, "mode": self.mode, "N": str(self.N),
                "rep": int(self.rep), "kind": self.kind, "effective_key": self.effective_key,
                "status": self.status, "est_hours": self.est_hours,
                "est_ram_gb": self.est_ram_gb, "est_read_gb": self.est_read_gb,
                "cut_step": int(self.cut_step), "products": " ".join(self.products),
                "note": self.note}


def regional_dataset(site, window):
    """``(dataset_id or None, reason)``: the CMEMS dataset that serves LM and
    L16 at a site over a whole window (``fetch.cmems.choose_dataset``; the
    AMM15 archive starts 2024-08-04, so the Dutch sites have W1 = Y2025
    only). None: NOT AVAILABLE there, never substituted (§1.6)."""
    from experiments.benchmark.fetch import cmems   # lazy: numpy/pandas
    w = settings.window(window)
    return cmems.choose_dataset(site, w.start, w.end)


def applicable(method, setting, site, window=None, nodelay=False):
    """``(ok, reason)``: can ``method`` run the cell ``setting`` at ``site``
    (and, given ``window``, over that window)? (spec §4: level sources, LG
    sites, Ferrol elevation; LM and L16 need a regional dataset covering the
    whole window: otherwise "NOT AVAILABLE: <reason>"). ``nodelay``: the E0
    'nodelay' variant (no delay runs under every source, MAREA under LB
    only, §1.6)."""
    m = METHODS[method]
    s = settings.Setting.parse(setting)
    if s.level != "LB":
        if not m.elevation:
            return False, "lag products do not read the level source"
        if site not in settings.LEVEL[s.level].sites:
            return False, f"{s.level} is not available at {site}"
        if s.level not in m.levels and not (nodelay and method == "E0"):
            return False, f"{method} does not run under {s.level} (§4.3)"
        if s.level == "LM" and window is not None:
            dsid, why = regional_dataset(site, window)
            if dsid is None:
                return False, f"NOT AVAILABLE: {why}"
    if method == "L16" and window is not None:
        dsid, why = regional_dataset(site, window)
        if dsid is None:
            return False, f"NOT AVAILABLE: {why}"
    if site == "ferrol" and m.kind == "elev":
        return False, "Ferrol has no elevation truth (elevation at P0 only)"
    return True, ""


def regional_note(site, window):
    """The plan note of an LM / L16 unit: its dataset, and for hourly means
    the measured stamp convention it needs (``fetch cmems``)."""
    from experiments.benchmark.fetch import cmems
    dsid, _ = regional_dataset(site, window)
    if dsid is None:
        return ""
    d = cmems.DATASETS[dsid]
    note = f"{d['model']} {d['resolution']} {d['step_min']} min ({dsid})"
    if d["kind"] == "m":
        note += "; hourly MEANS: needs the stamp convention measured by fetch cmems"
    return note


def _cell_units(design, sites, cells, methods, tier="1A'", **kw):
    """Units of ``methods`` in every (setting, window) of ``cells`` at
    ``sites`` (N = all), with not_applicable rows (the reason in the note;
    "NOT AVAILABLE: ..." where the regional archive does not cover the
    window)."""
    out = []
    for site in sites:
        for sid, wid in cells:
            for meth in methods:
                if site == "ferrol" and METHODS[meth].kind == "elev":
                    continue                    # Ferrol: elevation only in base
                s = settings.Setting.parse(sid)
                variant = kw.get("variant", "")
                nodelay = meth == "E0" and s.level != "LB"
                if nodelay:
                    # no delay (E0a) runs under every source; MAREA under LB only
                    variant = (variant + "+nodelay").lstrip("+")
                ok, why = applicable(meth, sid, site, wid, nodelay=nodelay)
                note = kw.get("note", "")
                if ok and (s.level == "LM" or meth == "L16"):
                    note = "; ".join(x for x in (note, regional_note(site, wid)) if x)
                fam = "rule" if s.scene_rule else kw.get("family", "rand")
                out.append(Unit(design=design, site=site, window=wid, setting=sid,
                                method=meth, variant=variant, family=fam,
                                status="planned" if ok else "not_applicable",
                                note=note if ok else why, tier=tier))
    return out


def _sid(**kw):
    return settings.Setting(**kw).id


P0 = settings.P0.id


# ─────────────────────────────────────────────────────────────────────────────
#  the designs (spec §5)
# ─────────────────────────────────────────────────────────────────────────────

#: the windows of Stage L (§6.6); W1 stands for Y2025 (the same dates)
LAG_WINDOWS = ("W3", "W1", "W2", "Y2023", "Y2024")


def max_span_days(wid):
    """The longest span (days) between two scene dates inside a window
    (end exclusive): 364 for a common calendar year, 365 for 2024."""
    import datetime as _dt
    w = settings.window(wid)
    a = _dt.date.fromisoformat(str(w.start)[:10])
    b = _dt.date.fromisoformat(str(w.end)[:10])
    return (b - a).days - 1


#: the variant tag of the not_applicable L09 row at a window too short for
#: HD26-H's span rule (the L09 unit there writes L10 only)
L09_SPAN_VARIANT = "span365"


def l09_span_rule(wid):
    """``(applicable, note)`` of HD26-H (L09) at a window: its published rule
    needs >= FROZEN hd26.min_span_days (365) between the first and the last
    usable scene; a window whose longest possible span is shorter cannot
    meet it whatever the data (spec §4.2 'window >= 365 d for L09'),
    so L09 is not applicable there. Decided from the window bounds alone,
    never from data or scores; the rule itself is not relaxed."""
    need = float(config.FROZEN["hd26"]["min_span_days"])
    span = max_span_days(wid)
    if span >= need:
        return True, ""
    return False, (f"HD26-H needs >= {need:g} days between its first and last scene (its "
                   f"published rule, FROZEN hd26.min_span_days); a {wid} window holds at most "
                   f"{span} days, so L09 is structurally not applicable here (spec §4.2); "
                   "the unit writes L10 only")


def design_lag(sites=LAG_SITES):
    """Stage L (§5.14, §6.6): the lag rows at P0 on W3 (MAREA's 30-draw
    bootstrap and the GM20 refits: variant boot30), W1, W2, Y2023, Y2024,
    the per-orbit MAREA and the models, physics and reference rows. L16
    (the regional models) is planned only where its archive covers the
    whole window (at the Dutch sites AMM15 starts 2024-08-04: W1 only);
    elsewhere it is a not_applicable "NOT AVAILABLE" row. At a window that
    cannot hold HD26-H's 365-day span (:func:`l09_span_rule`: W1, Y2023)
    the L09 unit writes L10 only and L09 is a not_applicable row."""
    units = []
    lag = ("E0", "L01v") + LAG_UNITS
    for site in sites:
        for wid in LAG_WINDOWS:
            for meth in lag:
                v = "boot30" if (wid == "W3" and meth in ("E0", "L08")) else ""
                note = "lag products scored where the judges cover"
                status = "planned"
                if meth == "L16":
                    ok, why = applicable(meth, P0, site, wid)
                    if ok:
                        note = f"{note}; {regional_note(site, wid)}"
                    else:
                        status, note = "not_applicable", why
                if meth == "L09":
                    ok, why = l09_span_rule(wid)
                    if not ok:
                        units.append(Unit(design="lag", site=site, window=wid, setting=P0,
                                          method=meth, variant=v, status=status,
                                          note=f"{note}; L10 only ({wid} is shorter than "
                                               "HD26-H's 365-day span)",
                                          only_products=("L10",)))
                        units.append(Unit(design="lag", site=site, window=wid, setting=P0,
                                          method=meth,
                                          variant=(f"{v}+" if v else "") + L09_SPAN_VARIANT,
                                          status="not_applicable", note=why,
                                          only_products=("L09",)))
                        continue
                units.append(Unit(design="lag", site=site, window=wid, setting=P0,
                                  method=meth, variant=v, status=status, note=note))
    return units


def design_base(sites=LAG_SITES):
    """Stage B core (§5.2): P0 at W3 for every cheap Tier-1 elevation unit
    at the elevation sites (and at Ferrol, for split-half repeatability and
    coverage), the seed-variance rows (4 extra seeds of the stochastic
    stages) and the alternate-scene halves. (The 'label instants'
    sensitivity of revision 2 is gone: the catalogue times are the default
    instants of every unit, user decision 2026-10-14.)"""
    units = []
    for site in sites:
        for meth in CHEAP_ELEV:
            v = "boot30" if meth == "E0" else ""
            units.append(Unit(design="base", site=site, window="W3", setting=P0, method=meth,
                              variant=v,
                              note="Ferrol: repeatability and coverage only"
                              if site == "ferrol" else ""))
            for half in (0, 1):
                units.append(Unit(design="base", site=site, window="W3", setting=P0,
                                  method=meth, family="half", rep=half,
                                  note="alternate-scene half (§7.3.4)"))
        for meth in ("E0", "L08"):
            for k in range(1, 5):
                units.append(Unit(design="base", site=site, window="W3", setting=P0,
                                  method=meth, variant=f"seed{k}",
                                  note="seed-variance floor (§9.1)"))
    return units


def design_published(sites=LAG_SITES):
    """The as-published rows (§5.3), one per competitor, native settings."""
    rows = (
        ("E0", "C2.raw.ndwi.D1", "W3", "", "production settings (catalogue instants)"),
        ("E1", "C4.floor0.ndwi.D1@LT", "W3", "published", "W9 in Tier 1B; TPXO9 if installed"),
        ("E2", "C1.floor0.ndwi.D1@LT", "W3", "published",
         "C9 needs the sidecar's cloud cover: C1 until then (declared)"),
        ("E4", "C1.floor0.mndwi.D2", "W3", "published",
         "+ frame cloud <= 70 % (declared); >= 27 images"),
        ("E5", P0, "W3", "published", "Flood2Topo, full 1-min series"),
        ("E5", "C1.floor0.mndwi.D1", "W3", "tseng", "Tseng: MNDWI, LB min/max"),
        ("E6a", "C8.floor0.ndwi.D2@LG", "W1", "published",
         "domain rule at >= 0.95 (declared); LB where no judges"),
        ("E6b", "C1.floor0.ndwi.D1@LG", "W3", "salameh", "summer subset; LB where no judges"),
        ("E7", "C1.floor0.ndwi.D1@LT", "W3", "published",
         "AOI coverage >= 99 %; W5 in Tier 1B; Eq. 1 as printed"),
        ("E8", "C6.floor0.nir.D1", "W1", "published", "a = -8, sat >= 0.3, IDW fill"),
    )
    units = []
    for site in sites:
        if site == "ferrol":
            continue
        for meth, sid, wid, variant, note in rows:
            s = settings.Setting.parse(sid)
            if s.level == "LG" and site not in settings.LEVEL["LG"].sites:
                sid = s.replace(level="LB").id
                note += " (LB here)"
            fam = "summer" if variant == "salameh" else ("rule" if settings.Setting.parse(
                sid).scene_rule else "rand")
            units.append(Unit(design="published", site=site, window=wid, setting=sid,
                              method=meth, variant=variant, family=fam, note=note))
        units.append(Unit(design="published", site=site, window="W3", setting="C6.floor0.nir.D1",
                          method="E3", variant="paper", family="rule", status="reuse",
                          note="v8 10 m runs reused (catalogue instants, as every run)"))
    return units


#: LM rows run at the first of these windows the regional archive covers
#: (§1.6: never substituted; the AMM15 archive starts 2024-08-04, so the
#: Dutch sites get W1 = Y2025, paired with MAREA(LB) of the lag design's E0
#: unit at W1). Pending the user's decision on the archive depth: W1 only,
#: or the AMM7 7 km reanalysis (another model) at W3.
LM_WINDOWS = ("W3", "W1")


def design_levels(sites=ELEV_SITES):
    """Level-source rows (§5.4): P0 under LT and LG at W3 (LG at the three
    Dutch sites) and under LM at the longest window of :data:`LM_WINDOWS`
    its regional archive covers (W3 at the rías: IBI reanalysis, hourly
    means, degenerate at Villaviciosa; W1 at the Dutch sites: AMM15, whose
    W3 rows are written as NOT AVAILABLE); no delay (E0+nodelay), never
    MAREA, never Granadeiro."""
    cells = [(_sid(level=lv), "W3") for lv in ("LT", "LM", "LG")]
    units = _cell_units("levels", sites, cells, LEVEL_UNITS)
    for site in sites:
        covered = [w for w in LM_WINDOWS if regional_dataset(site, w)[0] is not None]
        if covered and covered[0] != "W3":
            units += _cell_units("levels", (site,), [(_sid(level="LM"), covered[0])],
                                 LEVEL_UNITS,
                                 note=f"LM at {covered[0]}: the regional archive does not "
                                      "cover W3 (user decision pending: W1 only, or AMM7 7 km "
                                      "at W3)")
    out = []
    for u in units:
        if u.setting.endswith("@LM") and u.site == "villaviciosa" and u.status == "planned":
            u = replace(u, note="LM degenerate at Villaviciosa (flagged); " + u.note)
        if u.setting.endswith("@LG") and u.status == "planned":
            u = replace(u, note="uses in-estuary gauges (upper bound)")
        out.append(u)
    out += gauge_lag_units(sites)
    return out


#: the plan note of the E0 'gaugelag' unit
GAUGE_LAG_NOTE = ("MAREA's inversion with the judges' measured band lags (truth (a) at the "
                  "window, levels.gauge_band_lags); uses in-estuary gauges; attribution row")


def gauge_lag_units(sites=ELEV_SITES, window="W3"):
    """The gauge-lag attribution rows (user decision 2026-10-15): at every
    elevation site with in-estuary judges (the LG sites), one E0 'gaugelag'
    unit at P0 under LB on the same scenes, pixels and setting as the base
    E0 unit, writing E0g = MAREA's inversion with the judges' measured lags
    placed per band (:func:`levels.gauge_band_lags`) instead of the
    archive's. Elsewhere a not_applicable row (no judges to place)."""
    out = []
    for site in sites:
        if site == "ferrol":
            continue                                # no elevation truth
        judged = site in settings.LEVEL["LG"].sites
        out.append(Unit(design="levels", site=site, window=window, setting=P0, method="E0",
                        variant="gaugelag", status="planned" if judged else "not_applicable",
                        note=GAUGE_LAG_NOTE if judged
                        else "no in-estuary judges: the gauge lags cannot be placed"))
    return out


#: the native cells of Tier 1A' that are not in another design (§5.5)
NATIVE_CELLS = (("C1.floor0.mndwi.D2", "W3", "E4 LogiTide2DEM"),
                ("C1.floor0.ndwi.D2", "W1", "E6a Costa"),
                ("C1.floor0.nir.D1", "W1", "E8 Bue"))


def design_native(sites=LAG_SITES):
    """The native cells (§5.5): every cheap unit in each competitor's native
    cell (NIDEM's C4 cell and the P0 cells are in OFAT and base)."""
    units = []
    for sid, wid, who in NATIVE_CELLS:
        units += [replace(u, note=(u.note or f"native cell of {who}"))
                  for u in _cell_units("native", sites, [(sid, wid)],
                                       CHEAP_ELEV + IMAGE_LAG)]
    return units


def design_years(sites=LAG_SITES):
    """Calendar-year replicate windows Y2023-Y2025 at P0 (§5.8); Y2025 = W1."""
    cells = [(P0, w) for w in ("Y2023", "Y2024", "Y2025")]
    return _cell_units("years", sites, cells, CHEAP_ELEV + IMAGE_LAG)


#: §5.12 tuning grids: method -> number of configurations
TUNING_GRIDS = {"E1": 4, "E2": 6, "E4": 6, "E6a": 4, "E7": 6, "E8": 6}


def design_tuning():
    """The symmetric tuning budget (§5.12): each grid at Escalda P0, and the
    selected configuration as a 'tuned' row at the other elevation sites."""
    units = []
    for meth, n in TUNING_GRIDS.items():
        for k in range(n):
            units.append(Unit(design="tuning", site=TUNING_SITE, window="W3", setting=P0,
                              method=meth, variant=f"tune{k}",
                              note="objective CAE (F_const), Escalda only"))
        for site in TUNED_SITES:
            units.append(Unit(design="tuning", site=site, window="W3", setting=P0,
                              method=meth, variant="tuned",
                              note="selected at Escalda (fairness rung 2b)"))
    return units


#: OFAT cells of Tier 1A' (§5.6)
OFAT_1A = {
    "cloud": ("C0", "C1d", "C2", "C3", "C4", "C6", "C8", "C11"),
    "prep": ("raw",),
    "index": ("mndwi", "nir", "awei_sh", "ndwi20"),
    "detect": ("D2", "D3", "D5"),
}
#: frame methods get one matched-N draw, the others three (§2.2)
MATCHED_DRAWS = 3


def design_ofat(sites=LAG_SITES):
    """One-factor-at-a-time cells at W3 (§5.6): 16 cells, matched-N C1
    controls for the scene rules, the 'EOT20 only' boundary and the two
    Villaviciosa boundary variants."""
    cells = []
    for factor, levels_ in OFAT_1A.items():
        for lv in levels_:
            cells.append((_sid(**{factor: lv}), "W3"))
    methods = CHEAP_ELEV + IMAGE_LAG
    units = _cell_units("ofat", sites, cells, methods)
    units = [replace(u, note="evaluated on S0 only; never in a grid")
             if u.setting.startswith("C0.") and u.status == "planned" else u for u in units]
    for rule in ("C2", "C6", "C8"):
        for site in sites:
            for meth in methods:
                if site == "ferrol" and METHODS[meth].kind == "elev":
                    continue
                draws = 1 if METHODS[meth].frame else MATCHED_DRAWS
                for d in range(draws):
                    units.append(Unit(design="ofat", site=site, window="W3", setting=P0,
                                      method=meth, family=f"matched-{rule}",
                                      N=f"rule:{rule}", rep=d, kind="matched",
                                      note=f"C1 control at N_rule of {rule}"))
    for site in sites:
        for meth in methods:
            if site == "ferrol" and METHODS[meth].kind == "elev":
                continue
            units.append(Unit(design="ofat", site=site, window="W3", setting=P0, method=meth,
                              variant="bnd-eot20", note="boundary = EOT20 only (declared)"))
            if site == "villaviciosa":
                for v in ("bnd-gijon", "bnd-cantabrian"):
                    units.append(Unit(design="ofat", site=site, window="W3", setting=P0,
                                      method=meth, variant=v,
                                      note="Villaviciosa boundary variant"))
    return units


#: G1' = C x I x D at W3 (§5.7); C3 replaces C11 if OmniCloudMask is unavailable
G1_CLOUD = ("C1", "C2", "C4", "C11")
G1_INDEX = ("ndwi", "mndwi", "nir")
G1_DETECT = ("D1", "D3")


def design_g1(sites=LAG_SITES, ocm_available=True):
    """G1' (§5.7): the 24-cell C x I x D factorial at W3 for the cheap units
    (Granadeiro excluded); detector-blind units collapse to 12 effective
    cells (shared rows)."""
    clouds = G1_CLOUD if ocm_available else tuple("C3" if c == "C11" else c for c in G1_CLOUD)
    cells = [(_sid(cloud=c, index=i, detect=d), "W3")
             for c in clouds for i in G1_INDEX for d in G1_DETECT]
    note = "" if ocm_available else "C3 replaces C11 (OmniCloudMask unavailable; declared)"
    return _cell_units("g1", sites, cells, CHEAP_ELEV + IMAGE_LAG, note=note)


def design_curves_k1(sites=LAG_SITES):
    """K1' (§5.10): P0 at W3, A-mode, rand subsets at N in {20, 45, 90, 180,
    330} with 5d, 5d, 3d, 2d, 2n replicates (N = all is P0, in base and
    lag); every Tier-1 elevation and lag unit except Granadeiro."""
    cs = settings.CURVES["K1"]
    T = window_T("W3")
    units = []
    for N, R, _ in cs.grid:
        if N == "all":
            continue
        kind = settings.subset_kind(T, N, R)
        for site in sites:
            for meth in CHEAP_ELEV + IMAGE_LAG + ("L01v",):
                if site == "ferrol" and METHODS[meth].kind == "elev":
                    continue
                for r in range(R):
                    units.append(Unit(design="curves-k1", site=site, window="W3", setting=P0,
                                      method=meth, N=int(N), rep=r, kind=kind))
    return units


GNIR = "C1.floor0.nir.D1"


def design_granadeiro_base(sites=LAG_SITES):
    """G-base (§5.11): Granadeiro ``ctl`` at P0 at the five sites, alone
    except for two cheap workers."""
    return [Unit(design="granadeiro-base", site=s, window="W3", setting=P0, method="E3",
                 variant="ctl", note="own pane; <= 2 cheap workers beside") for s in sites]


def design_granadeiro_nir(sites=("villaviciosa", "escalda"), tier="1A'"):
    """G-nir: Granadeiro at C1 + NIR (its native cell)."""
    return [Unit(design="granadeiro-nir" if tier == "1A'" else "granadeiro-nir-1b", site=s,
                 window="W3", setting=GNIR, method="E3", variant="ctl", tier=tier,
                 note="Granadeiro's native cell") for s in sites]


# -- Tier 1B (separate campaign, after the 2017-2022 batch) -------------------

def design_windows(sites=LAG_SITES):
    """Window OFAT W2, W5, W9 at P0 and 'EOT20 only' at W9 (§5.6, §5.9)."""
    cells = [(P0, w) for w in ("W2", "W5", "W9")]
    units = _cell_units("windows", sites, cells, CHEAP_ELEV + IMAGE_LAG, tier="1B")
    units += [replace(u, variant="bnd-eot20", note="boundary = EOT20 only at W9")
              for u in _cell_units("windows", sites, [(P0, "W9")], CHEAP_ELEV + IMAGE_LAG,
                                   tier="1B")]
    return units


def design_g2(sites=LAG_SITES):
    """G2 = C x T at NDWI, D1 (§5.7): C in {C1, C2, C4, C11} x {W1, W3, W9}."""
    cells = [(_sid(cloud=c), w) for c in G1_CLOUD for w in ("W1", "W3", "W9")]
    return _cell_units("g2", sites, cells, CHEAP_ELEV + IMAGE_LAG, tier="1B")


def design_centred():
    """Survey-centred windows at P0 (§2.7, §5.9) at Escalda, Wadden and Ems."""
    units = []
    for site, wids in settings.SURVEY_WINDOWS.items():
        cells = [(P0, w) for w in wids]
        units += _cell_units("centred", (site,), cells, CHEAP_ELEV + IMAGE_LAG, tier="1B")
    return units


def design_years_1b(sites=LAG_SITES):
    """Calendar years Y2017-Y2022 at P0 (§5.8; final archive-stability)."""
    cells = [(P0, f"Y{y}") for y in range(2017, 2023)]
    return _cell_units("years-1b", sites, cells, CHEAP_ELEV + IMAGE_LAG, tier="1B")


def design_published_1b(sites=ELEV_SITES):
    """NIDEM at W9 and Coast X-Ray at W5, as published and native (§5.3, §5.5)."""
    units = []
    for site in sites:
        units.append(Unit(design="published-1b", site=site, window="W9",
                          setting="C4.floor0.ndwi.D1@LT", method="E1", variant="published",
                          tier="1B"))
        units.append(Unit(design="published-1b", site=site, window="W9",
                          setting="C7.floor0.ndwi.D1@LT", method="E1", variant="s2",
                          family="rule", tier="1B", note="NIDEM-S2 variant (C7)"))
        units.append(Unit(design="published-1b", site=site, window="W5",
                          setting="C1.floor0.ndwi.D1@LT", method="E7", variant="published",
                          tier="1B"))
    units += _cell_units("published-1b", LAG_SITES,
                         [("C4.floor0.ndwi.D1", "W9"), (P0, "W5")],
                         CHEAP_ELEV + IMAGE_LAG, tier="1B", note="native cell (Tier 1B)")
    return units


def design_ofat_1b(sites=LAG_SITES):
    """Tier-1B OFAT levels (§5.6): C5, C7, class-11-blocked C4, AWEI_nsh,
    WRI, D4 (naive baseline)."""
    cells = [(_sid(cloud="C5"), "W3"), (_sid(cloud="C7"), "W3"),
             (_sid(index="awei_nsh"), "W3"), (_sid(index="wri"), "W3"),
             (_sid(detect="D4"), "W3")]
    units = _cell_units("ofat-1b", sites, cells, CHEAP_ELEV + IMAGE_LAG, tier="1B")
    units += _cell_units("ofat-1b", sites, [(_sid(cloud="C4"), "W3")], CHEAP_ELEV + IMAGE_LAG,
                         tier="1B", variant="scl11-blocked",
                         note="class-11-blocked C4 sensitivity")
    return units


def design_ndwi20_slice(sites=LAG_SITES):
    """The NDWI@20 slice of G1' (§5.7): D1, C in {C2, C4, C11}."""
    cells = [(_sid(cloud=c, index="ndwi20"), "W3") for c in ("C2", "C4", "C11")]
    return _cell_units("ndwi20-slice", sites, cells, CHEAP_ELEV + IMAGE_LAG, tier="1B")


def design_ixt():
    """The I x T check at Escalda (§5.7): {NDWI, MNDWI, NIR} x {W1, W3, W5}."""
    cells = [(_sid(index=i), w) for i in G1_INDEX for w in ("W1", "W3", "W5")]
    return _cell_units("ixt", ("escalda",), cells, CHEAP_ELEV + IMAGE_LAG, tier="1B")


def _curve_units(design, cs_id, sites, methods, tier="1B"):
    cs = settings.CURVES[cs_id]
    units = []
    for sid in cs.settings:
        for wid in cs.windows:
            T = window_T(wid)
            for N, R, _ in cs.grid:
                if N == "all":
                    continue
                kind = settings.subset_kind(T, N, R)
                for site in (cs.sites or sites):
                    for meth in methods:
                        if site == "ferrol" and METHODS[meth].kind == "elev":
                            continue
                        for r in range(R):
                            units.append(Unit(design=design, site=site, window=wid,
                                              setting=sid, method=meth, N=int(N), rep=r,
                                              kind=kind, tier=tier))
    return units


def design_curves_k3(sites=LAG_SITES):
    return _curve_units("curves-k3", "K3", sites, CHEAP_ELEV + IMAGE_LAG)


def design_curves_k4(sites=LAG_SITES):
    return _curve_units("curves-k4", "K4", sites, CHEAP_ELEV + IMAGE_LAG)


def design_curves_k5(sites=LAG_SITES):
    units = []
    for k in ("K5_W1", "K5_W2", "K5_W5", "K5_W9"):
        units += _curve_units("curves-k5", k, sites, CHEAP_ELEV + IMAGE_LAG)
    return units


#: the L9 rows of §5.11 (run 1 = P0)
L9_ROWS = (("C1", "ndwi", "W3"), ("C1", "nir", "W1"), ("C1", "mndwi", "W5"),
           ("C2", "ndwi", "W1"), ("C2", "nir", "W5"), ("C2", "mndwi", "W3"),
           ("C6", "ndwi", "W5"), ("C6", "nir", "W3"), ("C6", "mndwi", "W1"))


def design_granadeiro_l9(sites=("villaviciosa", "escalda")):
    """G-L9 (§5.11): Granadeiro on the L9(3^4) rows; the cheap units run
    every L9 cell too, so MAREA - Granadeiro contrasts exist at each."""
    units = []
    for k, (c, i, w) in enumerate(L9_ROWS, 1):
        sid = _sid(cloud=c, index=i)
        for site in sites:
            units.append(Unit(design="granadeiro-l9", site=site, window=w, setting=sid,
                              method="E3", variant="ctl", family="rule" if c != "C1" else "rand",
                              tier="1B", note=f"L9 run {k}"))
        units += _cell_units("granadeiro-l9", sites, [(sid, w)], CHEAP_ELEV + IMAGE_LAG,
                             tier="1B", note=f"cheap units at L9 run {k}")
    return units


def design_granadeiro_curves():
    return _curve_units("granadeiro-curves", "K6", ("villaviciosa", "escalda"), ("E3",))


def design_granadeiro_tune():
    """G-tune (§5.12): SD threshold {0.15, 0.20} x lag window {+-90, -45..180}."""
    return [Unit(design="granadeiro-tune", site=TUNING_SITE, window="W3", setting=P0,
                 method="E3", variant=f"tune{k}", tier="1B") for k in range(4)]


DESIGNS = {
    "lag": design_lag, "base": design_base, "published": design_published,
    "levels": design_levels, "native": design_native, "years": design_years,
    "tuning": design_tuning, "g1": design_g1, "ofat": design_ofat,
    "curves-k1": design_curves_k1, "granadeiro-base": design_granadeiro_base,
    "granadeiro-nir": design_granadeiro_nir,
    "windows": design_windows, "g2": design_g2, "centred": design_centred,
    "years-1b": design_years_1b, "published-1b": design_published_1b,
    "ofat-1b": design_ofat_1b, "ndwi20-slice": design_ndwi20_slice, "ixt": design_ixt,
    "curves-k3": design_curves_k3, "curves-k4": design_curves_k4,
    "curves-k5": design_curves_k5,
    "granadeiro-nir-1b": lambda: design_granadeiro_nir(("wadden", "ems"), tier="1B"),
    "granadeiro-l9": design_granadeiro_l9, "granadeiro-curves": design_granadeiro_curves,
    "granadeiro-tune": design_granadeiro_tune,
}
TIER_1A = QUEUE_1A
TIER_1B = QUEUE_1B
if set(DESIGNS) != set(QUEUE_1A) | set(QUEUE_1B):
    raise AssertionError("DESIGNS and the queues disagree")


# ─────────────────────────────────────────────────────────────────────────────
#  building a design: dedupe, cuts
# ─────────────────────────────────────────────────────────────────────────────

#: the pre-ordered cut list (§5.14); each predicate says whether a unit is
#: removed by that step
CUTS = (
    (1, "K1' reps to 3d, 3d, 2d, 2d, 1n, 1"),
    (2, "matched-N controls to 1 draw for every method"),
    (3, "G1' detection column D3 moved to Tier 1B (G1' = C x I at D1)"),
    (4, "G-nir at Escalda only in Tier 1A' (G-bijagos on one raster tile)"),
    (5, "OFAT AWEI_sh, NDWI@20 and C1d moved to Tier 1B"),
    (6, "LT and LM rows at the three Dutch sites only"),
    (7, "tuned rows at two sites"),
    (8, "Villaviciosa boundary variants moved to Tier 1B"),
)
NEVER_CUT = ("the MAREA, NIDEM, DEA and Granadeiro gates", "Stage L at W3",
             "P0 at every site", "the as-published rows",
             "the LB-vs-LM/LT practical comparison at three or more sites", "the native cells",
             "G-base at three or more sites")
_K1_CUT_R = {20: 3, 45: 3, 90: 2, 180: 2, 330: 1}


def cut_step(u):
    """The first cut step (1-8) that removes unit ``u``, 0 if never cut."""
    if u.design == "curves-k1" and isinstance(u.N, int) and u.rep >= _K1_CUT_R.get(u.N, 99):
        return 1
    if u.family.startswith("matched-") and u.rep >= 1:
        return 2
    if u.design == "g1" and settings.Setting.parse(u.setting).detect == "D3":
        return 3
    if u.design == "granadeiro-nir" and u.site != "escalda":
        return 4
    if u.design == "ofat":
        s = settings.Setting.parse(u.setting)
        if s.index in ("awei_sh", "ndwi20") or s.cloud == "C1d":
            return 5
    if u.design == "levels" and u.site not in DUTCH_SITES:
        s = settings.Setting.parse(u.setting)
        if s.level in ("LT", "LM"):
            return 6
    if u.design == "tuning" and u.variant == "tuned" and u.site not in TUNED_KEEP_AFTER_CUT:
        return 7
    if u.design == "ofat" and u.variant in ("bnd-gijon", "bnd-cantabrian"):
        return 8
    return 0


def build(design, sites=None, **kw):
    """The units of ``design`` with dedupe (``shared``) and cut steps.

    Parameters
    ----------
    design : str
        A key of :data:`DESIGNS`.
    sites : sequence, optional
        Restrict to these sites.
    **kw
        Passed to the design function (e.g. ``ocm_available`` for g1).
    """
    if design not in DESIGNS:
        raise KeyError(f"unknown design {design!r}; known: {tuple(DESIGNS)}")
    fn = DESIGNS[design]
    units = fn(**kw) if kw else fn()
    if sites:
        sites = tuple(sites)
        bad = [s for s in sites if s not in config.SITES]
        if bad:
            raise ValueError(f"unknown site(s) {bad}")
        units = [u for u in units if u.site in sites]
    seen = {}
    out = []
    for u in units:
        u = replace(u, cut_step=cut_step(u))
        if u.status == "planned":
            k = u.dedupe_key
            if k in seen:
                u = replace(u, status="shared", note=f"shares unit {seen[k]}")
            else:
                seen[k] = u.unit_id
        out.append(u)
    return out


def plan_rows(design, sites=None, **kw):
    """CSV-ready rows of a design, in enumeration order."""
    pr = (QUEUE_1A + QUEUE_1B).index(design) + 1
    return [u.row(pr) for u in build(design, sites, **kw)]


def plan_csv(rows):
    """The CSV text of plan rows (LF, fixed column order, floats as repr)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def summary(units):
    """Counts and estimated hours of a list of units."""
    planned = [u for u in units if u.status == "planned"]
    cheap = sum(u.est_hours for u in planned if not u.m.expensive)
    gran = sum(u.est_hours for u in planned if u.m.expensive)
    n_inst = len({(u.site, u.window, u.setting, u.variant, u.family, u.mode, str(u.N), u.rep)
                  for u in planned})
    return {"n_rows": len(units), "n_planned": len(planned),
            "n_shared": sum(u.status == "shared" for u in units),
            "n_not_applicable": sum(u.status == "not_applicable" for u in units),
            "n_reuse": sum(u.status == "reuse" for u in units), "n_instances": n_inst,
            "cheap_wh": round(cheap, 2), "granadeiro_h": round(gran, 2),
            "max_ram_gb": max([u.est_ram_gb for u in planned], default=0.0),
            "read_gb": round(sum(u.est_read_gb for u in planned), 1)}


def _row_order_key(r):
    return (str(r.get("unit_id")), str(r.get("site")), str(r.get("window")),
            str(r.get("setting_id")), str(r.get("method")), str(r.get("variant")),
            str(r.get("status")), str(r.get("products")))


def merge_plan_rows(design, old_rows, new_rows, sites, **kw):
    """The rows of a plan written for ``sites`` only, merged into the plan
    already on disk: the other sites' rows are kept, ``sites``' rows replaced,
    in the order of the full design (every site)."""
    sites = set(sites)
    keep = [r for r in old_rows if r.get("site") not in sites]
    full = plan_rows(design, None, **kw)
    pos = {}
    for i, r in enumerate(full):
        pos.setdefault(_row_order_key(r), i)
    merged = list(new_rows) + keep
    n = len(full)
    order = sorted(range(len(merged)),
                   key=lambda i: (pos.get(_row_order_key(merged[i]), n), i))
    return [merged[i] for i in order]


def write_plan(design, root=V2_ROOT, sites=None, replace=False, **kw):
    """Write ``<root>/plans/<design>.csv`` and update ``plans/index.json``
    (the sha256 of each plan: FROZEN['design']). Returns ``(path, summary)``.

    With ``sites`` and an existing plan, the other sites' rows are kept (the
    file never shrinks to the sites of one call) unless ``replace``."""
    rows = plan_rows(design, sites, **kw)
    d = os.path.join(root, "plans")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{design}.csv")
    plan_sites = list(sites) if sites else None
    if sites and not replace and os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as f:
            old = list(csv.DictReader(f))
        if any(r.get("site") not in set(sites) for r in old):
            rows = merge_plan_rows(design, old, rows, sites, **kw)
            plan_sites = sorted({str(r["site"]) for r in rows},
                                key=lambda s: (config.SITES.index(s)
                                               if s in config.SITES else 99, s))
    text = plan_csv(rows)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    idx_path = os.path.join(d, "index.json")
    idx = {"format": PLAN_FORMAT, "plans": {}}
    if os.path.exists(idx_path):
        try:
            with open(idx_path, encoding="utf-8") as f:
                idx = json.load(f)
        except (OSError, ValueError):
            pass
    units = build(design, plan_sites, **kw)
    summ = summary(units)
    idx.setdefault("plans", {})[design] = {"sha256": sha, "rows": len(rows),
                                           "sites": list(plan_sites) if plan_sites else "all",
                                           "summary": summ}
    tmp = f"{idx_path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(idx, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, idx_path)
    return path, dict(summ, sha256=sha)


# ─────────────────────────────────────────────────────────────────────────────
#  the Tier 1A' projection and the cut order (§5.14, §14.2)
# ─────────────────────────────────────────────────────────────────────────────

def campaign_units(designs=QUEUE_1A, sites=None):
    """Every planned unit of ``designs``, each effective key counted once
    (the first design in queue order owns it)."""
    seen = set()
    out = []
    for d in designs:
        for u in build(d, sites):
            if u.status != "planned":
                continue
            k = u.dedupe_key
            if k in seen:
                continue
            seen.add(k)
            out.append(u)
    return out


def instance_hours(units):
    """Harness hours: one site-instance per (site, window, setting, variant,
    family, N, rep); N = all costs INSTANCE_WH / 5 (by frame), a curve
    instance CURVE_INSTANCE_H."""
    inst = {}
    for u in units:
        if u.m.expensive:
            continue
        k = (u.site, u.window, u.setting, u.variant, u.family, str(u.N), u.rep)
        if u.N == "all":
            h = INSTANCE_WH / len(FRAME_PX) * FRAME_PX[u.site] / _MEAN_PX * window_scale(u.window)
        else:
            h = CURVE_INSTANCE_H
        inst[k] = h
    return sum(inst.values())


def project(units, workers=WORKERS, utilisation=UTILISATION, beside=GRAN_BESIDE,
            fixed_days=None):
    """Projected Tier 1A' wall days of ``units`` (§14.2)."""
    fixed = sum((fixed_days or FIXED_DAYS).values())
    cheap = sum(u.est_hours for u in units if not u.m.expensive) + instance_hours(units)
    gran = sum(u.est_hours for u in units if u.m.expensive)
    per_day = workers * 24.0 * utilisation
    lost = gran / 24.0 * max(0.0, 1.0 - beside / float(workers))
    days = fixed + cheap / per_day + lost
    compute = cheap / per_day + lost
    return {"cheap_wh": round(cheap, 1), "granadeiro_h": round(gran, 1),
            "wh_per_day": round(per_day, 1), "fixed_days": round(fixed, 2),
            "days": round(days, 2),
            "days_with_contingency": round(fixed + compute * (1 + CONTINGENCY), 2)}


#: the P2 competitors' elevation units (P3 compares MAREA(LB) with them under
#: LM and LT, §9.2; Granadeiro is Tier 2 under LT/LM)
P2_UNITS = ("E1", "E2", "E4", "E5", "E6a", "E6b", "E7", "E8")


def never_cut_check(units, sites=None):
    """The never-cut items (§5.14) that plan rows can show, on ``units``
    (the kept planned units): ``{item: {"ok", "detail"}}``. "LB-vs-LM/LT at
    three or more sites" counts the sites where a P2 competitor runs under
    LM (LT) in a window where MAREA runs under LB (the E0 unit at P0); the
    gates are V2/V3 code, not plan rows (``ok`` None)."""
    planned = [u for u in units if u.status == "planned"]
    elev_sites = [s for s in ELEV_SITES if sites is None or s in sites]
    lag_sites = [s for s in LAG_SITES if sites is None or s in sites]
    need3 = min(3, len(elev_sites))

    def has(pred):
        return any(pred(u) for u in planned)

    marea_lb = {(u.site, u.window) for u in planned if u.method == "E0"
                and u.setting == P0 and u.N == "all" and u.family == "rand"
                and "nodelay" not in u.variant and "gaugelag" not in u.variant}
    out = {}
    out["Stage L at W3"] = {"ok": all((s, "W3") in marea_lb for s in lag_sites),
                            "detail": "E0 (the L01 lag) at P0, W3 at every site"}
    miss = [(s, m) for s in elev_sites for m in CHEAP_ELEV
            if not has(lambda u, s=s, m=m: u.site == s and u.method == m and u.setting == P0
                       and u.window == "W3" and u.N == "all" and u.family == "rand")]
    out["P0 at every site"] = {"ok": not miss, "detail": f"missing {miss}" if miss else "ok"}
    out["the as-published rows"] = {"ok": has(lambda u: u.design == "published"),
                                    "detail": "the published design's rows"}
    for lv in ("LM", "LT"):
        ss = sorted({u.site for u in planned if u.method in P2_UNITS
                     and settings.Setting.parse(u.setting).level == lv
                     and (u.site, u.window) in marea_lb})
        out[f"LB-vs-{lv} at three or more sites"] = {
            "ok": len(ss) >= need3, "detail": f"{len(ss)} site(s): {ss}"}
    cells = {(sid, wid) for sid, wid, _ in NATIVE_CELLS}
    got = {(u.setting, u.window) for u in planned if u.design == "native"}
    out["the native cells"] = {"ok": cells <= got,
                               "detail": f"missing {sorted(cells - got)}" if cells - got
                               else "ok"}
    g = sorted({u.site for u in planned if u.method == "E3" and u.setting == P0
                and u.window == "W3"})
    out["G-base at three or more sites"] = {"ok": len(g) >= min(3, len(lag_sites)),
                                            "detail": f"{len(g)} site(s): {g}"}
    out["the MAREA, NIDEM, DEA and Granadeiro gates"] = {
        "ok": None, "detail": "gates are V2/V3 code, not plan rows"}
    return out


def projection(designs=QUEUE_1A, sites=None, cap_days=CAP_DAYS, workers=WORKERS,
               utilisation=UTILISATION, wh_per_day=None):
    """Apply the cut list in order until the Tier 1A' projection fits.

    Parameters
    ----------
    wh_per_day : float, optional
        The measured cheap throughput (wh per day); overrides workers x 24
        x utilisation.

    Returns
    -------
    dict
        ``steps``: one entry per cut step applied (0 = no cut) with the
        projection; ``applied``: the last step needed (None if even all
        cuts do not fit); ``fits`` (within the cap AND every checkable
        never-cut item holds); ``never_cut_check`` (:func:`never_cut_check`
        on the units kept at the last step).
    """
    if wh_per_day is not None:
        utilisation = float(wh_per_day) / (workers * 24.0)
    units = campaign_units(designs, sites)
    steps = []
    applied = None
    kept = units
    for step in range(0, len(CUTS) + 1):
        kept = [u for u in units if u.cut_step == 0 or u.cut_step > step]
        p = project(kept, workers, utilisation)
        p.update(step=step, cut=("none" if step == 0 else CUTS[step - 1][1]),
                 n_units=len(kept), fits=bool(p["days"] <= cap_days))
        steps.append(p)
        if p["fits"]:
            applied = step
            break
    nc = never_cut_check(kept, sites)
    nc_ok = all(v["ok"] is not False for v in nc.values())
    return {"cap_days": cap_days, "workers": workers, "utilisation": round(utilisation, 3),
            "steps": steps, "applied": applied, "fits": applied is not None and nc_ok,
            "within_cap": applied is not None, "never_cut": list(NEVER_CUT),
            "never_cut_check": nc, "never_cut_ok": nc_ok}


def planned_settings(designs=QUEUE_1A):
    """The setting ids of ``designs`` (sorted; the level suffix kept)."""
    out = set()
    for d in designs:
        for u in build(d):
            if u.status == "planned":
                out.add(u.setting)
    return sorted(out)


def observation_keys(designs=QUEUE_1A):
    """The pixel observation keys the package counts for ``designs``:
    ``<pixel cloud>.<prep>.<index>.<detect>`` (scene rules mapped to their
    pixel rule C1, the level dropped; detector-blind units still count D1),
    sorted."""
    keys = set()
    for sid in planned_settings(designs):
        s = settings.Setting.parse(sid)
        keys.add(settings.Setting(cloud=s.pixel_rule, prep=s.prep, index=s.index,
                                  detect=s.detect).id)
    return sorted(keys)


# ─────────────────────────────────────────────────────────────────────────────
#  error attribution: which rows answer "noise, lag or elevation estimation?"
# ─────────────────────────────────────────────────────────────────────────────

#: the paired contrasts of the attribution (spec §9.6 items 2 and 8), each
#: as (name, product A, product B, what A - B isolates)
ATTRIBUTION_CONTRASTS = (
    ("lag effect", "E0b@LB", "E0a@LB",
     "MAREA's band lags vs no delay: the lag correction alone (family P1)"),
    ("lag headroom", "E0a@LG", "E0b@LB",
     "the measured gauge level (no delay under LG, labelled upper bound) vs MAREA's lags: "
     "what a perfect local level would still gain over the lag"),
    ("level error", "E0a@LG", "E0a@LB",
     "the measured gauge level vs the boundary tide, same inversion: the whole level error"),
    # the gauge-lag row E0g (user decision 2026-10-15) splits the level error:
    ("ignoring the lag", "E0g@LB", "E0a@LB",
     "the judges' measured band lags vs no delay: what ignoring a KNOWN lag costs"),
    ("lag estimation", "E0b@LB", "E0g@LB",
     "MAREA's archive band lags vs the judges' measured band lags, same inversion: the cost "
     "of estimating the lag from the archive"),
    ("band model", "E0a@LG", "E0g@LB",
     "the measured gauge level vs the judges' band lags: the cost of one lag per 3-km band "
     "(the bathtub model); what remains at LG against the truth is the elevation-estimation "
     "error"),
)


def attribution_rows(sites=ELEV_SITES, designs=QUEUE_1A):
    """The Tier 1A' rows that attribute an elevation error to noise, to the
    lag or to the elevation estimation (the user's question of 2026-10-14;
    spec §9.1, §9.6), per elevation site.

    Every contrast of :data:`ATTRIBUTION_CONTRASTS` compares products of the
    SAME inversion on the same scenes (one record, N = all), the same pixels
    (the harness domain of the same setting cell; the scorer's pair views)
    and the same setting (P0 at W3), differing only in the water level:

    * ``E0a@LB`` (no delay, tau = 0) and ``E0b@LB`` (MAREA's band lags) are
      written by ONE E0 unit of the base design;
    * ``E0a@LG`` is the E0 'nodelay' unit of the levels design at P0@LG,
      planned where the judges make LG available, ``not_applicable``
      elsewhere; its level rows (``levels_LG.npz``) are built by ``prepare
      --step sources`` once ``truth`` has written the judge table.
    * ``E0g@LB`` (user decision 2026-10-15) is the E0 'gaugelag' unit of the
      levels design at P0, W3: MAREA's inversion with the judges' measured
      band lags (:func:`levels.gauge_band_lags`), at the same sites as LG.
      With it the level error splits into ignoring the lag (E0g - E0a), the
      lag estimation (E0b - E0g) and the band model (LG - E0g); what is
      left at LG against the truth is the elevation-estimation error.

    The noise floors are the base design's seed rows (E0 seed1..seed4),
    its alternate-scene halves and the calendar-year windows Y2023-Y2025
    of the years design. The block-bootstrap CIs of the paired differences
    are pre-registered (FROZEN['inference']: B = 20,000 for the
    confirmatory families, at least 25 blocks) and computed by the scoring
    milestone, not by a design row.

    Returns
    -------
    dict
        ``{site: {"rows": {key: [unit ids]}, "present": {key: bool},
        "missing": [str]}}`` with keys ``E0a_E0b_LB`` (the base E0 unit),
        ``E0g_LB`` (the gauge-lag unit), ``E0a_LG``, ``seeds``, ``halves``,
        ``years``, ``years_LG``.
    """
    units = {d: build(d) for d in designs}
    out = {}
    for site in sites:
        rows = {k: [] for k in ("E0a_E0b_LB", "E0g_LB", "E0a_LG", "seeds", "halves", "years",
                                "years_LG")}
        na, na_g = [], []
        for d, us in units.items():
            for u in us:
                if u.site != site or u.method != "E0":
                    continue
                s = settings.Setting.parse(u.setting)
                base_cell = (s.cloud, s.prep, s.index, s.detect) == ("C1", "floor0", "ndwi", "D1")
                if not base_cell:
                    continue
                gaugelag = "gaugelag" in u.variant.split("+")
                if u.status == "not_applicable":
                    if gaugelag:
                        na_g.append(u.note)
                    elif s.level == "LG":
                        na.append(u.note)
                    continue
                if u.status not in ("planned", "shared"):
                    continue
                if gaugelag:
                    if s.level == "LB" and u.window == "W3" and u.N == "all":
                        rows["E0g_LB"].append(u.unit_id)
                    continue
                if d == "base" and u.window == "W3" and s.level == "LB" and u.N == "all":
                    if u.family == "half":
                        rows["halves"].append(u.unit_id)
                    elif u.variant.startswith("seed"):
                        rows["seeds"].append(u.unit_id)
                    elif u.family == "rand" and u.rep == 0:
                        rows["E0a_E0b_LB"].append(u.unit_id)
                elif d == "levels" and s.level == "LG" and u.window == "W3":
                    rows["E0a_LG"].append(u.unit_id)
                elif d == "years" and u.window.startswith("Y"):
                    rows["years_LG" if s.level == "LG" else "years"].append(u.unit_id)
        present = {k: bool(v) for k, v in rows.items()}
        missing = []
        if not present["E0a_E0b_LB"]:
            missing.append("the base E0 unit (no delay and MAREA from one run) at P0, W3")
        if not present["E0a_LG"]:
            missing.append("no delay under LG (measured gauge level): "
                           + ("; ".join(sorted(set(na))) if na else "no LG row planned"))
        if not present["E0g_LB"]:
            missing.append("MAREA with the judges' measured lags (E0g): "
                           + ("; ".join(sorted(set(na_g))) if na_g
                              else "no gauge-lag row planned"))
        if not present["seeds"] or not present["halves"]:
            missing.append("seed or alternate-scene rows of E0 at P0, W3")
        if len(rows["years"]) < 3:
            missing.append("E0 at P0 in the three calendar years 2023-2025")
        if present["E0a_LG"] and not present["years_LG"]:
            missing.append("LG rows in the calendar-year windows (the between-year spread of "
                           "the LG headroom is not planned)")
        out[site] = {"rows": rows, "present": present, "missing": missing}
    return out
