# -*- coding: utf-8 -*-
"""Validation v2, Stage L -- HD26: L09 HD26-H and L10 HD26-R2 (spec §3.3, §3.3.1, §4.2, §6.6).

Question
--------
Hart-Davis et al. (2026) read tidal harmonic constants from a satellite
proxy of the water level. Applied to the flooded share of each 3-km band,
how far behind h_B does the band's M2 tide run?

One unit (method L09 of the design catalogue) writes two products from the same band
aggregates (``aggregate.band_aggregates``):

* **L09 HD26-H** (F2, pending its fidelity gate G-hd26; spec §3.3.1 "L09"):
  the M2 phase of the band wet fraction, relative to h_B's own M2 phase on
  the same scenes (``rel``, the primary);
* **L10 HD26-R2** (F4): the delay-and-smear response with an isotonic link,
  :func:`experiments.benchmark.lags.aggregate.l10_hd26r2` (its rules and
  card live there; ``CARDS['L10']`` is :data:`aggregate.CARD_L10`).

Pre-registered rules of L09 (FROZEN['hd26'], repeated in :data:`CONFIG`)
------------------------------------------------------------------------
* proxy zeta_k(t) = the band wet fraction on the scenes whose band is at
  least 80 % valid (``band_valid_share``; sensitivities 0.5 and 0.9 are card
  variants, not scheduled);
* ``truth.hd26_m2_phase(days_since_1992(t), zeta_k)``: linear detrend, one
  3-sigma outlier pass, at least 30 points over at least 365 days, then
  ``pyTMD.solve.constants`` with M2, N2, K2, O1, Q1 and GOT nodal
  corrections; the M2 Greenwich phase g_zeta;
* ``rel``: g_h0 = the same function on h0 = h_B(t) at the same scenes (the
  scenes the proxy's 3-sigma pass kept, :func:`proxy_kept`; the V2 build of
  2026-10-19 fitted h0 on every usable scene, declared fix of 2026-10-20);
  tau_k = wrap180(g_zeta - g_h0) x 60 / 28.9841042 min (a phase lag:
  circular, period one M2 cycle; the scorer wraps it about 67.5 min);
* a band whose record is too short (span < 365 d, fewer than 30 points) is
  undefined; when every band is, the product is NOT COMPUTABLE (stage
  ``span_lt_365d`` or ``points_lt_30``). A window that cannot hold a
  365-day span at all (a common calendar year, W1: at most 364 days) never
  reaches this rule: the lag design plans L09 there as not applicable and
  the unit writes L10 only (``designs.l09_span_rule``, spec §4.2 'window >=
  365 d for L09'; the published rule itself is not relaxed; spec §3.3.1:
  "A 1-year window is marginal, because under fixed solar-time sampling the
  M2 and O1 alias frequencies differ by 1 cycle per year");
* diagnostics, not gates: the same phase lag of the band-median Y (the HD26
  index proxy) and the variance inflation factor of the M2 pair in the
  five-constituent design at the band's instants (flag above 2,
  FROZEN['hd26']['vif_flag']).

The output is a band lag broadcast to the band's keep pixels (kind 'band',
lag_kind 'phase'). Nothing is tuned on truth.
"""
from __future__ import annotations

import numpy as np

from experiments.benchmark import config, truth
from experiments.benchmark.lags import aggregate, common

_HD = config.FROZEN["hd26"]

#: angular speeds (deg/h) of the five constituents (for the VIF diagnostic only;
#: the fit itself is pyTMD's)
SPEEDS_DEG_H = {"m2": 28.9841042, "n2": 28.4397295, "k2": 30.0821373,
                "o1": 13.9430356, "q1": 13.3986609}

CONFIG = {
    "L09": {
        "proxy": "wetfrac",
        "index_proxy_diag": "median_Y",
        "band_valid_share": float(_HD["band_valid_share"]),
        "band_valid_share_sensitivities": [float(x) for x in
                                           _HD["band_valid_share_sensitivities"]],
        "constituents": [str(c) for c in _HD["constituents"]],
        "corrections": str(_HD["corrections"]),
        "outlier_sigma": float(_HD["outlier_sigma"]),
        "min_points": int(_HD["min_points"]),
        "min_span_days": float(_HD["min_span_days"]),
        "primary": str(_HD["primary"]),
        "tau": "wrap180(g_zeta - g_h0) * 60 / 28.9841042 (min)",
        "function": "truth.hd26_m2_phase",
        "vif_flag": float(_HD["vif_flag"]),
        "vif_speeds_deg_h": dict(SPEEDS_DEG_H),
    },
    "L10": aggregate.CONFIG["L10"],
}
del _HD

CODE_DEPS = ("experiments/benchmark/lags/aggregate.py", "experiments/benchmark/truth.py",
             "pyintertidal/tide_estimators.py")

_SPEC = "docs/validation_v2_spec.md"

CARD_L09 = {
    "id": "L09",
    "name": "HD26-H: M2 phase of the band wet fraction (Hart-Davis et al. 2026), rel",
    "family": "harmonic",
    "kind": "lag",
    "tier": "1",
    "fidelity": "F2",
    "evidence": ("F2 rests on G-hd26 (the port on CoastSat NZ transects against TICON-4 M2 "
                 "phases, median |dg| <= 10 deg; spec §10.1), which needs new downloads and "
                 "is not run in milestone V2: until it passes the row carries the F3 evidence "
                 "only (unit tests in tests/test_benchmark_lags_hd26.py on synthetic tides: "
                 "planted 0/45/120/200 min recovered; the 365-day rule) and the positive "
                 "control of spec §10.2"),
    "references": [
        {"cite": "Hart-Davis MG, Monahan T, Vos K, Andersen OB (2026) Commun. Earth Environ.",
         "doi_or_url": "doi:10.1038/s43247-026-03943-9"},
        {"cite": "Hart-Davis et al. (2026) preprint",
         "doi_or_url": "doi:10.21203/rs.3.rs-8967674/v1"},
        {"cite": "pyTMD (Sutterley): solve.constants, GOT nodal corrections",
         "doi_or_url": "https://github.com/tsutterley/pyTMD"},
        {"cite": "Vos K et al. (2019) CoastSat, Environ. Model. Softw. 122:104528 (G-hd26 data)",
         "doi_or_url": "doi:10.1016/j.envsoft.2019.104528"},
        {"cite": "TICON-4 tide gauge constants (G-hd26 reference)",
         "doi_or_url": "doi:10.17882/109129"},
    ],
    "output": "band",
    "lag_kind": "phase",
    "signature": aggregate.card_signature("L09"),
    "scan_window": None,
    "published_window": None,
    "as_published": ("harmonic least squares with nodal corrections for M2, N2, K2, O1 and Q1 "
                     "of a satellite level proxy after a linear detrend and a 3-sigma outlier "
                     "filter; M2 phase compared with gauge harmonic constants"),
    "native_cell": None,
    "common_conditions": ("proxy = band wet fraction (W & V) on the harness domain bands, scenes "
                          "with >= 80 % valid band; rel = minus the M2 phase of h0 on the same "
                          "scenes; band lag broadcast to the band's keep pixels"),
    "deviations": [
        {"paper": "level proxy from satellite shorelines along transects (CoastSat)",
         "ours": "the band wet fraction zeta_k of the 3-km harness band",
         "why": "one proxy per band, on the same instance as every lag method (spec §5.1)",
         "rows": "L09",
         "sensitivity": "index proxy (band-median Y) in diag; transect variant Tier 2"},
        {"paper": "absolute M2 phase of the proxy, validated against gauge constants",
         "ours": "rel: g_zeta - g_h0 on the same scenes",
         "why": "the lag relative to h_B, the scorer's convention; removes the sampling phase "
                "common to proxy and boundary",
         "rows": "L09", "sensitivity": "'faithful' variant (minus the continuous M2 phase of "
                                       "h_B; card only, not scheduled)"},
        {"paper": "every usable image of the record",
         "ours": "scenes whose band is >= 80 % valid",
         "why": "a partially clouded band biases the wet fraction",
         "rows": "L09", "sensitivity": "0.5 and 0.9 (card only, not scheduled)"},
    ],
    "qc": {"band_valid_share": 0.8, "min_points": 30, "min_span_days": 365.0,
           "outlier_sigma": 3.0, "vif_flag": 2.0, "vif_role": "flag only, never a gate",
           "applicability": ("not applicable at a window whose longest possible span is "
                             "under 365 days (a common calendar year, W1): structural, from "
                             "the window bounds (designs.l09_span_rule; spec §4.2)"),
           "rel_scenes": "h0 fitted on exactly the scenes the proxy's 3-sigma pass kept"},
    "n_min": 30,
    "tuning": [],
    "control_offset_min": 0.0,
    "control_offset_why": ("0 min: in the gating world (spec §10.2) the band proxy is a "
                           "monotone function of h_B(t - tau_k), whose M2 phase is h0's plus "
                           "omega_M2 tau_k; rel removes h0's"),
    "cost": aggregate.card_cost("L09"),
    "variants": {"faithful": "minus the continuous M2 phase of h_B (card only)",
                 "local": "I = nbhd ∩ keep instead of the band (card only)",
                 "cover0.5": "band scene rule 0.5 (card only)",
                 "cover0.9": "band scene rule 0.9 (card only)",
                 "index": "band-median Y proxy (reported in diag)",
                 "transect": "HD26 transect geometry (Tier 2)",
                 "rtide": "RTide, HD26-R1 (Tier 2)"},
}

CARDS = {"L09": CARD_L09, "L10": aggregate.CARD_L10}

#: variants of the unit's products that are listed on the cards but not scheduled in V2
_CARD_ONLY = ("faithful", "local", "cover0.5", "cover0.9", "transect", "rtide")


def external_inputs(inst, unit):
    """The installed pyTMD enters the inputs key (its nodal corrections fix
    the phases): ``{"pyTMD": sha256 of its version}``."""
    try:
        import pyTMD
        version = str(getattr(pyTMD, "__version__", "unknown"))
    except ImportError:                                     # pragma: no cover
        return {"pyTMD": None}
    return {"pyTMD": config.sha256_json({"pyTMD": version})}


def vif_m2(t_days):
    """Variance inflation factor of the M2 cos/sin pair in the design [1, t,
    cos/sin of M2, N2, K2, O1, Q1] at the given instants (the larger of the
    two; NaN when the design has fewer rows than columns + 1)."""
    t = np.asarray(t_days, np.float64)
    t = t[np.isfinite(t)]
    cols = [np.ones_like(t), (t - t.mean()) / max(float(np.ptp(t)), 1.0)]
    names = ["const", "trend"]
    th = 24.0 * t
    for c in ("m2", "n2", "k2", "o1", "q1"):
        w = np.deg2rad(SPEEDS_DEG_H[c])
        cols += [np.cos(w * th), np.sin(w * th)]
        names += [f"{c}_cos", f"{c}_sin"]
    X = np.column_stack(cols)
    if X.shape[0] <= X.shape[1]:
        return float("nan")
    out = []
    for j in (names.index("m2_cos"), names.index("m2_sin")):
        y = X[:, j]
        Z = np.delete(X, j, axis=1)
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        res = y - Z @ beta
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum(res ** 2)) / sst if sst > 0 else np.nan
        out.append(1.0 / max(1.0 - r2, 1e-12) if np.isfinite(r2) else np.nan)
    return float(np.nanmax(out)) if np.isfinite(out).any() else float("nan")


def proxy_kept(t_days, y):
    """The scenes ``truth.hd26_m2_phase`` keeps for a series (finite, then
    one 3-sigma pass on the linearly detrended values, the same operations
    in the same order), as a mask over the inputs."""
    t = np.asarray(t_days, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    out = np.zeros(len(y), bool)
    if m.sum() < 3:
        out[m] = True
        return out
    tt, yy = t[m], y[m]
    r = yy - np.polyval(np.polyfit(tt, yy, 1), tt)
    keep = np.abs(r - r.mean()) <= float(config.FROZEN["hd26"]["outlier_sigma"]) * r.std()
    out[np.flatnonzero(m)[keep]] = True
    return out


def l09_band(t_days, proxy, h0):
    """L09 on one band's usable scenes: ``{"tau", "g_proxy", "g_h0", "n",
    "span_days", "reason"}`` (``truth.hd26_m2_phase`` on both series). The
    h0 phase is fitted on EXACTLY the scenes the proxy's 3-sigma pass kept
    (rel: 'h0 at the same scenes'; :func:`proxy_kept`); h0's own pass in the
    same function removes nothing from a tide, and ``n_h0_dropped`` reports
    it if it ever does."""
    cfg = CONFIG["L09"]
    proxy = np.asarray(proxy, np.float64)
    t_days = np.asarray(t_days, np.float64)
    h0 = np.asarray(h0, np.float64)
    rp = truth.hd26_m2_phase(t_days, proxy)
    kept = proxy_kept(t_days, proxy) & np.isfinite(h0)
    rh = truth.hd26_m2_phase(t_days[kept], h0[kept])
    gp, gh = float(rp["phase_deg"]), float(rh["phase_deg"])
    span = float(rp["span_days"])
    n = int(rp["n"])
    if np.isfinite(gp) and np.isfinite(gh):
        if not np.ptp(proxy) > 0:
            # a constant proxy has no tide: its fitted phase is atan2(0, 0)
            tau, gp, reason = float("nan"), float("nan"), "proxy_constant"
        else:
            tau = float(truth.wrap180(gp - gh) * common.MIN_PER_DEG_M2)
            reason = "ok"
    else:
        tau = float("nan")
        if not np.isfinite(span):                    # fewer than 3 points
            reason = f"points_lt_{cfg['min_points']}"
        elif span < cfg["min_span_days"]:
            reason = "span_lt_365d"
        elif n < cfg["min_points"]:
            reason = f"points_lt_{cfg['min_points']}"
        elif not np.isfinite(gp):
            reason = "harmonic_failed"
        else:
            reason = "h0_phase_failed"
    return {"tau": tau, "g_proxy": gp, "g_h0": gh, "n": n, "span_days": span,
            "amp": float(rp["amp"]), "reason": reason,
            "n_h0_dropped": int(kept.sum()) - int(rh["n"]) if np.isfinite(gh) else 0}


def run_l09(inst):
    """L09 HD26-H on every band of the instance (module docstring)."""
    cfg = CONFIG["L09"]
    with aggregate._stage(inst, "aggregate"):
        agg = aggregate.band_aggregates(inst)
    t_days = truth.days_since_1992(inst.t)
    h0 = np.asarray(inst.h0, np.float64)
    K = agg["K"]
    taus, reasons, rows, tau_idx, vif, n_use = [], [], [], [], [], []
    with aggregate._stage(inst, "harmonic"):
        for k in range(K):
            use = np.isfinite(agg["cover"][:, k]) & (agg["cover"][:, k] >= cfg["band_valid_share"])
            use &= np.isfinite(agg["wetfrac"][:, k])
            r = l09_band(t_days[use], agg["wetfrac"][use, k], h0[use])
            taus.append(r["tau"])
            reasons.append(r["reason"])
            rows.append(r)
            n_use.append(int(use.sum()))
            mi = use & np.isfinite(agg["median_Y"][:, k])
            tau_idx.append(l09_band(t_days[mi], agg["median_Y"][mi, k].astype(np.float64),
                                    h0[mi])["tau"])
            vif.append(vif_m2(t_days[use]) if use.sum() > 12 else float("nan"))
            aggregate._progress(inst, 1)
    vif = np.asarray(vif, np.float64)
    diag = {"n_usable_scenes": np.asarray(n_use, np.int64),
            "n_points": np.asarray([r["n"] for r in rows], np.int64),
            "span_days": np.asarray([r["span_days"] for r in rows], np.float64),
            "g_proxy_deg": np.asarray([r["g_proxy"] for r in rows], np.float64),
            "g_h0_deg": np.asarray([r["g_h0"] for r in rows], np.float64),
            "amp_proxy": np.asarray([r["amp"] for r in rows], np.float64),
            "n_h0_dropped": np.asarray([r["n_h0_dropped"] for r in rows], np.int64),
            "tau_index_proxy_min": np.asarray(tau_idx, np.float64),
            "vif_m2": vif,
            "vif_flagged": np.asarray(np.isfinite(vif) & (vif > cfg["vif_flag"]), bool),
            "aggregate_path": list(agg["path"])}
    return aggregate.band_result(
        "L09", inst, taus, np.zeros(K, bool), agg["n_px"], reasons,
        ("span_lt_365d", f"points_lt_{cfg['min_points']}"), lag_kind="phase", diag=diag)


def run_unit(inst, unit, opts=None):
    """The L09 unit: products L09 (HD26-H) and L10 (HD26-R2), in the unit's order."""
    aggregate.require_domain(inst)
    products = aggregate.unit_products(unit, ("L09", "L10"))
    card_only = [p for p in aggregate.variant_parts(unit) if p in _CARD_ONLY]
    if card_only:
        raise ValueError(f"HD26 variants {card_only} are listed on the card but not "
                         "scheduled in milestone V2")
    out = []
    for pid in products:
        if pid == "L09":
            out.append(run_l09(inst))
        else:
            out.append(aggregate.l10_hd26r2(inst, proxy="wet", product_id="L10"))
    return out
