# -*- coding: utf-8 -*-
"""Validation v2 — the lag tables: fixed-column CSVs and the markdown report of
Stage L (spec §6.4-§6.5, §9.7, §11.2 tables T6-T7; scoring spec §9).

Question
--------
How are the lag scores written so that every row of the lag plan appears,
the printing rule is applied, and the pre-lock tables can never be mistaken
for the pre-registered analysis?

Rules
-----
* CSV files with the fixed column lists of the scoring spec
  (:data:`POINTS_COLUMNS`, :data:`SUMMARY_COLUMNS`, :data:`FAMILY_COLUMNS`,
  :data:`REPRO_COLUMNS`, :data:`COST_COLUMNS`). Pre-lock, every file starts
  with the banner line ``# PRE-LOCK, descriptive; not the pre-registered
  analysis`` (:func:`read_csv` skips it) and ``lag_meta.json`` records it.
* ``lag_table.md``: the banner (pre-lock), the table notes (MAREA's home
  advantage; gauge agreement is weak evidence, §6; the catalogue instants
  read every imagery lag low by the datatake-to-sensing offset), a "first
  lag table" section (the four gauge sites at their available windows, the
  CO-PRIMARY truths (a), tau_ab and (b) side by side, plus a cross-site
  comparable summary over the calendar-year windows only, one block per
  co-primary truth), the imagery-against-non-imagery line per site, family
  L-P1, one table per site x window x judge with rows = methods grouped in
  panels (MAREA and the in-house estimators F0; literature F1-F3; in-house
  adaptations F4; models, physics and references), the truth-free
  reproducibility and coherence tables, the efficiency table and the §6.4 /
  §6.5 items milestone V2 does not compute. A fidelity whose check has not
  passed prints so (L08 'F3 (paper check pending)', L09 'F2 pending G-hd26
  (F3 evidence)'). A blinded window (pre-lock W3) reads BLINDED.
* No row is ever blank: it reads "NOT COMPUTABLE (<stage>)", "not
  applicable (<plan note>)", "NOT AVAILABLE (<reason>)", "judge excluded
  (<rules>)" or "implementation not validated" (no positive control PASS or
  "control mismatch" flag, or a required gate not passed; such a row
  carries no numbers unless the development view is asked for).
* Method names, tiers, fidelity and families come from the method cards
  (``lags.card``) when the adapter module is built, else from the catalogue
  of spec §3.3 (:data:`CATALOGUE`).
"""
from __future__ import annotations

import functools
import io
import math
import os

import numpy as np
import pandas as pd

from experiments.benchmark import product
from experiments.benchmark.inference import FAMILY_COLUMNS

BANNER = "PRE-LOCK, descriptive; not the pre-registered analysis"
META_FORMAT = "v2-lag-scores-1"

POINTS_COLUMNS = ("site", "window", "judge", "location", "s_g_km", "mouth", "development",
                  "product", "method", "variant", "family", "fidelity", "lag_kind", "status",
                  "reason", "validated", "readout_step", "n_readout_px", "edge", "tau_m",
                  "tau_m_band", "tau_m_band_interp", "tau_m_step1", "truth", "tau_true", "I_lo",
                  "I_hi", "e", "abs_e", "e_lo", "e_hi", "err_int", "E0_cm", "Etau_cm", "Emin_cm",
                  "dE_cm", "dE_lo_cm", "dE_hi_cm", "Eall_tau_cm", "Eall_min_cm", "dE_all_cm",
                  "LLE_cm", "LLE_lo_cm", "LLE_hi_cm", "explained", "AGF", "AGF_lo", "AGF_hi",
                  "material", "IS", "covered", "n_instants", "B", "flags")
#: the key columns, then the scoring spec §9 list with the CIs §5 asks for
#: next to RMS dE and RMS LLE and the mean explained share of §5
SUMMARY_KEYS = ("site", "window", "product", "method", "variant", "family", "fidelity",
                "lag_kind", "validated", "status", "truth", "judge_set", "label")
SUMMARY_METRICS = ("n_locations", "MAE", "MAE_lo", "MAE_hi", "median_abs_e", "bias", "bias_lo",
                   "bias_hi", "max_abs_e", "hit10", "hitI", "rms_dE_cm", "rms_dE_lo_cm",
                   "rms_dE_hi_cm", "rms_dE_all_cm", "rms_LLE_cm", "rms_LLE_lo_cm",
                   "rms_LLE_hi_cm", "mean_AGF", "mean_AGF_lo", "mean_AGF_hi", "mean_explained",
                   "edge_share", "step_shares", "slope", "ccc", "shape_err", "mono_violations",
                   "celerity_m_min_per_km", "celerity_true_min_per_km")
SUMMARY_COLUMNS = SUMMARY_KEYS + SUMMARY_METRICS
REPRO_COLUMNS = ("site", "window", "kind", "product", "band", "centre_km", "n_groups", "groups",
                 "values", "median_abs_diff", "icc21", "spread_sd", "spread_range", "label")
COST_COLUMNS = ("product", "method", "n_instances", "wall_s_median", "cpu_self_s_median",
                "peak_rss_mb_median", "peak_anon_mb_median", "unit_wall_s_median",
                "share_not_computable", "share_fallback0", "label")
#: the co-primary truths of the first table (spec §0.3 item 7, §6.2)
CO_PRIMARY = ("a", "ab", "b")
CO_PRIMARY_NAMES = {"a": "(a)", "ab": "tau_ab", "b": "(b)"}
#: fidelity pending its check: what the table prints until the check passes
FIDELITY_PENDING = {"L08": "F3 (paper check pending)",
                    "L09": "F2 pending G-hd26 (F3 evidence)"}
FILES = {"points": "lag_points.csv", "summary": "lag_summary.csv", "family": "lag_family.csv",
         "repro": "lag_repro.csv", "cost": "lag_cost.csv", "table_md": "lag_table.md",
         "meta": "lag_meta.json"}

TABLE_NOTES = (
    "Home advantage: MAREA's internal parameters were developed on these sites, judges and "
    "truths (P0 is its development setting; spec §0.8, §9.7, §9.9).",
    "Gauge agreement is weak evidence about MAREA's lag: the lag absorbs wet/dry departures "
    "(ebb retention, drainage, high flats lagging low flats) and truth (a) is close to MAREA's "
    "own estimand; the decisive, non-circular test is the elevation contrast MAREA - no delay "
    "against external heights (family P1, spec §6).",
    "e = tau_m - tau_truth in minutes (circular for phase-type outputs, wrapped about the "
    "truth-free 67.5 min); E, dE, LLE in cm; intervals are truth uncertainty only "
    "(calendar-month block bootstrap of the gauge's S2 instants, paired across methods).",
    "Scene instants are the catalogue (datatake-start) times (user decision 2026-10-14). "
    "The tile is sensed 0-10.6 min later (median 6.5 min at the Ems), so every imagery lag "
    "(L01-L12, fitted to h_B at the catalogue instant) is expected to read about that offset "
    "LOW against truths (a) and (b), which compare the gauge with h_B at the same instants; "
    "the model, physics and reference rows (L14-L20) carry no such offset. Nothing is "
    "corrected (attribute first): the section 'Imagery against non-imagery rows' compares "
    "the two groups per site.",
)
#: the note every lag table carries (spec §6, §9.7, §11)
SHORT_NOTE = ("*Note: home advantage (MAREA was developed on these sites, judges and truths); "
              "gauge agreement is weak evidence about MAREA's lag (spec §6); CIs are truth "
              "uncertainty only.*")
PANELS = ("MAREA and in-house estimators (F0)", "Literature (F1-F3)",
          "In-house adaptations (F4)", "Models, physics and references")

#: spec §3.3: id -> (name, tier, fidelity, family, lag_kind)
CATALOGUE = {
    "L01": ("MAREA m2a", "-", "F0", "likelihood", "linear"),
    "L01v": ("MAREA per relative orbit", "1", "F0", "in-house variants", "linear"),
    "L02": ("m2b concordance", "1", "F0", "order statistic", "linear"),
    "L03": ("m2c flooded-area rank", "1", "F0", "order statistic", "linear"),
    "L04w": ("XC-wet", "1", "F4", "order statistic", "linear"),
    "L04i": ("XC-index", "1", "F4", "order statistic", "linear"),
    "L07": ("LOOP", "1", "F4", "self-consistency", "linear"),
    "L08": ("GM20 (Geyman & Maloof 2020)", "1", "F3", "harmonic", "phase"),
    "L09": ("HD26-H (Hart-Davis et al. 2026)", "1", "F2", "harmonic", "phase"),
    "L10": ("HD26-R2", "1", "F4", "harmonic/response", "linear"),
    "L11": ("PPL phase-plane logistic", "1", "F4", "harmonic", "phase"),
    "L12": ("BT26-shift", "1", "F4", "order statistic", "linear"),
    "L14": ("global tide model (primary)", "1", "-", "models", "linear"),
    "L15": ("multi-model ensemble", "1", "-", "models", "linear"),
    "L16": ("regional model (CMEMS)", "1", "-", "models", "linear"),
    "L17": ("zero lag", "1", "-", "physics", "linear"),
    "L18": ("shallow-water celerity", "1", "-", "physics", "linear"),
    "L20": ("REF gauge leave-one-out", "1", "-", "reference", "linear"),
}
PRODUCT_METHOD = {"L01": "E0", "L01v": "L01v", "L02": "L02", "L03": "L03", "L04w": "L04",
                  "L04i": "L04", "L07": "L07", "L08": "L08", "L09": "L09", "L10": "L09",
                  "L11": "L11", "L12": "L12", "L14": "L14", "L15": "L14", "L16": "L16",
                  "L17": "L17", "L18": "L18", "L20": "L20"}
#: read the method cards (``lags.card``) when the adapters are built
USE_CARDS = True


def panel_of(fidelity):
    f = str(fidelity or "")
    if f == "F0":
        return PANELS[0]
    if f in ("F1", "F2", "F3"):
        return PANELS[1]
    if f == "F4":
        return PANELS[2]
    return PANELS[3]


@functools.lru_cache(maxsize=None)
def _card_info(base):
    if not USE_CARDS:
        return {}
    try:
        from experiments.benchmark import lags
        c = lags.card(base)
    except Exception:
        return {}
    return {k: c.get(k) for k in ("name", "tier", "fidelity", "family", "lag_kind") if
            c.get(k) is not None}


def product_info(pid):
    """Name, tier, fidelity, family, lag kind, method and panel of a product
    id (descriptive ids '<base>@<x>' take their base's)."""
    base = str(pid).split("@")[0]
    name, tier, fid, fam, lk = CATALOGUE.get(base, (base, "", "", "", "linear"))
    info = {"product": str(pid), "base": base, "name": name, "tier": tier, "fidelity": fid,
            "family": fam, "lag_kind": lk, "method": PRODUCT_METHOD.get(base, base)}
    info.update(_card_info(base))
    if "@" in str(pid):
        info["name"] = f"{info['name']} [{str(pid).split('@', 1)[1]}]"
    info["panel"] = panel_of(info["fidelity"])
    # what the tables print: a fidelity whose check has not passed says so
    info["fidelity_label"] = FIDELITY_PENDING.get(base, info["fidelity"])
    return info


def _cell(pid, info):
    return f"{pid} {info['name']} ({info['tier'] or '-'}, {info.get('fidelity_label') or '-'})"


# ─────────────────────────────────────────────────────────────────────────────
#  CSV
# ─────────────────────────────────────────────────────────────────────────────

def _frame(obj, columns):
    df = pd.DataFrame(obj) if not isinstance(obj, pd.DataFrame) else obj.copy()
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df[list(columns)]


def write_text_lf(path, text):
    """Write a text file atomically with LF line ends (the same bytes on the
    laptop and the server)."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def write_csv(df, path, pre_lock):
    """Write ``df`` (LF line ends); pre-lock, the banner is the first line."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    text = (f"# {BANNER}\n" if pre_lock else "") + buf.getvalue()
    return write_text_lf(path, text)


def read_csv(path):
    """Read a lag CSV written by :func:`write_csv` (the banner line skipped)."""
    with open(path, encoding="utf-8") as f:
        first = f.readline()
    skip = 1 if first.startswith("# ") else 0
    return pd.read_csv(path, skiprows=skip)


# ─────────────────────────────────────────────────────────────────────────────
#  markdown
# ─────────────────────────────────────────────────────────────────────────────

def _num(x, fmt="{:.1f}", dash="–"):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    if not math.isfinite(v):
        return dash
    return fmt.format(v)


def _ci(lo, hi, fmt="{:.1f}"):
    if _num(lo) == "–" or _num(hi) == "–":
        return ""
    return f" [{_num(lo, fmt)}, {_num(hi, fmt)}]"


def _truthy(v):
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return False
    except TypeError:
        pass
    return bool(v)


def _row_status(r, show_unvalidated):
    st = str(r.get("status", ""))
    if st == "ok" and not _truthy(r.get("validated")):
        return "implementation not validated", show_unvalidated
    return st, st == "ok"


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return out


def _judge_table(g, order, show_unvalidated):
    """The rows of one site x window x judge table (panels in order)."""
    header = ["method (tier, fidelity)", "status", "tau_m (step)", "tau_a [I_a]", "e_a [CI]",
              "e_ab", "tau_b ± SE", "e_b", "E0 / E(tau_m) / E(tau_a) (cm)", "dE (cm) [CI]",
              "AGF % [CI]", "LLE (cm)", "flags"]
    by_p = {p: d for p, d in g.groupby("product", sort=False)}
    prods = sorted(by_p, key=lambda p: (order.get(p.split("@")[0], 999), "@" in p, p))
    panels = {}
    for p in prods:
        panels.setdefault(product_info(p)["panel"], []).append(p)
    rows = []
    for panel in PANELS:
        if panel not in panels:
            continue
        rows.append([f"**{panel}**"] + [""] * (len(header) - 1))
        for p in panels[panel]:
            d = by_p[p].set_index("truth", drop=False)
            ra = d.loc["a"] if "a" in d.index else d.iloc[0]
            info = product_info(p)
            status, show = _row_status(ra, show_unvalidated)
            cell = _cell(p, info)
            if not show:
                rows.append([cell, status] + ["–"] * (len(header) - 3) +
                            [str(ra.get("flags") or "") if str(ra.get("status")) == "ok"
                             else ""])
                continue
            rb = d.loc["b"] if "b" in d.index else None
            rab = d.loc["ab"] if "ab" in d.index else None
            step = ra.get("readout_step")
            tau = _num(ra.get("tau_m"))
            if rb is not None and _num(rb.get("tau_m")) not in (tau, "–"):
                tau += f" (b: {_num(rb.get('tau_m'))})"
            tau += f" ({step})" if step is not None and str(step) != "nan" else ""
            ta = f"{_num(ra.get('tau_true'), '{:.0f}')}" + _ci(ra.get("I_lo"), ra.get("I_hi"),
                                                                "{:.0f}")
            ea = _num(ra.get("e"), "{:+.1f}") + _ci(ra.get("e_lo"), ra.get("e_hi"), "{:+.1f}")
            eab = _num(None if rab is None else rab.get("e"), "{:+.1f}")
            if rb is not None:
                se = (float(rb.get("I_hi")) - float(rb.get("I_lo"))) / (2 * 1.96) \
                    if _num(rb.get("I_hi")) != "–" and _num(rb.get("I_lo")) != "–" else np.nan
                tb = f"{_num(rb.get('tau_true'))} ± {_num(se)}"
                eb = _num(rb.get("e"), "{:+.1f}")
            else:
                tb = eb = "–"
            E = (f"{_num(ra.get('E0_cm'))} / {_num(ra.get('Etau_cm'))} / "
                 f"{_num(ra.get('Emin_cm'))}")
            dE = _num(ra.get("dE_cm"), "{:.2f}") + _ci(ra.get("dE_lo_cm"), ra.get("dE_hi_cm"),
                                                       "{:.2f}")
            agf = ra.get("AGF")
            if _num(agf) == "–" and "lag not material" in str(ra.get("flags") or ""):
                agf_s = "lag not material"
            else:
                agf_s = _num(None if agf is None else 100 * float(agf), "{:.0f}") + \
                    _ci(None if ra.get("AGF_lo") is None else 100 * float(ra.get("AGF_lo")),
                        None if ra.get("AGF_hi") is None else 100 * float(ra.get("AGF_hi")),
                        "{:.0f}")
            flags = str(ra.get("flags") or "")
            if status != "ok":
                flags = (status + "; " + flags).strip("; ")
            rows.append([cell, "ok" if status == "ok" else "not validated (shown)", tau, ta, ea,
                         eab, tb, eb, E, dE, agf_s, _num(ra.get("LLE_cm")), flags])
    return header, rows


def _sorted_products(products, order):
    return sorted(set(products), key=lambda p: (order.get(str(p).split("@")[0], 999), str(p)))


def _first_table(summary, context, order, show_unvalidated):
    """The first lag table: per judged site at its descriptive L-P1 window,
    MAE [CI], bias and hit10 against each CO-PRIMARY truth ((a), tau_ab,
    (b); spec §0.3 item 7, §6.2) side by side, then RMS dE, RMS LLE and the
    mean AGF once (they are defined against tau_a for every truth), and the
    cross-site summary with one MAE block per co-primary truth."""
    lines = []
    lpw = context.get("lp1_windows") or {}
    s = summary
    if s.empty:
        return ["(no summary rows)"]
    head = s[s["judge_set"] == "headline"]
    lines.append("Headline judges, co-primary truths (a), tau_ab and (b) side by side; site-level "
                 "means over locations (each location once; the Terneuzen pair is one "
                 "location); CIs are truth uncertainty only. RMS dE, RMS LLE and AGF are level "
                 "metrics relative to tau_a for every truth.")
    for site, w in lpw.items():
        sub = head[(head["site"] == site) & (head["window"] == w)]
        lines += ["", f"**{site} · {w}**", ""]
        rows = []
        for p in _sorted_products(sub["product"], order):
            by_t = {T: sub[(sub["product"] == p) & (sub["truth"] == T)] for T in CO_PRIMARY}
            ra = by_t["a"].iloc[0] if len(by_t["a"]) else None
            if ra is None:
                continue
            info = product_info(p)
            st, show = _row_status(ra, show_unvalidated)
            cell = _cell(p, info)
            n = _num(ra.get("n_locations"), "{:.0f}")
            if not show or n in ("–", "0"):
                rows.append([cell, st, n] + ["–"] * (3 * len(CO_PRIMARY) + 3))
                continue
            cells = [cell, "ok" if st == "ok" else "not validated (shown)", n]
            for T in CO_PRIMARY:
                r = by_t[T].iloc[0] if len(by_t[T]) else None
                if r is None or _num(r.get("n_locations"), "{:.0f}") in ("–", "0"):
                    cells += ["–", "–", "–"]
                    continue
                cells += [_num(r.get("MAE")) + _ci(r.get("MAE_lo"), r.get("MAE_hi")),
                          _num(r.get("bias"), "{:+.1f}"),
                          _num(None if pd.isna(r.get("hit10")) else 100 * r.get("hit10"),
                               "{:.0f}")]
            cells += [_num(ra.get("rms_dE_cm"), "{:.2f}") + _ci(ra.get("rms_dE_lo_cm"),
                                                                ra.get("rms_dE_hi_cm"), "{:.2f}"),
                      _num(ra.get("rms_LLE_cm")),
                      _num(None if pd.isna(ra.get("mean_AGF")) else 100 * ra.get("mean_AGF"),
                           "{:.0f}")]
            rows.append(cells)
        hdr = ["method (tier, fidelity)", "status", "n loc"]
        for T in CO_PRIMARY:
            nm = CO_PRIMARY_NAMES[T]
            hdr += [f"MAE {nm} (min) [CI]", f"bias {nm}", f"hit10 {nm} %"]
        hdr += ["RMS dE vs tau_a (cm) [CI]", "RMS LLE vs tau_a (cm)", "mean AGF vs tau_a %"]
        lines += _table(hdr, rows)
        lines += ["", SHORT_NOTE]
    lines += ["", "**Cross-site comparable summary (calendar-year windows only; pooled = the "
                  "mean of the site values; one block per co-primary truth)**", ""]
    for w in ("Y2023", "Y2024", "Y2025"):
        for T in CO_PRIMARY:
            sub = head[(head["window"] == w) & (head["truth"] == T)]
            sites = [x for x in dict.fromkeys(sub["site"]) if x != "pooled"]
            if not sites:
                continue
            pooled = sub[sub["site"] == "pooled"].set_index("product")
            rows = []
            for p in _sorted_products(sub["product"], order):
                cells = [p]
                for site in sites:
                    r = sub[(sub["site"] == site) & (sub["product"] == p)]
                    if not len(r):
                        cells.append("–")
                        continue
                    r = r.iloc[0]
                    st, show = _row_status(r, show_unvalidated)
                    cells.append(_num(r.get("MAE")) if show else st)
                if p in pooled.index:
                    pr = pooled.loc[p]
                    if isinstance(pr, pd.DataFrame):
                        pr = pr.iloc[0]
                    st, show_p = _row_status(pr, show_unvalidated)
                    cells.append(_num(pr.get("MAE")) if show_p else st)
                else:
                    cells.append("–")
                rows.append(cells)
            lines += [f"*{w}* — MAE (min) against truth {CO_PRIMARY_NAMES[T]}", ""]
            lines += _table(["product"] + sites + ["pooled"], rows) + [""]
    return lines


def markdown(points, summary, family, repro, cost, plan_rows, pre_lock, context=None):
    """The text of lag_table.md."""
    context = dict(context or {})
    show = bool(context.get("show_unvalidated"))
    pts = _frame(points, POINTS_COLUMNS)
    summ = _frame(summary, SUMMARY_COLUMNS)
    order = {}
    for r in plan_rows or []:
        for p in str(r.get("products", "")).split():
            order.setdefault(p, len(order))
    for k in CATALOGUE:
        order.setdefault(k, len(order))
    L = ["# Lag benchmark — validation v2, Stage L", ""]
    if pre_lock:
        L += [f"> **{BANNER}**", ""]
    L += ["Notes:", ""] + [f"- {n}" for n in TABLE_NOTES]
    if context.get("B") is not None:
        L.append(f"- Truth-side bootstrap B = {context['B']}.")
    for b in context.get("blinded") or []:
        L.append(f"- {b['site']} {b['window']}: BLINDED before lock-A (the confirmatory window "
                 "of L-P1; `score --unblind` is the user's decision and is disclosed)")
    if context.get("unblinded"):
        L.append(f"- UNBLINDED before lock-A by the user's decision: "
                 f"{', '.join(context['unblinded'])} (recorded in scores/prelock_ledger.json; "
                 "lock-A lists it under forking_paths and needs an amendment)")
    L += ["", "## First lag table", ""]
    L += _first_table(summ, context, order, show)
    # imagery against non-imagery rows (the catalogue-to-sensing offset)
    ol = context.get("offset_lines") or []
    if ol:
        L += ["", "## Imagery against non-imagery rows (catalogue instants)", "",
              "Median signed e against truth (a) over the ok rows of each group: the imagery "
              "rows (L01-L12) and the lag estimates that read no imagery (the primary global "
              "model L14, the regional model L16, the celerity L18; zero lag is a null "
              "baseline). Descriptive, for the attribution step; nothing is corrected. The "
              "sensing offset is the site's median datatake-to-site time of the optional "
              "`diag time`, where it exists.", ""]
        rows = []
        for r in ol:
            sens = r.get("sensing") or {}
            rows.append([r["site"], r["window"],
                         _num(r.get("imagery_median_e"), "{:+.1f}") +
                         f" ({int(r.get('n_imagery') or 0)})",
                         _num(r.get("non_imagery_median_e"), "{:+.1f}") +
                         f" ({int(r.get('n_non_imagery') or 0)})",
                         (_num(sens.get("median_min"), "{:+.1f}") + f" (n {sens.get('n')})")
                         if sens else "no diag time output"])
        L += _table(["site", "window", "imagery e median (rows)", "L14/L16/L18 e median "
                     "(rows)", "sensing offset (min)"], rows)
    # family L-P1
    L += ["", "## Family L-P1 (MAREA L01 against zero lag, the primary global model, the "
              "regional model, GM20 and HD26-H)", ""]
    lp = context.get("lp1") or {}
    if lp:
        L.append(f"Windows: {', '.join(f'{k} {v}' for k, v in (lp.get('windows') or {}).items())}"
                 f"; comparisons: {', '.join(lp.get('comparisons') or [])}; "
                 + ("confirmatory (LOCK phase A, W3)." if lp.get("confirmatory") else
                    "descriptive only: no verdict is issued."))
        L.append("")
    fam = pd.DataFrame(family)
    if not fam.empty:
        for label, g in fam.groupby("label", sort=False):
            L += [f"*{label}*", ""]
            rows = []
            for _, r in g.iterrows():
                rows.append([r["truth"], r["comparison"], _num(r["n"], "{:.0f}"),
                             _num(r["k_zero"], "{:.0f}"), _num(r["S"], "{:+.2f}"),
                             _num(r["p"], "{:.4f}"), _num(r["p_min"], "{:.2g}"),
                             _num(r["holm_p"], "{:.4f}"),
                             "–" if r["reject"] is None or pd.isna(r["reject"])
                             else str(r["reject"]),
                             r["site_signs"] or "–", _num(r["agree_sites"], "{:.0f}"),
                             "–" if r["confirmatory"] is None or pd.isna(r["confirmatory"])
                             else str(r["confirmatory"])])
            L += _table([c for c in FAMILY_COLUMNS if c != "label"], rows) + [""]
    else:
        L.append("(no family rows: MAREA or the comparisons have no scored location)")
    # per site, window and judge
    L += ["", "## Per site, window and judge", ""]
    notes = context.get("site_notes") or {}
    for (site, w), g in pts.groupby(["site", "window"], sort=False):
        L += [f"### {site} · {w}", ""]
        sn = notes.get(site) or {}
        for k in ("package", "truth", "judges"):
            if sn.get(k):
                L.append(f"- {k}: {sn[k]}")
        excl = g[g["status"].astype(str).str.startswith("judge excluded")]
        ex_j = excl.drop_duplicates("judge")
        for _, r in ex_j.iterrows():
            L.append(f"- {r['judge']}: {r['status']}")
        st_ = g["status"].astype(str)
        guard = g[st_.str.startswith("NOT AVAILABLE: store years") | st_.str.startswith("BLINDED")]
        if len(guard):
            L.append(f"- every row: {guard['status'].iloc[0]}")
        L.append("")
        inc_j = [j for j in dict.fromkeys(g["judge"])
                 if j not in set(ex_j["judge"]) and not len(guard)]
        if len(guard):
            prods = list(dict.fromkeys(g["product"]))
            L += _table(["method (tier, fidelity)", "status"],
                        [[f"{p} {product_info(p)['name']}", guard["status"].iloc[0]]
                         for p in prods]) + [""]
            continue
        for j in inc_j:
            gj = g[g["judge"] == j]
            r0 = gj.iloc[0]
            ra = gj[(gj["truth"] == "a") & gj["tau_true"].notna()]
            truth_line = ""
            if len(ra):
                x = ra.iloc[0]
                truth_line = (f"; tau_a = {_num(x['tau_true'], '{:.0f}')} min "
                              f"[{_num(x['I_lo'], '{:.0f}')}, {_num(x['I_hi'], '{:.0f}')}]")
            flags = []
            if _truthy(r0.get("mouth")):
                flags.append("mouth")
            if _truthy(r0.get("development")):
                flags.append("development judge")
            L += [f"#### {site} · {w} · {j} ({r0['location']}; s_g {_num(r0['s_g_km'])} km"
                  f"{'; ' + ', '.join(flags) if flags else ''}{truth_line})", ""]
            header, rows = _judge_table(gj, order, show)
            L += _table(header, rows) + ["", SHORT_NOTE, ""]
    # coherence and repro
    rp = pd.DataFrame(repro)
    coh_status = context.get("coherence_status") or []
    valid = context.get("validated") or {}

    def shown(site, pid):
        return show or bool(valid.get(f"{site}|{pid}", True))

    L += ["## Truth-free reproducibility and coherence", ""]
    if not rp.empty:
        allr = rp[rp["band"].astype(str) == "all"]
        if len(allr):
            rows = []
            for _, r in allr.iterrows():
                if shown(r["site"], r["product"]):
                    rows.append([r["site"], r["window"], r["kind"], r["product"],
                                 _num(r["n_groups"], "{:.0f}"), _num(r["median_abs_diff"]),
                                 _num(r["icc21"], "{:.2f}"), r["label"]])
                else:
                    rows.append([r["site"], r["window"], r["kind"], r["product"],
                                 _num(r["n_groups"], "{:.0f}"), "–", "–",
                                 "implementation not validated"])
            L += _table(["site", "window", "kind", "product", "groups", "median |dtau| (min)",
                         "ICC(2,1)", "label"], rows) + [""]
        coh = rp[rp["kind"] == "coherence"]
        for (site, w), g in coh.groupby(["site", "window"], sort=False):
            L += [f"**{site} · {w} — lag coherence only (no included judge)**", ""]
            pids = str(g["groups"].iloc[0]).split(";")
            ok = [p for p in pids if shown(site, p)]
            hidden = [p for p in pids if p not in ok]
            rows = []
            for _, r in g.iterrows():
                vals = dict(kv.split("=") for kv in str(r["values"]).split(";") if "=" in kv)
                fin = np.array([float(vals[p]) for p in ok if p in vals], float)
                fin = fin[np.isfinite(fin)]
                rows.append([r["band"], _num(r["centre_km"])] +
                            [_num(vals.get(p)) for p in ok] +
                            [_num(np.std(fin) if len(fin) > 1 else None),
                             _num(np.ptp(fin) if len(fin) > 1 else None)])
            L += _table(["band", "centre (km)"] + ok + ["SD across methods",
                                                        "range across methods"], rows)
            if hidden:
                L += ["", f"Not shown (implementation not validated): {', '.join(hidden)}."]
            L += [""]
    if coh_status:
        rows = []
        for s, w, p, st, v in coh_status:
            if st == "ok" and v is False and not show:
                st = "implementation not validated"
            rows.append([s, w, p, st])
        L += ["Products at windows without included judges:", ""]
        L += _table(["site", "window", "product", "status"], rows) + [""]
    if rp.empty and not coh_status:
        L += ["(nothing to compare: no site has band lags at two calendar-year windows, no "
              "L01v result holds two relative orbits, and every scored window has an "
              "included judge)", ""]
    # efficiency
    cs = pd.DataFrame(cost)
    L += ["## Efficiency (contended production runs)", ""]
    if not cs.empty:
        rows = [[r["product"], r["method"], _num(r["n_instances"], "{:.0f}"),
                 _num(r["wall_s_median"]), _num(r["cpu_self_s_median"]),
                 _num(r["peak_rss_mb_median"], "{:.0f}"),
                 _num(r.get("peak_anon_mb_median"), "{:.0f}"), _num(r["unit_wall_s_median"]),
                 _num(r["share_not_computable"], "{:.2f}"), _num(r["share_fallback0"], "{:.2f}")]
                for _, r in cs.iterrows()]
        L += ["Median over instances of the method's own stages (lag.<method>.*; the "
              "products of one unit share them) and of the whole unit; contended production "
              "runs, never used in comparisons (spec §8.2). Peak anonymous memory leaves out "
              "the file pages of the memory-mapped package tiles (the RSS counts them).", ""]
        L += _table(["product", "method", "instances", "method wall (s)", "method CPU (s)",
                     "peak RSS (MB)", "peak anon (MB)", "unit wall (s)", "NOT COMPUTABLE share",
                     "fallback0 share"], rows) + [""]
    else:
        L += ["(no timings)", ""]
    deferred = context.get("deferred") or []
    if deferred:
        L += ["## Not computed in milestone V2 (spec §6.4, §6.5)", ""] + \
            [f"- {d}" for d in deferred] + [""]
    warns = context.get("warnings") or []
    if warns:
        L += ["## Warnings", ""] + [f"- {w}" for w in warns] + [""]
    if pre_lock:
        L += [f"> **{BANNER}**", ""]
    return "\n".join(L) + "\n"


def write(points, summary, family, repro, plan_rows, out_dir, pre_lock, *, cost=None,
          context=None):
    """Write the lag score files (scoring spec §9).

    Parameters
    ----------
    points, summary, family, repro : DataFrame or records
    plan_rows : list of dict
        ``plans/lag.csv`` rows (the full expected row set and its order).
    out_dir : str
        ``<root>/scores``.
    pre_lock : bool
        Banner on every file; no confirmatory verdict.
    cost : DataFrame or records, optional
    context : dict, optional
        Extra facts for the markdown (lp1 windows, site notes, coherence
        statuses, warnings, B, show_unvalidated).

    Returns
    -------
    dict {name: path}
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    frames = {"points": _frame(points, POINTS_COLUMNS),
              "summary": _frame(summary, SUMMARY_COLUMNS),
              "family": _frame(family, FAMILY_COLUMNS),
              "repro": _frame(repro, REPRO_COLUMNS),
              "cost": _frame(cost if cost is not None else [], COST_COLUMNS)}
    for name, df in frames.items():
        paths[name] = write_csv(df, os.path.join(out_dir, FILES[name]), pre_lock)
    md = markdown(frames["points"], frames["summary"], frames["family"], frames["repro"],
                  frames["cost"], plan_rows, pre_lock, context)
    paths["table_md"] = write_text_lf(os.path.join(out_dir, FILES["table_md"]), md)
    ctx = dict(context or {})
    meta = {"format": META_FORMAT, "pre_lock": bool(pre_lock),
            "banner": BANNER if pre_lock else None,
            "files": {k: os.path.basename(v) for k, v in paths.items()},
            "created_utc": product.utc_stamp(),
            "lp1": {k: v for k, v in (ctx.get("lp1") or {}).items() if k != "claims"},
            "lp1_claims": (ctx.get("lp1") or {}).get("claims"),
            "blinded": list(ctx.get("blinded") or []),
            "unblinded": list(ctx.get("unblinded") or []),
            "warnings": list(ctx.get("warnings") or []), "B": ctx.get("B")}
    paths["meta"] = os.path.join(out_dir, FILES["meta"])
    product.write_json(paths["meta"], meta)
    return paths
