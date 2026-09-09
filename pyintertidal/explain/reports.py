"""
reports.py — The bookkeeping panels of a run
=============================================

Every stage of the pipeline throws away data on purpose: dates that are too
cloudy, pixels outside the intertidal window, scenes with too few clear
observations. Those decisions are where a study is won or lost, and a number
like "305 usable dates" hides which 305 and why the other 1,074 went.

These panels put that back on the page. They are deliberately verbose where
verbosity is cheap — a collapsible list costs one line until you open it —
and they follow the same visual language as the rest of
:mod:`pyintertidal.explain`, so a notebook reads as one document rather than
a mix of tables and print statements.

Each function takes finished products and returns a displayable object; none
of them computes anything the analysis has not already computed.
"""

from __future__ import annotations

import numpy as np

from .htmlrepr import _CSS, _Html, _section

#: Extra styling for the per-date listings, on top of the shared panel CSS.
_EXTRA = """
<style>
.pit-repr table.dates{border-collapse:collapse;font-size:11px;width:100%;
 font-family:ui-monospace,monospace}
.pit-repr table.dates td{padding:1px 8px;white-space:nowrap}
.pit-repr table.dates tr:nth-child(even){background:#f7f9fa}
.pit-repr .ref{color:#1a6b3c;font-weight:600}
.pit-repr .keep{color:#1f5fa8;font-weight:600}
.pit-repr .drop{color:#b03030}
.pit-repr .bar{height:9px;border-radius:2px;display:inline-block;
 vertical-align:middle}
.pit-repr .legend{font-size:11.5px;color:#39424b;margin:6px 0 2px}
.pit-repr .big{font-size:20px;font-weight:600;color:#1c1f23}
.pit-repr .cols{display:flex;gap:26px;flex-wrap:wrap;align-items:flex-start}
.pit-repr .stat{min-width:104px}
.pit-repr .stat small{display:block;color:#6b747d;font-size:11px}
</style>
"""


def _stat(value, label):
    return (f"<div class='stat'><span class='big'>{value}</span>"
            f"<small>{label}</small></div>")


def _bar(fractions_colours, width=520):
    """A single stacked bar; ``fractions_colours`` is [(fraction, colour)]."""
    parts = "".join(
        f"<span class='bar' style='width:{max(f, 0) * width:.0f}px;"
        f"background:{c}'></span>" for f, c in fractions_colours)
    return f"<div style='margin:4px 0 8px'>{parts}</div>"


# ─────────────────────────────────────────────────────────────────────────────
#  1 · What the archive offers
# ─────────────────────────────────────────────────────────────────────────────

def dates_available(dates, time_extent=None, title="Archive"):
    """What the satellite archive holds for this area and period.

    The revisit statistics matter more than the raw count: a decade with a
    long gap is not the same archive as a decade sampled evenly, and the
    gaps are what later limit which epochs can be fitted.
    """
    dates = sorted(str(d)[:10] for d in dates)
    n = len(dates)
    if not n:
        return _Html(f"{_CSS}<div class='pit-repr'>No dates found.</div>")

    days = np.array([np.datetime64(d) for d in dates], dtype="datetime64[D]")
    gaps = np.diff(days).astype(int)
    years = {}
    for d in dates:
        years[d[:4]] = years.get(d[:4], 0) + 1

    span = f"{dates[0]} → {dates[-1]}"
    if time_extent:
        span = f"{time_extent[0]} → {time_extent[1]}"

    header = (f"<div class='hdr'><b>{title}</b> <span class='dims'>"
              f"{span}</span></div>")
    stats = ("<div class='cols'>"
             + _stat(f"{n:,}", "acquisitions")
             + _stat(f"{np.median(gaps):.0f} d", "median revisit")
             + _stat(f"{gaps.max():.0f} d", "longest gap")
             + _stat(f"{len(years)}", "years covered")
             + "</div>")

    per_year = "<table class='dates'>" + "".join(
        f"<tr><td>{y}</td><td>{c}</td><td>"
        f"<span class='bar' style='width:{c / max(years.values()) * 260:.0f}px;"
        f"background:#4a78c4'></span></td></tr>"
        for y, c in sorted(years.items())) + "</table>"

    listing = "<table class='dates'>" + "".join(
        "<tr>" + "".join(f"<td>{d}</td>" for d in dates[i:i + 6]) + "</tr>"
        for i in range(0, n, 6)) + "</table>"

    return _Html(f"{_CSS}{_EXTRA}<div class='pit-repr'>{header}{stats}"
                 f"{_section('Acquisitions per year', len(years), per_year, True)}"
                 f"{_section('Every date', n, listing)}</div>")


# ─────────────────────────────────────────────────────────────────────────────
#  2 · What the cloud screening kept, and why
# ─────────────────────────────────────────────────────────────────────────────

def cloud_screening(cloud_pct, usable_dates, clean_scene_max_bad=0.05,
                    cloud_threshold=0.10, title="Cloud screening"):
    """Which dates survived, split by the reason they survived.

    Three outcomes, and the middle one is the point of the whole design:

    ``reference``
        clean over the WHOLE scene, so the date is trusted enough to vote in
        the reference map;
    ``recovered``
        too cloudy globally, but clear over the transition zone — an
        observation a whole-scene filter would have thrown away;
    ``discarded``
        cloudy where it matters.

    Parameters
    ----------
    cloud_pct : dict {date: fraction}
        Per-date cloudiness over the transition zone, from
        :func:`pyintertidal.reference_and_clouds`.
    usable_dates : sequence
        The dates the filter kept.
    """
    items = sorted((str(d)[:10], float(v)) for d, v in cloud_pct.items())
    keep = {str(d)[:10] for d in usable_dates}
    n = len(items)

    ref = [(d, v) for d, v in items if d in keep and v <= clean_scene_max_bad]
    rec = [(d, v) for d, v in items if d in keep and v > clean_scene_max_bad]
    out = [(d, v) for d, v in items if d not in keep]

    rows = []
    for d, v in items:
        if d in keep and v <= clean_scene_max_bad:
            tag, cls = "ref", "ref"
        elif d in keep:
            tag, cls = "✓", "keep"
        else:
            tag, cls = "✗", "drop"
        rows.append(f"<tr><td class='{cls}'>[{tag}]</td><td>{d}</td>"
                    f"<td style='text-align:right'>{v * 100:.1f} %</td></tr>")
    listing = "<table class='dates'>" + "".join(rows) + "</table>"

    total = max(n, 1)
    bar = _bar([(len(ref) / total, "#1a6b3c"),
                (len(rec) / total, "#4a78c4"),
                (len(out) / total, "#d8dde2")])

    header = (f"<div class='hdr'><b>{title}</b> <span class='dims'>"
              f"threshold ≤ {cloud_threshold:.0%} over the transition zone"
              f"</span></div>")
    stats = ("<div class='cols'>"
             + _stat(f"{len(ref):,}", f"reference (≤{clean_scene_max_bad:.0%} global)")
             + _stat(f"+{len(rec):,}", "recovered by the transition filter")
             + _stat(f"{len(ref) + len(rec):,}", "usable")
             + _stat(f"{len(out):,}", "discarded")
             + "</div>")
    gain = (100.0 * len(rec) / len(ref)) if ref else 0.0
    note = (f"<div class='note'>A whole-scene filter would have kept "
            f"<b>{len(ref):,}</b> dates. Measuring cloudiness inside the "
            f"transition zone instead recovers <b>{len(rec):,}</b> more "
            f"(<b>+{gain:.0f} %</b>): scenes that are cloudy somewhere else "
            f"but clear over the estuary.</div>")

    return _Html(f"{_CSS}{_EXTRA}<div class='pit-repr'>{header}{stats}{bar}"
                 f"<div class='legend'><span class='ref'>[ref]</span> votes in "
                 f"the reference map &nbsp; <span class='keep'>[✓]</span> "
                 f"recovered &nbsp; <span class='drop'>[✗]</span> discarded"
                 f"</div>{_section('Every date, with its cloudiness', n, listing)}"
                 f"{note}</div>")


# ─────────────────────────────────────────────────────────────────────────────
#  3 · How the transition zone was split
# ─────────────────────────────────────────────────────────────────────────────

def intertidal_breakdown(reference_map, water_frequency, intertidal,
                         wf_low, wf_high, transform=None,
                         title="Intertidal zone"):
    """Where every pixel of the transition zone ended up.

    The transition class is only a CANDIDATE zone; the water-frequency window
    then splits it into ground that is almost never wet, ground that is
    almost always wet, and the intertidal proper. Reporting the three
    together is what makes the window a defensible choice rather than a
    magic number — widen it and you can see exactly what you gained.
    """
    ref = np.asarray(reference_map)
    wf = np.asarray(water_frequency, dtype=float)
    inter = np.asarray(intertidal, dtype=bool)

    transition = ref == 0
    known = transition & np.isfinite(wf)
    terrestrial = int((known & (wf < wf_low)).sum())
    aquatic = int((known & (wf > wf_high)).sum())
    n_inter = int(inter.sum())
    n_trans = int(transition.sum())

    px_km2 = abs(transform.a * transform.e) / 1e6 if transform is not None else None

    def area(px):
        return f" &nbsp;<span style='color:#6b747d'>{px * px_km2:.2f} km²</span>" \
            if px_km2 else ""

    header = (f"<div class='hdr'><b>{title}</b> <span class='dims'>"
              f"water-frequency window [{wf_low:.2f}, {wf_high:.2f}]"
              f"</span></div>")
    stats = ("<div class='cols'>"
             + _stat(f"{n_trans:,}", "transition (candidates)")
             + _stat(f"{n_inter:,}", "intertidal")
             + _stat(f"{terrestrial:,}", f"terrestrial (WF < {wf_low:.2f})")
             + _stat(f"{aquatic:,}", f"aquatic (WF > {wf_high:.2f})")
             + "</div>")
    total = max(n_trans, 1)
    bar = _bar([(n_inter / total, "#2b8cbe"),
                (terrestrial / total, "#d2b48c"),
                (aquatic / total, "#1f4e79")])

    rows = [("intertidal", n_inter, "#2b8cbe"),
            ("terrestrial", terrestrial, "#d2b48c"),
            ("aquatic", aquatic, "#1f4e79")]
    table = "<table class='dates'>" + "".join(
        f"<tr><td><span class='bar' style='width:9px;background:{c}'></span></td>"
        f"<td>{name}</td><td style='text-align:right'>{px:,} px</td>"
        f"<td>{area(px)}</td>"
        f"<td style='text-align:right'>{100 * px / total:.1f} %</td></tr>"
        for name, px, c in rows) + "</table>"

    return _Html(f"{_CSS}{_EXTRA}<div class='pit-repr'>{header}{stats}{bar}"
                 f"{table}</div>")
