"""
report.py — A self-contained HTML report per study area
========================================================

When a campaign produces dozens of tiles, nobody opens dozens of notebooks.
This module writes ONE HTML file per area containing the maps, the resolved
parameters, the diagnostics and the process graph — everything needed to
judge a result without re-running it.

Self-contained on purpose: images are embedded as base64 and diagrams are
inline SVG, so the file can be emailed, archived next to the data, or
attached to a paper's supplementary material and still render years later.
"""

from __future__ import annotations

import base64
import io
import os
from datetime import datetime

_CSS = """
body{font-family:system-ui,-apple-system,sans-serif;color:#1c1f23;margin:0;
 background:#f6f7f9}
.wrap{max-width:1000px;margin:0 auto;padding:28px 22px 60px}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:16px;margin:30px 0 10px;padding-bottom:6px;
 border-bottom:1px solid #dfe3e8}
.sub{color:#68727d;font-size:13px;margin-bottom:18px}
.card{background:#fff;border:1px solid #dfe3e8;border-radius:8px;
 padding:14px 16px;margin:12px 0}
.grid{display:flex;flex-wrap:wrap;gap:12px}
.grid .card{flex:1 1 300px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:5px 10px 5px 0;border-bottom:1px solid #eef1f4}
th{color:#68727d;font-weight:600}
td.n{font-family:ui-monospace,monospace}
img{max-width:100%;height:auto;border-radius:5px}
.kv{font-family:ui-monospace,monospace;font-size:12.5px}
.note{color:#68727d;font-size:12.5px;margin-top:6px}
.badge{display:inline-block;padding:2px 9px;border-radius:11px;font-size:12px;
 font-weight:600}
.ok{background:#e3f4e9;color:#1c6b38}.warn{background:#fdecea;color:#8a2318}
"""


def _figure_to_base64(fig, dpi=130):
    """Encode a matplotlib figure as an inline base64 PNG."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _table(rows, columns=None):
    if not rows:
        return "<p class='note'>no data</p>"
    if isinstance(rows, dict):
        rows = [{"key": k, "value": v} for k, v in rows.items()]
    columns = columns or list(rows[0].keys())
    head = "".join(f"<th>{c}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(
            f"<td class='n'>{r.get(c, '')}</td>" for c in columns) + "</tr>"
        for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_report(out_path, title, subtitle="", sections=None):
    """Assemble an HTML report from ordered sections.

    Each section is a dict with a ``title`` and one of:

    ``figures``    list of matplotlib figures (embedded as PNG)
    ``svg``        raw SVG string or a :class:`pyintertidal.explain.Diagram`
    ``table``      list of dicts (or a dict) rendered as a table
    ``params``     dict rendered as a parameter list
    ``html``       arbitrary HTML
    ``note``       explanatory paragraph

    Returns the path written.
    """
    parts = [f"<!doctype html><html><head><meta charset='utf-8'>"
             f"<title>{title}</title><style>{_CSS}</style></head><body>"
             f"<div class='wrap'><h1>{title}</h1>"
             f"<div class='sub'>{subtitle}"
             f"{' · ' if subtitle else ''}"
             f"generated {datetime.now():%Y-%m-%d %H:%M}</div>"]

    for section in sections or []:
        parts.append(f"<h2>{section.get('title', '')}</h2>")
        if section.get("note"):
            parts.append(f"<p class='note'>{section['note']}</p>")
        if section.get("params"):
            parts.append("<div class='card kv'>" + "".join(
                f"<div><b>{k}</b> = {v}</div>"
                for k, v in section["params"].items()) + "</div>")
        if section.get("table") is not None:
            parts.append("<div class='card'>"
                         + _table(section["table"], section.get("columns"))
                         + "</div>")
        if section.get("svg") is not None:
            svg = section["svg"]
            svg = svg.svg if hasattr(svg, "svg") else svg
            parts.append(f"<div class='card'>{svg}</div>")
        if section.get("figures"):
            parts.append("<div class='grid'>")
            for fig in section["figures"]:
                parts.append(f"<div class='card'><img src='data:image/png;"
                             f"base64,{_figure_to_base64(fig)}'/></div>")
            parts.append("</div>")
        if section.get("html"):
            parts.append(section["html"])

    parts.append("</div></body></html>")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("".join(parts))
    return out_path


def site_report(out_path, aoi, cube=None, params=None, figures=None,
                coverage=None, verdict=None, validation=None, oplog=None,
                zones=None, notes=None):
    """Standard report for one study area.

    Collects the things a reader needs to trust a result: what was processed,
    with which parameters, how well the tide was sampled, what the products
    look like, how they validated, and the process graph that produced them.
    """
    sections = []

    overview = {"site": aoi.name, "area_km2": f"{aoi.area_km2:.1f}",
                "bbox": ", ".join(f"{v:.3f}" for v in aoi.bbox.values())}
    if cube is not None:
        overview.update({"period": f"{cube.time_extent[0]} → {cube.time_extent[1]}",
                         "water detector": cube.water,
                         "bands": ", ".join(cube.bands)})
        try:
            overview["scenes"] = f"{len(cube.dates):,}"
            overview["grid"] = f"{cube.shape[0]}×{cube.shape[1]} @ " \
                               f"{abs(cube.grid[0].a):g} m"
        except Exception:
            pass
    sections.append({"title": "Study area", "params": overview,
                     "note": notes})

    if params:
        sections.append({"title": "Parameters used",
                         "params": params,
                         "note": "Every value that shaped the results, so the "
                                 "run can be reproduced exactly."})

    if coverage is not None:
        badge = ""
        if verdict is not None:
            cls = "ok" if verdict.get("suitable") else "warn"
            label = "suitable" if verdict.get("suitable") else "limited"
            reasons = "".join(f"<li>{r}</li>" for r in verdict.get("reasons", []))
            badge = (f"<p><span class='badge {cls}'>tidal sampling: {label}"
                     f"</span></p>" + (f"<ul class='note'>{reasons}</ul>"
                                       if reasons else ""))
        rows = [
            {"metric": "range coverage",
             "value": f"{coverage['range_coverage']['coverage_pct']:.1f} %"},
            {"metric": "entropy (normalised)",
             "value": f"{coverage['entropy']['entropy_normalised']:.3f}"},
            {"metric": "dispersion (VMR)",
             "value": f"{coverage['dispersion']['vmr']:.2f} "
                      f"({coverage['dispersion']['pattern']})"},
            {"metric": "vertical sampling resolution",
             "value": f"{coverage['resolution']['vsr']:.3f} m "
                      f"({coverage['resolution']['quality']})"},
            {"metric": "representativity",
             "value": f"{coverage['representativity']['representativity_pct']:.0f} %"},
        ]
        sections.append({"title": "Tidal sampling", "table": rows,
                         "html": badge,
                         "note": "How well the satellite archive sampled the "
                                 "tidal frame — the ceiling on what any "
                                 "elevation method can achieve here."})

    if figures:
        for group, figs in (figures.items() if isinstance(figures, dict)
                            else [("Products", figures)]):
            sections.append({"title": group, "figures": list(figs)})

    if zones:
        sections.append({"title": "Ecological zonation", "table": zones,
                         "note": "Zones defined by hydroperiod — the fraction "
                                 "of time each pixel is submerged."})

    if validation:
        sections.append({"title": "Validation", "table": validation,
                         "note": "Bias is dominated by the vertical-datum "
                                 "difference; RMSE and MAE are computed after "
                                 "removing it."})

    if oplog is not None and len(oplog):
        from .explain import draw_graph
        sections.append({"title": "How this was produced",
                         "svg": draw_graph(oplog),
                         "note": "Every operation, in order, with the "
                                 "parameters it ran with."})

    return build_report(out_path, f"Intertidal report · {aoi.name}",
                        subtitle="pyintertidal", sections=sections)
