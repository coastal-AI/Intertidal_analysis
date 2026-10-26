# -*- coding: utf-8 -*-
"""Validation v2 — ``fetch pnoa-meta``: when the Villaviciosa LiDAR was flown (§1.8) [V18].

Question
--------
On which day, and at what time, did the PNOA LiDAR flight that produced
the MDT02 sheet over the Villaviciosa ría (MDT02-WGS84-0015-3-COB2) cross
the ría, and what vertical accuracy does CNIG state for it, so that the
water level at flight time can mask the truth that is really water?

Pre-registered rules (§1.8)
---------------------------
* Sources, in order: (1) CNIG ISO 19139 metadata XML files of the product
  (downloaded by the user from the CNIG download centre; the GeoTIFF itself
  carries no date); (2) a manual flight table transcribed by the user from
  the CNIG flight information (``strip, start_utc, end_utc`` with an
  explicit UTC offset or 'Z'). Every source file is recorded with its
  sha256.
* The temporal extent (``gml:beginPosition`` / ``gml:endPosition``, or
  ``gml:timePosition``) and the dates of the XML give the flight DATES; a
  time of day is used only when a source gives one with a time zone. A
  time without a zone is reported and NOT used.
* ``flight_known`` is True only with at least one zoned flight instant.
  Without it, the truth applies only the constant-patch mask and the
  Villaviciosa elevation stays descriptive (§1.8 item 6).
* The vertical accuracy (m) is read from
  ``DQ_AbsoluteExternalPositionalAccuracy`` when its measure names the
  vertical (Z, altimetric, vertical) component; it feeds the PNOA noise of
  the truth-error model.

Output: ``results/validation_v2/common/pnoa_meta.json``.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

import pandas as pd

from experiments.benchmark import config, product

OUT_PATH = "results/validation_v2/common/pnoa_meta.json"
META_FORMAT = "validation-v2-pnoa-meta-1"
SHEET = "MDT02-WGS84-0015-3-COB2"
_VERTICAL = re.compile(r"(?i)(vertical|altim|\bz\b|cota|height|altura)")
_ZONED = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


def log(msg):
    print(msg, flush=True)


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _texts(el, name):
    return [(c.text or "").strip() for c in el.iter() if _local(c.tag) == name and c.text]


def parse_iso19139(text):
    """Dates and the vertical accuracy of an ISO 19139 metadata document.

    Returns
    -------
    dict
        title, begin, end (temporal extent, as written), time_positions,
        dates (every gco:Date / gco:DateTime), vertical_rmse_m (float or
        None) with its measure name.
    """
    root = ET.fromstring(text)
    out = {"title": None, "begin": None, "end": None, "time_positions": [], "dates": [],
           "vertical_rmse_m": None, "vertical_measure": None}
    titles = [t for c in root.iter() if _local(c.tag) == "title"
              for t in _texts(c, "CharacterString")]
    out["title"] = titles[0] if titles else None
    for c in root.iter():
        name = _local(c.tag)
        if name == "beginPosition" and c.text:
            out["begin"] = c.text.strip()
        elif name == "endPosition" and c.text:
            out["end"] = c.text.strip()
        elif name == "timePosition" and c.text:
            out["time_positions"].append(c.text.strip())
        elif name in ("Date", "DateTime") and c.text:
            out["dates"].append(c.text.strip())
    for acc in root.iter():
        if _local(acc.tag) != "DQ_AbsoluteExternalPositionalAccuracy":
            continue
        names = " ".join(_texts(acc, "CharacterString"))
        vals = _texts(acc, "Decimal") + _texts(acc, "Real")
        if vals and _VERTICAL.search(names):
            try:
                out["vertical_rmse_m"] = float(vals[0])
                out["vertical_measure"] = names[:200]
                break
            except ValueError:
                continue
    return out


def parse_instant(s):
    """``(naive-UTC Timestamp or None, status)`` of a written instant: a
    date only -> (None, "date only"); a time without a zone -> (None, "no
    time zone"); a zoned time -> (UTC, "zoned")."""
    s = str(s).strip()
    if not s:
        return None, "empty"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return None, "date only"
    if not _ZONED.search(s):
        return None, "no time zone"
    t = pd.Timestamp(s)
    if t.tzinfo is None:
        return None, "no time zone"
    return t.tz_convert("UTC").tz_localize(None), "zoned"


def read_flight_table(path):
    """A user-transcribed flight table (CSV with ``strip, start_utc,
    end_utc``) as a list of dicts."""
    df = pd.read_csv(path, dtype=str)
    need = {"strip", "start_utc", "end_utc"}
    if not need <= set(df.columns):
        raise ValueError(f"{path}: expected the columns {sorted(need)}, got {list(df.columns)}")
    return [{"strip": r["strip"], "start": r["start_utc"], "end": r["end_utc"]}
            for _, r in df.iterrows()]


def flights_from(xml_docs, table):
    """The flights from the parsed XML documents and the manual table."""
    flights = []
    for src, d in xml_docs:
        if d.get("begin") or d.get("end"):
            flights.append({"strip": None, "start": d.get("begin"), "end": d.get("end"),
                            "source": src})
        for tp in d.get("time_positions", []):
            flights.append({"strip": None, "start": tp, "end": tp, "source": src})
    for row in table or []:
        flights.append(dict(row, source="manual table"))
    for f in flights:
        a, sa = parse_instant(f.get("start") or "")
        b, sb = parse_instant(f.get("end") or "")
        f["start_utc"] = None if a is None else str(a)
        f["end_utc"] = None if b is None else str(b)
        f["status"] = sa if sa == sb else f"{sa} / {sb}"
        f["usable"] = a is not None and b is not None
    return flights


def flight_instants(doc):
    """Mid-strip naive-UTC instants of the usable flights of a pnoa_meta
    document (empty when the flight time is unknown)."""
    out = []
    for f in doc.get("flights", []):
        if f.get("usable"):
            a, b = pd.Timestamp(f["start_utc"]), pd.Timestamp(f["end_utc"])
            out.append(a + (b - a) / 2)
    return out


def build(xml_paths=(), table_path=None, out=OUT_PATH, sheet=SHEET):
    """``fetch pnoa-meta``: parse the CNIG metadata and the manual table.

    Returns
    -------
    dict
        The written document: sheet, flights, flight_known, dates,
        vertical_rmse_m, sources.
    """
    docs, sources = [], []
    for p in xml_paths:
        with open(p, encoding="utf-8") as f:
            docs.append((str(p).replace("\\", "/"), parse_iso19139(f.read())))
        sources.append({"path": str(p).replace("\\", "/"), "sha256": product.sha256_file(p)})
    table = None
    if table_path:
        table = read_flight_table(table_path)
        sources.append({"path": str(table_path).replace("\\", "/"),
                        "sha256": product.sha256_file(table_path)})
    flights = flights_from(docs, table)
    vert = next((d["vertical_rmse_m"] for _, d in docs if d.get("vertical_rmse_m") is not None),
                None)
    dates = sorted({str(x)[:10] for _, d in docs for x in
                    [d.get("begin"), d.get("end"), *d.get("time_positions", [])] if x})
    doc = {"format": META_FORMAT, "sheet": sheet, "flights": flights,
           "flight_known": any(f["usable"] for f in flights), "flight_dates": dates,
           "vertical_rmse_m": vert, "sources": sources,
           "rule": ("zoned instants only; without one the PNOA truth uses the constant-patch "
                    "mask only and Villaviciosa elevation is descriptive")}
    doc["sha256"] = config.sha256_json({k: v for k, v in doc.items() if k != "sha256"})
    if out:
        product.write_json(out, doc)
    log(f"[fetch pnoa-meta] {sheet}: {len(flights)} flight records, flight time "
        f"{'KNOWN' if doc['flight_known'] else 'unknown'}, dates {dates}, vertical RMSE "
        f"{vert} m" + (f" -> {out}" if out else ""))
    return doc


def load(path=OUT_PATH):
    """The pnoa_meta document (None when ``fetch pnoa-meta`` has not run)."""
    if not os.path.exists(path):
        return None
    doc = product.read_json(path)
    if doc.get("format") != META_FORMAT:
        raise ValueError(f"{path}: not a {META_FORMAT} file")
    return doc
