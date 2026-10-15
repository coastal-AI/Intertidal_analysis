# -*- coding: utf-8 -*-
"""Validation v2 — ``fetch ioc``: the boundary gauges over 2017-2025 (§1.5).

Question
--------
Which IOC sea-level chunks must exist before the boundary provider h_B is
built once over the union period, so that h_B is never changed silently by
a later download?

Pre-registered rules (§1.5)
---------------------------
* Codes: every boundary member (vlis, brsk, ters, bork, fer1) and Gijón
  (gij2, the Villaviciosa sensitivity): :data:`UNION_CODES`.
* The union provider reads its OWN cache, :data:`UNION_CACHE`
  (``data_v4/gauges/ioc_2017-2025``). ``load_cached_ioc`` concatenates EVERY
  chunk of a code in its cache directory and ``make_boundary`` demeans each
  gauge by the median of all of them, so adding 2017-2022 chunks to the
  production cache ``data_v4/gauges`` would change the production h_B (and
  the reproduction gates) silently. The production cache is only read.
* The production chunks of each code are COPIED byte for byte into the
  union cache, so the 2023-2025 samples are exactly production's; every
  calendar year not covered by a copied chunk is downloaded as one yearly
  chunk ``ioc_<code>_<Y>-01-01_<Y>-12-31.json`` with the production
  downloader ``pyintertidal.boundary.fetch_ioc``.
* A year counts as covered only when its data pass the coverage rule, not
  by the file NAME: ``fetch_ioc`` swallows every request error and still
  writes its yearly file (``[]`` after an outage). Every month of a
  downloaded year must hold data on at least :data:`MIN_DAYS_PER_MONTH`
  distinct days (:func:`assess_year`, counted over every chunk that meets
  the year). A year that fails is renamed ``*.incomplete-<utc>`` (never
  read by ``load_cached_ioc``) and downloaded again (two rounds per run);
  a real gauge gap is kept only with ``--accept-gaps`` (recorded). Copied
  production chunks are never renamed (production's h_B read them); their
  gaps are reported.
* When, and only when, every year 2017-2025 is covered, the union MARKER
  ``ioc_<code>_2017-01-01_2025-12-31.json`` is written with the content
  ``[]`` (and removed again when a year is not). ``make_boundary(period=
  ("2017-01-01", "2025-12-31"), cache_dir=UNION_CACHE)`` then finds the
  chunk ``fetch_ioc`` looks for and downloads nothing, and
  ``load_cached_ioc`` adds no sample from the empty marker.
  ``levels.union_provider`` refuses to build h_B without the markers and
  the per-year assessment of the last fetch (:func:`union_ready`).
* The manifest ``ioc_manifest.json`` lists every file with its sha256 and
  size and, per code and year, the samples and the days with data per
  month; the chunk hashes enter the package key (§1.5). The CLI exits 1
  while the cache is not ready.
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import math
import os
import re
import shutil

#: the v2 boundary cache (repository-relative; the CLI chdirs to ROOT)
UNION_CACHE = "data_v4/gauges/ioc_2017-2025"
#: the production cache (read only)
PRODUCTION_CACHE = "data_v4/gauges"
#: boundary members of every site plus Gijón (§1.5)
UNION_CODES = ("vlis", "brsk", "ters", "bork", "fer1", "gij2")
#: calendar years of the union period (settings.UNION_PERIOD, end exclusive)
UNION_YEARS = tuple(range(2017, 2026))
#: make_boundary's inclusive period for the union provider (names the marker)
UNION_FETCH_PERIOD = ("2017-01-01", "2025-12-31")
MANIFEST_NAME = "ioc_manifest.json"
MANIFEST_FORMAT = "validation-v2-ioc-2"
#: the coverage rule of a downloaded yearly chunk: every month holds data on
#: at least this many distinct days (an empty file, a failed request or a
#: proxy outage leaves whole months, or a 30-day request window, empty)
MIN_DAYS_PER_MONTH = 20
_CHUNK_RE = re.compile(r"^ioc_(?P<code>.+)_(?P<t0>\d{4}-\d{2}-\d{2})_(?P<t1>\d{4}-\d{2}-\d{2})\.json$")


def log(msg):
    print(msg, flush=True)


def marker_name(code):
    """The union marker's file name (the chunk ``fetch_ioc`` looks for)."""
    return f"ioc_{code}_{UNION_FETCH_PERIOD[0]}_{UNION_FETCH_PERIOD[1]}.json"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_marker(path):
    """True for the empty union marker (content ``[]``)."""
    if os.path.basename(path) != marker_name(_code_of(path) or ""):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip() == "[]"
    except OSError:
        return False


def _code_of(path):
    m = _CHUNK_RE.match(os.path.basename(path))
    return m.group("code") if m else None


def chunk_span(path):
    """``(code, t0, t1)`` dates from a chunk's file name, or None."""
    m = _CHUNK_RE.match(os.path.basename(path))
    if not m:
        return None
    return (m.group("code"), _dt.date.fromisoformat(m.group("t0")),
            _dt.date.fromisoformat(m.group("t1")))


def chunks(code, cache_dir):
    """Every data chunk of ``code`` in ``cache_dir`` (sorted; markers excluded).

    The glob is ``load_cached_ioc``'s, filtered to this code's name pattern
    (``ioc_fer1_*`` must not pick up a code ``fer1x``).
    """
    out = []
    for p in sorted(glob.glob(os.path.join(cache_dir, f"ioc_{code}_*.json"))):
        span = chunk_span(p)
        if span is None or span[0] != code or is_marker(p):
            continue
        out.append(p)
    return out


def years_covered(code, cache_dir, years=UNION_YEARS):
    """``{year: [chunk paths covering the whole calendar year]}`` (a year
    absent from the result is not covered by any one chunk)."""
    out = {}
    for p in chunks(code, cache_dir):
        _, t0, t1 = chunk_span(p)
        for y in years:
            if t0 <= _dt.date(y, 1, 1) and t1 >= _dt.date(y, 12, 31):
                out.setdefault(y, []).append(p)
    return out


def missing_years(code, cache_dir, years=UNION_YEARS):
    cov = years_covered(code, cache_dir, years)
    return [y for y in years if y not in cov]


def union_ready(codes, cache_dir=UNION_CACHE, years=UNION_YEARS, manifest=None):
    """``(ready, problems)``: every code covers every year, has its marker
    and was ASSESSED by the last ``fetch ioc`` (its manifest lists every
    year of the code as passing the coverage rule, accepted with
    ``--accept-gaps``, or copied from production). A cache filled by an
    older fetch, or a year that failed the rule, is not ready."""
    problems = []
    man = read_manifest(cache_dir) if manifest is None else manifest
    for code in codes:
        miss = missing_years(code, cache_dir, years)
        if miss:
            problems.append(f"{code}: years {miss} not in {cache_dir}")
        mk = os.path.join(cache_dir, marker_name(code))
        if not (os.path.exists(mk) and is_marker(mk)):
            problems.append(f"{code}: no union marker {mk}")
        rec = ((man or {}).get("codes") or {}).get(code)
        if rec is None or "years" not in rec:
            problems.append(f"{code}: not assessed by fetch ioc (run it again)")
            continue
        acc = set(int(y) for y in rec.get("accepted_gaps", []))
        bad = [int(y) for y, e in rec["years"].items()
               if not e.get("ok") and e.get("kind") == "downloaded" and int(y) not in acc]
        if bad:
            problems.append(f"{code}: years {sorted(bad)} fail the coverage rule")
    return (not problems), problems


def file_entry(path):
    return {"path": os.path.abspath(path).replace("\\", "/"), "name": os.path.basename(path),
            "sha256": _sha256(path), "size": int(os.path.getsize(path))}


def yearly_name(code, year):
    """The yearly chunk ``fetch_ioc`` writes for one calendar year."""
    return f"ioc_{code}_{int(year)}-01-01_{int(year)}-12-31.json"


def chunk_counts(path):
    """Per calendar year of one chunk: the valid samples (``slevel`` a finite
    number) and the distinct days with data per month.

    Returns
    -------
    dict
        ``{year: {"samples": int, "days": {"01".."12": int}}}``; an empty or
        unreadable chunk gives ``{}``.
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, list):
        return {}
    samples, days = {}, {}
    for r in raw:
        if not isinstance(r, dict):
            continue
        st = str(r.get("stime", ""))
        try:
            v = float(r.get("slevel"))
        except (TypeError, ValueError):
            continue
        if len(st) < 10 or not math.isfinite(v):
            continue
        y = int(st[:4])
        samples[y] = samples.get(y, 0) + 1
        days.setdefault((y, st[5:7]), set()).add(st[8:10])
    out = {y: {"samples": n, "days": {}} for y, n in samples.items()}
    for (y, m), ds in days.items():
        out[y]["days"][m] = len(ds)
    return out


def overlapping(code, cache_dir, year):
    """Every data chunk of ``code`` whose name span meets the calendar year
    (what ``load_cached_ioc`` reads for it)."""
    out = []
    for p in chunks(code, cache_dir):
        _, t0, t1 = chunk_span(p)
        if t0 <= _dt.date(year, 12, 31) and t1 >= _dt.date(year, 1, 1):
            out.append(p)
    return out


def assess_year(code, cache_dir, year, counts_cache=None, min_days=None):
    """The coverage of one code-year from the chunks that meet it.

    Returns
    -------
    dict
        ``samples``, ``days`` (per month, the largest count over the
        chunks), ``ok`` (every month has data on at least ``min_days``
        days, default :data:`MIN_DAYS_PER_MONTH`), ``thin`` (the months
        below), ``files`` and ``kind``: "downloaded" (the yearly chunk of
        this year exists) or "production" (copied chunks only).
    """
    min_days = MIN_DAYS_PER_MONTH if min_days is None else int(min_days)
    cc = {} if counts_cache is None else counts_cache
    files = overlapping(code, cache_dir, year)
    samples, days = 0, {}
    for p in files:
        if p not in cc:
            cc[p] = chunk_counts(p)
        e = cc[p].get(int(year))
        if not e:
            continue
        samples += e["samples"]
        for m, n in e["days"].items():
            days[m] = max(days.get(m, 0), n)
    thin = [f"{m:02d}" for m in range(1, 13) if days.get(f"{m:02d}", 0) < min_days]
    yearly = os.path.normcase(os.path.abspath(os.path.join(cache_dir, yearly_name(code, year))))
    kind = "downloaded" if any(os.path.normcase(os.path.abspath(p)) == yearly for p in files) \
        else "production"
    return {"samples": int(samples), "days": dict(sorted(days.items())), "ok": not thin,
            "thin": thin, "files": [os.path.basename(p) for p in files], "kind": kind}


def read_manifest(cache_dir=UNION_CACHE):
    """The union cache's manifest (the last ``fetch ioc``), or None."""
    p = os.path.join(cache_dir, MANIFEST_NAME)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fetch_union(codes=UNION_CODES, years=UNION_YEARS, cache_dir=UNION_CACHE,
                source_cache=PRODUCTION_CACHE, fetch=None, write_manifest=True,
                accept_gaps=False, attempts=2, min_days=None):
    """Fill the union cache: copy production chunks, download missing years,
    assess every year, write the markers.

    Parameters
    ----------
    codes : iterable of str
    years : iterable of int
    cache_dir, source_cache : str
        The union cache (written) and the production cache (read only).
    fetch : callable, optional
        ``fetch(code, t0, t1, cache_dir=...)``; default
        ``pyintertidal.boundary.fetch_ioc`` (the production downloader,
        which writes its yearly file even when every request failed).
        Tests inject a fake.
    accept_gaps : bool
        Keep a downloaded year that fails the coverage rule (a real gauge
        gap, after looking at the counts); recorded in the manifest.
    attempts : int
        Download rounds per run: a downloaded year that fails the rule is
        renamed ``*.incomplete-<utc>`` (``load_cached_ioc`` never reads it)
        and downloaded again in the next round.
    min_days : int, optional
        The coverage rule (default :data:`MIN_DAYS_PER_MONTH`).

    Returns
    -------
    dict
        ``{"format", "cache_dir", "codes": {code: {"copied", "downloaded",
        "incomplete", "accepted_gaps", "production_gaps", "years" (per-year
        :func:`assess_year`), "missing", "marker", "files"}}, "ready",
        "problems", "accept_gaps", "min_days_per_month"}``.
    """
    if fetch is None:
        from pyintertidal import net
        from pyintertidal.boundary import fetch_ioc as fetch
        net.use_system_certificates()
    min_days = MIN_DAYS_PER_MONTH if min_days is None else int(min_days)
    years = tuple(int(y) for y in years)
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.abspath(cache_dir) == os.path.abspath(source_cache):
        raise ValueError("the union cache must not be the production cache (§1.5)")
    report = {"format": MANIFEST_FORMAT, "cache_dir": cache_dir.replace("\\", "/"),
              "years": list(years), "codes": {}, "accept_gaps": bool(accept_gaps),
              "min_days_per_month": int(min_days)}
    for code in codes:
        rec = {"copied": [], "downloaded": [], "incomplete": [], "accepted_gaps": [],
               "production_gaps": [], "marker": None}
        for src in chunks(code, source_cache):
            dst = os.path.join(cache_dir, os.path.basename(src))
            if os.path.exists(dst) and _sha256(dst) == _sha256(src):
                continue
            if os.path.exists(dst):
                raise RuntimeError(f"{dst} differs from the production chunk {src}; "
                                   "the union cache must hold production's bytes")
            shutil.copyfile(src, dst)
            rec["copied"].append(os.path.basename(src))
            log(f"[fetch ioc] {code}: copied production chunk {os.path.basename(src)}")
        counts = {}
        assessed = {}
        for rnd in range(max(1, int(attempts))):
            for y in missing_years(code, cache_dir, years):
                t0, t1 = f"{y}-01-01", f"{y}-12-31"
                log(f"[fetch ioc] {code}: downloading {t0}..{t1}"
                    + (f" (attempt {rnd + 1})" if rnd else ""))
                fetch(code, t0, t1, cache_dir=cache_dir)
                rec["downloaded"].append(y)
            retry = False
            for y in years:
                e = assess_year(code, cache_dir, y, counts, min_days)
                assessed[str(y)] = e
                if e["ok"] or missing_years(code, cache_dir, [y]):
                    continue
                if e["kind"] == "production" or os.path.exists(
                        os.path.join(source_cache, yearly_name(code, y))):
                    # a copied production chunk (even one with a yearly name) is
                    # never renamed: the union cache holds production's bytes
                    e["kind"] = "production"
                    if y not in rec["production_gaps"]:
                        rec["production_gaps"].append(y)
                        log(f"[fetch ioc] {code} {y}: the copied production chunks have "
                            f"months with < {min_days} days of data {e['thin']} (kept: "
                            "production's bytes; reported)")
                    continue
                if accept_gaps:
                    if y not in rec["accepted_gaps"]:
                        rec["accepted_gaps"].append(y)
                        log(f"[fetch ioc] {code} {y}: months with < {min_days} days of data "
                            f"{e['thin']} ACCEPTED (--accept-gaps, recorded)")
                    continue
                p = os.path.join(cache_dir, yearly_name(code, y))
                bad = f"{p}.incomplete-{_dt.datetime.now(_dt.timezone.utc):%Y%m%dT%H%M%S%fZ}"
                os.replace(p, bad)
                counts.pop(p, None)
                rec["incomplete"].append({"year": y, "samples": e["samples"],
                                          "thin": e["thin"], "kept_as": os.path.basename(bad)})
                log(f"[fetch ioc] {code} {y}: only {e['samples']} samples, months with < "
                    f"{min_days} days of data {e['thin']}: kept as {os.path.basename(bad)}"
                    + ("; downloading it again" if rnd + 1 < max(1, int(attempts)) else ""))
                retry = True
            if not retry:
                break
        rec["years"] = assessed
        rec["missing"] = missing_years(code, cache_dir, years)
        mk = os.path.join(cache_dir, marker_name(code))
        if not rec["missing"] and tuple(years) == UNION_YEARS:
            if not os.path.exists(mk):
                with open(mk, "w", encoding="utf-8", newline="\n") as f:
                    f.write("[]")
                log(f"[fetch ioc] {code}: every year 2017-2025 cached and assessed; marker "
                    "written")
            rec["marker"] = os.path.basename(mk)
        else:
            if rec["missing"]:
                log(f"[fetch ioc] {code}: years {rec['missing']} missing or incomplete (no "
                    "marker): rerun 'fetch ioc' later, or, after checking the gaps, pass "
                    "--accept-gaps")
            if os.path.exists(mk) and is_marker(mk):
                os.remove(mk)
                log(f"[fetch ioc] {code}: the stale union marker was removed")
        rec["files"] = [file_entry(p) for p in chunks(code, cache_dir)]
        report["codes"][code] = rec
    if tuple(years) == UNION_YEARS:
        ready, problems = union_ready(codes, cache_dir, years, manifest=report)
    else:
        ready, problems = False, ["not the union years"]
    report["ready"], report["problems"] = ready, problems
    if write_manifest:
        path = os.path.join(cache_dir, MANIFEST_NAME)
        tmp = f"{path}.tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
    return report
