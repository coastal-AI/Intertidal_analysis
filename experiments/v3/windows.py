# -*- coding: utf-8 -*-
"""The windows of validation v3 and the class-balance rule of its scored pixels, as the design of 2026-10-29 fixes them
(experiments/validation_v3_cantabrian_2026-10-29.md, sections "Windows" and "The pixels scored").

Windows
-------
For a truth date d (the LiDAR flight; provisionally 15 July of the coverage year) and each n of NS, the window is the
n screened scenes nearest in time to d. The scenes are sorted by |t - d|, a tie by the earlier t (at the same distance
a scene before d comes before one after it; scenes of the same t keep their order in `dates`), and the first n are
taken. The largest of their distances is the half-width T, and [d - T, d + T] is the smallest window symmetric in time
about d that holds n scenes. It is symmetric in time, not in scenes: a cloudy spring before a summer flight leaves
fewer scenes before d than after it, and n_before and n_after report how many. A scene dated d itself is neither
before nor after (n - n_before - n_after of them). When the n-th and the (n+1)-th nearest scenes are both at distance
T, one at d - T and one at d + T, the earlier is taken: [d - T, d + T] then holds n + 1 screened scenes, n of them in
the window.

A window is reachable only when the cube covers [d - T, d + T] entirely, both ends included: with cover = (first,
last), the cube's first and last date, first <= d - T and d + T <= last. Outside the cube there could be scenes nearer
to d than the farthest one taken; the n nearest scenes of the cube would then not be the n nearest of the archive, and
the window would lean towards the side the cube covers. A window that is not reachable is reported (its half-width and
scenes) and not computed. With fewer than n screened scenes no window holds n of them: it is not reachable, its
half-width is None and idx holds every scene.

`dates`, `truth_date` and `cover` are ISO strings ('2020-07-15', or instants such as '2020-07-15T11:02:31') or numpy
datetime64 (arrays or scalars, or anything numpy turns into datetime64: datetime.date, pandas.Timestamp,
DatetimeIndex). They are compared exactly, at the finest resolution among them, and a day counts as its first instant
(00:00). Give them all as days (the screened list's dates, the truth date) or all as instants. A year or a month
('2020-07') is refused rather than read as its first day, and so is anything that is not a date (a number, None,
NaT). The screened scenes are the cube's, so every one must lie in `cover`; a scene outside it is an error.

The pixels scored
-----------------
Y and C are the window's looks: rows the window's scenes, columns the pixels (any set: the n-tables' pixels F, or every
extraction pixel). Y is the NDWI and C the clear looks (SCL 4-7; bool, or 0/1). A look is wet when it is clear and its
NDWI is > 0. A clear look whose NDWI is undefined is dry, as MAREA's inversion reads it (wet = C & (nan_to_num(Y) > 0)),
so Y may hold NaN or 0 there; what a cloudy look holds does not matter. The wet share is f = wet looks / clear looks,
and a pixel is kept when q <= f <= 1 - q, i.e. when neither class dominates its record in the window (the design's
q = 0.10: 0.10 <= f <= 0.90). A pixel without a clear look has no f and is never kept; q = 0 keeps every pixel with a
clear look, and the pixels kept at a larger q are a subset of those kept at a smaller one. The mask reads only the
looks, never a product or the truth, so it is the same for every method; the pixels scored are those of F that it
keeps.

The bounds are evaluated on the counts, as wet / clear >= q and (clear - wet) / clear >= q, each a quotient of two
whole numbers rounded once, so a share equal to q or to 1 - q is kept at every q: 1 of 10 and 9 of 10 at q = 0.10.
Comparing f with a computed 1 - q would not always keep it: at q = 0.07, the rounded 1 - 0.07 is less than the rounded
93/100.

The looks are counted in blocks of `block_px` pixels (BLOCK_PX by default), so that only one block of the window's
looks is in memory at a time. Y and C may be memory-mapped or lazy arrays that read [rows, columns] slices, and `rows`
(the window's idx) reads the window's scenes from the screened record in place, without copying them first. The result
does not depend on the blocking.

    symmetric_window        the window of n scenes about the truth date: idx, half_width_days, n_before, n_after,
                            reachable
    look_counts             the wet and clear looks of every pixel in the window
    neither_dominates       the class-balance rule on those counts
    class_mask              the pixels kept at one q (the design's 0.10)
    class_mask_sensitivity  the pixels kept at each q of the sensitivity (0, 0.05, 0.10, 0.15, 0.20), counted once

In a notebook (S_dates: the screened scenes' dates; Y_S, C_S: their looks, screened scenes x pixels; cube_dates: every
date of the cube):

    for n in NS:
        w = symmetric_window(S_dates, "2020-07-15", n, cover=(cube_dates[0], cube_dates[-1]))
        if not w["reachable"]:
            continue                                                    # reported, not computed
        keep = class_mask(Y_S, C_S, rows=w["idx"])                      # q = 0.10
        by_q = class_mask_sensitivity(Y_S, C_S, rows=w["idx"])          # {0.0: mask, 0.05: mask, ...}
"""
from __future__ import annotations

import numpy as np

#: the window sizes: screened scenes per window
NS = (20, 40, 60, 80, 100, 120)
#: the class-balance bound: a pixel is scored when Q_CLASS <= f <= 1 - Q_CLASS
Q_CLASS = 0.10
#: the bounds of the reported sensitivity (Q_CLASS among them)
Q_SENSITIVITY = (0.0, 0.05, 0.10, 0.15, 0.20)
#: pixels per block of the look counts (bool and NDWI temporaries of looks x BLOCK_PX)
BLOCK_PX = 50_000


# ─────────────────────────────────────────────────────────────────────────────
#  the window
# ─────────────────────────────────────────────────────────────────────────────

def _instants(x, what):
    """`x` as numpy datetime64, ISO strings parsed: a day or finer, no NaT."""
    a = np.asarray(x)
    if a.size == 0:
        return np.zeros(a.shape, "datetime64[D]")           # an empty record: np.asarray([]) is float64
    if a.dtype.kind != "M":
        if a.dtype.kind not in "USO":                       # strings, bytes, objects (datetime.date, Timestamp)
            raise ValueError(f"{what}: ISO dates or numpy datetime64 expected, got {a.dtype}")
        try:
            a = a.astype("datetime64")
        except (TypeError, ValueError) as e:
            raise ValueError(f"{what}: not ISO dates or numpy datetime64 ({e})") from None
    unit = np.datetime_data(a.dtype)[0]
    if unit == "generic" or np.isnat(a).any():              # numbers and None become generic datetime64 or NaT
        raise ValueError(f"{what}: a missing or unreadable date (NaT)")
    if unit in ("Y", "M"):
        raise ValueError(f"{what}: {a.dtype} holds years or months, not dates")
    return a


def _cover(cover):
    """(first, last) of `cover` as datetime64 scalars, first <= last."""
    try:
        first, last = cover
    except (TypeError, ValueError):
        raise ValueError(f"cover must be (first, last), the cube's first and last date, got {cover!r}") from None
    first, last = _instants(first, "cover"), _instants(last, "cover")
    if first.ndim or last.ndim:
        raise ValueError("cover must be two dates, the cube's first and last")
    first, last = first[()], last[()]
    if last < first:
        raise ValueError(f"cover ends ({last}) before it starts ({first})")
    return first, last


def _scenes(n):
    """n as a whole number of scenes >= 1."""
    try:
        k = int(n)
    except (TypeError, ValueError):
        raise ValueError(f"n must be a whole number of scenes >= 1, got {n!r}") from None
    if isinstance(n, (bool, np.bool_)) or k != n or k < 1:
        raise ValueError(f"n must be a whole number of scenes >= 1, got {n!r}")
    return k


def symmetric_window(dates, truth_date, n, cover):
    """The window of the n screened scenes nearest in time to `truth_date` (the rule: the module docstring).

    dates       the screened scenes' dates, 1-D, in any order; they are the cube's scenes, so they lie in `cover`
    truth_date  d, one date
    n           the scenes of the window, >= 1
    cover       (first, last), the cube's first and last date

    Returns dict(idx, half_width_days, n_before, n_after, reachable): idx (int64) the positions in `dates` of the
    window's scenes, ascending (chronological when `dates` is); half_width_days the half-width T in days (a float,
    whole when the dates are days); n_before and n_after the scenes of idx strictly before and strictly after d;
    reachable True when the cube covers [d - T, d + T]. With fewer than n scenes: idx every scene, half_width_days None,
    reachable False."""
    t = _instants(dates, "dates")
    if t.ndim != 1:
        raise ValueError(f"dates must be one-dimensional, got shape {t.shape}")
    d = _instants(truth_date, "truth_date")
    if d.ndim:
        raise ValueError(f"truth_date must be one date, got shape {d.shape}")
    d = d[()]
    first, last = _cover(cover)
    n = _scenes(n)
    if t.size and (t.min() < first or t.max() > last):
        raise ValueError(f"the screened scenes ({t.min()} .. {t.max()}) are the cube's, so they lie in cover "
                         f"({first} .. {last})")
    before, after = t < d, t > d
    if n > t.size:
        return dict(idx=np.arange(t.size, dtype=np.int64), half_width_days=None, n_before=int(before.sum()),
                    n_after=int(after.sum()), reachable=False)
    delta = t - d                                           # exact, at the finer of the two resolutions
    dist = np.abs(delta)
    order = np.lexsort((np.arange(t.size), delta, dist))    # |t - d|, then the earlier t, then the order in dates
    idx = np.sort(order[:n]).astype(np.int64)
    half = dist[order[n - 1]]
    return dict(idx=idx, half_width_days=float(half / np.timedelta64(1, "D")), n_before=int(before[idx].sum()),
                n_after=int(after[idx].sum()), reachable=bool(first <= d - half and d + half <= last))


# ─────────────────────────────────────────────────────────────────────────────
#  the class-balance mask
# ─────────────────────────────────────────────────────────────────────────────

def _bound(q):
    """q as a float in [0, 0.5] (above 0.5 no share satisfies q <= f <= 1 - q)."""
    q = float(q)
    if not 0.0 <= q <= 0.5:                                 # NaN fails too
        raise ValueError(f"q must lie in [0, 0.5] (kept: q <= f <= 1 - q), got {q!r}")
    return q


def _rows(rows, n_looks):
    """The row selector of a block read: every row (None) or the window's scenes (1-D positions, unique, in range)."""
    if rows is None:
        return slice(None)
    r = np.asarray(rows)
    if r.ndim != 1 or (r.size and r.dtype.kind not in "iu"):
        raise ValueError("rows must be the window's scene positions (symmetric_window's idx): 1-D integers")
    r = r.astype(np.int64)
    if r.size and (r.min() < 0 or r.max() >= n_looks):
        raise ValueError(f"rows must lie in 0 .. {n_looks - 1}, the rows of Y and C")
    if np.unique(r).size != r.size:
        raise ValueError("rows must not repeat a scene: its looks would count twice")
    return r


def look_counts(Y, C, *, rows=None, block_px=BLOCK_PX):
    """(wet, clear), int64 (pixels,): per pixel the clear looks of the window and the wet ones among them (clear and
    NDWI > 0; an undefined NDWI is dry). Y (looks x pixels) the NDWI, NaN or 0 where undefined; C (looks x pixels) the
    clear looks, bool or 0/1; rows the window's scenes among the rows of Y and C (None: every row). Read in blocks of
    `block_px` columns, Y[rows, columns] and C[rows, columns], never whole."""
    shape = tuple(np.shape(Y))                              # np.shape reads .shape: a lazy Y is not loaded
    if len(shape) != 2 or tuple(np.shape(C)) != shape:
        raise ValueError(f"Y and C must be looks x pixels of the same shape, got {shape} and {np.shape(C)}")
    block = int(block_px)
    if block < 1:
        raise ValueError(f"block_px must be >= 1, got {block_px!r}")
    r = _rows(rows, shape[0])
    P = shape[1]
    wet, clear = np.zeros(P, np.int64), np.zeros(P, np.int64)
    with np.errstate(invalid="ignore"):                     # NaN > 0 is False: an undefined NDWI is dry
        for b0 in range(0, P, block):
            cols = slice(b0, min(b0 + block, P))
            c = np.asarray(C[r, cols], bool)
            clear[cols] = np.count_nonzero(c, axis=0)
            wet[cols] = np.count_nonzero(c & (np.asarray(Y[r, cols]) > 0), axis=0)
    return wet, clear


def neither_dominates(wet, clear, q=Q_CLASS):
    """The class-balance rule on the counts of `look_counts`, bool (pixels,): q <= wet / clear <= 1 - q, evaluated as
    wet / clear >= q and (clear - wet) / clear >= q (the module docstring says why); False where clear = 0."""
    q = _bound(q)
    wet, clear = np.asarray(wet, np.int64), np.asarray(clear, np.int64)
    if wet.shape != clear.shape:
        raise ValueError(f"wet and clear must have the same shape, got {wet.shape} and {clear.shape}")
    if np.any(wet < 0) or np.any(wet > clear):
        raise ValueError("the counts must satisfy 0 <= wet <= clear")
    has = clear > 0
    den = np.where(has, clear, 1)
    return has & (wet / den >= q) & ((clear - wet) / den >= q)


def class_mask(Y, C, q=Q_CLASS, *, rows=None, block_px=BLOCK_PX):
    """The pixels whose record in the window holds both classes, bool (pixels,): kept when q <= f <= 1 - q, with f the
    wet share of the clear looks (the design's q = 0.10); a pixel without a clear look is not kept. Y, C, rows and
    block_px as in `look_counts`."""
    q = _bound(q)                                           # before anything is read
    wet, clear = look_counts(Y, C, rows=rows, block_px=block_px)
    return neither_dominates(wet, clear, q)


def class_mask_sensitivity(Y, C, qs=Q_SENSITIVITY, *, rows=None, block_px=BLOCK_PX):
    """{q: class_mask at q} for every q of `qs` (keys as floats, in the order given), from one count of the looks. Y,
    C, rows and block_px as in `look_counts`."""
    qs = [_bound(q) for q in qs]
    wet, clear = look_counts(Y, C, rows=rows, block_px=block_px)
    return {q: neither_dominates(wet, clear, q) for q in qs}
