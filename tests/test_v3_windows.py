# -*- coding: utf-8 -*-
"""Validation v3's windows and class-balance mask (experiments/v3/windows.py, the design of 2026-10-29).

The symmetric window on hand cases: the nearest scenes and the half-width, a tie at the edge and scenes of the same
date, asymmetric clouds, a scene on the truth date, a cube that ends before d + T or starts after d - T, fewer scenes
than n. It is also checked against a window widened day by day with plain loops on random records, for nesting over
NS, and on every date form. The mask is checked on hand records (shares on both bounds, an undefined NDWI and an NDWI
of 0 counted dry, cloudy looks ignored, no clear look) and against plain loops with exact fractions on a synthetic
tidal flat. Further cases: q = 0 keeps every pixel with a clear look, the rule at every share of up to 100 looks and
every q in steps of 0.01, the sensitivity's nested masks, and the reading (block sizes, a lazy array read block by
block, the window's rows read in place). The constants are checked against the design.

Run:  python -X utf8 -m pytest -q tests/test_v3_windows.py
"""
import datetime
import inspect
import os
import sys
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.v3 import windows as win  # noqa: E402

DESIGN = os.path.join(ROOT, "experiments", "validation_v3_cantabrian_2026-10-29.md")
D = "2020-07-15"                                    # the provisional truth date of the second coverage
KEYS = {"idx", "half_width_days", "n_before", "n_after", "reachable"}


def day(k, d=D):
    """The ISO date k days after d (before it when k < 0)."""
    return str(np.datetime64(d) + np.timedelta64(int(k), "D"))


def days(*ks):
    return np.array([day(k) for k in ks])


WIDE = (day(-3000), day(3000))                      # a cube that covers every window of the hand cases


def window(dates, n, cover=WIDE, d=D):
    """symmetric_window, with the types of its result checked."""
    w = win.symmetric_window(dates, d, n, cover=cover)
    assert set(w) == KEYS
    assert isinstance(w["idx"], np.ndarray) and w["idx"].dtype == np.int64
    assert type(w["n_before"]) is int and type(w["n_after"]) is int and type(w["reachable"]) is bool
    assert w["half_width_days"] is None or type(w["half_width_days"]) is float
    return w


# ─────────────────────────────────────────────────────────────────────────────
#  the window: hand cases
# ─────────────────────────────────────────────────────────────────────────────

def test_the_nearest_scenes_and_the_half_width():
    dates = days(-10, -3, 1, 4, 20)
    expect = {1: ([2], 1, 0, 1), 2: ([1, 2], 3, 1, 1), 3: ([1, 2, 3], 4, 1, 2), 4: ([0, 1, 2, 3], 10, 2, 2),
              5: ([0, 1, 2, 3, 4], 20, 2, 3)}
    for n, (idx, half, nb, na) in expect.items():
        w = window(dates, n)
        assert w["idx"].tolist() == idx and w["half_width_days"] == half, n
        assert (w["n_before"], w["n_after"], w["reachable"]) == (nb, na, True), n


def test_a_tie_at_the_edge_takes_the_earlier_scene():
    dates = days(5, -2, -5, 9)                      # d + 5 and d - 5 at the same distance
    w = window(dates, 2)
    assert w["idx"].tolist() == [1, 2] and w["half_width_days"] == 5.0      # d - 2, then d - 5 before d + 5
    assert (w["n_before"], w["n_after"]) == (2, 0)
    t = dates.astype("datetime64[D]")
    assert np.count_nonzero(np.abs(t - np.datetime64(D)) <= np.timedelta64(5, "D")) == 3
    # [d - T, d + T] holds n + 1 = 3 scenes; the window takes n of them, the earlier at the edge
    w = window(dates, 3)
    assert w["idx"].tolist() == [0, 1, 2] and w["half_width_days"] == 5.0
    assert (w["n_before"], w["n_after"]) == (2, 1)


def test_scenes_of_the_same_date_keep_their_order_in_dates():
    dates = np.array(["2020-07-20", "2020-07-10", "2020-07-10", "2020-07-12"])    # 5 after, 5 before twice, 3 before
    assert [window(dates, n)["idx"].tolist() for n in (1, 2, 3, 4)] == [[3], [1, 3], [1, 2, 3], [0, 1, 2, 3]]
    assert [window(dates, n)["half_width_days"] for n in (1, 2, 3, 4)] == [3.0, 5.0, 5.0, 5.0]
    # the second scene of 2020-07-10 still comes before 2020-07-20, the later date at the same distance
    assert [window(dates, n)["n_after"] for n in (1, 2, 3, 4)] == [0, 0, 0, 1]


def test_asymmetric_clouds_give_a_window_symmetric_in_time_not_in_scenes():
    # a cloudy spring: two screened scenes in the four months before a summer flight, a clear summer after it
    dates = days(-118, -40, 2, 5, 9, 12, 16, 23, 30, 37, 44, 51)
    w = window(dates, 6)
    assert w["half_width_days"] == 23.0 and (w["n_before"], w["n_after"]) == (0, 6)
    w = window(dates, 9)
    assert w["idx"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert w["half_width_days"] == 40.0 and (w["n_before"], w["n_after"]) == (1, 8)
    w = window(dates, 12)
    assert w["half_width_days"] == 118.0 and (w["n_before"], w["n_after"]) == (2, 10)


def test_a_scene_on_the_truth_date_is_neither_before_nor_after():
    dates = days(-4, 0, 3)
    w = window(dates, 1)
    assert w["idx"].tolist() == [1] and w["half_width_days"] == 0.0 and w["reachable"]
    assert (w["n_before"], w["n_after"]) == (0, 0)
    w = window(dates, 3)
    assert (w["n_before"], w["n_after"]) == (1, 1)              # n - n_before - n_after = 1 scene on d
    assert window(days(0), 1, cover=(D, D))["reachable"]        # a one-day cube covers [d, d]


def test_a_cube_that_ends_before_d_plus_t_is_not_reachable():
    dates = days(-30, -12, -7, 3, 8)                # the cube ends on d + 10; its last screened scene is d + 8
    cover = (day(-60), day(10))
    w = window(dates, 3, cover)                     # d + 3, d - 7, d + 8: T = 8, and d + 8 is inside the cube
    assert w["idx"].tolist() == [2, 3, 4] and w["half_width_days"] == 8.0 and w["reachable"]
    w = window(dates, 4, cover)                     # ... and d - 12: T = 12, and d + 12 is past the cube's end
    assert w["idx"].tolist() == [1, 2, 3, 4] and w["half_width_days"] == 12.0
    assert (w["n_before"], w["n_after"], w["reachable"]) == (2, 2, False)
    # a scene at d + 11 would be nearer than d - 12, and the cube cannot say whether there is one
    assert window(dates, 4, (day(-60), day(12)))["reachable"]   # both ends of the window included
    assert not window(dates, 4, (day(-60), day(11)))["reachable"]


def test_a_cube_that_starts_after_d_minus_t_is_not_reachable_even_with_every_scene_after_d():
    dates = days(2, 5, 9)
    assert window(dates, 3, (day(-9), day(30)))["reachable"]
    w = window(dates, 3, (day(-8), day(30)))        # no scene before d, yet d - 9 is outside the cube
    assert (w["n_before"], w["n_after"], w["half_width_days"], w["reachable"]) == (0, 3, 9.0, False)


def test_fewer_scenes_than_n_is_not_reachable():
    w = window(days(-3, 2, 6), 4)
    assert w["idx"].tolist() == [0, 1, 2] and w["half_width_days"] is None
    assert (w["n_before"], w["n_after"], w["reachable"]) == (1, 2, False)
    for empty in (np.array([], dtype=str), [], np.array([], dtype="datetime64[D]")):
        w = window(empty, 1)
        assert w["idx"].tolist() == [] and w["half_width_days"] is None and not w["reachable"]
        assert (w["n_before"], w["n_after"]) == (0, 0)


# ─────────────────────────────────────────────────────────────────────────────
#  the window: the plain-loop reference, nesting, date forms, errors
# ─────────────────────────────────────────────────────────────────────────────

def widened_window(offsets, n):
    """The window by plain loops, from the design's wording: widen [d - T, d + T] one day at a time from T = 0 until
    it holds n scenes; take every scene nearer than T and, of those at distance T, the earlier first (on the same date,
    the first in the list) until there are n. offsets: the scenes' days from d. None with fewer than n scenes."""
    if n > len(offsets):
        return None
    T = 0
    while sum(1 for k in offsets if abs(k) <= T) < n:
        T += 1
    nearer = [i for i, k in enumerate(offsets) if abs(k) < T]
    edge = sorted((k, i) for i, k in enumerate(offsets) if abs(k) == T)     # d - T before d + T, then the list order
    return sorted(nearer + [i for _, i in edge[:n - len(nearer)]]), T


def test_the_window_is_the_smallest_symmetric_window_holding_n_scenes():
    rng = np.random.default_rng(20261029)
    for _ in range(300):
        N = int(rng.integers(1, 40))
        offsets = [int(k) for k in rng.integers(-60, 61, N)]           # ties and repeated dates included
        lo = min(offsets) - int(rng.integers(0, 40))
        hi = max(offsets) + int(rng.integers(0, 40))
        dates = days(*offsets)
        for n in sorted({1, int(rng.integers(1, N + 1)), N, N + 1}):
            w = window(dates, n, (day(lo), day(hi)))
            ref = widened_window(offsets, n)
            if ref is None:
                assert w["idx"].tolist() == list(range(N)) and w["half_width_days"] is None and not w["reachable"]
                continue
            idx, T = ref
            assert w["idx"].tolist() == idx and w["half_width_days"] == T, (offsets, n)
            taken = [offsets[i] for i in idx]
            assert (w["n_before"], w["n_after"]) == (sum(k < 0 for k in taken), sum(k > 0 for k in taken))
            assert w["reachable"] == (lo <= -T and T <= hi), (offsets, n, lo, hi)


def test_the_windows_of_ns_are_nested():
    rng = np.random.default_rng(7)
    dates = days(*np.sort(rng.choice(np.arange(-900, 900), 400, replace=False)))
    prev, prev_half = set(), 0.0
    for n in win.NS:
        w = window(dates, n)
        assert len(w["idx"]) == n and prev <= set(w["idx"].tolist()) and w["half_width_days"] >= prev_half
        prev, prev_half = set(w["idx"].tolist()), w["half_width_days"]


def test_the_order_of_dates_does_not_change_the_scenes_taken():
    rng = np.random.default_rng(11)
    dates = days(*rng.choice(np.arange(-300, 300), 150, replace=False))   # distinct dates, in no order
    perm = rng.permutation(150)
    for n in win.NS:
        a, b = window(dates, n), window(dates[perm], n)
        assert set(perm[b["idx"]].tolist()) == set(a["idx"].tolist())
        assert (b["half_width_days"], b["n_before"], b["n_after"]) == (a["half_width_days"], a["n_before"],
                                                                       a["n_after"])


def test_every_date_form_gives_the_same_window():
    s = days(-41, -17, -16, -3, 2, 2, 9, 30, 31, 77)
    c0, c1 = day(-50), day(90)
    ref = window(s, 6, (c0, c1))                    # d + 2 twice, d - 3, d + 9, d - 16, d - 17
    assert ref["idx"].tolist() == [1, 2, 3, 4, 5, 6] and ref["half_width_days"] == 17.0
    assert (ref["n_before"], ref["n_after"], ref["reachable"]) == (3, 3, True)

    def iso(x):
        return datetime.date.fromisoformat(x)

    forms = [(s.astype("datetime64[D]"), np.datetime64(D), (np.datetime64(c0), np.datetime64(c1))),
             (list(s), D, [c0, c1]),
             ([iso(x) for x in s], iso(D), (iso(c0), iso(c1))),
             (pd.DatetimeIndex(s), pd.Timestamp(D), (pd.Timestamp(c0), pd.Timestamp(c1))),
             (s.astype("datetime64[s]"), np.datetime64(D, "s"), np.array([c0, c1], "datetime64[D]"))]
    for dates, d, cover in forms:
        w = win.symmetric_window(dates, d, 6, cover=cover)
        assert w["idx"].tolist() == ref["idx"].tolist() and w["half_width_days"] == ref["half_width_days"]
        assert (w["n_before"], w["n_after"], w["reachable"]) == (ref["n_before"], ref["n_after"], ref["reachable"])
    # every scene: T = 77, and d - 77 is before the cube's first date d - 50; a cube from d - 77 covers the window
    assert not win.symmetric_window(pd.DatetimeIndex(s), pd.Timestamp(D), 10, cover=(c0, c1))["reachable"]
    assert win.symmetric_window(pd.DatetimeIndex(s), pd.Timestamp(D), 10, cover=(day(-77), c1))["reachable"]


def test_instants_are_compared_at_their_resolution():
    dates = ["2020-07-14T11:00", "2020-07-15T11:00", "2020-07-16T10:00"]       # 13 h before, 11 h and 34 h after
    cover = ("2020-01-01T00:00", "2020-12-31T23:59")
    w = win.symmetric_window(dates, D, 1, cover=cover)
    assert w["idx"].tolist() == [1] and w["half_width_days"] == 11 / 24 and (w["n_before"], w["n_after"]) == (0, 1)
    w = win.symmetric_window(dates, D, 2, cover=cover)
    assert w["idx"].tolist() == [0, 1] and w["half_width_days"] == 13 / 24
    # a day counts as its first instant: from 18:00 on d, the day d + 1 (6 h) is nearer than the day d (18 h)
    w = win.symmetric_window(days(0, 1), "2020-07-15T18:00", 1, cover=(day(-1), day(2)))
    assert w["idx"].tolist() == [1] and w["half_width_days"] == 0.25


@pytest.mark.parametrize("n", [0, -1, 2.5, True, None, "3", np.nan])
def test_n_must_be_a_whole_number_of_scenes(n):
    with pytest.raises(ValueError):
        win.symmetric_window(days(-1, 1, 2), D, n, cover=WIDE)


def test_a_numpy_integer_n_is_a_number_of_scenes():
    assert window(days(-1, 1, 2), np.int64(2))["idx"].tolist() == [0, 1]
    assert window(days(-1, 1, 2), 2.0)["idx"].tolist() == [0, 1]


GOOD = days(-1, 1, 2)
#: (dates, truth date, cover) that symmetric_window refuses
BAD = {"dates in two dimensions": (np.array([GOOD, GOOD]), D, WIDE),
       "two truth dates": (GOOD, GOOD[:2], WIDE),
       "a NaT among the dates": (np.array([day(-1), "NaT"]), D, WIDE),
       "a None among the dates": ([day(-1), None], D, WIDE),
       "numbers for dates": ([1, 2, 3], D, WIDE),
       "floats for dates": (np.array([1.5, 2.5]), D, WIDE),
       "a month for the truth date": (GOOD, "2020-07", WIDE),
       "a year for the truth date": (GOOD, "2020", WIDE),
       "a date that is not ISO": (GOOD, "15/07/2020", WIDE),
       "a cover that ends before it starts": (GOOD, D, (day(10), day(-10))),
       "a cover that is one date": (GOOD, D, day(10)),
       "a cover of one element": (GOOD, D, (day(-10),)),
       "a scene before the cube's first date": (GOOD, D, (day(0), day(10))),
       "a scene after the cube's last date": (GOOD, D, (day(-10), day(1)))}


@pytest.mark.parametrize("case", list(BAD))
def test_bad_dates_and_covers_are_errors(case):
    dates, d, cover = BAD[case]
    with pytest.raises(ValueError):
        win.symmetric_window(dates, d, 2, cover=cover)


# ─────────────────────────────────────────────────────────────────────────────
#  the mask: hand records and the rule on the counts
# ─────────────────────────────────────────────────────────────────────────────

#: one look: (clear, NDWI)
LOOK = {"w": (True, 0.4), "d": (True, -0.3), "z": (True, 0.0), "n": (True, np.nan), "c": (False, 0.6)}


def records(*cols):
    """Pixels written as strings of looks, one character per look: w wet (clear, NDWI 0.4), d dry (clear, -0.3), z
    clear with an NDWI of exactly 0, n clear with an undefined NDWI, c cloudy (NDWI 0.6, water-like: a cloudy look
    never counts). Shorter records are completed with cloudy looks. Returns (Y float32, C bool), looks x pixels."""
    T = max(len(s) for s in cols)
    Y = np.zeros((T, len(cols)), np.float32)
    C = np.zeros((T, len(cols)), bool)
    for p, s in enumerate(cols):
        for k, ch in enumerate(s.ljust(T, "c")):
            C[k, p], Y[k, p] = LOOK[ch]
    return Y, C


#: (record, wet looks, clear looks, kept at q = 0, 0.05, 0.10, 0.15, 0.20)
HAND = [("w" + "d" * 9, 1, 10, "TTTFF"),            # f = 0.10, on the lower bound at q = 0.10
        ("d" * 10, 0, 10, "TFFFF"),                 # always dry
        ("w" * 9 + "d", 9, 10, "TTTFF"),            # f = 0.90, on the upper bound at q = 0.10
        ("w" * 10, 10, 10, "TFFFF"),                # always wet
        ("wd" * 5, 5, 10, "TTTTT"),
        ("c" * 10, 0, 0, "FFFFF"),                  # no clear look: never kept, not even at q = 0
        ("w" + "n" * 9, 1, 10, "TTTFF"),            # an undefined NDWI is dry (dropped, f would be 1 of 1)
        ("w" + "z" * 9, 1, 10, "TTTFF"),            # an NDWI of 0 is dry
        ("w" + "c" * 8 + "d", 1, 2, "TTTTT"),       # cloudy looks do not count, water-like or not
        ("w" + "d" * 19, 1, 20, "TTFFF"),           # 0.05
        ("w" * 3 + "d" * 17, 3, 20, "TTTTF"),       # 0.15
        ("w" * 4 + "d" * 16, 4, 20, "TTTTT"),       # 0.20
        ("w" * 16 + "d" * 4, 16, 20, "TTTTT"),      # 0.80
        ("w" * 17 + "d" * 3, 17, 20, "TTTTF"),      # 0.85
        ("w" * 19 + "d", 19, 20, "TTFFF"),          # 0.95
        ("n" * 5, 0, 5, "TFFFF"),                   # every clear look undefined: always dry
        ("w", 1, 1, "TFFFF")]                       # one clear look


def test_the_mask_on_hand_records():
    Y, C = records(*(h[0] for h in HAND))
    wet, clear = win.look_counts(Y, C)
    assert wet.dtype == clear.dtype == np.int64
    assert wet.tolist() == [h[1] for h in HAND] and clear.tolist() == [h[2] for h in HAND]
    by_q = win.class_mask_sensitivity(Y, C)
    for k, q in enumerate(win.Q_SENSITIVITY):
        want = [h[3][k] == "T" for h in HAND]
        assert by_q[q].tolist() == want, q
        assert win.class_mask(Y, C, q).tolist() == want, q
    assert win.class_mask(Y, C).tolist() == [h[3][2] == "T" for h in HAND]      # the default, q = 0.10


def test_the_rule_at_every_share_of_up_to_100_looks():
    pairs = [(w, c) for c in range(0, 101) for w in range(0, c + 1)]
    wet, clear = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    share = [Fraction(w, c) if c else None for w, c in pairs]
    for k in range(51):
        Q = Fraction(k, 100)                        # the decimal q; k / 100 is the float nearest to it
        want = [f is not None and Q <= f <= 1 - Q for f in share]
        assert win.neither_dominates(wet, clear, k / 100).tolist() == want, k
    # the form f <= 1 - q would lose 93 of 100 at q = 0.07 (the module docstring)
    assert not 93 / 100 <= 1 - 0.07
    assert win.neither_dominates(np.array([93, 7]), np.array([100, 100]), 0.07).tolist() == [True, True]


# ─────────────────────────────────────────────────────────────────────────────
#  the mask: synthetic records
# ─────────────────────────────────────────────────────────────────────────────

def tidal_flat(rng, T=60, P=600):
    """The looks of a synthetic tidal flat, looks x pixels (float32 NDWI with NaN, bool clear). Elevations run from
    -2 to 2 m and a tide of 1.4 m amplitude is sampled at the scene instants. A look is wet when the level tops the
    pixel (NDWI 0.35 +- 0.1) and dry otherwise (-0.3 +- 0.1), so the channel below -1.4 m is always wet and the land
    above 1.4 m always dry. About 30 % of the looks are cloudy, some scenes cloudy throughout, 1 % of the NDWI
    undefined, and the first 5 pixels have no clear look."""
    z = np.linspace(-2.0, 2.0, P)
    h = 1.4 * np.sin(2 * np.pi * rng.random(T))
    flooded = h[:, None] > z[None, :]
    Y = np.where(flooded, rng.normal(0.35, 0.1, (T, P)), rng.normal(-0.3, 0.1, (T, P))).astype(np.float32)
    C = rng.random((T, P)) > 0.3
    C[rng.random(T) < 0.1] = False
    Y[rng.random((T, P)) < 0.01] = np.nan
    C[:, :5] = False
    return Y, C


def plain_counts(Y, C):
    """(wet, clear) per pixel by plain loops over the looks."""
    Yl, Cl = Y.tolist(), C.tolist()
    T, P = Y.shape
    wet, clear = [0] * P, [0] * P
    for p in range(P):
        for s in range(T):
            if Cl[s][p]:
                clear[p] += 1
                if Yl[s][p] > 0:                    # False for NaN: an undefined NDWI is dry
                    wet[p] += 1
    return wet, clear


def test_the_mask_is_the_plain_count_on_synthetic_records():
    Y, C = tidal_flat(np.random.default_rng(20261029))
    wet, clear = plain_counts(Y, C)
    got_wet, got_clear = win.look_counts(Y, C)
    assert got_wet.tolist() == wet and got_clear.tolist() == clear
    for q in (*win.Q_SENSITIVITY, 0.07, 0.32, 0.5):
        Q = Fraction(str(q))
        want = [c > 0 and Q <= Fraction(w, c) <= 1 - Q for w, c in zip(wet, clear)]
        assert win.class_mask(Y, C, q).tolist() == want, q
    # the flat holds every case: no clear look, always dry, always wet, both classes
    f = np.array([w / c if c else np.nan for w, c in zip(wet, clear)])
    assert np.isnan(f).sum() == 5 and (f == 0).any() and (f == 1).any() and ((f > 0.1) & (f < 0.9)).any()


def test_q_0_keeps_every_pixel_with_a_clear_look():
    Y, C = tidal_flat(np.random.default_rng(3))
    has_clear = C.any(axis=0)
    assert not has_clear.all()
    for m in (win.class_mask(Y, C, 0), win.class_mask(Y, C, 0.0), win.class_mask_sensitivity(Y, C)[0]):
        assert m.dtype == bool and np.array_equal(m, has_clear)
    # with the pixels always wet or always dry, which every q > 0 drops
    wet, clear = win.look_counts(Y, C)
    one_class = (clear > 0) & ((wet == 0) | (wet == clear))
    assert one_class.sum() > 50 and not win.class_mask(Y, C, 0.05)[one_class].any()


def test_the_sensitivity_is_the_mask_at_each_q_and_the_masks_are_nested():
    Y, C = tidal_flat(np.random.default_rng(5))
    by_q = win.class_mask_sensitivity(Y, C)
    assert list(by_q) == [0.0, 0.05, 0.10, 0.15, 0.20] and all(type(q) is float for q in by_q)
    for q, m in by_q.items():
        assert m.dtype == bool and m.shape == (Y.shape[1],)
        assert np.array_equal(m, win.class_mask(Y, C, q))
    masks = list(by_q.values())
    for a, b in zip(masks, masks[1:]):
        assert not (b & ~a).any() and b.sum() < a.sum()     # a larger q keeps a subset, here a smaller one
    assert np.array_equal(by_q[0.10], win.class_mask(Y, C))
    other = win.class_mask_sensitivity(Y, C, qs=(0.2, 0))
    assert list(other) == [0.2, 0.0] and np.array_equal(other[0.2], by_q[0.20])


def test_nan_or_zero_for_an_undefined_ndwi_and_any_clear_dtype_give_the_same_mask():
    Y, C = tidal_flat(np.random.default_rng(9))
    m = win.class_mask(Y, C)
    assert np.array_equal(win.class_mask(np.nan_to_num(Y, nan=0.0), C), m)       # the inversion's input
    assert np.array_equal(win.class_mask(Y.astype(np.float64), C), m)
    for Cx in (C.astype(np.uint8), C.astype(np.float64)):                        # a 0/1 clear mask
        assert np.array_equal(win.class_mask(Y, Cx), m)
    for fill in (np.nan, 0.9, -0.9):                                             # what the cloudy looks hold
        Y2 = Y.copy()
        Y2[~C] = fill
        assert np.array_equal(win.class_mask(Y2, C), m)


# ─────────────────────────────────────────────────────────────────────────────
#  the mask: reading in blocks and the window's rows
# ─────────────────────────────────────────────────────────────────────────────

class Lazy:
    """An array read only by [rows, columns] slices, as a memory-mapped or chunked store: it refuses to be read whole
    and records the columns of every read."""

    def __init__(self, a):
        self.a, self.shape, self.ndim, self.reads = a, a.shape, a.ndim, []

    def __getitem__(self, key):
        rows, cols = key
        self.reads.append((cols.start, cols.stop))
        return self.a[rows, cols]

    def __array__(self, *args, **kwargs):
        raise AssertionError("the whole array was read")


def test_the_blocks_do_not_change_the_mask_and_only_blocks_are_read():
    Y, C = tidal_flat(np.random.default_rng(13), P=257)
    ref = win.class_mask_sensitivity(Y, C)
    for block in (1, 7, 64, 256, 257, 10_000):
        got = win.class_mask_sensitivity(Y, C, block_px=block)
        assert all(np.array_equal(got[q], ref[q]) for q in ref), block
        Yl, Cl = Lazy(Y), Lazy(C)
        assert np.array_equal(win.class_mask(Yl, Cl, block_px=block), ref[0.10]), block
        assert Yl.reads == Cl.reads
        spans = Yl.reads
        assert spans[0][0] == 0 and spans[-1][1] == 257 and all(b - a <= block for a, b in spans)
        assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))


def test_rows_read_the_windows_scenes_in_place():
    rng = np.random.default_rng(17)
    Y_S, C_S = tidal_flat(rng, T=150, P=300)                             # the screened record
    dates = days(*np.sort(rng.choice(np.arange(-700, 700), 150, replace=False)))
    w = window(dates, 60)
    i = w["idx"]
    assert np.array_equal(win.class_mask(Y_S, C_S, rows=i), win.class_mask(Y_S[i], C_S[i]))
    a, b = win.class_mask_sensitivity(Y_S, C_S, rows=i), win.class_mask_sensitivity(Y_S[i], C_S[i])
    assert all(np.array_equal(a[q], b[q]) for q in a)
    Yl, Cl = Lazy(Y_S), Lazy(C_S)
    assert np.array_equal(win.class_mask(Yl, Cl, rows=i, block_px=50), win.class_mask(Y_S[i], C_S[i]))
    assert np.array_equal(win.look_counts(Y_S, C_S, rows=np.arange(150))[1], win.look_counts(Y_S, C_S)[1])
    assert not win.class_mask(Y_S, C_S, rows=[]).any()                   # no look: no pixel kept


def test_bad_mask_inputs_are_errors():
    Y, C = tidal_flat(np.random.default_rng(1), T=8, P=12)
    for Yb, Cb in ((Y, C[:, :11]), (Y, C[:7]), (Y[0], C[0]), (Y[None], C[None])):
        with pytest.raises(ValueError):
            win.class_mask(Yb, Cb)
    for q in (-0.01, 0.51, 1.0, np.nan):
        with pytest.raises(ValueError):
            win.class_mask(Y, C, q)
        with pytest.raises(ValueError):
            win.class_mask_sensitivity(Y, C, qs=(0.1, q))
        with pytest.raises(ValueError):
            win.neither_dominates(np.array([1]), np.array([2]), q)
    with pytest.raises(ValueError):
        win.class_mask(Y, C, block_px=0)
    for rows in ([0, 8], [-1, 2], [1, 1, 2], np.ones(8, bool), [[0, 1]], [0.5, 1.5]):
        with pytest.raises(ValueError):
            win.class_mask(Y, C, rows=rows)
    for wet, clear in (([1], [0]), ([-1], [3]), ([1, 2], [3])):
        with pytest.raises(ValueError):
            win.neither_dominates(np.array(wet), np.array(clear))


# ─────────────────────────────────────────────────────────────────────────────
#  the constants
# ─────────────────────────────────────────────────────────────────────────────

def test_the_constants_and_defaults_are_the_designs():
    assert win.NS == (20, 40, 60, 80, 100, 120)
    assert win.Q_CLASS == 0.10 and win.Q_SENSITIVITY == (0.0, 0.05, 0.10, 0.15, 0.20)
    assert inspect.signature(win.class_mask).parameters["q"].default == win.Q_CLASS
    assert inspect.signature(win.neither_dominates).parameters["q"].default == win.Q_CLASS
    assert inspect.signature(win.class_mask_sensitivity).parameters["qs"].default == win.Q_SENSITIVITY
    with open(DESIGN, encoding="utf-8") as f:
        text = " ".join(f.read().split())
    assert "For each n in (20, 40, 60, 80, 100, 120)" in text
    assert "(sorted by |t - d|, ties by the earlier t)" in text and "covers [d - T, d + T] entirely" in text
    assert "0.10 <= f <= 0.90" in text and "q = 0, 0.05, 0.10, 0.15 and 0.20" in text
