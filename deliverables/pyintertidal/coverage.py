"""
coverage.py — Is this site mappable? Tidal-sampling diagnostics
===============================================================

Satellites do not sample the tide evenly. A sun-synchronous orbit crosses at
roughly the same local solar time, so over some sites the archive repeatedly
catches the same phase of the tide, and a decade of imagery can still leave
whole metres of the tidal frame unobserved. Elevation methods cannot invent
what was never seen: **the quality of an intertidal DEM is bounded by how
well the satellite sampled the tidal range.**

Run these diagnostics BEFORE a study to know what is achievable, and report
them WITH the results so readers can judge them.

The metrics, and what each one catches
--------------------------------------
:func:`tidal_range_coverage`
    Fraction of the full tidal frame spanned by the usable dates. Catches
    the blunt failure: the archive never saw low water.
:func:`ks_uniformity`
    Kolmogorov–Smirnov test against a uniform distribution. Catches sampling
    that spans the range but clusters at mid-tide.
:func:`shannon_entropy`
    Normalised entropy over height bins (0–1). A smooth, bin-based view of
    the same idea, robust to outliers.
:func:`dispersion_index`
    Variance-to-mean ratio of the gaps between consecutive levels: < 1
    regular, ≈ 1 random, > 1 clustered.
:func:`gap_statistics`
    The gaps themselves — where the biggest hole in the vertical coverage is.
:func:`vertical_sampling_resolution`
    Mean spacing between consecutive tide levels: the effective vertical
    resolution of a waterline-style reconstruction (Mason et al. 1995;
    Heygster et al. 2010). Below ~0.1 m is detailed, above ~0.3 m coarse.
:func:`representativity`
    Fraction of quantiles of the full distribution containing at least one
    sample — complements coverage by looking at the middle, not the extremes.

:func:`coverage_metrics` runs them all; :func:`coverage_verdict` turns them
into a pass/fail judgement; :func:`print_coverage_report` prints a readable
summary.
"""

from __future__ import annotations

import warnings

import numpy as np


def _clean(values):
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


# ─────────────────────────────────────────────────────────────────────────────
#  Individual metrics
# ─────────────────────────────────────────────────────────────────────────────

def tidal_range_coverage(sampled, reference):
    """Fraction of the full tidal range spanned by the sampled dates.

    ``sampled`` are the tide heights of the usable satellite dates;
    ``reference`` is the full tidal distribution (e.g. an hourly model series
    over the same period, from
    :meth:`pyintertidal.tides.TideService.series`).

    A coverage of 1.0 means the archive saw both the lowest and the highest
    water of the period; 0.6 means 40 % of the vertical frame was never
    imaged and any DEM there is extrapolation.
    """
    s, r = _clean(sampled), _clean(reference)
    if s.size == 0 or r.size == 0:
        # Same keys as the normal path, so callers never have to special-case
        # the empty result (e.g. when no date resolved a tide height).
        return {"coverage": 0.0, "coverage_pct": 0.0,
                "sampled_range": 0.0, "sampled_min": np.nan,
                "sampled_max": np.nan, "reference_range": 0.0,
                "reference_min": np.nan, "reference_max": np.nan,
                "unsampled_low": np.nan, "unsampled_high": np.nan}
    smin, smax = float(s.min()), float(s.max())
    rmin, rmax = float(r.min()), float(r.max())
    total = rmax - rmin
    frac = (smax - smin) / total if total > 0 else 0.0
    return {
        "coverage": float(frac), "coverage_pct": float(frac * 100),
        "sampled_range": smax - smin, "sampled_min": smin, "sampled_max": smax,
        "reference_range": total, "reference_min": rmin, "reference_max": rmax,
        "unsampled_low": max(smin - rmin, 0.0),
        "unsampled_high": max(rmax - smax, 0.0),
    }


def ks_uniformity(values, reference_range=None):
    """Kolmogorov–Smirnov test of the sampling against a uniform spread.

    The null hypothesis is "the tide heights are uniformly distributed over
    the range". A LOW statistic (and a p-value above 0.05) means the sampling
    is close to uniform, which is what a waterline reconstruction wants.
    """
    from scipy import stats

    v = _clean(values)
    if v.size < 3:
        return {"ks_statistic": np.nan, "ks_pvalue": np.nan,
                "uniform": False, "interpretation": "too few samples"}
    lo, hi = reference_range if reference_range else (v.min(), v.max())
    if hi <= lo:
        return {"ks_statistic": np.nan, "ks_pvalue": np.nan,
                "uniform": False, "interpretation": "degenerate range"}
    stat, p = stats.kstest((v - lo) / (hi - lo), "uniform")
    uniform = bool(p > 0.05)
    return {"ks_statistic": float(stat), "ks_pvalue": float(p),
            "uniform": uniform,
            "interpretation": ("consistent with uniform sampling" if uniform
                               else "significantly non-uniform")}


def shannon_entropy(values, n_bins=20, reference_range=None):
    """Normalised Shannon entropy of the tide-height distribution (0–1).

    1.0 = every height bin equally populated (ideal); low values = the
    archive keeps re-imaging the same few tide levels. Reported alongside the
    number of EMPTY bins, which is the actionable part: those are the
    elevations no image constrains.
    """
    v = _clean(values)
    if v.size == 0:
        return {"entropy": 0.0, "entropy_normalised": 0.0,
                "bins_occupied": 0, "bins_empty": int(n_bins), "n_bins": int(n_bins)}
    rng = reference_range if reference_range else (v.min(), v.max())
    counts, _ = np.histogram(v, bins=n_bins, range=rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probs = counts / counts.sum() if counts.sum() else counts.astype(float)
        probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0
    entropy_max = float(np.log2(n_bins))
    occupied = int((counts > 0).sum())
    return {
        "entropy": entropy, "entropy_max": entropy_max,
        "entropy_normalised": entropy / entropy_max if entropy_max else 0.0,
        "n_bins": int(n_bins), "bins_occupied": occupied,
        "bins_empty": int(n_bins) - occupied,
    }


def gap_statistics(values):
    """Statistics of the gaps between consecutive sorted tide levels.

    ``max_gap`` and its location tell you exactly which band of the tidal
    frame is unconstrained — often the actionable output of the whole module
    (e.g. "we never imaged between +0.8 and +1.6 m").
    """
    v = np.sort(_clean(values))
    if v.size < 2:
        return {"n_gaps": 0, "mean_gap": np.nan, "median_gap": np.nan,
                "max_gap": np.nan, "max_gap_between": (np.nan, np.nan)}
    gaps = np.diff(v)
    imax = int(np.argmax(gaps))
    return {
        "n_gaps": int(gaps.size), "mean_gap": float(gaps.mean()),
        "median_gap": float(np.median(gaps)), "std_gap": float(gaps.std()),
        "min_gap": float(gaps.min()), "max_gap": float(gaps.max()),
        "max_gap_between": (float(v[imax]), float(v[imax + 1])),
        "p90_gap": float(np.percentile(gaps, 90)),
    }


def dispersion_index(values):
    """Variance-to-mean ratio (VMR) of the gaps — regular, random or clustered.

    ``< 1`` the levels are more evenly spaced than random (good);
    ``≈ 1`` random; ``> 1`` clustered, i.e. dense sampling in a few bands and
    holes elsewhere.
    """
    v = np.sort(_clean(values))
    if v.size < 3:
        return {"vmr": np.nan, "pattern": "too few samples"}
    gaps = np.diff(v)
    mean = float(gaps.mean())
    var = float(gaps.var())
    vmr = var / mean if mean > 0 else np.nan
    if not np.isfinite(vmr):
        pattern = "undefined"
    elif vmr < 0.8:
        pattern = "regular"
    elif vmr <= 1.2:
        pattern = "random"
    else:
        pattern = "clustered"
    return {"vmr": float(vmr), "pattern": pattern,
            "mean_gap": mean, "variance_gap": var}


def vertical_sampling_resolution(values):
    """Effective vertical resolution of the sampling (metres).

    The mean spacing between consecutive observed tide levels. In a
    waterline reconstruction each observation contributes one contour, so
    this IS the vertical resolution of the resulting DEM.

    Guide: ≤ 0.1 m detailed · 0.1–0.3 m moderate · > 0.3 m coarse.
    """
    v = np.sort(_clean(values))
    if v.size < 2:
        return {"vsr": np.nan, "quality": "too few samples"}
    gaps = np.diff(v)
    vsr = float(gaps.mean())
    quality = ("detailed" if vsr <= 0.1 else
               "moderate" if vsr <= 0.3 else "coarse")
    return {"vsr": vsr, "vsr_median": float(np.median(gaps)),
            "vsr_min": float(gaps.min()), "vsr_max": float(gaps.max()),
            "quality": quality, "n_levels": int(v.size)}


def representativity(sampled, reference, n_quantiles=10):
    """Fraction of the reference distribution's quantiles that were sampled.

    Coverage looks at the extremes; this looks at the MIDDLE. Splitting the
    full tidal distribution into deciles and checking each one has at least
    one observation catches the case of a range that is nominally covered but
    hollow inside.
    """
    s, r = _clean(sampled), _clean(reference)
    if s.size == 0 or r.size == 0:
        return {"representativity": 0.0, "representativity_pct": 0.0,
                "quantiles_covered": 0, "quantiles_total": int(n_quantiles)}
    edges = np.quantile(r, np.linspace(0, 1, n_quantiles + 1))
    edges[0] -= 1e-9
    covered = sum(1 for i in range(n_quantiles)
                  if np.any((s > edges[i]) & (s <= edges[i + 1])))
    frac = covered / n_quantiles
    return {"representativity": float(frac),
            "representativity_pct": float(frac * 100),
            "quantiles_covered": int(covered),
            "quantiles_total": int(n_quantiles),
            "empty_quantiles": [i + 1 for i in range(n_quantiles)
                                if not np.any((s > edges[i]) & (s <= edges[i + 1]))]}


# ─────────────────────────────────────────────────────────────────────────────
#  Combined report
# ─────────────────────────────────────────────────────────────────────────────

def coverage_metrics(sampled, reference, n_bins=20, n_quantiles=10):
    """Run every diagnostic and return them in one dictionary.

    ``sampled`` = tide heights of the dates that survived cloud filtering;
    ``reference`` = the full tidal distribution over the same period.
    """
    s, r = _clean(sampled), _clean(reference)
    rng = (float(r.min()), float(r.max())) if r.size else None
    return {
        "n_sampled": int(s.size), "n_reference": int(r.size),
        "range_coverage": tidal_range_coverage(s, r),
        "uniformity": ks_uniformity(s, rng),
        "entropy": shannon_entropy(s, n_bins, rng),
        "dispersion": dispersion_index(s),
        "gaps": gap_statistics(s),
        "resolution": vertical_sampling_resolution(s),
        "representativity": representativity(s, r, n_quantiles),
    }


def coverage_verdict(metrics, min_coverage=0.85, min_entropy=0.70,
                     max_vmr=1.20, max_vsr=0.25, min_representativity=0.80):
    """Turn the metrics into a pass/fail judgement with reasons.

    The thresholds are defaults from our own work, not universal law — a
    macrotidal site can afford a coarser VSR than a microtidal one. Pass your
    own and say which you used when reporting.

    Returns ``{"suitable": bool, "checks": {...}, "reasons": [...]}``.
    """
    checks = {
        "range_coverage": metrics["range_coverage"]["coverage"] >= min_coverage,
        "entropy": metrics["entropy"]["entropy_normalised"] >= min_entropy,
        "dispersion": (np.isfinite(metrics["dispersion"]["vmr"])
                       and metrics["dispersion"]["vmr"] <= max_vmr),
        "resolution": (np.isfinite(metrics["resolution"]["vsr"])
                       and metrics["resolution"]["vsr"] <= max_vsr),
        "representativity": (metrics["representativity"]["representativity"]
                             >= min_representativity),
    }
    reasons = []
    if not checks["range_coverage"]:
        c = metrics["range_coverage"]
        reasons.append(f"only {c['coverage_pct']:.0f}% of the tidal range was "
                       f"imaged ({c['unsampled_low']:.2f} m missing at the "
                       f"bottom, {c['unsampled_high']:.2f} m at the top)")
    if not checks["entropy"]:
        reasons.append(f"sampling concentrates on few tide levels "
                       f"({metrics['entropy']['bins_empty']} empty height bins)")
    if not checks["dispersion"]:
        reasons.append(f"levels are {metrics['dispersion']['pattern']} "
                       f"(VMR {metrics['dispersion']['vmr']:.2f})")
    if not checks["resolution"]:
        reasons.append(f"vertical resolution {metrics['resolution']['vsr']:.2f} m "
                       f"is coarse for detailed topography")
    if not checks["representativity"]:
        reasons.append(f"{metrics['representativity']['quantiles_total'] - metrics['representativity']['quantiles_covered']} "
                       f"quantiles of the tidal distribution have no observation")
    return {"suitable": all(checks.values()), "checks": checks,
            "reasons": reasons}


def print_coverage_report(metrics, verdict=None, title=""):
    """Print a readable summary of the diagnostics (and optional verdict)."""
    if title:
        print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")
    c = metrics["range_coverage"]
    print(f"\nTIDAL RANGE COVERAGE")
    print(f"  sampled   {c['sampled_range']:.2f} m  "
          f"[{c['sampled_min']:+.2f} → {c['sampled_max']:+.2f}]")
    print(f"  reference {c['reference_range']:.2f} m  "
          f"[{c['reference_min']:+.2f} → {c['reference_max']:+.2f}]")
    print(f"  coverage  {c['coverage_pct']:.1f}%   "
          f"(unsampled: {c['unsampled_low']:.2f} m low, "
          f"{c['unsampled_high']:.2f} m high)")

    u = metrics["uniformity"]
    print(f"\nUNIFORMITY (Kolmogorov–Smirnov)")
    print(f"  statistic {u['ks_statistic']:.4f} · p {u['ks_pvalue']:.4f} → "
          f"{u['interpretation']}")

    e = metrics["entropy"]
    print(f"\nENTROPY")
    print(f"  normalised {e['entropy_normalised']:.3f} "
          f"({e['bins_occupied']}/{e['n_bins']} height bins occupied)")

    d = metrics["dispersion"]
    print(f"\nDISPERSION")
    print(f"  VMR {d['vmr']:.3f} → {d['pattern']} spacing")

    g = metrics["gaps"]
    print(f"\nGAPS")
    print(f"  mean {g['mean_gap']:.3f} m · max {g['max_gap']:.3f} m "
          f"between {g['max_gap_between'][0]:+.2f} and "
          f"{g['max_gap_between'][1]:+.2f} m")

    r = metrics["resolution"]
    print(f"\nVERTICAL SAMPLING RESOLUTION")
    print(f"  VSR {r['vsr']:.3f} m → {r['quality']} ({r['n_levels']} levels)")

    p = metrics["representativity"]
    print(f"\nREPRESENTATIVITY")
    print(f"  {p['quantiles_covered']}/{p['quantiles_total']} quantiles sampled "
          f"({p['representativity_pct']:.0f}%)")

    if verdict is not None:
        print(f"\nVERDICT: {'SUITABLE' if verdict['suitable'] else 'LIMITED'}")
        for reason in verdict["reasons"]:
            print(f"  - {reason}")
    return metrics
