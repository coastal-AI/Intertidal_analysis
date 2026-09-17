"""
boundary.py — The water level at the mouth, from wherever you can get it
========================================================================

Everything this package computes inland starts from one number per date: the
water level at the entrance of the estuary. Until now that number came from a
global ocean tide model chosen once and wired in. That is a bad place for a
choice to live, for three reasons:

* the choice is not universal. EOT20 is good on the Cantabrian shelf; FES2022
  or TPXO may be better elsewhere, and on a coast nobody has evaluated you do
  not know which in advance;
* a hard-wired model makes its errors indistinguishable from physics. If the
  model is 4 % high in M2 at the mouth, any estuarine transfer fitted on top
  absorbs that error and stops being a property of the estuary — so it stops
  transferring to the next one;
* sometimes the best boundary is not a model at all. A harbour tide gauge, or
  a survey of the waterline, measures the thing directly.

So the boundary becomes an object with a contract rather than a constant.
Anything that can answer "what was the water level here, at these instants"
is a valid boundary, and the rest of the package never learns which kind it
got.

The contract
------------
A provider answers two questions:

``levels(times)``
    water level in metres for arbitrary instants;
``harmonics()``
    amplitude and phase per constituent, plus an uncertainty per constituent
    where one is available — ``None`` when the provider cannot decompose.

The second is what a physical transfer operator needs, because propagation up
an estuary happens constituent by constituent: each has its own frequency, and
friction and convergence act differently on each.

What a sun-synchronous archive can and cannot see
-------------------------------------------------
Sentinel-2 crosses at essentially the same solar time every pass, so a
constituent whose period divides the solar day is observed at one fixed phase
forever and is invisible no matter how long the archive runs. This is
arithmetic, not a data problem, and :func:`alias_periods` reports it:

===========  ============  ==================================
constituent  alias period  estimable from a daily-at-fixed-hour archive?
===========  ============  ==================================
M2           14.8 d        yes
N2            9.6 d        yes
O1           14.2 d        yes
Q1            9.4 d        yes
M4            7.4 d        yes
M6            4.9 d        yes
K2          182.6 d        hard
K1          365.2 d        confounded with the seasonal cycle
P1          365.2 d        confounded with the seasonal cycle
**S2**      **infinite**   **no — frozen at one phase**
===========  ============  ==================================

S2 is the solar semidiurnal constituent and its period is exactly half a solar
day. Any interior tide model estimated from imagery must take S2 from outside;
this module makes that explicit rather than leaving it to be discovered.
"""

from __future__ import annotations

import numpy as np

#: Fixed reference instant for every phase in this package. Any constant
#: would do; what matters is that it is the SAME one everywhere.
EPOCH = np.datetime64("1992-01-01T00:00:00")


def hours_since_epoch(times):
    """Hours from :data:`EPOCH`, the common phase reference."""
    import pandas as pd

    idx = pd.DatetimeIndex(pd.to_datetime(times))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return (idx.values - EPOCH) / np.timedelta64(1, "h")


#: Constituent periods in hours (Doodson's principal set).
PERIODS_H = {
    "M2": 12.4206012, "S2": 12.0000000, "N2": 12.6583475, "K2": 11.9672348,
    "K1": 23.9344697, "O1": 25.8193417, "P1": 24.0658902, "Q1": 26.8683567,
    "M4": 6.2103006, "M6": 4.1402004, "MS4": 6.1033393, "MN4": 6.2691739,
}


def alias_periods(samples_per_day=1.0):
    """Alias period of each constituent for a fixed-local-time sensor.

    A constituent of frequency ``f`` sampled at ``samples_per_day`` folds to
    ``|f - round(f)|`` cycles per day. When that is zero the constituent never
    changes phase between samples and cannot be recovered at all.

    Returns ``{name: alias_period_days}``, with ``inf`` for the unobservable.
    """
    out = {}
    for name, per in PERIODS_H.items():
        f = 24.0 / per / max(samples_per_day, 1e-9)
        fold = abs(f - round(f))
        out[name] = float("inf") if fold < 1e-9 else float(1.0 / fold)
    return out


def estimable_constituents(samples_per_day=1.0, max_alias_days=100.0):
    """Constituents a fixed-local-time archive can actually resolve."""
    al = alias_periods(samples_per_day)
    return sorted(k for k, v in al.items() if v <= max_alias_days)


# ─────────────────────────────────────────────────────────────────────────────
#  The contract
# ─────────────────────────────────────────────────────────────────────────────

class BoundaryProvider:
    """Water level at the estuary mouth. Subclasses implement ``levels``.

    Deliberately not an abstract base class: a provider is anything with these
    two methods, and a user with their own data source should be able to pass
    a plain object without importing anything from here.
    """

    name = "boundary"

    def levels(self, times):
        """Water level in metres at the given instants (array-like of datetimes)."""
        raise NotImplementedError

    def harmonics(self):
        """``{constituent: (amplitude_m, phase_rad, sigma_m)}`` or ``None``."""
        return None

    def __repr__(self):
        return f"<{type(self).__name__} {self.name!r}>"


class PyTMDBoundary(BoundaryProvider):
    """A global ocean tide model, through pyTMD/eo-tides.

    Any model pyTMD serves works here — EOT20, FES2022, GOT5.6, TPXO — because
    they share one API. Which one is best is a question to answer with data,
    not to settle in an import.

    Parameters
    ----------
    model : str
        Model name as pyTMD knows it.
    lat, lon : float
        Prediction point. Estuary centroids often fall on a LAND cell of the
        model grid; :mod:`pyintertidal.tidecheck` measures how far the nearest
        ocean cell really is, and it is worth checking before trusting this.
    directory : str
        Where the model files live.
    """

    def __init__(self, model, lat, lon, directory="tide_models"):
        self.name = model
        self.model = model
        self.lat = float(lat)
        self.lon = float(lon)
        self.directory = directory

    def levels(self, times):
        import pandas as pd
        from eo_tides.model import model_tides

        idx = pd.DatetimeIndex(pd.to_datetime(times))
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        df = model_tides(x=[self.lon], y=[self.lat], time=idx,
                         model=self.model, directory=self.directory,
                         crs="EPSG:4326", extrapolate=True, cutoff=np.inf,
                         parallel=False)
        return (df.reset_index().sort_values("time")["tide_height"]
                .to_numpy(float))

    def harmonics(self, constituents=None, times=None):
        """Amplitudes and phases, obtained by fitting the model's own output.

        The parameter order matters and is part of the contract: every
        provider takes ``constituents`` first, so callers can pass it
        positionally without knowing which kind of boundary they hold. An
        earlier version put ``times`` first here and the constituent tuple
        silently arrived as a list of timestamps.

        Going through a synthetic hourly year rather than reading the model's
        coefficients keeps this identical in form to what a gauge produces, so
        a model and a gauge really are interchangeable downstream.
        """
        import pandas as pd

        if times is None:
            times = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="1h")
        h = self.levels(times)
        return fit_harmonics(times, h, constituents)


class GaugeBoundary(BoundaryProvider):
    """A measured record: a harbour tide gauge, or a surveyed waterline.

    The point of this class is that a measurement and a model become the same
    kind of object. A transfer operator fitted with a gauge at the mouth is
    then free of ocean-model error entirely, which is the cleanest boundary
    available and the one to prefer wherever a gauge exists.
    """

    def __init__(self, times, levels, name="gauge", max_gap="1h"):
        import pandas as pd

        idx = pd.DatetimeIndex(pd.to_datetime(times))
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        order = np.argsort(idx.values)
        self.times = idx[order]
        self.values = np.asarray(levels, dtype=float)[order]
        self.name = name
        # a record with holes must not be bridged by a straight line
        # across them: an instant farther than max_gap from both of its
        # neighbouring samples reads NaN
        self.max_gap_s = float(pd.Timedelta(max_gap).total_seconds())

    @classmethod
    def from_file(cls, path, name=None, **kw):
        """Load through :func:`pyintertidal.gauges.load_gauge`."""
        from .gauges import load_gauge

        df = load_gauge(path, **kw)
        return cls(df["time"], df["level_m"], name=name or str(path))

    def levels(self, times):
        import pandas as pd

        idx = pd.DatetimeIndex(pd.to_datetime(times))
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        x = self.times.view("int64").astype(float) / 1e9
        q = idx.view("int64").astype(float) / 1e9
        out = np.interp(q, x, self.values, left=np.nan, right=np.nan)
        j = np.searchsorted(x, q)
        j0 = np.clip(j - 1, 0, len(x) - 1)
        j1 = np.clip(j, 0, len(x) - 1)
        gap = np.minimum(np.abs(q - x[j0]), np.abs(x[j1] - q))
        out[gap > self.max_gap_s] = np.nan
        return out

    def harmonics(self, constituents=None):
        return fit_harmonics(self.times, self.values, constituents)


class EnsembleBoundary(BoundaryProvider):
    """Several providers averaged, with their disagreement as the uncertainty.

    The spread between independent ocean products is a free and honest
    estimate of how wrong any one of them might be at this coast — an
    estimate almost nobody uses, and the natural prior for a boundary
    correction term. It is not a guarantee: models sharing an assimilation
    dataset can agree and be wrong together.
    """

    def __init__(self, providers, name="ensemble"):
        self.providers = list(providers)
        self.name = name

    def levels(self, times):
        stack = np.vstack([p.levels(times) for p in self.providers])
        return np.nanmean(stack, axis=0)

    def spread(self, times):
        """Standard deviation across members — the uncertainty of the mean."""
        stack = np.vstack([p.levels(times) for p in self.providers])
        return np.nanstd(stack, axis=0)

    def harmonics(self, constituents=None):
        hs = [p.harmonics() for p in self.providers]
        hs = [h for h in hs if h]
        if not hs:
            return None
        out = {}
        for k in hs[0]:
            amps = np.array([h[k][0] for h in hs if k in h])
            phs = np.array([h[k][1] for h in hs if k in h])
            # phases average on the unit circle, not on the real line
            ph = float(np.angle(np.mean(np.exp(1j * phs))))
            out[k] = (float(np.mean(amps)), ph, float(np.std(amps)))
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Harmonic analysis
# ─────────────────────────────────────────────────────────────────────────────

def fit_harmonics(times, levels, constituents=None):
    """Least-squares harmonic fit at the astronomical frequencies.

    Plain linear least squares on sines and cosines of known frequencies —
    the frequencies come from astronomy, so nothing here is being searched
    for. Nodal corrections are not applied, which limits absolute accuracy
    over multi-year records but leaves the relative comparison between a
    boundary and an interior station, which is what a transfer operator
    needs, unaffected.

    Returns ``{constituent: (amplitude_m, phase_rad, sigma_m)}``.
    """
    import pandas as pd

    idx = pd.DatetimeIndex(pd.to_datetime(times))
    y = np.asarray(levels, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 50:
        return None
    # Phases are referenced to a FIXED epoch, never to the start of whatever
    # series happened to be passed. Referencing to the series start looks
    # harmless and is not: harmonics fitted on one window and then evaluated
    # over another acquire an arbitrary phase shift equal to the gap between
    # the two starts. It only stays hidden while both happen to begin at the
    # same instant, which is exactly the case a validation loop falls into.
    t_h = hours_since_epoch(idx)
    names = list(constituents or PERIODS_H)
    cols = [np.ones_like(t_h)]
    for n in names:
        w = 2 * np.pi / PERIODS_H[n]
        cols += [np.cos(w * t_h), np.sin(w * t_h)]
    A = np.column_stack(cols)[ok]
    coef, res, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    dof = max(int(ok.sum()) - A.shape[1], 1)
    sigma = float(np.sqrt(np.sum((y[ok] - A @ coef) ** 2) / dof))
    try:
        cov = sigma ** 2 * np.linalg.pinv(A.T @ A)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(A.shape[1], np.nan)
    out = {}
    for i, n in enumerate(names):
        a, b = coef[1 + 2 * i], coef[2 + 2 * i]
        amp = float(np.hypot(a, b))
        # error on the amplitude, propagated from the two coefficients
        s = float(np.hypot(se[1 + 2 * i] * (a / amp if amp else 0),
                           se[2 + 2 * i] * (b / amp if amp else 0)))
        out[n] = (amp, float(np.arctan2(-b, a)), s)
    return out


def transfer_from_gauges(mouth, inner, constituents=None):
    """Measured amplification and lag, constituent by constituent.

    Given two records — one at the mouth, one inland — this is what an
    estuarine transfer operator must reproduce, obtained without any model or
    imagery. It is the yardstick, not the method.

    Returns ``{constituent: {"gain", "gain_sigma", "lag_hours"}}``, where gain
    above one means the estuary amplifies that constituent.
    """
    hm = mouth.harmonics(constituents) if hasattr(mouth, "harmonics") else mouth
    hi = inner.harmonics(constituents) if hasattr(inner, "harmonics") else inner
    if not hm or not hi:
        return None
    out = {}
    for k in hm:
        if k not in hi or hm[k][0] <= 1e-6:
            continue
        am, ph_m, sm = hm[k]
        ai, ph_i, si = hi[k]
        gain = ai / am
        dphi = float(np.angle(np.exp(1j * (ph_i - ph_m))))
        lag = dphi / (2 * np.pi) * PERIODS_H[k]
        out[k] = {"gain": float(gain),
                  "gain_sigma": float(gain * np.hypot(si / max(ai, 1e-9),
                                                      sm / max(am, 1e-9))),
                  "lag_hours": float(-lag)}
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  One boundary interface for every estimator: a tide model, N nearest
#  gauges, or the consensus of both
# ─────────────────────────────────────────────────────────────────────────────

IOC_BASE = "http://www.ioc-sealevelmonitoring.org/service.php"


def nearest_gauges(lat, lon, n, stations_path="data_v4/gauges/ioc_stations.json",
                   max_km=400.0):
    """The ``n`` IOC stations closest to (lat, lon): [(code, name, km)]."""
    import json

    with open(stations_path, encoding="utf-8") as f:
        st = json.load(f)
    rows = st if isinstance(st, list) else st.get("stations", st)
    out = []
    for s in rows:
        try:
            la, lo = float(s["Lat"]), float(s["Lon"])
        except (KeyError, TypeError, ValueError):
            continue
        dlat = np.radians(la - lat)
        dlon = np.radians(lo - lon)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(lat)) * np.cos(np.radians(la))
             * np.sin(dlon / 2) ** 2)
        km = 2 * 6371.0 * np.arcsin(np.sqrt(a))
        if km <= max_km:
            out.append((s["Code"], s.get("Location", s["Code"]), float(km)))
    out.sort(key=lambda r: r[2])
    return out[:int(n)]


def fetch_ioc(code, t0, t1, cache_dir="data_v4/gauges", chunk_days=30):
    """Download (once) and load one IOC sea-level record for [t0, t1].

    Cached as ``ioc_{code}_{t0}_{t1}.json`` in ``cache_dir`` so the loader
    :func:`pyintertidal.gauges.load_cached_ioc` finds it; returns its tidy
    frame (time, level_m) or None when the station has no data.
    """
    import json
    import os
    import time
    import urllib.request

    import pandas as pd

    from .gauges import load_cached_ioc

    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, f"ioc_{code}_{t0}_{t1}.json")
    if not os.path.exists(out):
        raw = []
        edges = list(pd.date_range(t0, t1, freq=f"{chunk_days}D")) + [
            pd.Timestamp(t1) + pd.Timedelta(days=1)]
        for a, b in zip(edges, edges[1:]):
            url = (f"{IOC_BASE}?query=data&code={code}"
                   f"&timestart={a.strftime('%Y-%m-%dT%H:%M:%S')}"
                   f"&timestop={b.strftime('%Y-%m-%dT%H:%M:%S')}&format=json")
            for _ in range(3):
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "pyintertidal"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        chunk = json.loads(r.read().decode("utf-8", "replace"))
                    if isinstance(chunk, list):
                        raw += chunk
                    break
                except Exception:
                    time.sleep(5)
            time.sleep(0.3)
        with open(out, "w") as f:
            json.dump(raw, f)
    return load_cached_ioc(code, cache_dir=cache_dir)


class InterpolatedGauges(BoundaryProvider):
    """The N nearest gauges, demeaned and blended by inverse distance.

    Demeaning removes each harbour's datum; the blend keeps each gauge's
    own amplitude and phase weighted by 1/d^2, so the nearest dominates
    and a far one only nudges. Members that lack data at an instant simply
    drop out of the average there.
    """

    def __init__(self, gauges, distances_km, name="gauges"):
        self.gauges = list(gauges)
        d = np.maximum(np.asarray(distances_km, float), 1.0)
        self.weights = (1.0 / d ** 2) / np.sum(1.0 / d ** 2)
        self.name = name

    def levels(self, times):
        stack = np.vstack([g.levels(times) for g in self.gauges])
        w = self.weights[:, None] * np.isfinite(stack)
        num = np.nansum(np.nan_to_num(stack) * w, axis=0)
        den = w.sum(axis=0)
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)

    def harmonics(self, constituents=None):
        return EnsembleBoundary(self.gauges).harmonics(constituents)


def make_boundary(aoi, tide_model="EOT20", n_gauges=None,
                  period=("2023-01-01", "2025-12-31"), directory="tide_models",
                  cache_dir="data_v4/gauges", exclude=(), verbose=True):
    """The boundary level every estimator consumes, from two switches.

    ``tide_model``  a pyTMD model name at the site, or None.
    ``n_gauges``    the N nearest IOC gauges (fetched and cached for
                    ``period``), inverse-distance blended, or None.
    Both given → the CONSENSUS (mean of the two); both None → error.
    ``exclude``     gauge codes never used, however near — the judges a
                    validation holds out (e.g. the inner gauge of an estuary).
    Returns a provider with ``.levels(times)`` and ``.harmonics()``.
    """
    from .aoi import as_aoi

    aoi = as_aoi(aoi)
    lat, lon = aoi.centroid          # AOI.centroid is (lat, lon)
    parts = []
    if tide_model:
        parts.append(PyTMDBoundary(tide_model, lat, lon, directory=directory))
    if n_gauges:
        members, dists = [], []
        excl = {str(c).lower() for c in exclude}
        near = [g for g in nearest_gauges(lat, lon, n_gauges + len(excl))
                if g[0].lower() not in excl][:int(n_gauges)]
        for code, name, km in near:
            df = fetch_ioc(code, period[0], period[1], cache_dir=cache_dir)
            if df is None or len(df) < 1000:
                if verbose:
                    print(f"[boundary] gauge {code} ({name}): no data, skipped")
                continue
            from .gauges import despike
            n_raw = len(df)
            df = despike(df)
            lev = df["level_m"].to_numpy(float)
            members.append(GaugeBoundary(df["time"], lev - np.nanmedian(lev),
                                         name=code))
            dists.append(km)
            if verbose:
                print(f"[boundary] gauge {code} ({name}) at {km:.0f} km: "
                      f"{len(df):,} samples ({n_raw - len(df)} spikes "
                      f"dropped)")
        if members:
            parts.append(InterpolatedGauges(
                members, dists,
                name=(members[0].name if len(members) == 1 else
                      f"{len(members)} gauges ("
                      + "+".join(m.name for m in members) + ")")))
    if not parts:
        raise ValueError("make_boundary needs a tide model, gauges, or both")
    if len(parts) == 1:
        return parts[0]
    return EnsembleBoundary(parts, name="consensus " + " + ".join(
        getattr(q, "name", "?") for q in parts))
