# -*- coding: utf-8 -*-
"""v13 benchmark — the common input package of a site (§1.4, §2.2, §2.5-2.11).

Question
--------
What exactly does every method receive at a site? One package per site and
grid, built once, hashed and then only read: the scene record, the boundary
levels at the record instants (exact, for every integer lag the methods can
use), the 1-min boundary series and its marks, the raw band values and SCL
at every pixel any method's per-pixel domain can reach, the full-frame
validity bits, the full-record counters and geometry, and the record subsets
every method shares. Methods read observations only through it
(``obs.observe`` on the stored raw values), so the mask rule and the index
mode stay run-time options and are never baked into the package.

Pre-registered rules (FROZEN['record'], ['package'], ['levels'], ['subsets'])
-----------------------------------------------------------------------------
* Record (C1, §2.2): every cube date of 2023-01-01..2025-12-31 with an
  overpass time (``v8.to_utc_naive``) and a finite boundary level, the
  levels from ONE provider call on the sorted candidate instants. The
  extraction's dates must be the cube's (``v7.cube_grid``).
* Two streaming passes over the record scenes, one full frame of raw int16
  B03, B08 and SCL at a time (``mask_and_scale=False``):
  - pass 1: the primary counters ``clr_n``, ``wet_n`` and the packed frame
    validity bits (marea_clear + floor0, FROZEN['package']['counters'] and
    ['frame_valid']), the U_rec counters (blocklist + floor0) and the scene
    statistics;
  - pass 2: raw B03 and B08 (int16) and SCL (uint8, fill -> 255) at the
    package domain, each scene checked against its pass-1 validity bits.
* Package domain (v2 spec §2.10; sorted unique flat indices):
  ``dom = S0 ∪ dilate(U_rec, 3 px) ∪ keep_prod`` with the evaluation
  support ``S0 = NB ∪ dilate(keep_full, 300 m)`` (dilations by a disk on the
  pixel centres, :func:`dilate_disk`). keep_full = the full-record domain
  under the primary observation (P0); ``U_rec = {clr >= 31, wet >= 4,
  dry >= 4}`` over the full record under blocklist + floor0, a superset of
  every A-mode keep under either mask rule, hence of U_rec of the curve
  settings P0 and C2; NB = the frame pixels of the judges' neighbourhoods
  (FROZEN['package']['NB'], the judges' rule R = max(500, d_min + 300) m,
  widened by 250 m to >= 100 keep_full pixels while R < 2,000 m), for every
  listed station inside the frame; keep_prod = the production extraction's
  keep (legacy gates). Ω (the D2 domain) lies inside dilate(keep_full,
  300 m). Not yet here: D_truth in S0 (the v2 truth module freezes it at
  lock-A) and U_rec of C11/C3 (the v2 stores carry CLD).
* Full-record geometry (§2.6, ``domain.from_counts``), its strata on the
  whole frame (``domain.band_map_frame``) and the production extraction's
  keep, sea and inter.
* Level table (§2.7, ``levels.build_level_table``): integer tau in
  [-90, max(240, 180 (K_full - 1))] with the G-levels check; LEV[0] equals
  the record's h0 bitwise.
* Subsets (§2.9): rand ``np.sort(default_rng([20261008, SITE_ID, r])
  .permutation(T)[:N])``; win ``arange(s, s + N)`` with s =
  ``default_rng([20261012, SITE_ID, N, r]).integers(0, T - N + 1)``; rules
  tz (the 10 m MAREA transition-zone record_dates), g21 (frame cloud < 0.10
  and no data <= 0.02) and ard (load_ard good fraction >= 0.5).
* Keys (§2.11): ``package_key`` = sha256 of the sha256 of every keyed
  package file, of the inputs (cube, extraction, overpass, IOC chunks, the
  judges' coordinates, the transition record), ``obs_sha`` and the FROZEN
  shas of record, package, obs, domain, levels, subsets
  (``product.package_key``). Git is provenance only.
* Every keyed file is deterministic (no times, hosts or durations; JSON and
  CSV with LF line ends; byte-stable npz) and published atomically, so the
  same inputs give the same package key on the laptop and on the server.

Steps
-----
``record, series, pass1, geometry, judges, levels, pass2, subsets,
manifest``, resumable: ``steps.json`` records each step's outputs (size,
mtime_ns, sha256) and the sha256 of the package files it read; a step whose
outputs changed or whose inputs were rebuilt is stale and runs again. Any
step that runs removes ``manifest.json`` first, so a half-updated package
can never be loaded. ``judges.json`` (the score-side judges cache written by
``judges.build``) is not a package file and is not in the manifest.

Validation v2 (docs/validation_v2_spec.md §2.10, §13.1-13.4; the section at
the end of this module, :func:`prepare_v2`, :class:`V2Package`)
-------------------------------------------------------------------------
* Inputs through ``inputs.ZarrSource`` (per-scene BOA offsets, lattice20);
  the record holds the shard years with a finite h_B. Scene instants: the
  catalogue / product-name times (the datatake start) by default, as every
  production result (user decision of 2026-10-14); a ``sensing_times.json``
  (the optional solar-angle diagnostic) only when its path is given
  (``V2Site.instants``; CLI ``--sensing-times``), and then its sha256
  enters the shard keys.
* Local software checks (never reported numbers): ``V2Site(source=
  "netcdf")`` builds the same package from a legacy netCDF cube (B03, B08,
  SCL; ``v2_site(..., res=20)``) with the production boundary provider over
  2023-2025 (``boundary="production"``). Only the observation keys whose
  bands the source delivers are counted (the rest are listed in
  record.json), the OCM and C3 steps cannot run (no B04/B8A, no CLD), dom
  is S0 ∪ dilate(U_rec of P0), and the tiles hold B03, B08 and SCL.
* Pass 1 per YEAR: uint8 counters (H, W) per observation key of the
  planned settings (``designs.observation_keys``; no year has more than
  255 scenes), C3 at both candidate cut-offs (p = 20 and 35, before the
  cross-tab decides), frame bits (P0 V, SCL bad, the production wet/dry of
  the C2 transition zone, C3), SCL x CLD cross-tabs and scene statistics.
  The C3/C5 neighbourhoods are computed with a 16-row halo, so a row block
  equals the frame computation.
* The OmniCloudMask step (C11) per year on the bounding box of the pre-dom
  plus 1 km: clear bits, C11 counters and frame bits (the C1 analog
  outside the region, declared); package version and licence recorded.
* S0 = D_truth ∪ NB ∪ dilate(keep_full(P0, W3), 300 m) ∪ Omega; dom = S0 ∪
  dilate(U_rec(W3) of P0 (C2 inside) and C11 (C3 without OmniCloudMask),
  3 px), closed under the 20 m lattice blocks; dom_version = sha256 of dom
  and its rule; frozen at lock-A (``freeze_dom``), never replaced after.
* Pass 2 per year and dom_version: pixel-major tiles (n_tiles, T_Y, 4096)
  per band (B02, B03, B04, B08, B8A, B11, B12 int16; SCL, CLD uint8) and
  one uint8 of cloud bits per observation; each shard has its own
  manifest and package_key (record slice, files, obs core, FROZEN shas,
  store fingerprint, IOC chunks, the sensing file if one was given,
  dom_version).
* The settings pass (t_NIR, s_k from per-year histograms on keep_full; D2
  on Omega and D3 on the bounding box of dom plus 1 km, per scene), the
  at-dom counters of the threshold detectors, the MNAR table, the subsets
  per window and the level table.

Memory: pass 1 holds one frame of three int16 bands, four int16 counters and
the float32 NDWI (about 0.5 GB at the Ems 10 m frame of 10.9 M px); the
(T, P_dom) arrays are written row by row into ``np.lib.format.open_memmap``
files; nothing ever holds a whole cube or a whole (T, P_dom) array.
"""
from __future__ import annotations

import errno
import gc
import glob
import hashlib
import json
import os
import re
import shutil
import socket
import time
from functools import cached_property

import numpy as np
import pandas as pd

from experiments.benchmark import config, domain, levels, obs, product, settings

FORMAT = "v13-package-1"
RECORD_FORMAT = "v13-record-1"
PIXELS_FORMAT = "v13-pixels-1"
COUNTS_FORMAT = "v13-counts-1"
NB_FORMAT = "v13-nb-2"
SUBSETS_FORMAT = "v13-subsets-1"
STEPS_FORMAT = "v13-steps-1"

STEPS = ("record", "series", "pass1", "geometry", "judges", "levels", "pass2",
         "subsets", "manifest")

RECORD_JSON = "record.json"
SERIES_CACHE = "boundary_1min_cache.npz"
MARKS_NPZ = "marks.npz"
COUNTS_NPZ = "counts.npz"
FRAME_VALID_NPY = "frame_valid.npy"
SCENE_STATS_CSV = "scene_stats.csv"
PIXELS_NPZ = "pixels.npz"
NB_NPZ = "nb.npz"
DOM_NPY = "dom.npy"
LEVELS_NPZ = "levels.npz"
B03_NPY = "b03.npy"
B08_NPY = "b08.npy"
SCL_NPY = "scl.npy"
SUBSETS_NPZ = "subsets.npz"
MANIFEST_JSON = "manifest.json"
#: judges.build's cache (score side; not a package file, not in the manifest)
JUDGES_JSON = "judges.json"
#: step bookkeeping (not a package file)
STEPS_JSON = "steps.json"
LOCK_FILE = "prepare.lock"

#: the files each step publishes
STEP_OUTPUTS = {
    "record": (RECORD_JSON,),
    "series": (SERIES_CACHE, MARKS_NPZ),
    "pass1": (COUNTS_NPZ, FRAME_VALID_NPY, SCENE_STATS_CSV),
    "geometry": (PIXELS_NPZ,),
    "judges": (NB_NPZ, DOM_NPY),
    "levels": (LEVELS_NPZ,),
    "pass2": (B03_NPY, B08_NPY, SCL_NPY),
    "subsets": (SUBSETS_NPZ,),
    "manifest": (MANIFEST_JSON,),
}
#: the package files the manifest keys (every step output but the manifest)
KEYED_FILES = tuple(f for s in STEPS[:-1] for f in STEP_OUTPUTS[s])
#: the package files each step reads (staleness and prerequisites)
STEP_READS = {
    "record": (),
    "series": (RECORD_JSON,),
    "pass1": (RECORD_JSON, MARKS_NPZ),
    "geometry": (RECORD_JSON, COUNTS_NPZ),
    "judges": (RECORD_JSON, COUNTS_NPZ, PIXELS_NPZ),
    "levels": (RECORD_JSON, PIXELS_NPZ),
    "pass2": (RECORD_JSON, DOM_NPY, FRAME_VALID_NPY),
    "subsets": (RECORD_JSON, SCENE_STATS_CSV),
    "manifest": KEYED_FILES,
}
_PRODUCER = {f: s for s, fs in STEP_OUTPUTS.items() for f in fs}

_P = config.FROZEN["package"]
#: the primary observation of the counters and the frame bits (D1, D2)
PRIMARY_MASK = str(_P["frame_valid"]["mask"])
PRIMARY_INDEX = str(_P["frame_valid"]["index"])
#: the observation of the package-domain counters U_rec
UREC_MASK = str(_P["U_rec"]["mask"])
UREC_INDEX = str(_P["U_rec"]["index"])
UREC_MIN_CLEAR = int(_P["U_rec"]["min_clear"])
UREC_MIN_WET = int(_P["U_rec"]["min_wet"])
UREC_MIN_DRY = int(_P["U_rec"]["min_dry"])
#: the judges' neighbourhood rule as frozen in the package section
NB_RULE = {k: v for k, v in _P["NB"].items() if k != "rule"}
#: S0 holds keep_full dilated by this radius (m), dom U_rec dilated by this
#: many pixels (v2 §2.10)
S0_KEEP_DILATION_M = float(_P["S0"]["keep_dilation_m"])
DOM_UREC_DILATION_PX = float(_P["dom_rule"]["urec_dilation_px"])
if (PRIMARY_MASK, PRIMARY_INDEX) != (obs.MASK_RULE, obs.INDEX_MODE):
    raise AssertionError("FROZEN['package']['frame_valid'] is not the primary observation")
if (_P["counters"]["mask"], _P["counters"]["index"]) != (PRIMARY_MASK, PRIMARY_INDEX):
    raise AssertionError("the package counters and the frame bits use different observations")
if UREC_INDEX != PRIMARY_INDEX:
    # one NDWI per frame serves both rules in pass 1, and the superset
    # argument (marea_clear counts <= blocklist counts) needs one index
    raise AssertionError("U_rec must use the primary index mode")
del _P

_S = config.FROZEN["subsets"]
SEED_SUBSETS = int(_S["rand"]["seed"])
SEED_WINDOWS = int(_S["win"]["seed"])
#: rand permutations stored in subsets.npz (more reps are drawn on the fly)
STORED_REPS = int(_S["rand"]["stored_reps"])
RULES = ("tz", "g21", "ard")
FAMILY_RULE = {f"rule-{r}": r for r in RULES}
del _S

#: the ``sens`` keys that change the observation (held at keep_full)
SCENE_STATS_LOG_EVERY = 25
_SCL_CODES = list(range(12))


def log(msg):
    print(msg, flush=True)


class PackageError(RuntimeError):
    """A package is missing, incomplete, inconsistent or tampered with."""


class StalePackageError(PackageError):
    """The package was built with another obs.py or other FROZEN package
    sections: it must be rebuilt (``prepare --force``)."""


# ─────────────────────────────────────────────────────────────────────────────
#  small IO helpers (atomic, deterministic)
# ─────────────────────────────────────────────────────────────────────────────

def _tmp(path):
    return f"{path}.tmp-{socket.gethostname()}-{os.getpid()}"


def _write_bytes(path, data):
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = _tmp(path)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "items"):
        return dict(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def write_json_lf(path, obj):
    """JSON with sorted keys and LF line ends, written atomically: the same
    content gives the same bytes on every machine (a keyed file)."""
    text = json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False,
                      default=_json_default) + "\n"
    _write_bytes(path, text.encode("utf-8"))


def _save_npy(path, arr):
    tmp = _tmp(path)
    try:
        with open(tmp, "wb") as f:
            np.lib.format.write_array(f, np.asarray(arr), allow_pickle=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _discard_after_failure(paths, exc):
    """Remove the temporary files of a failed pass without masking the error.

    Windows cannot delete a file that a live memory map still holds
    (WinError 32). The caller has dropped its own names for the maps; a map
    can still be held by a frame of the traceback (an error raised inside
    ``np.memmap.flush`` keeps the map as that frame's ``self``), so the
    locals of the finished frames are cleared and garbage is collected
    first. A file that still cannot be removed is reported and left in
    place: the caller re-raises the original error, never this one.
    """
    import traceback

    traceback.clear_frames(exc.__traceback__)
    gc.collect()
    for p in paths:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError as e:
                log(f"[package] warning: could not remove {p} after a failure ({e})")


def memmap_bytes(shape, dtype):
    """Bytes of an ``open_memmap`` file of this shape and dtype (data plus a
    4 KiB allowance for the .npy header)."""
    return int(np.prod([int(v) for v in shape], dtype=np.int64)) * np.dtype(dtype).itemsize \
        + 4096


def space_margin_gb():
    """The free-space margin of every pass (:data:`SPACE_MARGIN_GB`), or
    ``$VALIDATION_V2_SPACE_MARGIN_GB`` (local scratch checks on a small disk;
    the server keeps the default). Read at call time."""
    env = os.environ.get("VALIDATION_V2_SPACE_MARGIN_GB")
    return float(env) if env not in (None, "") else float(SPACE_MARGIN_GB)


def check_space(directory, need_bytes, what, margin_gb=None):
    """Refuse a step whose files would not fit: the free space of
    ``directory`` must exceed ``need_bytes`` plus the margin
    (:func:`space_margin_gb`; spec §14.4). A memory-mapped write that hits a
    full disk is a SIGBUS on Linux, not an exception, so this runs before
    any file is created."""
    margin = float(space_margin_gb() if margin_gb is None else margin_gb) * 1e9
    d = os.path.abspath(directory)
    while not os.path.isdir(d):
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    free = shutil.disk_usage(d).free
    if free < int(need_bytes) + margin:
        raise PackageError(
            f"{what} writes {need_bytes / 1e9:.2f} GB into {directory}, but only "
            f"{free / 1e9:.2f} GB are free there (need {need_bytes / 1e9:.2f} GB plus a "
            f"{margin / 1e9:.0f} GB margin): free space, or move the output root "
            "(VALIDATION_V2_ROOT)")
    return free


def reserve_file(path):
    """Allocate a memory-mapped file's blocks now (``os.posix_fallocate``),
    so a full disk raises here (:class:`PackageError`, inside the caller's
    cleanup path) instead of a SIGBUS when a page is written. No-op where
    the call does not exist (Windows); other errors (a filesystem without
    fallocate support) are logged and ignored."""
    fa = getattr(os, "posix_fallocate", None)
    if fa is None:
        return False
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDWR)
    try:
        fa(fd, 0, size)
    except OSError as e:
        if e.errno == errno.ENOSPC:
            raise PackageError(f"no space left to allocate {path} ({size / 1e9:.2f} GB)") from e
        log(f"[package] warning: posix_fallocate({path}) failed ({e}); continuing")
        return False
    finally:
        os.close(fd)
    return True


#: ``<file>.tmp-<host>-<pid>`` (:func:`_tmp`), optionally with an extension
_TMP_RE = re.compile(r"\.tmp-(?P<host>.+)-(?P<pid>\d+)(?:\.[A-Za-z0-9]+)?$")


def clean_stale_tmps(directory, recursive=False):
    """Remove the temporary files of killed runs in ``directory``:
    ``*.tmp-<this host>-<pid>`` whose process no longer exists (SIGKILL, a
    full disk, a reboot or a closed terminal skip the Python cleanup). Files
    of live processes and of other hosts are kept. Returns the removed
    paths."""
    if not os.path.isdir(directory):
        return []
    try:
        import psutil
    except ImportError:                       # cannot tell a dead pid: keep everything
        return []
    host = socket.gethostname()
    pattern = os.path.join(directory, "**", "*.tmp-*") if recursive \
        else os.path.join(directory, "*.tmp-*")
    out = []
    for p in glob.glob(pattern, recursive=recursive):
        m = _TMP_RE.search(os.path.basename(p))
        if not m or m.group("host") != host or not os.path.isfile(p):
            continue
        pid = int(m.group("pid"))
        if pid == os.getpid() or psutil.pid_exists(pid):
            continue
        try:
            size = os.path.getsize(p)
            os.remove(p)
            out.append(p)
            log(f"[package] removed the leftover {p} ({size / 1e9:.2f} GB) of the dead "
                f"process {pid}")
        except OSError as e:
            log(f"[package] warning: could not remove the leftover {p} ({e})")
    return out


def sha256_raw(path, block=1 << 22):
    """sha256 of a file's bytes (inputs: no content hashing of npz)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _abs(path):
    """A path relative to the repository root (production) or absolute."""
    path = os.fspath(path)
    return path if os.path.isabs(path) else os.path.join(config.ROOT, path)


def _rel(path):
    a = os.path.abspath(_abs(path))
    try:
        r = os.path.relpath(a, config.ROOT)
    except ValueError:                       # another drive on Windows
        return a.replace("\\", "/")
    return a.replace("\\", "/") if r.startswith("..") else r.replace("\\", "/")


def input_entry(path, sha=True):
    """Manifest entry of an input file: path, size, mtime_ns, raw sha256."""
    p = _abs(path)
    st = os.stat(p)
    e = {"path": _rel(p), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    if sha:
        e["sha256"] = sha256_raw(p)
    return e


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:                        # psutil absent: not worth failing for
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
#  the cube and the production extraction
# ─────────────────────────────────────────────────────────────────────────────

def open_cube(path):
    """``(dataset, t_dim)`` of a cube opened RAW (``mask_and_scale=False``);
    B03, B08 and SCL must be (t, y, x)."""
    import xarray as xr

    ds = xr.open_dataset(_abs(path), mask_and_scale=False)
    try:
        t_dim = [k for k in ds["B03"].dims if k not in ("x", "y")][0]
        for v in ("B03", "B08", "SCL"):
            if ds[v].dims != (t_dim, "y", "x"):
                raise ValueError(f"{path}: {v} dims {ds[v].dims} are not (t, y, x)")
    except BaseException:
        ds.close()
        raise
    return ds, t_dim


def cube_encoding(path):
    """The raw encoding of a cube: int16 bands with a ``_FillValue`` and no
    scale or offset (``v8.band_encoding``, asserted), and its sizes."""
    from experiments import v8_granadeiro2021 as v8

    ds, t_dim = open_cube(path)
    try:
        fill = {v: int(v8.band_encoding(ds, v)) for v in ("B03", "B08", "SCL")}
        return {"t_dim": t_dim, "T": int(ds.sizes[t_dim]), "H": int(ds.sizes["y"]),
                "W": int(ds.sizes["x"]), "dtype": "int16", "fill": fill}
    finally:
        ds.close()


def read_scene(ds, t_dim, t):
    """Raw (H, W) int16 B03, B08 and SCL of cube scene ``t``."""
    sl = {t_dim: int(t)}
    return tuple(np.asarray(ds[v].isel(sl).values) for v in ("B03", "B08", "SCL"))


def extraction_header(cfg):
    """The production extraction's header arrays (dates, keep, shape, sea,
    inter), read member by member without pickles; None without an
    extraction (the positive control)."""
    if cfg.extract is None:
        return None
    with np.load(_abs(cfg.extract), allow_pickle=False) as z:
        out = {k: np.asarray(z[k]) for k in ("dates", "keep", "shape", "sea", "inter")}
    out["dates"] = np.array([str(d) for d in out["dates"]])
    out["keep"] = out["keep"].astype(np.int64)
    out["shape"] = tuple(int(v) for v in out["shape"])
    H, W = out["shape"]
    k = out["keep"]
    if k.size and (k[0] < 0 or k[-1] >= H * W or np.any(np.diff(k) <= 0)):
        raise ValueError(f"{cfg.extract}: keep is not sorted unique flat indices")
    for m in ("sea", "inter"):
        if out[m].shape != (H, W):
            raise ValueError(f"{cfg.extract}: {m} has shape {out[m].shape}, not {(H, W)}")
        out[m] = out[m].astype(bool)
    return out


def read_overpass(path):
    """``{date: naive UTC Timestamp}`` of an overpass_times.json
    (``v8.to_utc_naive``)."""
    from experiments import v8_granadeiro2021 as v8

    with open(_abs(path), encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): v8.to_utc_naive(v) for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  the record (C1)
# ─────────────────────────────────────────────────────────────────────────────

def build_record(cfg, provider, ext_dates):
    """The common record (§2.2).

    Parameters
    ----------
    cfg : config.SiteConfig
        Its ``overpass`` file gives the acquisition instants.
    provider : object with ``levels(times)``
        The site's boundary provider.
    ext_dates : array of 'YYYY-MM-DD'
        The cube's dates (the production extraction's, asserted equal).

    Returns
    -------
    dict
        ``time_index`` (cube positions, int64), ``dates``, ``t`` (naive UTC
        DatetimeIndex, ns, strictly increasing), ``h0`` (float64, from ONE
        provider call on the sorted candidate instants), the excluded dates
        by reason and the counts.
    """
    ext_dates = np.array([str(d) for d in np.asarray(ext_dates)])
    if len(ext_dates) > 1 and np.any(ext_dates[1:] <= ext_dates[:-1]):
        raise ValueError("the cube dates must be strictly increasing")
    over = read_overpass(cfg.overpass)
    p0, p1 = config.PERIOD
    in_period = (ext_dates >= p0) & (ext_dates <= p1)
    has_time = np.array([d in over for d in ext_dates], bool)
    cand = np.flatnonzero(in_period & has_time)
    if not len(cand):
        raise ValueError(f"{cfg.site}: no cube date in {config.PERIOD} has an overpass time")
    t = pd.DatetimeIndex([over[d] for d in ext_dates[cand]])
    t = (t.tz_convert("UTC").tz_localize(None) if t.tz is not None else t).as_unit("ns")
    if not (np.diff(t.asi8) > 0).all():
        raise ValueError("the overpass instants of the record are not strictly increasing")
    # ONE call on the sorted candidate instants (PyTMDBoundary answers in
    # time order; an ensemble averages its members only on aligned input)
    h = np.asarray(provider.levels(t), np.float64)
    if h.shape != (len(t),):
        raise AssertionError(f"provider.levels returned shape {h.shape} for {len(t)} instants")
    fin = np.isfinite(h)
    time_index = cand[fin].astype(np.int64)
    return {"time_index": time_index, "dates": ext_dates[time_index], "t": t[fin],
            "h0": h[fin],
            "exclusions": {
                "outside_period": ext_dates[~in_period].tolist(),
                "without_overpass": ext_dates[in_period & ~has_time].tolist(),
                "nonfinite_level": ext_dates[cand[~fin]].tolist()},
            "n_cube_dates": int(len(ext_dates)), "n_candidates": int(len(cand))}


# ─────────────────────────────────────────────────────────────────────────────
#  pass 1: counters, frame validity bits, scene statistics
# ─────────────────────────────────────────────────────────────────────────────

SCENE_STATS_COLUMNS = (
    ["q", "t_index", "date", "utc", "h0", "rising"]
    + [f"scl_{c}" for c in _SCL_CODES] + ["scl_nodata"]
    + [f"keep_scl_{c}" for c in _SCL_CODES] + ["keep_scl_nodata"]
    + ["g21_cloud", "g21_nodata", "ard_good", "v_frame", "v_keep_prod",
       "v_frame_blocklist", "n_b03_neg", "n_b08_neg", "n_both_le0",
       "n_raw_abs_ndwi_gt1"])


def _share(mask):
    return float(np.count_nonzero(mask)) / mask.size if mask.size else float("nan")


def scene_stats_row(q, t_index, date, utc, h0, rising, g, n, s_raw, s8, v_primary,
                    v_blocklist, keep_prod, fill, fill_scl):
    """One scene's statistics (§2.5), every value on the whole frame unless
    named ``keep_*`` (the production keep).

    SCL counts 0..11 and no data; ``g21.scene_frame_shares`` (cloud, no
    data); ``wc.frame_good_fraction`` (load_ard); the shares of V under the
    primary observation (frame, keep_prod) and under blocklist (frame); the
    non-fill negative B03 / B08 counts, both bands <= 0 (undefined under
    floor0) and raw |NDWI| > 1.
    """
    from pyintertidal import granadeiro2021 as g21
    from pyintertidal import waterline_contour as wc

    cnt = obs.scl_counts(s8)
    if keep_prod.size:
        sk = s8.ravel()[keep_prod]
        cnt_k = obs.scl_counts(sk)
        v_keep = _share(v_primary.ravel()[keep_prod])
    else:
        cnt_k = np.zeros(13, np.int64)
        v_keep = float("nan")
    cloud, nodata = g21.scene_frame_shares(s_raw, fill_scl)
    gp, np_ = g != fill, n != fill
    with np.errstate(invalid="ignore"):
        nd_raw = obs.ndwi(*obs.bands(g, n, fill, "raw"))
        gt1 = int(np.count_nonzero(np.abs(nd_raw) > 1))
    row = {"q": int(q), "t_index": int(t_index), "date": str(date), "utc": str(utc),
           "h0": float(h0), "rising": int(bool(rising))}
    row.update({f"scl_{c}": int(cnt[c]) for c in _SCL_CODES})
    row["scl_nodata"] = int(cnt[12])
    row.update({f"keep_scl_{c}": int(cnt_k[c]) for c in _SCL_CODES})
    row["keep_scl_nodata"] = int(cnt_k[12])
    row.update({"g21_cloud": float(cloud), "g21_nodata": float(nodata),
                "ard_good": float(wc.frame_good_fraction(s8)),
                "v_frame": _share(v_primary), "v_keep_prod": v_keep,
                "v_frame_blocklist": _share(v_blocklist),
                "n_b03_neg": int(np.count_nonzero(gp & (g < 0))),
                "n_b08_neg": int(np.count_nonzero(np_ & (n < 0))),
                "n_both_le0": int(np.count_nonzero(gp & np_ & (g <= 0) & (n <= 0))),
                "n_raw_abs_ndwi_gt1": gt1})
    return row


def _write_csv_lf(path, rows, columns):
    df = pd.DataFrame(rows, columns=list(columns))
    _write_bytes(path, df.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def read_scene_stats(path):
    """``scene_stats.csv`` with every float exactly as pass 1 computed it.

    ``to_csv`` writes the shortest repr that round-trips, but pandas' default
    C float parser can land one ulp away from it (73/720 = 0.10138888888888889
    reads back as 0.1013888888888888), and the scene rules compare these
    shares with their thresholds (g21: cloud < 0.10). ``float_precision=
    "round_trip"`` parses with Python's own conversion, which is exact.
    """
    return pd.read_csv(path, float_precision="round_trip")


def _rising(t, marks_lp):
    """Direction of the tide at each record instant from the low-pass marks
    (``g21.eq1_bracket``: the previous mark is a low water)."""
    from pyintertidal import granadeiro2021 as g21

    return np.asarray(g21.eq1_bracket(levels.minutes_since_epoch(t), marks_lp)["rising"],
                      bool)


def pass1(cfg, record, out_dir, keep_prod=None, marks_lp=None, log_every=SCENE_STATS_LOG_EVERY):
    """Pass 1 over the record scenes (§2.5).

    Writes ``counts.npz`` (``clr_n``, ``wet_n`` under the primary observation
    and ``clr_n_blocklist``, ``wet_n_blocklist`` under U_rec's, int16
    (H, W)), ``frame_valid.npy`` ((T, ceil(H W / 8)) uint8, ``np.packbits``
    of the primary V of each record scene) and ``scene_stats.csv`` into
    ``out_dir``.

    Parameters
    ----------
    cfg : config.SiteConfig
    record : dict
        The content of record.json (``time_index``, ``dates``, ``t_ns``,
        ``h0``, ``encoding``, ``grid``).
    out_dir : str
    keep_prod : array, optional
        Production keep (flat indices) for the keep statistics.
    marks_lp : dict, optional
        Low-pass marks for the ``rising`` column (NaN-free required).

    Returns
    -------
    dict
        Summary (counts of observations, seconds); not part of any key.
    """
    t0 = time.time()
    time_index = np.asarray(record["time_index"], np.int64)
    T = len(time_index)
    H, W = int(record["grid"]["H"]), int(record["grid"]["W"])
    fill = int(record["encoding"]["fill"]["B03"])
    fill_scl = int(record["encoding"]["fill"]["SCL"])
    if T >= np.iinfo(np.int16).max:
        raise ValueError(f"{T} record scenes do not fit the int16 counters")
    keep_prod = np.zeros(0, np.int64) if keep_prod is None else np.asarray(keep_prod, np.int64)
    t = pd.DatetimeIndex(np.asarray(record["t_ns"], np.int64).view("M8[ns]"))
    rising = (_rising(t, marks_lp) if marks_lp is not None
              else np.zeros(T, bool))
    counters = {k: np.zeros((H, W), np.int16)
                for k in ("clr_n", "wet_n", "clr_n_blocklist", "wet_n_blocklist")}
    fv_path = os.path.join(out_dir, FRAME_VALID_NPY)
    fv_tmp = _tmp(fv_path)
    fv = np.lib.format.open_memmap(fv_tmp, mode="w+", dtype=np.uint8,
                                   shape=(T, (H * W + 7) // 8))
    rows = []
    ds, t_dim = open_cube(cfg.cube)
    try:
        if (ds.sizes["y"], ds.sizes["x"]) != (H, W):
            raise ValueError("the cube's frame differs from the record's grid")
        for q, ti in enumerate(time_index):
            g, n, s = read_scene(ds, t_dim, ti)
            obs.check_boa(g, fill, "B03")
            obs.check_boa(n, fill, "B08")
            s8 = obs.scl_u8(s, fill_scl)
            nd, V = obs.observe_rules(g, n, s8, fill, mode=PRIMARY_INDEX,
                                      rules=(PRIMARY_MASK, UREC_MASK))
            vp, vb = V[PRIMARY_MASK], V[UREC_MASK]
            counters["clr_n"] += vp
            counters["wet_n"] += obs.wet(nd, vp)
            counters["clr_n_blocklist"] += vb
            counters["wet_n_blocklist"] += obs.wet(nd, vb)
            fv[q] = np.packbits(vp.ravel())
            rows.append(scene_stats_row(
                q, ti, record["dates"][q], record["t_utc"][q], record["h0"][q], rising[q],
                g, n, s, s8, vp, vb, keep_prod, fill, fill_scl))
            del g, n, s, s8, nd, V, vp, vb
            if (q + 1) % log_every == 0 or q + 1 == T:
                log(f"  [pass 1] {q + 1}/{T} scenes ({time.time() - t0:.0f} s, "
                    f"RSS {_rss_mb():.0f} MB)")
        fv.flush()
    except BaseException as exc:
        ds.close()
        del fv
        _discard_after_failure([fv_tmp], exc)
        raise
    ds.close()
    del fv
    gc.collect()
    meta = {"format": COUNTS_FORMAT, "T": T,
            "primary": {"mask": PRIMARY_MASK, "index": PRIMARY_INDEX},
            "blocklist": {"mask": UREC_MASK, "index": UREC_INDEX}}
    product.save_npz(os.path.join(out_dir, COUNTS_NPZ),
                     {"meta": np.array(json.dumps(meta, sort_keys=True)), **counters})
    _write_csv_lf(os.path.join(out_dir, SCENE_STATS_CSV), rows, SCENE_STATS_COLUMNS)
    os.replace(fv_tmp, fv_path)
    return {"T": T, "n_obs_primary": int(counters["clr_n"].sum(dtype=np.int64)),
            "n_obs_blocklist": int(counters["clr_n_blocklist"].sum(dtype=np.int64)),
            "seconds": round(time.time() - t0, 1)}


# ─────────────────────────────────────────────────────────────────────────────
#  the package domain: U_rec and the judges' neighbourhoods
# ─────────────────────────────────────────────────────────────────────────────

def urec_mask(clr_b, wet_b):
    """U_rec = {clr >= 31, wet >= 4, dry >= 4} on the blocklist counters."""
    clr = np.asarray(clr_b).astype(np.int64)
    wet = np.asarray(wet_b).astype(np.int64)
    return (clr >= UREC_MIN_CLEAR) & (wet >= UREC_MIN_WET) & (clr - wet >= UREC_MIN_DRY)


def station_coordinates(cfg, stations_path=None):
    """``[(code, lon, lat)]`` of the site's listed judges (cfg.judges) from
    ``estuary_truth/stations.json`` (or ``cfg.extra['stations_json']``)."""
    codes = tuple(cfg.judges)
    if not codes:
        return [], None
    path = stations_path or cfg.extra.get("stations_json", config.STATIONS_JSON)
    p = _abs(path)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{path} is missing: copy data_v4/gauges/estuary_truth/ to this machine "
            "(spec §7.5, step 0); the package domain holds the judges' neighbourhoods")
    with open(p, encoding="utf-8") as f:
        st = {s["code"]: s for s in json.load(f)}
    missing = [c for c in codes if c not in st]
    if missing:
        raise KeyError(f"{path} has no station {missing}")
    return [(c, float(st[c]["lon"]), float(st[c]["lat"])) for c in codes], path


def station_xy(crs_wkt, lon, lat):
    """Grid coordinates of a station (§2.8: ``Transformer.from_crs(
    "EPSG:4326", grid crs, always_xy=True)``)."""
    import pyproj

    tr = pyproj.Transformer.from_crs("EPSG:4326", pyproj.CRS.from_wkt(crs_wkt),
                                     always_xy=True)
    gx, gy = tr.transform(lon, lat)
    return float(gx), float(gy)


def frame_cell(xs, ys, gx, gy):
    """``(row, col, inside)`` of a point: the frame pixel whose cell holds it
    (xs, ys are pixel centres, regular, as v7.cube_grid's transform)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
    col = int(np.floor((gx - (xs[0] - dx / 2)) / dx))
    row = int(np.floor((gy - (ys[0] - dy / 2)) / dy))
    inside = 0 <= row < len(ys) and 0 <= col < len(xs)
    return row, col, bool(inside)


def neighbourhood(xs, ys, keep_full, gx, gy, rule=None):
    """The judges' neighbourhood of one station (§2.8, FROZEN NB rule).

    ``R = max(r0, d_min + pad)`` with d_min the distance to the nearest
    keep_full pixel centre, widened by ``step`` while fewer than
    ``min_keep_px`` keep_full pixels lie within R and R < cap; the
    neighbourhood is every frame pixel whose centre is within R.

    Returns
    -------
    dict
        ``d_min_m``, ``R_m``, ``n_keep_in_nbhd``, ``nbhd`` (ascending flat
        indices, int64); R is NaN and the neighbourhood empty without keep
        pixels.
    """
    r = dict(NB_RULE if rule is None else rule)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    W = len(xs)
    keep_full = np.asarray(keep_full, np.int64)
    if not keep_full.size:
        return {"d_min_m": float("nan"), "R_m": float("nan"), "n_keep_in_nbhd": 0,
                "nbhd": np.zeros(0, np.int64)}
    d_keep = np.hypot(xs[keep_full % W] - gx, ys[keep_full // W] - gy)
    d_min = float(d_keep.min())
    R = max(float(r["r0_m"]), d_min + float(r["pad_m"]))
    while np.count_nonzero(d_keep <= R) < int(r["min_keep_px"]) and R < float(r["cap_m"]):
        R += float(r["step_m"])
    cc = np.flatnonzero(np.abs(xs - gx) <= R)
    rr = np.flatnonzero(np.abs(ys - gy) <= R)
    if cc.size and rr.size:
        inside = np.hypot(xs[cc][None, :] - gx, ys[rr][:, None] - gy) <= R
        nbhd = (rr[:, None].astype(np.int64) * W + cc[None, :])[inside]
    else:
        nbhd = np.zeros(0, np.int64)
    return {"d_min_m": d_min, "R_m": float(R),
            "n_keep_in_nbhd": int(np.count_nonzero(d_keep <= R)),
            "nbhd": np.asarray(nbhd, np.int64)}


def build_nb(cfg, crs_wkt, xs, ys, keep_full, stations_path=None):
    """NB: the union of the neighbourhoods of the listed stations inside the
    frame, with each station's geometry. Returns ``(nb, rows, path)``."""
    coords, path = station_coordinates(cfg, stations_path)
    rows, parts = [], []
    for code, lon, lat in coords:
        gx, gy = station_xy(crs_wkt, lon, lat)
        row, col, inside = frame_cell(xs, ys, gx, gy)
        e = {"code": code, "lon": lon, "lat": lat, "gx": gx, "gy": gy, "row": row,
             "col": col, "in_frame": inside}
        if inside:
            nb = neighbourhood(xs, ys, keep_full, gx, gy)
            parts.append(nb["nbhd"])
            e.update({k: nb[k] for k in ("d_min_m", "R_m", "n_keep_in_nbhd")})
            e["nbhd"] = nb["nbhd"]
        else:
            e.update({"d_min_m": float("nan"), "R_m": float("nan"), "n_keep_in_nbhd": 0,
                      "nbhd": np.zeros(0, np.int64)})
        rows.append(e)
    nb = (np.unique(np.concatenate(parts)).astype(np.int64) if parts
          else np.zeros(0, np.int64))
    return nb, rows, path


def coords_sha(coords):
    """sha256 of the station coordinates the package domain used."""
    return config.sha256_json([[c, lon, lat] for c, lon, lat in coords])


# ─────────────────────────────────────────────────────────────────────────────
#  the evaluation support S0 and the stored domain dom (v2 spec §2.10)
# ─────────────────────────────────────────────────────────────────────────────

def dilate_disk(mask, radius_px):
    """Binary dilation of ``mask`` by a disk: every pixel whose centre lies
    within ``radius_px`` pixel steps (Euclidean) of the centre of a True
    pixel.

    Exact on any grid size: squared distances between pixel centres are
    integers, so the rounded square of the exact Euclidean distance
    transform is compared with ``radius_px ** 2``. Costs O(H W) whatever the
    radius (a 300 m disk is 61 x 61 px at 10 m).

    Parameters
    ----------
    mask : (H, W) bool
    radius_px : float
        Radius in pixel steps (0 returns the mask itself).

    Returns
    -------
    (H, W) bool
    """
    from scipy import ndimage

    m = np.asarray(mask, bool)
    if float(radius_px) < 0:
        raise ValueError(f"radius_px must be >= 0, got {radius_px!r}")
    if not m.any():
        return np.zeros(m.shape, bool)
    d = ndimage.distance_transform_edt(~m)
    np.square(d, out=d)
    np.rint(d, out=d)
    return d <= float(radius_px) ** 2


def _flat_mask(flat, shape):
    out = np.zeros(int(shape[0]) * int(shape[1]), bool)
    out[np.asarray(flat, np.int64)] = True
    return out.reshape(shape)


def support_s0(shape, keep_full, nb, pixel_m, d_truth=None):
    """S0, the fixed evaluation support (v2 §2.10), as sorted flat indices.

    ``S0 = D_truth ∪ NB ∪ dilate(keep_full(P0, W3), 300 m) ∪ Ω``. Ω, the D2
    domain (keep_full plus a 300 m ring of permanent water and stable land,
    v2 §2.6), lies inside the 300 m dilation of keep_full by its definition,
    so it adds no pixel and is not computed here.

    Parameters
    ----------
    shape : (H, W)
    keep_full : flat indices
        The full-record domain under the primary observation (P0).
    nb : flat indices
        The judges' neighbourhoods (NB).
    pixel_m : float
        Grid spacing (m); the dilation radius is FROZEN 300 m / pixel_m.
    d_truth : flat indices, optional
        The truth pixels inside the range filter (v2 §1.8), frozen at
        lock-A by the v2 truth module; this package has none yet (None).

    Returns
    -------
    (n,) int64
    """
    r_px = S0_KEEP_DILATION_M / float(pixel_m)
    s0 = dilate_disk(_flat_mask(keep_full, shape), r_px).ravel()
    s0[np.asarray(nb, np.int64)] = True
    if d_truth is not None:
        s0[np.asarray(d_truth, np.int64)] = True
    return np.flatnonzero(s0).astype(np.int64)


def package_domain(shape, s0, urec, keep_prod=()):
    """dom, the stored package domain (v2 §2.10), as sorted flat indices:
    ``S0 ∪ dilate(U_rec, 3 px) ∪ keep_prod``.

    U_rec (FROZEN['package']['U_rec'], blocklist + floor0 over the record)
    contains U_rec of the A-mode curve settings P0 and C2 (both read the
    marea_clear observations of the record or of a subset of it); the C11/C3
    term joins with the v2 package, whose stores carry CLD. keep_prod, the
    production extraction's keep, is kept for the legacy gates (G-keep,
    G-marea read the stored values there).

    Parameters
    ----------
    shape : (H, W)
    s0 : flat indices
    urec : (H, W) bool
    keep_prod : flat indices
    """
    dom = dilate_disk(np.asarray(urec, bool).reshape(shape), DOM_UREC_DILATION_PX).ravel()
    dom[np.asarray(s0, np.int64)] = True
    dom[np.asarray(keep_prod, np.int64)] = True
    return np.flatnonzero(dom).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
#  pass 2: raw values at the package domain
# ─────────────────────────────────────────────────────────────────────────────

def pass2(cfg, record, dom, out_dir, check_frame=True, log_every=SCENE_STATS_LOG_EVERY):
    """Pass 2 (§2.5): raw B03, B08 (int16) and SCL (uint8, fill -> 255) at
    ``dom``, one row per record scene, into ``b03.npy``, ``b08.npy``,
    ``scl.npy`` (``np.lib.format.open_memmap``, published atomically).

    With ``check_frame`` each scene's primary V at dom, recomputed from the
    gathered values, must equal its pass-1 validity bits (the two passes
    read the same data).
    """
    t0 = time.time()
    time_index = np.asarray(record["time_index"], np.int64)
    T, P = len(time_index), len(dom)
    H, W = int(record["grid"]["H"]), int(record["grid"]["W"])
    HW = H * W
    fill = int(record["encoding"]["fill"]["B03"])
    fill_scl = int(record["encoding"]["fill"]["SCL"])
    dom = np.asarray(dom, np.int64)
    names = (B03_NPY, B08_NPY, SCL_NPY)
    dtypes = (np.int16, np.int16, np.uint8)
    paths = [os.path.join(out_dir, nm) for nm in names]
    tmps = [_tmp(p) for p in paths]
    mms = [np.lib.format.open_memmap(tp, mode="w+", dtype=dt, shape=(T, P))
           for tp, dt in zip(tmps, dtypes)]
    fv = (np.load(os.path.join(out_dir, FRAME_VALID_NPY), mmap_mode="r")
          if check_frame else None)
    # the flush loop's variable keeps the last map open after the loop; on
    # Windows os.replace or os.remove of a mapped file fails (WinError 32),
    # so every name that refers to a map is deleted before the files are
    # published or discarded (v2 spec §12.3, defect 1)
    mm = None
    ds, t_dim = open_cube(cfg.cube)
    try:
        for q, ti in enumerate(time_index):
            g, n, s = read_scene(ds, t_dim, ti)
            if g.dtype != np.int16 or n.dtype != np.int16:
                raise TypeError("the cube bands must be raw int16")
            gd, nd_, sd = g.ravel()[dom], n.ravel()[dom], obs.scl_u8(s.ravel()[dom], fill_scl)
            mms[0][q], mms[1][q], mms[2][q] = gd, nd_, sd
            if fv is not None:
                _, vd = obs.observe(gd, nd_, sd, fill, PRIMARY_MASK, PRIMARY_INDEX)
                bits = np.unpackbits(np.asarray(fv[q]), count=HW)[dom].astype(bool)
                if not np.array_equal(vd, bits):
                    raise PackageError(
                        f"scene {q} (cube t = {ti}): {int(np.count_nonzero(vd != bits))} "
                        "package pixels differ from the pass-1 validity bits; the cube "
                        "changed between the passes")
            del g, n, s, gd, nd_, sd
            if (q + 1) % log_every == 0 or q + 1 == T:
                log(f"  [pass 2] {q + 1}/{T} scenes ({time.time() - t0:.0f} s, "
                    f"RSS {_rss_mb():.0f} MB)")
        for mm in mms:
            mm.flush()
    except BaseException as exc:
        ds.close()
        del mm, mms, fv
        _discard_after_failure(tmps, exc)
        raise
    ds.close()
    del mm, mms, fv
    gc.collect()
    for tp, p in zip(tmps, paths):
        os.replace(tp, p)
    return {"T": T, "P_dom": P, "seconds": round(time.time() - t0, 1)}


# ─────────────────────────────────────────────────────────────────────────────
#  subsets (§2.9)
# ─────────────────────────────────────────────────────────────────────────────

def rand_perm(site, rep, T):
    """``default_rng([20261008, SITE_ID[site], rep]).permutation(T)``."""
    rng = np.random.default_rng([SEED_SUBSETS, config.SITE_ID[site], int(rep)])
    return rng.permutation(int(T)).astype(np.int64)


def win_start(site, N, rep, T):
    """``default_rng([20261012, SITE_ID[site], N, rep]).integers(0, T - N + 1)``."""
    if not 1 <= int(N) <= int(T):
        raise ValueError(f"a window of {N} scenes does not fit a record of {T}")
    rng = np.random.default_rng([SEED_WINDOWS, config.SITE_ID[site], int(N), int(rep)])
    return int(rng.integers(0, int(T) - int(N) + 1))


def subset_positions(site, T, family, N, rep=0, rules=None):
    """Sorted record positions of the subset ``(family, N, rep)`` (§2.9).

    ``rand``: ``np.sort(perm[rep][:N])`` (nested in N within a rep);
    ``win``: ``arange(start, start + N)``; N = "all" gives arange(T) in both;
    ``rule-tz`` / ``rule-g21`` / ``rule-ard``: the stored rule subset
    (``rules[name]``), at N = "all" only.
    """
    T = int(T)
    if family in FAMILY_RULE:
        if N != config.N_ALL:
            raise ValueError("the scene-rule subsets are used at N = 'all' only")
        if rules is None or rules.get(FAMILY_RULE[family]) is None:
            raise PackageError(f"the {family} subset is not available at {site}")
        return np.asarray(rules[FAMILY_RULE[family]], np.int64)
    if family not in ("rand", "win"):
        raise ValueError(f"unknown subset family {family!r}")
    if N == config.N_ALL:
        return np.arange(T, dtype=np.int64)
    N = int(N)
    if not 1 <= N <= T:
        raise ValueError(f"N = {N} does not fit a record of {T} scenes")
    if family == "rand":
        return np.sort(rand_perm(site, rep, T)[:N])
    s = win_start(site, N, rep, T)
    return np.arange(s, s + N, dtype=np.int64)


def transition_dates(cfg):
    """``(record_dates, path)`` of the 10 m MAREA transition-zone record
    (``cfg.transition_result``, the local server mirror if absent), or
    ``(None, None)`` for a site without one (the positive control)."""
    if cfg.transition_result is None:
        return None, None
    p = cfg.transition_result
    if not os.path.isabs(p):
        p = config.resolve(p)
    if not os.path.exists(_abs(p)):
        raise FileNotFoundError(f"{cfg.transition_result}: the transition-zone record "
                                "(rule-tz subset) is missing")
    with open(_abs(p), encoding="utf-8") as f:
        dates = [str(d) for d in json.load(f)["record_dates"]]
    return dates, p


def rule_subsets(dates, scene_stats, tz_dates=None):
    """Record positions passing each scene rule (§2.9): tz (date in the
    transition-zone record), g21 (``g21.scene_rule`` on the frame shares),
    ard (``wc.s2_scene_rule`` on the load_ard good fraction). tz is None
    without a transition record."""
    from pyintertidal import granadeiro2021 as g21
    from pyintertidal import waterline_contour as wc

    dates = np.asarray(dates).astype(str)
    T = len(dates)
    st = scene_stats
    if len(st) != T or not np.array_equal(st["q"].to_numpy(), np.arange(T)):
        raise ValueError("scene_stats.csv is not one row per record scene in order")
    ok_g21 = g21.scene_rule(st["g21_cloud"].to_numpy(float), st["g21_nodata"].to_numpy(float),
                            np.ones(T, bool), np.ones(T, bool))
    try:
        ok_ard = wc.s2_scene_rule(st["ard_good"].to_numpy(float))
    except wc.EmptyIntervalError:           # no scene passes: an empty subset
        ok_ard = np.zeros(T, bool)
    out = {"g21": np.flatnonzero(ok_g21).astype(np.int64),
           "ard": np.flatnonzero(ok_ard).astype(np.int64),
           "tz": None}
    if tz_dates is not None:
        out["tz"] = np.flatnonzero(np.isin(dates, np.asarray(tz_dates, str))).astype(np.int64)
    return out


def build_subsets(T, site, rules=None, n_reps=STORED_REPS):
    """The stored subsets of a site: ``perm`` (n_reps, T), the window starts
    of every N_GRID value that fits (n_N, n_reps), the scene-rule subsets
    and the seeds. Any other (family, N, rep) is drawn on the fly from the
    same formulas (:func:`subset_positions`)."""
    T = int(T)
    sid = config.SITE_ID[site]
    perm = np.stack([rand_perm(site, r, T) for r in range(n_reps)])
    win_N = np.array([N for N in config.N_GRID if N <= T], np.int64)
    win_s = np.array([[win_start(site, N, r, T) for r in range(n_reps)] for N in win_N],
                     np.int64).reshape(len(win_N), n_reps)
    out = {"format": np.array(SUBSETS_FORMAT), "T": np.array(T, np.int64),
           "site_id": np.array(sid, np.int64),
           "seed_subsets": np.array(SEED_SUBSETS, np.int64),
           "seed_windows": np.array(SEED_WINDOWS, np.int64),
           "perm": perm, "win_N": win_N, "win_start": win_s}
    for name in RULES:
        sel = None if rules is None else rules.get(name)
        out[f"rule_{name}_available"] = np.array(sel is not None)
        out[f"rule_{name}"] = (np.zeros(0, np.int64) if sel is None
                               else np.asarray(sel, np.int64))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  steps
# ─────────────────────────────────────────────────────────────────────────────

class _Context:
    """What the steps share: the config, the output folder, the provider."""

    def __init__(self, cfg, provider=None, level_chunk=None):
        self.cfg = cfg
        self.dir = cfg.common_dir
        self._provider = provider
        self.production_provider = provider is None
        self.level_chunk = int(level_chunk or levels.CHUNK)

    def path(self, name):
        return os.path.join(self.dir, name)

    def provider(self):
        if self._provider is None:
            levels.check_boundary_spec(self.cfg)
            log(f"[package] building the {self.cfg.site} boundary provider")
            self._provider = levels.make_provider(self.cfg)
        return self._provider

    def record(self):
        return product.read_json(self.path(RECORD_JSON))


def _step_record(ctx):
    from experiments import v7_waterline_contour as v7

    cfg = ctx.cfg
    grid = v7.cube_grid(_abs(cfg.cube))
    enc = cube_encoding(cfg.cube)
    if enc["fill"]["B03"] != enc["fill"]["B08"]:
        raise PackageError("B03 and B08 have different fill values")
    ext = extraction_header(cfg)
    if ext is not None:
        if not np.array_equal(ext["dates"], grid["dates"]):
            raise PackageError(f"{cfg.extract}: its dates are not the cube's")
        if ext["shape"] != (grid["H"], grid["W"]):
            raise PackageError(f"{cfg.extract}: shape {ext['shape']} is not the cube's")
        ext_dates = ext["dates"]
    else:
        ext_dates = grid["dates"]
    provider = ctx.provider()
    rec = build_record(cfg, provider, ext_dates)
    name = levels.provider_name(cfg, provider)
    ioc = levels.ioc_chunks(cfg, provider=provider if ctx.production_provider else None)
    xs, ys = grid["xs"], grid["ys"]
    t = rec["t"]
    doc = {
        "format": RECORD_FORMAT, "site": cfg.site, "res_m": cfg.res_m,
        "scenario": cfg.scenario, "rule": config.FROZEN["record"]["rule"],
        "inputs": {"cube": _rel(cfg.cube), "overpass": _rel(cfg.overpass),
                   "extract": None if cfg.extract is None else _rel(cfg.extract)},
        "T": int(len(rec["time_index"])), "n_cube_dates": rec["n_cube_dates"],
        "n_candidates": rec["n_candidates"],
        "time_index": rec["time_index"].tolist(), "dates": rec["dates"].tolist(),
        "t_utc": [ts.isoformat() for ts in t], "t_ns": t.asi8.astype(np.int64).tolist(),
        "h0": [float(v) for v in rec["h0"]],
        "exclusions": rec["exclusions"],
        "boundary": {"spec": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in cfg.boundary.items()},
                     "name": name, "description": levels.describe_provider(provider)},
        "ioc_chunks": ioc,
        "grid": {"H": int(grid["H"]), "W": int(grid["W"]), "x0": float(xs[0]),
                 "y0": float(ys[0]), "dx": float(xs[1] - xs[0]), "dy": float(ys[1] - ys[0]),
                 "crs_wkt": str(grid["wkt"])},
        "encoding": {"dtype": enc["dtype"], "fill": enc["fill"], "t_dim": enc["t_dim"],
                     "T_cube": enc["T"]},
        "pixel_m": float(cfg.pixel_m), "mouth_side": cfg.mouth_side,
    }
    write_json_lf(ctx.path(RECORD_JSON), doc)
    ex = rec["exclusions"]
    log(f"[record] {cfg.site}: T = {doc['T']} of {rec['n_cube_dates']} cube dates "
        f"({len(ex['outside_period'])} outside the period, {len(ex['without_overpass'])} "
        f"without an overpass time, {len(ex['nonfinite_level'])} with a non-finite level); "
        f"boundary {name}; {len(ioc)} IOC file(s)")
    return {"T": doc["T"]}


def _step_series(ctx):
    rec = ctx.record()
    name = rec["boundary"]["name"]
    t_min, h, cached = levels.boundary_series_1min(ctx.provider(), name,
                                                   ctx.path(SERIES_CACHE))
    mk = levels.build_marks(t_min, h)
    levels.save_marks(ctx.path(MARKS_NPZ), **mk)
    log(f"[series] 1-min series {'cached' if cached else 'built'}: {len(h):,} samples; "
        f"marks_lp {mk['marks_lp']['n_marks']:,}, marks_raw {mk['marks_raw']['n_marks']:,}")
    return {"cached": bool(cached), "n_marks_lp": int(mk["marks_lp"]["n_marks"]),
            "n_marks_raw": int(mk["marks_raw"]["n_marks"])}


def _step_pass1(ctx):
    rec = ctx.record()
    ext = extraction_header(ctx.cfg)
    keep_prod = ext["keep"] if ext is not None else np.zeros(0, np.int64)
    marks_lp = levels.load_marks(ctx.path(MARKS_NPZ))["marks_lp"]
    log(f"[pass 1] {ctx.cfg.site}: {rec['T']} scenes of {rec['grid']['H']} x "
        f"{rec['grid']['W']} px")
    return pass1(ctx.cfg, rec, ctx.dir, keep_prod=keep_prod, marks_lp=marks_lp)


def _step_geometry(ctx):
    from experiments import v7_waterline_contour as v7

    cfg = ctx.cfg
    rec = ctx.record()
    cnt = product.load_npz(ctx.path(COUNTS_NPZ))
    d = domain.from_counts(cnt["clr_n"], cnt["wet_n"], cfg.pixel_m, cfg.mouth_side,
                           mode="F", n_sel=rec["T"], with_frame=True)
    band_frame = domain.band_map_frame(d, d.s_frame)
    fin = np.isfinite(d.s_km)
    bk = band_frame.ravel()[d.keep]
    if not (np.array_equal(bk[fin], d.band_of[fin]) and np.all(bk[~fin] == -1)):
        raise AssertionError("the frame strata do not reproduce the bands at keep_full")
    ext = extraction_header(cfg)
    H, W = int(rec["grid"]["H"]), int(rec["grid"]["W"])
    if ext is None:
        keep_prod = np.zeros(0, np.int64)
        sea_prod = inter_prod = np.zeros((H, W), bool)
        keep_is_inter = None
    else:
        keep_prod, sea_prod, inter_prod = ext["keep"], ext["sea"], ext["inter"]
        keep_is_inter = bool(np.array_equal(np.flatnonzero(inter_prod.ravel()), keep_prod))
    grid = v7.cube_grid(_abs(cfg.cube))
    meta = {"format": PIXELS_FORMAT, "pixel_m": float(cfg.pixel_m),
            "mouth_side": cfg.mouth_side, "status": d.status,
            "min_intertidal": domain.MIN_INTERTIDAL, "min_finite_s": domain.MIN_FINITE_S,
            "band_km": domain.BAND_KM, "min_px_band": domain.MIN_PX_BAND,
            "keep_prod_is_inter_prod": keep_is_inter,
            "s_frame": "km, float32 copy of the float64 s; band_frame from the float64 s"}
    product.save_npz(ctx.path(PIXELS_NPZ), {
        "meta": np.array(json.dumps(meta, sort_keys=True)),
        "keep_full": d.keep, "sea_full": d.sea, "s_km": d.s_km, "band_of": d.band_of,
        "centers": d.centers, "edges": d.edges, "K": np.array(d.K, np.int64),
        "status": np.array(d.status),
        "s_frame": d.s_frame.astype(np.float32), "band_frame": band_frame,
        "keep_prod": keep_prod, "sea_prod": sea_prod, "inter_prod": inter_prod,
        "xs": np.asarray(grid["xs"], np.float64), "ys": np.asarray(grid["ys"], np.float64)})
    log(f"[geometry] keep_full {d.n_keep:,} px ({d.status}), K = {d.K}, centres "
        f"{np.round(d.centers, 2).tolist()} km; isolated {int(np.count_nonzero(~fin)):,}; "
        f"keep_prod {len(keep_prod):,}")
    return {"n_keep_full": d.n_keep, "K": d.K, "status": d.status}


def _step_judges(ctx):
    """NB, S0 and dom (v2 §2.10): ``nb.npz`` (the neighbourhoods and S0) and
    ``dom.npy``."""
    cfg = ctx.cfg
    rec = ctx.record()
    H, W = int(rec["grid"]["H"]), int(rec["grid"]["W"])
    px = product.load_npz(ctx.path(PIXELS_NPZ))
    cnt = product.load_npz(ctx.path(COUNTS_NPZ))
    U = urec_mask(cnt["clr_n_blocklist"], cnt["wet_n_blocklist"])
    keep_full, keep_prod = px["keep_full"], px["keep_prod"]
    if not U.ravel()[keep_full].all():
        raise AssertionError("keep_full is not inside U_rec: the superset rule is broken")
    nb, rows, spath = build_nb(cfg, rec["grid"]["crs_wkt"], px["xs"], px["ys"], keep_full)
    s0 = support_s0((H, W), keep_full, nb, cfg.pixel_m)
    dom = package_domain((H, W), s0, U, keep_prod)
    n_urec = int(np.count_nonzero(U))
    coords = [(r["code"], r["lon"], r["lat"]) for r in rows]
    nbhd = [r.pop("nbhd") for r in rows]
    ptr = np.concatenate([[0], np.cumsum([len(a) for a in nbhd])]).astype(np.int64)
    meta = {"format": NB_FORMAT, "rule": NB_RULE,
            "stations_json": None if spath is None else _rel(spath),
            "coords_sha256": coords_sha(coords) if coords else None,
            "stations": rows, "n_nb": int(len(nb)), "n_urec": n_urec,
            "n_keep_full": int(len(keep_full)), "n_keep_prod": int(len(keep_prod)),
            "n_s0": int(len(s0)), "n_dom": int(len(dom)),
            "s0_keep_dilation_m": S0_KEEP_DILATION_M,
            "dom_urec_dilation_px": DOM_UREC_DILATION_PX}
    product.save_npz(ctx.path(NB_NPZ), {
        "meta": np.array(json.dumps(meta, sort_keys=True, default=_json_default)),
        "nb": nb, "nbhd_ptr": ptr,
        "nbhd": np.concatenate(nbhd).astype(np.int64) if nbhd else np.zeros(0, np.int64),
        "s0": s0})
    _save_npy(ctx.path(DOM_NPY), dom)
    for r in rows:
        log(f"[judges] {r['code']}: in frame {r['in_frame']}, d_min "
            f"{r['d_min_m']:.0f} m, R {r['R_m']:.0f} m, {r['n_keep_in_nbhd']:,} keep px")
    ratio = len(dom) / len(s0) if len(s0) else float("nan")
    log(f"[judges] S0 = NB {len(nb):,} ∪ keep_full {len(keep_full):,} dilated "
        f"{S0_KEEP_DILATION_M:.0f} m = {len(s0):,} px; dom = S0 ∪ U_rec {n_urec:,} dilated "
        f"{DOM_UREC_DILATION_PX:g} px ∪ keep_prod {len(keep_prod):,} = {len(dom):,} px "
        f"({len(dom) / (H * W):.1%} of the frame; |dom| / |S0| = {ratio:.2f})")
    return {"P_dom": int(len(dom)), "n_s0": int(len(s0)), "n_urec": n_urec,
            "n_nb": int(len(nb))}


def _step_levels(ctx):
    rec = ctx.record()
    K = int(product.load_npz(ctx.path(PIXELS_NPZ))["K"])
    t = pd.DatetimeIndex(np.asarray(rec["t_ns"], np.int64).view("M8[ns]"))
    h0 = np.asarray(rec["h0"], np.float64)
    tab = levels.build_level_table(ctx.provider(), t, config.tau_tab_max(K),
                                   chunk=ctx.level_chunk, h0=h0)
    tab.save(ctx.path(LEVELS_NPZ))
    return {"tau_tab_max": tab.tau_hi, "batch_invariant": tab.batch_invariant,
            "seconds": tab.build_seconds}


def _step_pass2(ctx):
    rec = ctx.record()
    dom = np.load(ctx.path(DOM_NPY))
    log(f"[pass 2] {ctx.cfg.site}: {rec['T']} scenes x {len(dom):,} package px "
        f"({rec['T'] * len(dom) * 5 / 1e9:.2f} GB)")
    return pass2(ctx.cfg, rec, dom, ctx.dir)


def _step_subsets(ctx):
    cfg = ctx.cfg
    rec = ctx.record()
    st = read_scene_stats(ctx.path(SCENE_STATS_CSV))
    tz, tz_path = transition_dates(cfg)
    rules = rule_subsets(rec["dates"], st, tz)
    sub = build_subsets(rec["T"], cfg.site, rules)
    n_tz_missing = (None if tz is None
                    else int(len(set(tz) - set(np.asarray(rec["dates"], str)))))
    meta = {"format": SUBSETS_FORMAT, "transition_result": None if tz_path is None
            else _rel(tz_path), "n_tz_dates": None if tz is None else len(tz),
            "n_tz_dates_not_in_record": n_tz_missing,
            "rules": dict(config.FROZEN["subsets"]["rules"])}
    sub["meta"] = np.array(json.dumps(meta, sort_keys=True))
    product.save_npz(ctx.path(SUBSETS_NPZ), sub)
    log("[subsets] rule subsets: " + ", ".join(
        f"{r} {'n/a' if rules[r] is None else len(rules[r])}" for r in RULES)
        + (f" ({n_tz_missing} tz dates not in the record)" if n_tz_missing else ""))
    return {r: (None if rules[r] is None else int(len(rules[r]))) for r in RULES}


def _step_manifest(ctx):
    key = write_manifest(ctx.dir)
    return {"package_key": key}


_RUN = {"record": _step_record, "series": _step_series, "pass1": _step_pass1,
        "geometry": _step_geometry, "judges": _step_judges, "levels": _step_levels,
        "pass2": _step_pass2, "subsets": _step_subsets, "manifest": _step_manifest}


# ─────────────────────────────────────────────────────────────────────────────
#  step bookkeeping, lock, prepare
# ─────────────────────────────────────────────────────────────────────────────

def _read_book(d):
    p = os.path.join(d, STEPS_JSON)
    if os.path.exists(p):
        try:
            book = product.read_json(p)
            if book.get("format") == STEPS_FORMAT:
                return book
        except (OSError, ValueError):
            pass
    return {"format": STEPS_FORMAT, "steps": {}}


def _write_book(d, book):
    product.write_json(os.path.join(d, STEPS_JSON), book)


def _output_ok(d, name, e):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return False
    st = os.stat(p)
    if st.st_size != e["size"]:
        return False
    if st.st_mtime_ns != e["mtime_ns"]:
        # copied or touched: the content decides
        return product.sha256_file(p) == e["sha256"]
    return True


def _step_done(d, book, step, _seen=None):
    """True when the step ran, its outputs are intact and every file it read
    is the one its producer last published (recursively up the chain)."""
    seen = {} if _seen is None else _seen
    if step in seen:
        return seen[step]
    rec = book["steps"].get(step)
    ok = rec is not None and all(_output_ok(d, n, e) for n, e in rec["outputs"].items())
    if ok:
        for name, sha in rec.get("reads", {}).items():
            prod = _PRODUCER[name]
            prec = book["steps"].get(prod)
            if (prec is None or prec["outputs"].get(name, {}).get("sha256") != sha
                    or not _step_done(d, book, prod, seen)):
                ok = False
                break
    seen[step] = ok
    return ok


def _prerequisites(step):
    return sorted({_PRODUCER[f] for f in STEP_READS[step]}, key=STEPS.index)


def step_status(d):
    """``{step: "done" | "stale" | "pending"}`` of a package folder."""
    book = _read_book(d)
    out = {}
    for s in STEPS:
        if s not in book["steps"]:
            out[s] = "pending"
        else:
            out[s] = "done" if _step_done(d, book, s) else "stale"
    return out


class _Lock:
    """One prepare per package folder at a time (O_CREAT | O_EXCL)."""

    def __init__(self, d):
        self.path = os.path.join(d, LOCK_FILE)

    def __enter__(self):
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._stale():
                    log(f"[package] removing the stale lock {self.path}")
                    os.remove(self.path)
                    continue
                raise PackageError(f"{self.path} exists: another prepare is running on this "
                                   "package (delete the file if it is not)")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"host": socket.gethostname(), "pid": os.getpid(),
                           "start_utc": product.utc_stamp()}, f)
            return self
        raise PackageError(f"could not take {self.path}")

    def _stale(self):
        try:
            info = product.read_json(self.path)
            if info.get("host") != socket.gethostname():
                return False
            import psutil
            return not psutil.pid_exists(int(info["pid"]))
        except Exception:                    # unreadable or no psutil: not stale
            return False

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass
        return False


def prepare(site, res_m, steps=STEPS, force=False, scenario=None, *, cfg=None,
            provider=None, level_chunk=None):
    """Build (or finish) the common package of a site (§2).

    Parameters
    ----------
    site, res_m, scenario
        The site (``scenario`` for the positive control).
    steps : sequence of str
        The steps to run, in STEPS order; their prerequisites must be done.
    force : bool
        Run the listed steps even when done (later steps then become stale).
    cfg : config.SiteConfig, optional
        Another configuration (tests, the positive control).
    provider : object with ``levels``, optional
        Use this boundary provider instead of building the production one
        (tests); the IOC inventory then does not check its members.
    level_chunk : int, optional
        Instants per batched provider call of the level table (default
        150,000; a laptop may need less; it enters levels.npz).

    Returns
    -------
    str or None
        The package key when the manifest exists after the run (verified
        with a full rehash), else None.
    """
    cfg = cfg or config.site_config(site, res_m, scenario)
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise ValueError(f"unknown step(s) {unknown}; expected {STEPS}")
    d = cfg.common_dir
    os.makedirs(d, exist_ok=True)
    t_all = time.time()
    log(f"[package] {cfg.site} {cfg.res_m} m -> {d}: steps {list(steps)}"
        f"{' (force)' if force else ''}")
    with _Lock(d):
        book = _read_book(d)
        ctx = _Context(cfg, provider=provider, level_chunk=level_chunk)
        for step in STEPS:
            if step not in steps:
                continue
            if not force and _step_done(d, book, step):
                log(f"[package] step {step}: done")
                continue
            missing = [s for s in _prerequisites(step) if not _step_done(d, book, s)]
            if missing:
                raise PackageError(f"step {step} needs {missing} first (run prepare with "
                                   "those steps, or without --steps)")
            man = os.path.join(d, MANIFEST_JSON)
            if step != "manifest" and os.path.exists(man):
                os.remove(man)
                log("[package] manifest.json removed: the package changes")
            t0 = time.time()
            log(f"[package] step {step} ...")
            info = _RUN[step](ctx) or {}
            reads = {}
            for name in STEP_READS[step]:
                prod_rec = book["steps"].get(_PRODUCER[name])
                reads[name] = prod_rec["outputs"][name]["sha256"]
            outputs = {name: product.file_entry(os.path.join(d, name))
                       for name in STEP_OUTPUTS[step]}
            book["steps"][step] = {"outputs": outputs, "reads": reads,
                                   "seconds": round(time.time() - t0, 1),
                                   "done_utc": product.utc_stamp(),
                                   "host": socket.gethostname(),
                                   "info": info}
            _write_book(d, book)
            log(f"[package] step {step} done in {time.time() - t0:.1f} s "
                f"(RSS {_rss_mb():.0f} MB)")
        key = None
        if os.path.exists(os.path.join(d, MANIFEST_JSON)):
            key = verify_manifest(d, full=True)
    log(f"[package] {cfg.site} {cfg.res_m} m: {time.time() - t_all:.0f} s"
        + (f"; package_key {key}" if key else "; no manifest yet"))
    return key


# ─────────────────────────────────────────────────────────────────────────────
#  manifest
# ─────────────────────────────────────────────────────────────────────────────

def _manifest_inputs(d, rec):
    inputs = {}
    for name in ("cube", "extract", "overpass"):
        p = rec["inputs"].get(name)
        if p is not None:
            inputs[name] = input_entry(p)
    for e in rec.get("ioc_chunks", []):
        now = input_entry(e["path"])
        if now["sha256"] != e["sha256"]:
            raise PackageError(f"{e['path']} changed after the record step: rerun prepare "
                               "with --force")
        inputs[f"ioc:{e['path']}"] = now
    nbm = json.loads(str(product.load_npz(os.path.join(d, NB_NPZ))["meta"]))
    if nbm.get("coords_sha256"):
        inputs["judges_coords"] = {"path": nbm["stations_json"],
                                   "sha256": nbm["coords_sha256"],
                                   "hash": "sha256_json of [code, lon, lat] of cfg.judges"}
    sm = json.loads(str(product.load_npz(os.path.join(d, SUBSETS_NPZ))["meta"]))
    if sm.get("transition_result"):
        inputs["transition_result"] = input_entry(sm["transition_result"])
    return inputs


def write_manifest(common_dir, rehash=False):
    """Write ``manifest.json`` (§2.11) and return the package key.

    Files: size, mtime_ns and sha256 of every keyed package file (the sha256
    of steps.json is reused while size and mtime are unchanged, unless
    ``rehash``); inputs: the cube, extraction, overpass file, every IOC
    chunk (asserted unchanged since the record step), the judges'
    coordinates and the transition record; ``obs_sha``; the FROZEN shas of
    the package sections; git (provenance).
    """
    d = common_dir
    rec = product.read_json(os.path.join(d, RECORD_JSON))
    book = _read_book(d)
    cached = {}
    for s in book["steps"].values():
        cached.update(s["outputs"])
    files = {}
    for name in KEYED_FILES:
        p = os.path.join(d, name)
        if not os.path.exists(p):
            raise PackageError(f"{p} is missing: the package is incomplete")
        st = os.stat(p)
        e = cached.get(name)
        if (not rehash and e is not None and e["size"] == st.st_size
                and e["mtime_ns"] == st.st_mtime_ns):
            sha = e["sha256"]
        else:
            sha = product.sha256_file(p)
        files[name] = {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns), "sha256": sha}
    man = {"format": FORMAT, "site": rec["site"], "res_m": rec["res_m"],
           "scenario": rec.get("scenario"), "files": files,
           "inputs": _manifest_inputs(d, rec), "obs_sha": obs.source_sha256(),
           "frozen": config.frozen_shas(config.PACKAGE_SECTIONS),
           "git": product.git_state(), "created_utc": product.utc_stamp(),
           "host": socket.gethostname(),
           "summary": {"T": rec["T"], "H": rec["grid"]["H"], "W": rec["grid"]["W"],
                       "P_dom": int(np.load(os.path.join(d, DOM_NPY), mmap_mode="r").shape[0])}}
    man["package_key"] = product.package_key(man)
    write_json_lf(os.path.join(d, MANIFEST_JSON), man)
    log(f"[manifest] {rec['site']} {rec['res_m']} m: package_key {man['package_key']}")
    return man["package_key"]


def verify_manifest(common_dir, full=False, strict=True):
    """Check a package against its manifest; return its package key.

    Parameters
    ----------
    full : bool
        Rehash every package file (and every input whose size or mtime
        changed); otherwise size and mtime_ns of the package files and the
        size of the inputs. A full check that finds only new mtimes with
        equal contents (a copied package) records them in the manifest.
    strict : bool
        Refuse a package built with another ``obs.py`` or other FROZEN
        package sections (:class:`StalePackageError`).
    """
    d = common_dir
    p = os.path.join(d, MANIFEST_JSON)
    if not os.path.exists(p):
        raise PackageError(f"{p} is missing: run prepare (the package is incomplete)")
    man = product.read_json(p)
    if man.get("format") != FORMAT:
        raise PackageError(f"{p}: unknown format {man.get('format')!r}")
    key = product.package_key(man)
    if key != man.get("package_key"):
        raise PackageError(f"{p}: its entries do not give its package_key (edited by hand?)")
    if strict:
        stale = []
        if man["obs_sha"] != obs.source_sha256():
            stale.append("obs.py")
        cur = config.frozen_shas(config.PACKAGE_SECTIONS)
        stale += [f"FROZEN[{s!r}]" for s in config.PACKAGE_SECTIONS
                  if man["frozen"].get(s) != cur[s]]
        if stale:
            raise StalePackageError(f"{d} was built with another {', '.join(stale)}: "
                                    "rerun prepare with --force")
    bad, restamp = [], {}
    for name, e in man["files"].items():
        fp = os.path.join(d, name)
        if not os.path.exists(fp):
            bad.append(f"{name}: missing")
            continue
        st = os.stat(fp)
        if st.st_size != e["size"]:
            bad.append(f"{name}: size {st.st_size} != {e['size']}")
        elif full:
            if product.sha256_file(fp) != e["sha256"]:
                bad.append(f"{name}: content differs from the manifest")
            elif st.st_mtime_ns != e["mtime_ns"]:
                restamp[name] = int(st.st_mtime_ns)
        elif st.st_mtime_ns != e["mtime_ns"]:
            bad.append(f"{name}: modified after the manifest (mtime); a copied package "
                       "needs one full check (prepare)")
    for name, e in man["inputs"].items():
        if "size" not in e:
            continue
        ip = _abs(e["path"])
        if not os.path.exists(ip):
            log(f"[package] warning: input {e['path']} is not on this machine")
            continue
        st = os.stat(ip)
        if st.st_size != e["size"]:
            bad.append(f"input {name}: size {st.st_size} != {e['size']}")
        elif full and st.st_mtime_ns != e["mtime_ns"] and sha256_raw(ip) != e["sha256"]:
            bad.append(f"input {name}: content differs from the manifest")
    if bad:
        raise PackageError(f"{d} does not match its manifest:\n  " + "\n  ".join(bad))
    if restamp:
        for name, m in restamp.items():
            man["files"][name]["mtime_ns"] = m
        write_json_lf(p, man)
        log(f"[package] {len(restamp)} file time(s) updated in the manifest (same contents)")
    return key


# ─────────────────────────────────────────────────────────────────────────────
#  the package as the methods see it
# ─────────────────────────────────────────────────────────────────────────────

def _take(mm, sel, pos=None):
    """``mm[sel][:, pos]`` from a (T, P) memmap, reading row segments.

    ``pos`` sorted unique positions or None (every column). A dense column
    range is read as one slice per row; a sparse one element-wise.
    """
    if pos is None:
        return np.asarray(mm[sel])
    pos = np.asarray(pos, np.int64)
    if not pos.size:
        return np.empty((len(sel), 0), mm.dtype)
    lo, hi = int(pos[0]), int(pos[-1]) + 1
    if hi - lo == pos.size:
        return np.asarray(mm[sel, lo:hi])
    if hi - lo <= 4 * pos.size:
        return np.asarray(mm[sel, lo:hi])[:, pos - lo]
    return np.asarray(mm[np.ix_(np.asarray(sel), pos)])


class CommonPackage:
    """A built package, read-only (§1.4).

    Attributes
    ----------
    site, res_m, cfg, dir
    keys : dict
        package_key, obs_sha, the FROZEN shas, git (provenance).
    dates, time_index, t, T, h0
        The record, sorted by time (h0 = h_B at the instants, float64).
    shape, H, W, fill
        The frame and the bands' fill value.
    dom : (P_dom,) int64
        The package domain (flat indices, ascending): S0 ∪ dilate(U_rec,
        3 px) ∪ keep_prod (v2 §2.10).
    B03, B08 : np.memmap (T, P_dom) int16
        Raw cube values at dom.
    SCL : np.memmap (T, P_dom) uint8
        SCL at dom, no data = 255.
    frame_valid : wc.PackedRows (T, H*W)
        V under the primary rule and mode, for the whole frame.
    clr_n, wet_n : (H, W) int16
        Full-record counters under the primary rule and mode.
    full : domain.Domain
        The full-record domain (§2.6).
    keep_prod, sea_prod, inter_prod
        The production extraction (gates only).
    levels, series1, marks_lp, marks_raw, series10, judges, scene_stats,
    grid, nb, s0, s_frame, band_frame
        Lazy.
    """

    def __init__(self, cfg, manifest, provider=None):
        self.cfg = cfg
        self.site, self.res_m = cfg.site, cfg.res_m
        self.dir = cfg.common_dir
        self.manifest = manifest
        self.keys = {"package_key": manifest["package_key"], "obs_sha": manifest["obs_sha"],
                     "frozen": dict(manifest["frozen"]), "git": manifest.get("git")}
        rec = product.read_json(self._p(RECORD_JSON))
        if (rec["site"], int(rec["res_m"])) != (cfg.site, int(cfg.res_m)):
            raise PackageError(f"{self.dir} holds {rec['site']} {rec['res_m']} m")
        self.record = rec
        self.dates = np.asarray(rec["dates"], str)
        self.time_index = np.asarray(rec["time_index"], np.int64)
        self.t = pd.DatetimeIndex(np.asarray(rec["t_ns"], np.int64).view("M8[ns]"))
        self.T = int(len(self.t))
        self.h0 = np.asarray(rec["h0"], np.float64)
        self.H, self.W = int(rec["grid"]["H"]), int(rec["grid"]["W"])
        self.shape = (self.H, self.W)
        self.fill = int(rec["encoding"]["fill"]["B03"])
        self.dom = np.load(self._p(DOM_NPY)).astype(np.int64)
        self.P = int(len(self.dom))
        self.B03 = np.load(self._p(B03_NPY), mmap_mode="r")
        self.B08 = np.load(self._p(B08_NPY), mmap_mode="r")
        self.SCL = np.load(self._p(SCL_NPY), mmap_mode="r")
        for name, mm, dt in (("b03", self.B03, np.int16), ("b08", self.B08, np.int16),
                             ("scl", self.SCL, np.uint8)):
            if mm.shape != (self.T, self.P) or mm.dtype != dt:
                raise PackageError(f"{name}.npy is {mm.shape} {mm.dtype}, expected "
                                   f"{(self.T, self.P)} {np.dtype(dt)}")
        self._provider = provider

    def _p(self, name):
        return os.path.join(self.dir, name)

    def __repr__(self):
        return (f"<CommonPackage {self.site} {self.res_m} m: T = {self.T}, P_dom = {self.P:,}, "
                f"frame {self.H} x {self.W}, key {self.keys['package_key'][:12]}>")

    def close(self):
        """Drop the memory maps (Windows cannot replace a mapped file)."""
        for name in ("B03", "B08", "SCL"):
            setattr(self, name, None)
        for name in ("frame_valid", "levels"):
            self.__dict__.pop(name, None)
        gc.collect()

    # lazy pieces ----------------------------------------------------------
    def _npz_member(self, fname, member):
        with np.load(self._p(fname), allow_pickle=False) as z:
            return z[member]

    @cached_property
    def frame_valid(self):
        from pyintertidal import waterline_contour as wc

        packed = np.load(self._p(FRAME_VALID_NPY), mmap_mode="r")
        if packed.shape != (self.T, (self.H * self.W + 7) // 8):
            raise PackageError("frame_valid.npy has the wrong shape")
        return wc.PackedRows(packed, self.H * self.W)

    @cached_property
    def clr_n(self):
        return self._npz_member(COUNTS_NPZ, "clr_n")

    @cached_property
    def wet_n(self):
        return self._npz_member(COUNTS_NPZ, "wet_n")

    @cached_property
    def clr_n_blocklist(self):
        return self._npz_member(COUNTS_NPZ, "clr_n_blocklist")

    @cached_property
    def wet_n_blocklist(self):
        return self._npz_member(COUNTS_NPZ, "wet_n_blocklist")

    @cached_property
    def full(self):
        """The stored full-record domain (equals ``domain.full(self)``)."""
        z = product.load_npz(self._p(PIXELS_NPZ))
        keep = z["keep_full"].astype(np.int64)
        inter = np.zeros(self.H * self.W, bool)
        inter[keep] = True
        return domain.Domain("F", keep, self.positions(keep), z["sea_full"],
                             inter.reshape(self.shape), s_km=z["s_km"],
                             band_of=z["band_of"], centers=z["centers"], edges=z["edges"],
                             K=int(z["K"]), status=str(z["status"]), n_sel=self.T)

    @cached_property
    def keep_prod(self):
        return self._npz_member(PIXELS_NPZ, "keep_prod").astype(np.int64)

    @cached_property
    def sea_prod(self):
        return self._npz_member(PIXELS_NPZ, "sea_prod")

    @cached_property
    def inter_prod(self):
        return self._npz_member(PIXELS_NPZ, "inter_prod")

    @cached_property
    def s_frame(self):
        """s (km, float32) of every frame pixel, NaN unreachable."""
        return self._npz_member(PIXELS_NPZ, "s_frame")

    @cached_property
    def band_frame(self):
        """Strata band (int16) of every frame pixel, -1 = no s (§2.6)."""
        return self._npz_member(PIXELS_NPZ, "band_frame")

    @cached_property
    def grid(self):
        """``v7.cube_grid``'s geometry without opening the cube: xs, ys, H,
        W, wkt, crs, transform."""
        import pyproj
        from rasterio.transform import Affine

        xs = self._npz_member(PIXELS_NPZ, "xs")
        ys = self._npz_member(PIXELS_NPZ, "ys")
        dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
        wkt = self.record["grid"]["crs_wkt"]
        return dict(xs=xs, ys=ys, H=self.H, W=self.W, wkt=wkt, crs=pyproj.CRS.from_wkt(wkt),
                    transform=Affine(dx, 0.0, float(xs[0]) - dx / 2, 0.0, dy,
                                     float(ys[0]) - dy / 2))

    @cached_property
    def nb(self):
        """The neighbourhood superset: ``nb`` (flat), per-station geometry
        (``stations``) and ``nbhd(code)`` -> that station's pixels."""
        z = product.load_npz(self._p(NB_NPZ))
        meta = json.loads(str(z["meta"]))
        ptr, allpx = z["nbhd_ptr"], z["nbhd"]
        codes = [s["code"] for s in meta["stations"]]
        per = {c: allpx[ptr[i]:ptr[i + 1]] for i, c in enumerate(codes)}
        return {"nb": z["nb"], "stations": meta["stations"], "rule": meta["rule"],
                "nbhd": per}

    @cached_property
    def s0(self):
        """S0, the fixed evaluation support (v2 §2.10; sorted flat indices,
        int64; a subset of dom)."""
        return self._npz_member(NB_NPZ, "s0").astype(np.int64)

    @cached_property
    def levels(self):
        """The exact level table (``levels.LevelTable``); beyond the table it
        calls the live provider (:meth:`provider`)."""
        return levels.LevelTable.load(self._p(LEVELS_NPZ), provider=self.provider)

    @cached_property
    def series1(self):
        """The 1-min boundary series ``(t_min, h)`` (v8's grid and tag)."""
        return levels.load_series_1min(self._p(SERIES_CACHE),
                                       name=self.record["boundary"]["name"])

    @cached_property
    def series10(self):
        """The 10-min boundary series ``(DatetimeIndex, h)``."""
        return levels.series_10min(*self.series1)

    @cached_property
    def marks(self):
        return levels.load_marks(self._p(MARKS_NPZ))

    @property
    def marks_lp(self):
        return self.marks["marks_lp"]

    @property
    def marks_raw(self):
        return self.marks["marks_raw"]

    @cached_property
    def scene_stats(self):
        """Pass 1's per-scene statistics, floats bit-exact (:func:`read_scene_stats`)."""
        return read_scene_stats(self._p(SCENE_STATS_CSV))

    @cached_property
    def subsets_stored(self):
        return product.load_npz(self._p(SUBSETS_NPZ))

    @cached_property
    def judges(self):
        """``judges.build(self)`` (the score-side judges cache)."""
        try:
            from experiments.benchmark import judges as _judges
        except ImportError as e:
            raise PackageError("experiments.benchmark.judges is not available") from e
        return _judges.build(self)

    def provider(self):
        """The live boundary provider (built once; asserted to be the one
        that built the package)."""
        if self._provider is None:
            p = levels.make_provider(self.cfg)
            name = levels.provider_name(self.cfg, p)
            if name != self.record["boundary"]["name"]:
                raise PackageError(f"the live provider is {name!r}, the package was built "
                                   f"with {self.record['boundary']['name']!r}")
            self._provider = p
        return self._provider

    # reading ----------------------------------------------------------------
    def _sel(self, sel):
        sel = np.arange(self.T) if sel is None else np.asarray(sel)
        if sel.dtype == bool:
            if sel.shape != (self.T,):
                raise ValueError(f"a boolean sel must have length {self.T}")
            sel = np.flatnonzero(sel)
        sel = sel.astype(np.int64)
        if sel.ndim != 1 or (sel.size and (sel[0] < 0 or sel[-1] >= self.T
                                           or np.any(np.diff(sel) <= 0))):
            raise ValueError("sel must be sorted unique record positions in [0, T)")
        return sel

    def positions(self, flat):
        """Flat frame indices -> positions in dom (KeyError if absent)."""
        return domain.positions_in(self.dom, flat)

    def observe(self, sel, pos, rule=obs.MASK_RULE, mode=obs.INDEX_MODE):
        """``(ndwi float32 (n, m), V bool (n, m))`` of the record scenes
        ``sel`` at the package positions ``pos`` (``obs.observe`` on the
        stored raw values)."""
        sel = self._sel(sel)
        pos = np.asarray(pos, np.int64)
        g = _take(self.B03, sel, pos)
        n = _take(self.B08, sel, pos)
        s = _take(self.SCL, sel, pos)
        return obs.observe(g, n, s, self.fill, rule, mode)

    def blocks(self, sel, pos, size=200_000, rule=obs.MASK_RULE, mode=obs.INDEX_MODE):
        """Yield ``(a, b, ndwi, V)`` for consecutive column blocks
        ``pos[a:b]`` of at most ``size`` positions."""
        sel = self._sel(sel)
        pos = np.asarray(pos, np.int64)
        for a in range(0, len(pos), int(size)):
            b = min(a + int(size), len(pos))
            nd, V = self.observe(sel, pos[a:b], rule, mode)
            yield a, b, nd, V

    def count(self, sel=None, pos=None, rule=obs.MASK_RULE, mode=obs.INDEX_MODE,
              budget_mb=256):
        """Valid and wet observation counts per column over the scenes
        ``sel`` (int32 ``(clr, wet)``), at ``pos`` or every package pixel,
        read in row chunks of about ``budget_mb``."""
        sel = self._sel(sel)
        m = self.P if pos is None else len(pos)
        clr = np.zeros(m, np.int32)
        wet = np.zeros(m, np.int32)
        k = max(1, int(budget_mb * 1e6 // (max(m, 1) * 20)))
        for a in range(0, len(sel), k):
            rows = sel[a:a + k]
            g = _take(self.B03, rows, pos)
            n = _take(self.B08, rows, pos)
            s = _take(self.SCL, rows, pos)
            nd, V = obs.observe(g, n, s, self.fill, rule, mode)
            clr += V.sum(axis=0, dtype=np.int32)
            wet += obs.wet(nd, V).sum(axis=0, dtype=np.int32)
        return clr, wet

    # subsets ------------------------------------------------------------------
    def subset(self, family, N, rep=0):
        """Sorted record positions of ``(family, N, rep)`` (§2.9); a stored
        draw is asserted equal to the formula."""
        z = self.subsets_stored
        rules = {r: (z[f"rule_{r}"] if bool(z[f"rule_{r}_available"]) else None)
                 for r in RULES}
        sel = subset_positions(self.site, self.T, family, N, rep, rules)
        if family == "rand" and N != config.N_ALL and int(rep) < len(z["perm"]):
            if not np.array_equal(sel, np.sort(z["perm"][int(rep)][:int(N)])):
                raise PackageError("the stored permutation differs from the formula")
        if family == "win" and N != config.N_ALL and int(N) in z["win_N"].tolist() \
                and int(rep) < z["win_start"].shape[1]:
            i = z["win_N"].tolist().index(int(N))
            if int(z["win_start"][i, int(rep)]) != int(sel[0]):
                raise PackageError("the stored window start differs from the formula")
        return sel

    @staticmethod
    def subset_key(sel):
        """``sha256(sel.astype("<i8").tobytes())``."""
        return product.subset_key(sel)


def load(site, res_m, scenario=None, *, cfg=None, provider=None, full=False, strict=True):
    """The site's package, checked against its manifest on EVERY call.

    Each call re-stats every keyed file (size and mtime_ns) and the inputs
    (size), so a file changed after an earlier load in the same process is
    refused; ``full=True`` also rehashes every file. Nothing is memoised:
    the cheap check costs one ``os.stat`` per file.

    Parameters
    ----------
    site, res_m, scenario
    cfg : config.SiteConfig, optional
        Another configuration (tests, the positive control).
    provider : optional
        The live provider for beyond-table levels (default: built on first
        use with the production call).
    full, strict
        See :func:`verify_manifest`.
    """
    cfg = cfg or config.site_config(site, res_m, scenario)
    d = cfg.common_dir
    mp = os.path.join(d, MANIFEST_JSON)
    if not os.path.exists(mp):
        raise PackageError(f"{d} has no manifest: run prepare {cfg.site} --res {cfg.res_m}")
    key = verify_manifest(d, full=full, strict=strict)
    man = product.read_json(mp)
    if man.get("package_key") != key:
        raise PackageError(f"{mp} changed while the package was being loaded")
    return CommonPackage(cfg, man, provider=provider)


def status(site, res_m, scenario=None, cfg=None):
    """Steps and manifest state of a site's package (for the CLI)."""
    cfg = cfg or config.site_config(site, res_m, scenario)
    d = cfg.common_dir
    out = {"dir": d, "steps": step_status(d) if os.path.isdir(d) else
           {s: "pending" for s in STEPS}, "package_key": None, "manifest": "absent"}
    mp = os.path.join(d, MANIFEST_JSON)
    if os.path.exists(mp):
        try:
            out["package_key"] = verify_manifest(d, full=False, strict=True)
            out["manifest"] = "ok"
        except StalePackageError as e:
            out["manifest"] = f"stale: {e}"
        except PackageError as e:
            out["manifest"] = f"invalid: {e}"
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  G-keep, data part (§4.1)
# ─────────────────────────────────────────────────────────────────────────────

def check_extraction(pkg, tmpdir=None, block=100_000, s_km_reference=None):
    """The package against the production extraction at keep_prod (G-keep).

    Raw-mode NDWI from the stored arrays must equal ``ex["Y"][time_index]``
    and the marea_clear mask ``ex["C"][time_index]``, bitwise (NaN payloads
    included); the extraction's members are streamed one at a time
    (``v7.npz_rows``, memmapped in ``tmpdir`` when large). The production
    s_km recomputed from keep_prod, sea_prod and inter_prod (§2.6) must
    equal the stored one as float32, NaN equal to NaN (``s_km_reference``,
    else the extraction's ``s_km`` when it has one).

    Returns
    -------
    dict
        What was compared; raises AssertionError at the first difference.
    """
    import tempfile

    from experiments import v7_waterline_contour as v7

    cfg = pkg.cfg
    if cfg.extract is None:
        raise ValueError(f"{pkg.site} has no production extraction")
    path = _abs(cfg.extract)
    sel = np.arange(pkg.T)
    pos = pkg.positions(pkg.keep_prod)
    P = len(pos)
    out = {"n_scenes": pkg.T, "n_keep_prod": P, "n_nan_ndwi": 0, "n_clear": 0}
    with tempfile.TemporaryDirectory(dir=tmpdir, prefix="v13_gkeep_") as td:
        Y = v7.npz_rows(path, "Y", pkg.time_index, td)
        for a in range(0, P, block):
            b = min(a + block, P)
            nd, _ = pkg.observe(sel, pos[a:b], rule="marea_clear", mode="raw")
            y = np.ascontiguousarray(Y[:, a:b], np.float32)
            if not np.array_equal(nd.view(np.int32), y.view(np.int32)):
                i, j = np.argwhere(nd.view(np.int32) != y.view(np.int32))[0]
                raise AssertionError(f"raw NDWI differs from the extraction's Y at record "
                                     f"scene {i}, keep_prod pixel {a + j}: {nd[i, j]!r} != "
                                     f"{y[i, j]!r}")
            out["n_nan_ndwi"] += int(np.count_nonzero(np.isnan(nd)))
        del Y
        gc.collect()
        C = v7.npz_rows(path, "C", pkg.time_index, td)
        for a in range(0, P, block):
            b = min(a + block, P)
            m = obs.mask_valid(_take(pkg.SCL, sel, pos[a:b]), "marea_clear")
            c = np.asarray(C[:, a:b], bool)
            if not np.array_equal(m, c):
                i, j = np.argwhere(m != c)[0]
                raise AssertionError(f"the marea_clear mask differs from the extraction's C "
                                     f"at record scene {i}, keep_prod pixel {a + j}")
            out["n_clear"] += int(np.count_nonzero(m))
        del C
        gc.collect()
    ref = s_km_reference
    if ref is None:
        with np.load(path, allow_pickle=False) as z:
            ref = z["s_km"] if "s_km" in z.files else None
    if ref is not None:
        prod = domain.production(pkg)
        mine = prod.s_km.astype(np.float32)
        ref = np.asarray(ref, np.float32)
        if not np.array_equal(mine, ref, equal_nan=True):
            n = int(np.count_nonzero(~((mine == ref) | (np.isnan(mine) & np.isnan(ref)))))
            raise AssertionError(f"the recomputed production s_km differs at {n} keep_prod px")
        out["s_km"] = {"n": int(len(ref)), "n_nan": int(np.count_nonzero(np.isnan(ref))),
                       "K_prod": prod.K}
    else:
        out["s_km"] = "no stored s_km to compare"
    out["result"] = "identical (raw NDWI, marea_clear mask)"
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  VALIDATION V2: year shards of the common package (spec §2.10, §13.1-13.4)
# ═════════════════════════════════════════════════════════════════════════════
#
# Layout under <root>/common/<site>/ (root = results/validation_v2):
#
#   record.json                      the record over the shard years (instants,
#                                    h_B, rising, offsets, lattice20, grid,
#                                    store fingerprints, IOC chunks)
#   years/<Y>/pass1/                 uint8 counters per observation key, frame
#                                    bits (P0 V, SCL bad, production wet/dry,
#                                    C3 at both candidate p), scene_stats.csv,
#                                    SCL x CLD cross-tabs
#   years/<Y>/ocm/                   OmniCloudMask clear bits on the OCM region,
#                                    C11 counters and frame bits, ocm.json
#   geometry.npz                     keep_full(P0, W3), sea, s, bands, Omega
#   support.npz                      NB, D_truth (when the truth exists), S0,
#                                    pre-dom, the OCM region
#   dom/<dom_version>/dom.npz        dom (closed under the 20 m lattice blocks)
#   dom/current.json                 the current dom_version, frozen at lock-A
#   shards/<dom_version>/<Y>/        pixel-major tiles per band
#                                    (n_tiles, T_Y, 4096), cloud bits, counters
#                                    of the threshold detectors, manifest.json
#   settings.json, years/<Y>/settings/thresholds.npz
#   mnar.csv, mnar.json, subsets_<window>.npz, levels.npz
#   boundary_1min.npz (+ .key.json), marks.npz   the 1-min h_B and its marks
#   levels_<LB|LT|LG|LM_<dataset>>.npz, sources.json   the level sources (§1.6)
#   steps_v2.json                    step bookkeeping (not a package file): per
#                                    step its outputs, the upstream steps it
#                                    read and the sha256 of its other inputs
#                                    (instants, stores, offsets, IOC chunks,
#                                    D_truth, OCM region, CMEMS manifests, ...)
#
# Every pass checks the free disk space before creating its memory-mapped
# files (SPACE_MARGIN_GB) and allocates them (posix_fallocate), so a full
# disk is an error, not a SIGBUS; leftovers of killed runs are removed when
# the step runs again. The OCM step checkpoints after every stored time
# block and resumes there.

import dataclasses as _dc
from typing import Mapping as _Mapping, Optional as _Optional

V2_FORMAT = "v2-package-1"
V2_RECORD_FORMAT = "v2-record-1"
V2_ROOT = "results/validation_v2"
#: pixels per pixel-major tile (spec §13.4)
TILE = 4096
#: the bands stored at dom (B8A added to the spec's six: NDWI@20 and the
#: OmniCloudMask card read it; declared)
TILE_BANDS = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12", "SCL", "CLD")
TILE_DTYPES = {**{b: np.int16 for b in ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")},
               "SCL": np.uint8, "CLD": np.uint8}
TILE_FILL = {**{b: obs.FILL for b in ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")},
             "SCL": obs.SCL_NODATA_U8, "CLD": obs.CLD_NODATA_U8}
#: the cloud bits stored per observation at dom (one uint8)
CLOUDBITS = {"C3p20": 1, "C3p35": 2, "ocm_clear": 4, "ocm_region": 8}
#: the two candidate C3 cut-offs of the cross-tab rule (§2.2): both are
#: counted before the rule decides
C3_P_CANDIDATES = (20, 35)
#: rows read beyond a row block for the C3/C5 neighbourhood operations:
#: their reach is 2 open_px + cloud_px = 12 px < 16, so the rows kept equal
#: the whole-frame computation
C3_HALO = 16
#: the OmniCloudMask region and the D3 region: the bounding box plus 1 km
REGION_PAD_M = 1000.0
#: pass 2 is refused above this |dom| / |S0| (spec §2.10)
DOM_MAX_OVER_S0 = 1.5
#: the projected W9 package of all sites must fit this (spec §14.4: W9
#: ~130 GB, plus 15 %)
PACKAGE_W9_BUDGET_GB = 150.0
W9_SCENES = 1390
#: bytes per stored observation: 7 int16 bands, SCL, CLD, the cloud bits
BYTES_PER_OBS = 7 * 2 + 3
#: uint8 counters: no year holds more scenes
YEAR_MAX_SCENES = 255
P0_KEY = "C1.floor0.ndwi.D1"
W3_YEARS = (2023, 2024, 2025)
#: SCL classes of the C2 transition-zone rule and its production observation
TZ_BAD_SCL = (3, 8, 9, 10)
TZ_CLEAR_SCL = (4, 5, 6)
#: Omega (D2 domain): a 300 m ring of permanent water / stable land
OMEGA_RING_M = 300.0
OMEGA_WATER = 0.98
OMEGA_LAND = 0.02
#: the rising flag: h_B(t + 10 min) > h_B(t - 10 min)
RISING_HALF_MIN = 10
#: Tier-1A' pixel cloud levels of the MNAR table (scene rules share C1's)
MNAR_CLOUDS = ("C0", "C1", "C1d", "C3", "C4", "C11")
MNAR_DECILES = 10
V2_STEPS = ("record", "pass1", "geometry", "support", "ocm", "dom", "pass2", "settings",
            "counters", "mnar", "subsets", "levels", "sources", "manifest")
#: steps run only when asked (``truth`` asks for them): the 1-min h_B series
#: over the provider's period and its marks, read by the lag truths
V2_EXTRA_STEPS = ("series",)
SERIES_V2 = "boundary_1min.npz"
MARKS_V2 = "marks.npz"
YEAR_STEPS = ("pass1", "ocm", "pass2", "counters", "manifest")
V2_BOOK = "steps_v2.json"
CROSSTAB_JSON = "crosstab.json"
TRUTH_ELEV_V2 = "truth_elev_v2.npz"
V2_SCENE_COLUMNS = (["q", "local", "date", "year", "offset", "h0", "rising"]
                    + [f"scl_{c}" for c in range(12)] + ["scl_nodata", "cld_fill",
                    "g21_cloud", "g21_nodata", "ard_good", "frame_bad", "v_p0"])
#: free disk space a step must leave beyond the bytes it writes (v13's
#: SPACE_MARGIN_GB): checked before any memory-mapped file is created
SPACE_MARGIN_GB = 20.0
#: the Campaign-1 windows whose EOT20-only share of h_B the record reports
COVERAGE_WINDOWS = ("W3", "W2", "W1", "Y2023", "Y2024", "Y2025")
#: the OmniCloudMask checkpoint of a year (resume after a kill)
OCM_PROGRESS = "ocm_progress.json"
OCM_PARTIAL = ".partial"
#: the level sources of the 'sources' step (spec §1.6) and their index
SOURCES_JSON = "sources.json"
#: the Campaign-1 elevation windows whose LM dataset sources.json resolves
#: (fetch.cmems.choose_dataset per window; none covering: NOT AVAILABLE)
LEVEL_WINDOWS = ("W3", "W1", "Y2023", "Y2024", "Y2025")


class DomFrozenError(PackageError):
    """The frozen dom (lock-A) would change: new years add shards for the
    same dom; anything else needs the user's decision."""


# ─────────────────────────────────────────────────────────────────────────────
#  the site of a v2 package
# ─────────────────────────────────────────────────────────────────────────────

@_dc.dataclass(frozen=True)
class V2Site:
    """A site of validation v2 and where its package lives.

    Attributes
    ----------
    site : str
    cfg : config.SiteConfig
        Boundary recipe, judges, mouth side and pixel size (10 m); its
        ``root`` is set to ``root`` so that ``cfg.truth_dir`` and
        ``cfg.common_dir`` point into the v2 tree.
    root : str
        The validation v2 output root.
    cubes_root : str, optional
        The Zarr cubes root (``zarr_cubes.cubes_root``).
    instants : "catalogue" (default), "labels", a path or a mapping
        The scene instants (``inputs.resolve_instants``): the catalogue /
        product-name times (datatake start) by default, as production (user
        decision of 2026-10-14); a ``sensing_times.json`` only when its
        path is given (CLI ``--sensing-times``).
    offsets : mapping, optional
        Frozen ``{date: 0 or 1000}``; default: the inputs report's, if it
        exists, else the provisional decision (recorded).
    lattice : "measure", (py, px) or None
    years : tuple of int
        The shard years to build (W3: 2023-2025).
    keys : tuple of str, optional
        Observation keys (default ``designs.observation_keys()``, restricted
        to the bands of the source).
    stations_json : str, optional
        The judges' stations file (default the config's).
    source : "zarr" (default) or "netcdf"
        The scene source: the per-year Zarr stores (the reported 10 m runs)
        or the legacy netCDF cube of ``cfg`` (B03, B08, SCL; the local runs
        on the 20 m cubes, never reported numbers).
    boundary : "union" (default) or "production"
        The h_B provider: built once over the union period 2017-2025 after
        ``fetch ioc`` (§1.5), or the production provider over 2023-2025
        (``levels.make_provider``; the local runs on the legacy cubes,
        declared in record.json).
    cube, labels : str, optional
        The legacy cube and its catalogue times (default the config's).
    block_rows : int
        Rows per pass block; must be even (no 20 m block straddles two row
        blocks).
    allow_provisional_offsets : bool
        Build a Zarr record with the provisional per-scene offsets (no
        inputs report for the site); refused otherwise (recorded).
    """

    site: str
    cfg: object
    root: str = V2_ROOT
    cubes_root: _Optional[str] = None
    instants: object = "catalogue"
    offsets: _Optional[_Mapping] = None
    lattice: object = "measure"
    cache_mb: float = 1024.0
    block_rows: int = 256
    years: tuple = W3_YEARS
    keys: _Optional[tuple] = None
    stations_json: _Optional[str] = None
    source: str = "zarr"
    boundary: str = "union"
    cube: _Optional[str] = None
    labels: _Optional[str] = None
    allow_provisional_offsets: bool = False

    def __post_init__(self):
        object.__setattr__(self, "cfg", _dc.replace(self.cfg, root=str(self.root)))
        object.__setattr__(self, "years", tuple(int(y) for y in self.years))
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}; expected one of {SOURCES}")
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"unknown boundary {self.boundary!r}; expected one of {BOUNDARIES}")
        if int(self.block_rows) <= 0 or int(self.block_rows) % 2:
            raise ValueError(f"block_rows must be a positive even number, not {self.block_rows} "
                             "(a 20 m block must not straddle two row blocks)")

    def __hash__(self):
        return hash((self.site, self.root, self.years))

    @property
    def dir(self):
        return f"{self.root}/common/{self.site}"

    @property
    def truth_dir(self):
        return f"{self.root}/truth/{self.site}"

    @property
    def pixel_m(self):
        return float(self.cfg.pixel_m)

    @property
    def bands(self):
        """The bands the source delivers (in the tiles' order)."""
        from experiments.benchmark import inputs
        have = inputs.LEGACY_BANDS if self.source == "netcdf" else TILE_BANDS
        return tuple(b for b in TILE_BANDS if b in have)

    @property
    def legacy_cube(self):
        return self.cube or self.cfg.cube

    @property
    def legacy_labels(self):
        return self.labels or config.resolve(self.cfg.overpass)

    def _planned_keys(self):
        if self.keys is not None:
            return tuple(self.keys)
        from experiments.benchmark import designs
        return tuple(designs.observation_keys())

    def observation_keys(self):
        """The observation keys the package counts: the planned keys whose
        bands the source delivers (:meth:`unavailable_keys` lists the rest)."""
        have = set(self.bands)
        return tuple(k for k in self._planned_keys()
                     if set(settings.Setting.parse(k).bands) <= have)

    def unavailable_keys(self):
        """``{key: missing bands}`` of the planned keys the source cannot serve."""
        have = set(self.bands)
        out = {}
        for k in self._planned_keys():
            miss = sorted(set(settings.Setting.parse(k).bands) - have)
            if miss:
                out[k] = miss
        return out


#: scene sources and boundary providers of a V2Site
SOURCES = ("zarr", "netcdf")
BOUNDARIES = ("union", "production")


def v2_site(site, root=V2_ROOT, cubes_root=None, res=10, **kw):
    """The :class:`V2Site` of a real site: the Zarr stores on the 10 m grid
    (``res=10``, the reported runs), or the legacy netCDF cube of the
    ``res`` grid with the production boundary provider (``res=20``: local
    software checks only; ``source``/``boundary`` override)."""
    if int(res) != 10:
        kw.setdefault("source", "netcdf")
        kw.setdefault("boundary", "production")
    return V2Site(site=site, cfg=config.site_config(site, int(res)), root=root,
                  cubes_root=cubes_root, **kw)


def boundary_provider(site2):
    """The h_B provider of a site: the union provider (§1.5) or, for
    ``boundary="production"``, ``levels.make_provider`` over the production
    period 2023-2025 (local runs on the legacy cubes)."""
    if site2.boundary == "production":
        return levels.make_provider(site2.cfg)
    return union_provider(site2.cfg)


def boundary_period(site2):
    """The inclusive period of the 1-min h_B series of a site's provider."""
    if site2.boundary == "production":
        return tuple(config.PERIOD)
    return ("2017-01-01", "2025-12-31")


def _p(site2, *parts):
    return os.path.join(site2.dir, *[str(p) for p in parts])


def _year_dir(site2, year, *parts):
    return _p(site2, "years", str(int(year)), *parts)


def _shard_dir(site2, dv, year, *parts):
    return _p(site2, "shards", dv, str(int(year)), *parts)


# ─────────────────────────────────────────────────────────────────────────────
#  observation keys
# ─────────────────────────────────────────────────────────────────────────────

def threshold_key(key):
    """True when the key's detector or index needs the settings pass (D2,
    D3, or NIR's data-dependent t_k): counted at dom after it."""
    s = settings.Setting.parse(key)
    return settings.DETECT[s.detect].per_scene or settings.INDEX[s.index].t_k is None


def counter_name(key, p=None):
    """File stem of a counter: the key, ``~p<p>`` for C3's candidates."""
    return key if p is None else f"{key}~p{int(p)}"


def pass1_plan(keys):
    """``[(name, Setting, p)]`` counted in pass 1: every key whose pixel rule
    is not C11 and that needs no settings pass, the C1 analog of each such
    C11 key, P0, and C3 at both candidate cut-offs."""
    out = {}
    want = set(keys) | {P0_KEY}
    for k in list(want):
        s = settings.Setting.parse(k)
        if s.pixel_rule == "C11":
            want.add(s.replace(cloud="C1").id)
    for k in sorted(want):
        s = settings.Setting.parse(k)
        if s.pixel_rule == "C11" or threshold_key(k):
            continue
        if s.pixel_rule == "C3":
            for p in C3_P_CANDIDATES:
                out[counter_name(k, p)] = (s, p)
        else:
            out[k] = (s, None)
    return [(n, s, p) for n, (s, p) in sorted(out.items())]


def ocm_plan(keys):
    """``[(name, Setting)]`` of the C11 keys counted in the OCM step (the
    threshold ones are counted at dom after the settings pass); C11's P0
    analog always."""
    want = {k for k in keys if settings.Setting.parse(k).pixel_rule == "C11"}
    want.add(settings.P0.replace(cloud="C11").id)
    return [(k, settings.Setting.parse(k)) for k in sorted(want) if not threshold_key(k)]


def dom_counter_plan(keys):
    """Keys counted at dom after the settings pass (threshold detectors and NIR)."""
    return [k for k in sorted(set(keys)) if threshold_key(k)]


# ─────────────────────────────────────────────────────────────────────────────
#  the observation of a block, with shared preprocessing and indices
# ─────────────────────────────────────────────────────────────────────────────

class _ObsCache:
    """``obs.observe_setting`` for many settings of one block, computing each
    preprocessed band, index and pixel rule once (same arithmetic; tested
    equal to observe_setting)."""

    def __init__(self, raw, offset, lattice, fill=obs.FILL):
        self.raw = raw
        self.offset = offset
        self.lattice = lattice
        self.fill = fill
        self.scl8 = obs._as_scl8(raw["SCL"], fill)
        self._prep, self._X, self._cv = {}, {}, {}

    def prepared(self, prep, band):
        k = (prep, band)
        if k not in self._prep:
            self._prep[k] = obs.prep_band(self.raw[band], self.offset, prep, self.fill)
        return self._prep[k]

    def X(self, prep, index):
        k = (prep, index)
        if k not in self._X:
            e = settings.INDEX[index]
            self._X[k] = obs.compute_index({b: self.prepared(prep, b) for b in e.bands},
                                           index, self.lattice)
        return self._X[k]

    def cloud(self, level):
        if level not in self._cv:
            self._cv[level] = obs.pixel_valid(level, self.scl8, self.raw.get("CLD"), None,
                                              None, self.fill)
        return self._cv[level]

    def observe(self, s, cloud_valid=None, params=None, threshold=None):
        """``(X, V, W)`` of setting ``s`` (a Setting)."""
        de = settings.DETECT[s.detect]
        if params is not None:
            t_k = float(params["t_k"])
        else:
            # V and W need only t_k (s_k scales Y); a fixed t_k is the registry's
            t_k = settings.INDEX[s.index].t_k
            if t_k is None:
                raise ValueError(f"{s.index}: t_k is data-dependent; pass params")
            t_k = float(t_k)
        cv = self.cloud(s.pixel_rule) if cloud_valid is None else np.asarray(cloud_valid, bool)
        X = self.X(s.prep, s.index)
        V = cv & np.isfinite(X)
        for b in obs.spectral_bands(s):
            V &= ~np.isnan(self.prepared(s.prep, b))
        if s.detect == "D1":
            W = obs.detect_d1(X, V, t_k)
        elif de.per_scene:
            if threshold is None:
                raise ValueError(f"{s.detect} needs the per-scene thresholds")
            W = obs.detect_threshold(X, V, threshold)
        elif s.detect == "D4":
            W = obs.detect_d4(V, self.scl8)
        elif s.detect == "D5":
            W = obs.detect_d5(V, *[self.X(s.prep, i) for i in de.indices],
                              votes=int(de.params["votes"]))
        else:
            raise ValueError(f"no detector implementation for {s.detect}")
        return X, V, W


def _neighbourhood_valid(src, qs, r0, r1, level, p=None, cloud_params=None):
    """C3 (at cut-off ``p``) or C5 of record positions ``qs`` on rows
    [r0, r1), computed on the rows plus a halo of :data:`C3_HALO` rows, so
    the result equals the whole-frame computation."""
    H = src.grid.H
    a0, a1 = max(0, r0 - C3_HALO), min(H, r1 + C3_HALO)
    prm = dict(settings.CLOUD[level].params)
    prm.update(cloud_params or {})
    if 2 * int(prm["open_px"]) + float(prm["cloud_px"]) >= C3_HALO:
        raise AssertionError("the C3/C5 neighbourhood reach exceeds the halo")
    if level == "C3":
        d = src.read_block(qs, ["SCL", "CLD"], slice(a0, a1))
        v = obs.cloud_c3(d["SCL"], d["CLD"], int(p if p is not None else prm["p"]),
                         int(prm["open_px"]), float(prm["cloud_px"]), float(prm["shadow_px"]))
    elif level == "C5":
        d = src.read_block(qs, ["SCL"], slice(a0, a1))
        v = obs.cloud_c5(d["SCL"], int(prm["open_px"]), float(prm["cloud_px"]),
                         float(prm["shadow_px"]))
    else:
        raise ValueError(f"{level} has no neighbourhood operation")
    return v[:, r0 - a0:r1 - a0, :]


def _pack_rows(mask):
    """``np.packbits`` along the last axis (frame bits are (T, H, ceil(W/8)))."""
    return np.packbits(np.asarray(mask, bool), axis=-1)


def _unpack_rows(packed, W):
    return np.unpackbits(np.asarray(packed), axis=-1, count=int(W)).astype(bool)


def _scl_codes13(scl8):
    """SCL as 0..11 with no data -> 12 (uint8)."""
    s = np.asarray(scl8)
    return np.where(s == obs.SCL_NODATA_U8, 12, s).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
#  step bookkeeping
# ─────────────────────────────────────────────────────────────────────────────

_SHA_LIMIT = 64 << 20


def _sig_entry(path):
    """Bookkeeping entry of an output: size, mtime_ns and sha256 (files above
    64 MB: size and mtime only; the manifest hashes them in full)."""
    st = os.stat(path)
    e = {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    if st.st_size <= _SHA_LIMIT:
        e["sha256"] = product.sha256_file(path)
    return e


class _V2Book:
    """``steps_v2.json``: per step key (``step`` or ``step:<year>`` or
    ``step:<dom_version>:<year>``) its outputs (relative paths), the
    signatures of the upstream steps it read, the sha256 of the other
    inputs it read (``extra``: files outside the package, the region, the
    plan, ...) and a short info.

    A step is done when its outputs are intact, it read the SAME upstream
    steps that it would read now (a step that gained an upstream, e.g. pass
    2 after the OCM step ran, is stale), their signatures are unchanged and,
    when the caller passes them, its other inputs are unchanged."""

    def __init__(self, site2):
        self.site2 = site2
        self.path = _p(site2, V2_BOOK)
        self.doc = {"format": "v2-steps-1", "steps": {}}
        if os.path.exists(self.path):
            try:
                d = product.read_json(self.path)
                if d.get("format") == "v2-steps-1":
                    self.doc = d
            except (OSError, ValueError):
                pass

    def rec(self, key):
        return self.doc["steps"].get(key)

    def sig(self, key):
        r = self.rec(key)
        return None if r is None else r["sig"]

    def _ok(self, rel, e):
        p = _p(self.site2, rel)
        if not os.path.exists(p):
            return False
        st = os.stat(p)
        if st.st_size != e["size"]:
            return False
        if st.st_mtime_ns != e["mtime_ns"]:
            return "sha256" in e and product.sha256_file(p) == e["sha256"]
        return True

    def why_stale(self, key, upstream=None, extra=None):
        """None when the step is done, else the reason it must run."""
        r = self.rec(key)
        if r is None:
            return "never ran"
        bad = [rel for rel, e in r["outputs"].items() if not self._ok(rel, e)]
        if bad:
            return f"outputs changed ({bad[:3]})"
        if upstream is not None and sorted(r["reads"]) != sorted(upstream):
            return (f"its upstream steps changed ({sorted(r['reads'])} -> "
                    f"{sorted(upstream)})")
        changed = [u for u, s in r["reads"].items() if self.sig(u) != s]
        if changed:
            return f"upstream {changed} rebuilt"
        if extra is not None and r.get("extra_sha256") != config.sha256_json(extra):
            old = r.get("extra") or {}
            diff = sorted(k for k in set(old) | set(extra) if old.get(k) != extra.get(k))
            return f"its inputs changed ({diff or 'format'})"
        return None

    def done(self, key, upstream=None, extra=None):
        return self.why_stale(key, upstream, extra) is None

    def record(self, key, outputs, upstream, info=None, seconds=None, extra=None):
        outs = {}
        for rel in outputs:
            rel = rel.replace("\\", "/")
            outs[rel] = _sig_entry(_p(self.site2, rel))
        sig = config.sha256_json({k: e.get("sha256", f"{e['size']}:{e['mtime_ns']}")
                                  for k, e in sorted(outs.items())})
        self.doc["steps"][key] = {"outputs": outs, "reads": {u: self.sig(u) for u in upstream},
                                  "sig": sig, "info": info or {}, "seconds": seconds,
                                  "extra": None if extra is None else config._plain(extra),
                                  "extra_sha256": None if extra is None
                                  else config.sha256_json(extra),
                                  "done_utc": product.utc_stamp(),
                                  "host": socket.gethostname()}
        product.write_json(self.path, self.doc)

    def status(self):
        return {k: ("done" if self.done(k) else "stale") for k in sorted(self.doc["steps"])}


def _rel2(site2, path):
    return os.path.relpath(path, site2.dir).replace("\\", "/")


# ─────────────────────────────────────────────────────────────────────────────
#  record (site)
# ─────────────────────────────────────────────────────────────────────────────

def _frozen_offsets_for(site2):
    """The inputs report's frozen offsets of the site, or None."""
    if site2.offsets is not None:
        return {str(k)[:10]: int(v) for k, v in dict(site2.offsets).items()}, "given"
    from experiments.benchmark import inputs
    p = os.path.join(site2.root, "common", "inputs_report.json")
    if not os.path.exists(p):
        return None, "provisional (no inputs report)"
    rep = product.read_json(p)
    s = rep.get("sites", {}).get(site2.site, rep if rep.get("site") == site2.site else None)
    if not s or "offsets" not in s:
        return None, "provisional (site not in the inputs report)"
    return inputs.frozen_offsets(s), f"inputs report {p}"


_UNSET = object()


def _open_source(site2, keep_dates=None, offsets=None, lattice=_UNSET, instants=None):
    """The site's scene source over its shard years: ``inputs.ZarrSource``,
    or ``inputs.NetCDFSource`` on the legacy cube (``source="netcdf"``; no
    20 m sub-lattice on a 20 m grid; o_s = 0, offset-harmonised values)."""
    from experiments.benchmark import inputs
    y0, y1 = min(site2.years), max(site2.years)
    inst = site2.instants if instants is None else instants
    lat = site2.lattice if lattice is _UNSET else lattice
    if site2.source == "netcdf":
        if lattice is _UNSET and site2.pixel_m >= 20:
            lat = None
        return inputs.NetCDFSource(site2.legacy_cube, site2.site, inst, start=f"{y0}-01-01",
                                   end=f"{y1 + 1}-01-01", keep_dates=keep_dates,
                                   labels_path=site2.legacy_labels, lattice=lat,
                                   cache_mb=site2.cache_mb)
    return inputs.ZarrSource(site2.site, start=f"{y0}-01-01", end=f"{y1 + 1}-01-01",
                             root=site2.cubes_root, instants=inst,
                             keep_dates=keep_dates, offsets=offsets, lattice=lat,
                             cache_mb=site2.cache_mb)


#: the v1 name (the Zarr stores were the only source)
_open_zarr = _open_source


def instants_identity(site2):
    """The instants a run asks for (``inputs.instants_identity``):
    ``{"source", "path", "sha256"}`` without opening a store; the sha256
    covers the dates of the site's years only (catalogue dates the
    downloader adds for other years change nothing)."""
    from experiments.benchmark import inputs
    if site2.source == "netcdf":
        labels = inputs.file_labels(site2.legacy_labels)
    else:
        labels = inputs.catalogue_labels(site2.site, site2.cubes_root)
    return inputs.instants_identity(site2.instants, labels, years=site2.years)


def _norm_path(p):
    return None if p is None else os.path.normcase(os.path.abspath(str(p)))


def check_record_instants(site2, rec):
    """Refuse a run whose instants are another request than the record's
    (another source, e.g. a ``--sensing-times`` file against catalogue
    times, or another file): the record is never reused silently with other
    instants. The same file with other content rebuilds the record (its
    step inputs change). Returns the requested identity."""
    want = instants_identity(site2)
    have_src, have_path = rec.get("instants"), rec.get("instants_path")
    if want["source"] != have_src or _norm_path(want["path"]) != _norm_path(have_path):
        have = have_src if not have_path else f"{have_src} ({have_path})"
        asked = want["source"] if not want["path"] else f"{want['source']} ({want['path']})"
        raise PackageError(
            f"{site2.site}: record.json was built with the {have} instants; this run asks "
            f"for the {asked} instants. Pass the record's instants, or rebuild it with "
            "--force --step record (every downstream step follows)")
    return want


def _file_id(path):
    """``{name, size, mtime_ns}`` of a file (None when absent): a cheap
    change detector for a step's inputs outside the package."""
    if path is None or not os.path.isfile(path):
        return None
    st = os.stat(path)
    return {"name": os.path.basename(path), "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns)}


def _small_sha(path, limit=_SHA_LIMIT):
    """sha256 of a small file, else its :func:`_file_id`; None when absent."""
    if path is None or not os.path.isfile(path):
        return None
    if os.path.getsize(path) <= limit:
        return product.sha256_file(path)
    return _file_id(path)


def _ioc_ids(site2):
    """File ids of every IOC chunk the site's boundary provider reads (the
    union cache, or the production cache for ``boundary="production"``)."""
    codes = levels.boundary_codes(site2.cfg) if \
        site2.cfg.boundary.get("kind") == "make_boundary" else []
    if not codes:
        return []
    if site2.boundary == "union":
        from experiments.benchmark.fetch import ioc as fioc
        cache = fioc.UNION_CACHE
    else:
        cache = config.GAUGE_CACHE
    out = []
    for code in codes:
        for p in sorted(glob.glob(os.path.join(cache, f"ioc_{code}_*.json"))):
            out.append(_file_id(p))
    return out


def record_inputs(site2):
    """What the record reads besides the code (its step inputs): the
    instants, the stores' consolidated metadata (or the cube), the frozen
    per-scene offsets and the IOC chunk files. A change rebuilds the record
    and everything downstream."""
    from experiments import zarr_cubes as zc
    out = {"instants": instants_identity(site2), "years": [int(y) for y in site2.years],
           "source": site2.source, "boundary": site2.boundary,
           "allow_provisional_offsets": bool(site2.allow_provisional_offsets)}
    if site2.source == "netcdf":
        out["cube"] = _file_id(str(site2.legacy_cube))
    else:
        out["stores"] = {str(y): _small_sha(str(zc.store_path(site2.site, y, site2.cubes_root)
                                                / ".zmetadata"))
                         for y in site2.years}
        offs, src = _frozen_offsets_for(site2)
        out["offsets"] = {"source": src,
                          "sha256": None if offs is None else config.sha256_json(offs)}
    out["ioc"] = _ioc_ids(site2)
    return out


def open_source(site2, rec):
    """The scene source of a built record: its dates (finite h_B), its
    offsets and its lattice, so record positions are the source's positions."""
    inst = {d: pd.Timestamp(t) for d, t in zip(rec["dates"], rec["t_utc"])}
    lat = rec["lattice20"]["phase"]
    src = _open_source(site2, keep_dates=rec["dates"],
                       offsets=dict(zip(rec["dates"], rec["offsets"])),
                       lattice=None if lat is None else tuple(lat), instants=inst)
    if src.dates != list(rec["dates"]):
        src.close()
        raise PackageError(f"{site2.site}: the source's dates differ from record.json "
                           "(inputs changed: rebuild with --force)")
    return src


def v2_record(site2, provider):
    """The record of the shard years (spec §1.4): dates with a valid instant
    (the catalogue times by default, user decision of 2026-10-14) and a
    finite h_B (ONE provider call on the sorted instants), the rising flag,
    the per-scene offsets, lattice20, the grid, the year slices and the
    store fingerprints. Returns the record dict."""
    from experiments.benchmark import inputs
    if site2.source == "netcdf":
        offs, off_src = None, "legacy cube: offset-harmonised values (o_s = 0)"
    else:
        offs, off_src = _frozen_offsets_for(site2)
        if offs is None and not site2.allow_provisional_offsets:
            raise PackageError(
                f"{site2.site}: the per-scene BOA offsets are {off_src}: run 'inputs --sites "
                f"{site2.site}' first (one process for all sites), or pass "
                "--allow-provisional-offsets (recorded in record.json)")
    src = _open_source(site2, offsets=offs)
    try:
        missing_req = sorted(set(site2.years) - {int(y) for y in src.years})
        if missing_req:
            raise PackageError(
                f"{site2.site}: the stores of {missing_req} are not complete in the cubes root "
                f"(present: {list(src.years)}): finish their download first, or pass "
                f"--years with the present years. Nothing was written")
        t = pd.DatetimeIndex(src.instants).as_unit("ns")
        if not len(t):
            raise PackageError(f"{site2.site}: no store date of {site2.years} has an instant")
        h = np.asarray(provider.levels(t), np.float64)
        if h.shape != (len(t),):
            raise AssertionError(f"provider.levels returned {h.shape} for {len(t)} instants")
        fin = np.isfinite(h)
        dt = pd.Timedelta(minutes=RISING_HALF_MIN)
        hm = np.asarray(provider.levels(t - dt), np.float64)
        hp = np.asarray(provider.levels(t + dt), np.float64)
        rising = hp > hm
        dates = [d for d, f in zip(src.dates, fin) if f]
        tab = src.offset_table()
        states = tab.loc[dates, "state"].tolist()
        unclear = [d for d, s_ in zip(dates, states) if s_ == "unclear"]
        if unclear:
            raise inputs.OffsetUnclear(f"{site2.site}: BOA offset unclear on {unclear[:5]} "
                                       f"({len(unclear)} scene(s)); run the inputs report")
        lat = src.lattice20
        lat_rep = src.lattice_report
        grid = src.grid
        years_present = list(src.years)
        missing = list(src.missing_years)
        fps = src.fingerprint()
        if site2.source == "netcdf":            # one cube for every shard year
            fps = {str(y): fps for y in years_present}
        inst_src = src.instant_source
        inst_path = getattr(src, "instant_path", None)
        inst_sha = src.instants_sha256
        sens_sha = (sha256_raw(inst_path) if inst_src == "sensing_times" and inst_path
                    else None)
        src_desc = {"kind": site2.source, "bands": list(src.bands),
                    "path": (str(site2.legacy_cube).replace("\\", "/")
                             if site2.source == "netcdf" else None),
                    "pixel_m": site2.pixel_m}
    finally:
        src.close()
    tt = t[fin]
    yrs = np.array([int(d[:4]) for d in dates])
    slices = {}
    for y in sorted(set(yrs.tolist())):
        idx = np.flatnonzero(yrs == y)
        slices[str(y)] = [int(idx[0]), int(idx[-1]) + 1]
        if len(idx) > YEAR_MAX_SCENES:
            raise PackageError(f"{site2.site} {y}: {len(idx)} scenes exceed the uint8 counters")
    offsets = tab.loc[dates, "offset"].astype(int).tolist()
    try:
        cache_dir = None
        if site2.boundary == "union":
            try:
                from experiments.benchmark.fetch import ioc as _fioc
                cache_dir = getattr(_fioc, "UNION_CACHE", None)
            except ImportError:
                pass
        ioc = levels.ioc_chunks(site2.cfg, provider=None, cache_dir=cache_dir)
    except Exception as e:                    # not fatal for tests / fake providers
        ioc = [{"error": f"{type(e).__name__}: {e}"}]
    name = getattr(provider, "name", None) or type(provider).__name__
    try:
        wins = {}
        for wid in COVERAGE_WINDOWS:
            w = settings.window(wid)
            m = (tt >= pd.Timestamp(w.start)) & (tt < pd.Timestamp(w.end))
            if m.any():
                wins[wid] = np.flatnonzero(m)
        cov = levels.boundary_coverage(provider, tt, wins)
        cov["windows"] = {k: round(float(v), 6) for k, v in cov["windows"].items()}
    except Exception as e:                    # diagnostic only: never fails the record
        cov = {"error": f"{type(e).__name__}: {e}"}
    if isinstance(cov.get("share"), float) and cov.get("gauge_parts"):
        log(f"[record] {site2.site}: h_B is EOT20-only (every gauge member missing) at "
            f"{cov['model_only']} of {cov['n']} instants; per window {cov['windows']}")
    return {
        "format": V2_RECORD_FORMAT, "site": site2.site, "years": sorted(slices, key=int),
        "years_requested": list(site2.years), "years_in_stores": years_present,
        "missing_years": missing, "T": len(dates), "dates": dates,
        "t_utc": [x.isoformat() for x in tt], "t_ns": tt.asi8.astype(np.int64).tolist(),
        "h0": [float(v) for v in h[fin]], "rising": [bool(v) for v in rising[fin]],
        "offsets": offsets, "offset_source": off_src,
        "offset_states": tab.loc[dates, "state"].tolist(),
        "year_slices": slices, "nonfinite_level": [d for d, f in zip(src.dates, fin) if not f],
        "lattice20": {"phase": None if lat is None else [int(v) for v in lat],
                      "report": lat_rep},
        "grid": {"H": int(grid.H), "W": int(grid.W), "xs": np.asarray(grid.xs).tolist(),
                 "ys": np.asarray(grid.ys).tolist(), "crs_wkt": grid.crs_wkt,
                 "sha256": grid.sha256()},
        "pixel_m": site2.pixel_m, "mouth_side": site2.cfg.mouth_side,
        "instants": inst_src, "instants_path": inst_path, "instants_sha256": inst_sha,
        "sensing_sha256": sens_sha,
        "fingerprints": fps, "source": src_desc,
        "boundary": {"name": str(name), "kind": site2.boundary,
                     "period": list(boundary_period(site2))},
        "boundary_coverage": cov,
        "keys_unavailable": site2.unavailable_keys(), "ioc_chunks": ioc,
        "provisional_offsets_allowed": bool(site2.source != "netcdf"
                                            and site2.allow_provisional_offsets),
    }


def _read_record(site2):
    p = _p(site2, "record.json")
    if not os.path.exists(p):
        raise PackageError(f"{p} is missing: run prepare {site2.site} --step record")
    return product.read_json(p)


def year_positions(rec, year):
    a, b = rec["year_slices"][str(int(year))]
    return np.arange(a, b, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
#  pass 1 (per year): uint8 counters, frame bits, scene statistics
# ─────────────────────────────────────────────────────────────────────────────

FRAME_BITS = ("P0", "bad", "tzwet", "tzdry")


def v2_pass1(site2, rec, year, keys, log_every=SCENE_STATS_LOG_EVERY):
    """Pass 1 of one year (spec §2.10): one streaming read of the store.

    Writes into ``years/<Y>/pass1/``: ``counters/<key>.npz`` (uint8 ``clr``
    and ``wet`` (H, W) of every :func:`pass1_plan` key), the frame bits
    ``bits_<name>.npy`` ((T_Y, H, ceil(W/8)) packed rows: P0's V, SCL bad
    {3, 8, 9, 10}, the production wet and dry of the C2 transition zone,
    and C3's V at each candidate p when C3 is planned), ``crosstab.npy``
    ((T_Y, 13, 102) int32 SCL x CLD counts, no data last), and
    ``scene_stats.csv``. Returns a summary.
    """
    t0 = time.time()
    qs = year_positions(rec, year)
    Ty = len(qs)
    plan = pass1_plan(keys)
    d = _year_dir(site2, year, "pass1")
    os.makedirs(os.path.join(d, "counters"), exist_ok=True)
    clean_stale_tmps(d)
    src = open_source(site2, rec)
    H, W = src.grid.shape
    Wp = (W + 7) // 8
    offs_all = np.asarray(rec["offsets"], np.int64)
    bands = set(["SCL", "CLD", "B03", "B08"])
    for _, s, _ in plan:
        bands |= set(s.bands)
    bands = [b for b in TILE_BANDS if b in bands and b in src.bands]   # no CLD in a legacy cube
    c3_ps = sorted({p for _, s, p in plan if p is not None})
    bit_names = list(FRAME_BITS) + [f"C3p{p}" for p in c3_ps]
    need = (len(bit_names) * memmap_bytes((Ty, H, Wp), np.uint8)
            + memmap_bytes((Ty, 13, 102), np.int32) + 2 * len(plan) * H * W)
    try:
        check_space(d, need, f"{site2.site}: pass 1 of {year}")
    except PackageError:
        src.close()
        raise
    clr = {n: np.zeros((H, W), np.uint8) for n, _, _ in plan}
    wet = {n: np.zeros((H, W), np.uint8) for n, _, _ in plan}
    paths = {n: os.path.join(d, f"bits_{n}.npy") for n in bit_names}
    tmps = {n: _tmp(p) for n, p in paths.items()}
    xt_path = os.path.join(d, "crosstab.npy")
    xt_tmp = _tmp(xt_path)
    mms, xt = {}, None
    scl_n = np.zeros((Ty, 13), np.int64)
    v_p0 = np.zeros(Ty, np.int64)
    p0 = settings.Setting.parse(P0_KEY)
    mm = None
    n_blocks, done_scenes = 0, 0
    try:
        for n, tp in tmps.items():
            mms[n] = np.lib.format.open_memmap(tp, mode="w+", dtype=np.uint8, shape=(Ty, H, Wp))
            reserve_file(tp)
        xt = np.lib.format.open_memmap(xt_tmp, mode="w+", dtype=np.int32, shape=(Ty, 13, 102))
        reserve_file(xt_tmp)
        for blk in src.row_blocks(bands, block_rows=site2.block_rows, positions=qs, pair20=True):
            loc = blk.positions - qs[0]
            r0, r1 = blk.rows.start, blk.rows.stop
            off = offs_all[blk.positions]
            cache = _ObsCache(blk.data, off, blk.lattice)
            nb = {}
            for p in c3_ps:
                nb[("C3", p)] = _neighbourhood_valid(src, blk.positions, r0, r1, "C3", p)
            for n, s, p in plan:
                cv = None
                if s.pixel_rule == "C3":
                    cv = nb[("C3", p)]
                elif s.pixel_rule == "C5":
                    if ("C5", None) not in nb:
                        nb[("C5", None)] = _neighbourhood_valid(src, blk.positions, r0, r1, "C5")
                    cv = nb[("C5", None)]
                _, V, Wt = cache.observe(s, cloud_valid=cv)
                clr[n][r0:r1] += V.sum(axis=0, dtype=np.int64).astype(np.uint8)
                wet[n][r0:r1] += Wt.sum(axis=0, dtype=np.int64).astype(np.uint8)
            _, Vp0, _ = cache.observe(p0)
            sl = slice(int(loc[0]), int(loc[-1]) + 1)
            if not np.array_equal(loc, np.arange(sl.start, sl.stop)):
                raise AssertionError("a time block's positions are not consecutive")
            mms["P0"][sl, r0:r1] = _pack_rows(Vp0)
            scl8 = cache.scl8
            mms["bad"][sl, r0:r1] = _pack_rows(np.isin(scl8, TZ_BAD_SCL))
            Xr = cache.X("raw", "ndwi")
            clear = np.isin(scl8, TZ_CLEAR_SCL) & np.isfinite(Xr)
            with np.errstate(invalid="ignore"):
                mms["tzwet"][sl, r0:r1] = _pack_rows(clear & (Xr > 0))
                mms["tzdry"][sl, r0:r1] = _pack_rows(clear & (Xr <= 0))
            for p in c3_ps:
                _, Vc3, _ = cache.observe(p0.replace(cloud="C3"), cloud_valid=nb[("C3", p)])
                mms[f"C3p{p}"][sl, r0:r1] = _pack_rows(Vc3)
            codes = _scl_codes13(scl8)
            if "CLD" in blk.data:
                cld = np.asarray(blk.data["CLD"]).astype(np.int64)
                cld = np.where(cld == obs.CLD_NODATA_U8, 101, cld)
            else:                                # a legacy cube has no CLD: all "fill"
                cld = np.full(codes.shape, 101, np.int64)
            for j in range(len(loc)):
                scl_n[loc[j]] += np.bincount(codes[j].ravel(), minlength=13)
                xt[loc[j]] += np.bincount((codes[j] * 102 + cld[j]).ravel(),
                                          minlength=13 * 102).reshape(13, 102).astype(np.int32)
                v_p0[loc[j]] += int(np.count_nonzero(Vp0[j]))
            del cache, nb
            if r1 == H:
                n_blocks += 1
                done_scenes = int(loc[-1]) + 1
                log(_progress(f"pass 1 {year}", done_scenes, Ty, t0, n_blocks))
        for mm in list(mms.values()) + [xt]:
            mm.flush()
    except BaseException as exc:
        src.close()
        del mm, mms, xt
        _discard_after_failure(list(tmps.values()) + [xt_tmp], exc)
        raise
    src.close()
    del mm, mms, xt
    gc.collect()
    for n in bit_names:
        os.replace(tmps[n], paths[n])
    os.replace(xt_tmp, xt_path)
    outs = [os.path.join(d, f"bits_{n}.npy") for n in bit_names] + [xt_path]
    for n, s, p in plan:
        f = os.path.join(d, "counters", f"{n}.npz")
        product.save_npz(f, {"clr": clr[n], "wet": wet[n], "key": np.array(s.id),
                             "p": np.array(-1 if p is None else int(p))})
        outs.append(f)
    rows = []
    HW = H * W
    for j, q in enumerate(qs):
        c = scl_n[j]
        row = {"q": int(q), "local": j, "date": rec["dates"][q], "year": int(year),
               "offset": int(offs_all[q]), "h0": float(rec["h0"][q]),
               "rising": int(bool(rec["rising"][q]))}
        row.update({f"scl_{k}": int(c[k]) for k in range(12)})
        row["scl_nodata"] = int(c[12])
        row["cld_fill"] = 0
        bad = int(c[3] + c[8] + c[9] + c[10])
        row.update({"g21_cloud": float(bad / HW), "g21_nodata": float((c[0] + c[12]) / HW),
                    "ard_good": float((HW - int(c[1] + c[3] + c[8] + c[9] + c[10])) / HW),
                    "frame_bad": float(bad / HW), "v_p0": float(v_p0[j] / HW)})
        rows.append(row)
    xtv = np.load(xt_path, mmap_mode="r")
    for j, row in enumerate(rows):
        row["cld_fill"] = int(np.asarray(xtv[j][:, 101]).sum())
    del xtv
    sp = os.path.join(d, "scene_stats.csv")
    _write_csv_lf(sp, rows, V2_SCENE_COLUMNS)
    outs.append(sp)
    meta = {"format": V2_FORMAT, "year": int(year), "T": Ty, "keys": [n for n, _, _ in plan],
            "bits": bit_names, "c3_p": c3_ps, "H": H, "W": W, "bands": bands,
            "source": site2.source}
    mp = os.path.join(d, "pass1.json")
    write_json_lf(mp, meta)
    outs.append(mp)
    return outs, {"T": Ty, "keys": len(plan), "seconds": round(time.time() - t0, 1)}


T_BLOCK_LOG = 64


def _progress(what, done, total, t0, blocks=None):
    """A progress line: scenes done, elapsed, ETA and RSS (one per finished
    time block in the passes, so a silent log means a hang, not work)."""
    el = time.time() - t0
    eta = el / done * (total - done) if done else float("nan")
    b = f", block {blocks}" if blocks is not None else ""
    return (f"  [{what}] {done}/{total} scenes{b} ({el:.0f} s elapsed, ETA {eta:.0f} s, "
            f"RSS {_rss_mb():.0f} MB)")


def read_scene_stats_v2(site2, year):
    return read_scene_stats(_year_dir(site2, year, "pass1", "scene_stats.csv"))


def _counter_path(site2, year, name, kind="pass1"):
    if kind == "pass1":
        return _year_dir(site2, year, "pass1", "counters", f"{name}.npz")
    return _year_dir(site2, year, "ocm", "counters", f"{name}.npz")


def year_counters(site2, year, key, p=None):
    """``(clr, wet)`` uint8 (H, W) of one year and observation key (pass 1,
    or the OCM step for C11 keys), or None when not counted."""
    name = counter_name(key, p)
    for kind in ("ocm", "pass1"):
        f = _counter_path(site2, year, name, kind)
        if os.path.exists(f):
            z = product.load_npz(f)
            return z["clr"], z["wet"]
    return None


def window_counters(site2, years, key, p=None):
    """Sum of the year counters (int16 (H, W)) over ``years``; None if any
    year lacks them."""
    clr = wet = None
    for y in years:
        c = year_counters(site2, y, key, p)
        if c is None:
            return None
        if clr is None:
            clr = c[0].astype(np.int16)
            wet = c[1].astype(np.int16)
        else:
            clr += c[0]
            wet += c[1]
    return clr, wet


# ─────────────────────────────────────────────────────────────────────────────
#  geometry and support (site)
# ─────────────────────────────────────────────────────────────────────────────

def w3_years(rec):
    """The record years inside W3 (2023-2025)."""
    w = settings.window("W3")
    return [int(y) for y in rec["years"] if int(y) in w.years]


def v2_geometry(site2, rec):
    """keep_full(P0, W3), its sea, s and bands (domain.from_counts on the
    summed P0 counters) and Omega, the D2 domain: keep_full plus the pixels
    of a 300 m ring that are permanent water (wet fraction >= 0.98) or
    stable land (<= 0.02) with at least 31 clear observations."""
    yrs = w3_years(rec)
    c = window_counters(site2, yrs, P0_KEY)
    if c is None:
        raise PackageError(f"{site2.site}: pass 1 of {yrs} is incomplete")
    clr, wetn = c
    H, W = int(rec["grid"]["H"]), int(rec["grid"]["W"])
    d = domain.from_counts(clr, wetn, site2.pixel_m, site2.cfg.mouth_side, mode="F",
                           n_sel=int(sum(len(year_positions(rec, y)) for y in yrs)),
                           with_frame=True)
    band_frame = domain.band_map_frame(d, d.s_frame)
    clr64 = clr.astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        wf = np.where(clr64 >= domain.MIN_CLEAR, wetn / np.maximum(clr64, 1), np.nan)
    keep_mask = _flat_mask(d.keep, (H, W))
    ring = dilate_disk(keep_mask, OMEGA_RING_M / site2.pixel_m) & ~keep_mask
    with np.errstate(invalid="ignore"):
        anchor = (wf >= OMEGA_WATER) | (wf <= OMEGA_LAND)
    omega = keep_mask | (ring & anchor)
    out = _p(site2, "geometry.npz")
    product.save_npz(out, {
        "keep_full": d.keep, "sea_full": d.sea, "s_km": d.s_km, "band_of": d.band_of,
        "centers": d.centers, "edges": d.edges, "K": np.array(d.K, np.int64),
        "status": np.array(d.status), "s_frame": d.s_frame.astype(np.float32),
        "band_frame": band_frame, "wf": wf.astype(np.float32),
        "omega": np.flatnonzero(omega.ravel()).astype(np.int64),
        "years": np.array(yrs, np.int64)})
    log(f"[geometry] {site2.site}: keep_full(P0, W3 = {yrs}) {d.n_keep:,} px ({d.status}), "
        f"K = {d.K}; Omega {int(omega.sum()):,} px")
    return [out], {"n_keep_full": d.n_keep, "K": d.K, "status": d.status,
                   "n_omega": int(omega.sum())}


def _geometry(site2):
    p = _p(site2, "geometry.npz")
    if not os.path.exists(p):
        raise PackageError(f"{p} is missing: run the geometry step")
    return product.load_npz(p)


def urec(clr, wet):
    """U_rec = {clr >= 31, wet >= 4, dry >= 4} on counts (v2 §2.10)."""
    return urec_mask(clr, wet)


def _bbox_region(mask, pad_px, shape, lattice=None):
    """``[r0, r1, c0, c1]`` of a mask's bounding box padded by ``pad_px``
    and clipped to the frame (the whole frame for an empty mask). With a
    20 m ``lattice`` (py, px) the edges move outwards to whole lattice
    blocks (a block starts at rows r with r % 2 == py), so the 20 m indices
    of a crop equal the frame's (only the frame's own edge blocks are
    partial, as in the frame computation)."""
    H, W = shape
    rr = np.flatnonzero(mask.any(axis=1))
    cc = np.flatnonzero(mask.any(axis=0))
    if not len(rr):
        return [0, H, 0, W]
    p = int(np.ceil(pad_px))
    r0, r1 = max(0, int(rr[0]) - p), min(H, int(rr[-1]) + 1 + p)
    c0, c1 = max(0, int(cc[0]) - p), min(W, int(cc[-1]) + 1 + p)
    if lattice is not None:
        py, px = (int(v) % 2 for v in lattice)
        r0, c0 = max(0, r0 - (r0 - py) % 2), max(0, c0 - (c0 - px) % 2)
        r1, c1 = min(H, r1 + (r1 - py) % 2), min(W, c1 + (c1 - px) % 2)
    return [r0, r1, c0, c1]


def load_d_truth(site2, shape):
    """D_truth (sorted flat indices) from the v2 truth
    (``truth/<site>/truth_elev_v2.npz``, member ``d_truth_union``), or None."""
    p = os.path.join(site2.truth_dir, TRUTH_ELEV_V2)
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        if "d_truth_union" not in z.files:
            return None
        d = np.asarray(z["d_truth_union"], np.int64)
    if d.size and (d.min() < 0 or d.max() >= shape[0] * shape[1]):
        raise PackageError(f"{p}: D_truth is not on this frame")
    return np.unique(d)


def v2_support(site2, rec):
    """NB, S0 = D_truth ∪ NB ∪ dilate(keep_full, 300 m) ∪ Omega, the
    pre-dom (S0 ∪ dilate(U_rec of P0 and of C3 at both p, 3 px)) and the
    OCM region (its bounding box plus 1 km)."""
    g = _geometry(site2)
    H, W = int(rec["grid"]["H"]), int(rec["grid"]["W"])
    xs, ys = np.asarray(rec["grid"]["xs"]), np.asarray(rec["grid"]["ys"])
    keep_full = g["keep_full"].astype(np.int64)
    cfg = site2.cfg
    nb, rows, spath = np.zeros(0, np.int64), [], None
    if cfg.judges:
        try:
            nb, rows, spath = build_nb(cfg, rec["grid"]["crs_wkt"], xs, ys, keep_full,
                                       stations_path=site2.stations_json)
        except FileNotFoundError as e:
            log(f"[support] {site2.site}: WARNING no judges' neighbourhoods ({e})")
    d_truth = load_d_truth(site2, (H, W))
    if d_truth is None and cfg.elevation_truth:
        log(f"[support] {site2.site}: WARNING D_truth is not available yet (run truth "
            f"{site2.site}); S0 and dom will change when it arrives")
    s0 = support_s0((H, W), keep_full, nb, site2.pixel_m, d_truth)
    s0 = np.union1d(s0, g["omega"]).astype(np.int64)
    yrs = w3_years(rec)
    U = np.zeros((H, W), bool)
    terms = {}
    c = window_counters(site2, yrs, P0_KEY)
    U |= urec(*c)
    terms[P0_KEY] = int(urec(*c).sum())
    for p in C3_P_CANDIDATES:
        c = window_counters(site2, yrs, settings.P0.replace(cloud="C3").id, p)
        if c is not None:
            u = urec(*c)
            U |= u
            terms[counter_name(settings.P0.replace(cloud="C3").id, p)] = int(u.sum())
    pre = dilate_disk(U, DOM_UREC_DILATION_PX).ravel()
    pre[s0] = True
    region = _bbox_region(pre.reshape(H, W), REGION_PAD_M / site2.pixel_m, (H, W),
                          rec["lattice20"]["phase"])
    nbhd = [r.pop("nbhd") for r in rows]
    meta = {"format": V2_FORMAT, "n_nb": int(len(nb)), "n_s0": int(len(s0)),
            "n_predom": int(pre.sum()), "d_truth": None if d_truth is None else int(len(d_truth)),
            "urec_terms": terms, "ocm_region": region, "stations": rows,
            "stations_json": None if spath is None else _rel(spath)}
    out = _p(site2, "support.npz")
    product.save_npz(out, {"meta": np.array(json.dumps(meta, sort_keys=True,
                                                       default=_json_default)),
                           "nb": nb, "s0": s0,
                           "nbhd": np.concatenate(nbhd).astype(np.int64) if nbhd
                           else np.zeros(0, np.int64),
                           "nbhd_ptr": np.concatenate([[0], np.cumsum([len(a) for a in nbhd])])
                           .astype(np.int64),
                           "predom": np.flatnonzero(pre).astype(np.int64),
                           "d_truth": np.zeros(0, np.int64) if d_truth is None else d_truth,
                           "has_d_truth": np.array(d_truth is not None),
                           "ocm_region": np.asarray(region, np.int64)})
    log(f"[support] {site2.site}: S0 {len(s0):,} px (NB {len(nb):,}, D_truth "
        f"{'pending' if d_truth is None else f'{len(d_truth):,}'}), pre-dom {int(pre.sum()):,} "
        f"px, OCM region rows {region[0]}:{region[1]}, cols {region[2]}:{region[3]}")
    return [out], meta


def _support(site2):
    p = _p(site2, "support.npz")
    if not os.path.exists(p):
        raise PackageError(f"{p} is missing: run the support step")
    z = product.load_npz(p)
    z["meta"] = json.loads(str(z["meta"]))
    return z


# ─────────────────────────────────────────────────────────────────────────────
#  OmniCloudMask (C11), per year
# ─────────────────────────────────────────────────────────────────────────────

ocm_model_dir = obs.ocm_model_dir


def ocm_provenance(predictor=None, weights=False):
    """Package name, version and licence of OmniCloudMask (installed
    metadata), the card, the versions of torch and of the libraries its
    models are built with, the torch thread count and, with ``weights``,
    the model files and their sha256 (``omnicloudmask.download_models.
    get_models``, which downloads them once into :func:`ocm_model_dir`);
    or the injected predictor's description."""
    import importlib.metadata as md
    out = {"package": "omnicloudmask", "card": dict(settings.CLOUD["C11"].params)}
    try:
        meta = md.metadata("omnicloudmask")
        out["version"] = md.version("omnicloudmask")
        out["licence"] = (meta.get("License-Expression") or meta.get("License")
                          or "; ".join(c for c in (meta.get_all("Classifier") or [])
                                       if c.startswith("License")) or "unknown")
    except md.PackageNotFoundError:
        out["version"] = None
        out["licence"] = "MIT (PyPI license_expression of 1.7.1, checked 2026-10-14)"
    if predictor is not None:
        out["predictor"] = getattr(predictor, "description", None) or repr(predictor)
        out["injected"] = True
    else:
        out["predictor"] = "omnicloudmask.predict_from_array (CPU, deterministic kernels)"
        out["injected"] = False
    out["libraries"] = {}
    for dist in ("torch", "torchvision", "timm", "segmentation-models-pytorch",
                 "huggingface-hub", "safetensors"):
        try:
            out["libraries"][dist] = md.version(dist)
        except md.PackageNotFoundError:
            out["libraries"][dist] = None
    out["torch"] = out["libraries"]["torch"]
    if out["torch"] is not None and predictor is None:
        try:
            import torch
            out["torch_threads"] = int(torch.get_num_threads())
        except ImportError:
            out["torch_threads"] = None
    if weights and predictor is None:
        from omnicloudmask.download_models import get_models
        d = ocm_model_dir()
        os.makedirs(d, exist_ok=True)
        mv = settings.CLOUD["C11"].params.get("model_version")
        files = get_models(model_dir=d, model_version=None if mv is None else float(mv))
        out["model_dir"] = str(d).replace("\\", "/")
        out["weights"] = [{"file": os.path.basename(str(f["Path"])),
                           "timm_model_name": f.get("timm_model_name"),
                           "sha256": product.sha256_file(str(f["Path"]))} for f in files]
    return out


def ocm_available(predictor=None):
    """True when the C11 step can run (an injected predictor, or the package
    importable)."""
    if predictor is not None:
        return True
    import importlib.util
    return importlib.util.find_spec("omnicloudmask") is not None


def _ocm_resume(prog_path, key, parts, shapes):
    """The checkpoint of an interrupted OCM year, or None: its progress file
    must carry ``key`` and every partial file the expected shape."""
    if not os.path.exists(prog_path):
        return None
    try:
        prog = product.read_json(prog_path)
    except (OSError, ValueError):
        return None
    if prog.get("key") != key or not os.path.exists(prog.get("snapshot", "")):
        return None
    for p, shape in zip(parts, shapes):
        if not os.path.exists(p):
            return None
        try:
            mm = np.load(p, mmap_mode="r")
            ok = mm.shape == tuple(shape) and mm.dtype == np.uint8
            del mm
        except (OSError, ValueError):
            return None
        if not ok:
            return None
    return prog


def v2_ocm(site2, rec, year, keys, region, predictor=None):
    """The C11 step of one year (spec §2.2): OmniCloudMask on the red, green
    and NIR bands of every scene over the OCM region (``ocm_classes``,
    floor0 DN - o_s), stored as packed clear bits; the C11 counters of the
    planned keys (C11 inside the region, the C1 analog's counts outside it,
    declared) and C11's P0 frame bits (likewise).

    Each stored time block (16 dates) of the region is read ONCE, band by
    band (every chunk decoded once whatever the cache), then classified
    scene by scene. After every time block the partial files are flushed
    and a checkpoint (counters, class histogram, ``ocm_progress.json``) is
    written: a killed run resumes at the last finished block, provided the
    year's dates, offsets, region, plan, card and predictor are unchanged.
    ``ocm.json`` records the provenance (versions, torch threads, weights
    sha256)."""
    t0 = time.time()
    qs = year_positions(rec, year)
    Ty = len(qs)
    plan = ocm_plan(keys)
    d = _year_dir(site2, year, "ocm")
    os.makedirs(os.path.join(d, "counters"), exist_ok=True)
    clean_stale_tmps(d)
    r0, r1, c0, c1 = (int(v) for v in region)
    h, w = r1 - r0, c1 - c0
    src = open_source(site2, rec)
    H, W = src.grid.shape
    Wp = (W + 7) // 8
    lat = src.lattice20
    local = None if lat is None else ((lat[0] - r0) % 2, (lat[1] - c0) % 2)
    offs_all = np.asarray(rec["offsets"], np.int64)
    ocm_bands = tuple(settings.CLOUD["C11"].params["bands"])
    bands = set(ocm_bands) | {"SCL"}
    for _, s in plan:
        bands |= set(s.bands)
    bands = [b for b in TILE_BANDS if b in bands]
    p1 = _year_dir(site2, year, "pass1")
    try:
        prov = ocm_provenance(predictor, weights=predictor is None)
    except BaseException:
        src.close()
        raise
    key = config.sha256_json({
        "year": int(year), "dates": [rec["dates"][int(q)] for q in qs],
        "offsets": [int(rec["offsets"][int(q)]) for q in qs], "region": [r0, r1, c0, c1],
        "lattice": None if lat is None else [int(v) for v in lat],
        "plan": [n for n, _ in plan], "card": prov["card"], "predictor": prov["predictor"],
        "version": prov.get("version"), "weights": prov.get("weights"),
        "pass1": _small_sha(os.path.join(p1, "bits_P0.npy"), limit=0)})
    path_bits = os.path.join(d, "bits_C11.npy")
    path_clear = os.path.join(d, "ocm_clear.npy")
    part_bits, part_clear = path_bits + OCM_PARTIAL, path_clear + OCM_PARTIAL
    snap = os.path.join(d, "ocm_state" + OCM_PARTIAL + ".npz")
    prog_path = os.path.join(d, OCM_PROGRESS)
    shapes = [(Ty, H, Wp), (Ty, h, (w + 7) // 8)]
    prog = _ocm_resume(prog_path, key, [part_bits, part_clear], shapes)
    mm_bits = mm_clear = None
    try:
        if prog is not None:
            st = product.load_npz(prog["snapshot"])
            clr = {n: st[f"clr.{n}"].copy() for n, _ in plan}
            wet = {n: st[f"wet.{n}"].copy() for n, _ in plan}
            hist = st["hist"].copy()
            start = int(prog["done"])
            mm_bits = np.load(part_bits, mmap_mode="r+")
            mm_clear = np.load(part_clear, mmap_mode="r+")
            log(f"[ocm] {site2.site} {year}: resuming at scene {start}/{Ty} from the "
                f"checkpoint of {prog.get('utc')}")
        else:
            for p in (part_bits, part_clear, snap, prog_path):
                if os.path.exists(p):
                    os.remove(p)
            check_space(d, sum(memmap_bytes(s_, np.uint8) for s_ in shapes),
                        f"{site2.site}: the OmniCloudMask step of {year}")
            clr, wet = {}, {}
            for n, s in plan:
                base = year_counters(site2, year, s.replace(cloud="C1").id)
                if base is None:
                    raise PackageError(f"{site2.site} {year}: pass 1 lacks the C1 analog of {n}")
                clr[n], wet[n] = base[0].copy(), base[1].copy()
                clr[n][r0:r1, c0:c1] = 0
                wet[n][r0:r1, c0:c1] = 0
            hist = np.zeros((Ty, 5), np.int64)
            start = 0
            mm_bits = np.lib.format.open_memmap(part_bits, mode="w+", dtype=np.uint8,
                                                shape=shapes[0])
            reserve_file(part_bits)
            mm_clear = np.lib.format.open_memmap(part_clear, mode="w+", dtype=np.uint8,
                                                 shape=shapes[1])
            reserve_file(part_clear)
    except BaseException as exc:
        src.close()
        mm_bits = mm_clear = None
        if prog is None:                       # a fresh start: nothing worth keeping
            _discard_after_failure([part_bits, part_clear], exc)
        else:
            gc.collect()
        raise
    bits_p0 = np.load(os.path.join(p1, "bits_P0.npy"), mmap_mode="r")
    p0c11 = settings.P0.replace(cloud="C11")
    done = start
    t_run, n_run = time.time(), 0
    try:
        for qblk in src.time_blocks(qs):
            js = (qblk - qs[0]).astype(np.int64)
            if int(js[-1]) < start:
                continue
            rawb = {}
            for b in bands:                    # one band at a time: half the peak memory
                a = src.read_block(qblk, [b], slice(r0, r1))[b]
                rawb[b] = np.ascontiguousarray(a[:, :, c0:c1])
                del a
            for k, q in enumerate(qblk):
                j = int(js[k])
                if j < start:
                    continue
                raw = {b: arr[k:k + 1] for b, arr in rawb.items()}
                cls = obs.ocm_classes({b: raw[b][0] for b in ocm_bands}, int(offs_all[q]),
                                      predictor=predictor)
                hist[j] = np.bincount(np.where(cls == obs.OCM_NODATA_U8, 4, cls).ravel(),
                                      minlength=5)[:5]
                cache = _ObsCache(raw, int(offs_all[q]), local)
                cv = obs.cloud_c11(cache.scl8, cls[None])
                for n, s in plan:
                    _, V, Wt = cache.observe(s, cloud_valid=cv)
                    clr[n][r0:r1, c0:c1] += V[0].astype(np.uint8)
                    wet[n][r0:r1, c0:c1] += Wt[0].astype(np.uint8)
                _, Vp, _ = cache.observe(p0c11, cloud_valid=cv)
                row = _unpack_rows(bits_p0[j], W)
                row[r0:r1, c0:c1] = Vp[0]
                mm_bits[j] = _pack_rows(row)
                mm_clear[j] = _pack_rows(cls == 0)
                n_run += 1
            del rawb
            mm_bits.flush()
            mm_clear.flush()
            done = int(js[-1]) + 1
            arrays = {"hist": hist}
            for n, _ in plan:
                arrays[f"clr.{n}"] = clr[n]
                arrays[f"wet.{n}"] = wet[n]
            product.save_npz(snap, arrays)
            write_json_lf(prog_path, {"key": key, "done": done, "T": Ty, "snapshot": snap,
                                      "utc": product.utc_stamp(),
                                      "host": socket.gethostname()})
            el = time.time() - t_run
            eta = el / max(n_run, 1) * (Ty - done)
            log(f"  [ocm {year}] {done}/{Ty} scenes ({el:.0f} s this run, ETA {eta:.0f} s, "
                f"RSS {_rss_mb():.0f} MB; checkpoint written)")
    except BaseException:
        src.close()
        mm_bits = mm_clear = bits_p0 = None
        gc.collect()
        log(f"[ocm] {site2.site} {year}: stopped after {done}/{Ty} scenes; the partial files "
            f"and the checkpoint are kept: the next run resumes there")
        raise
    src.close()
    mm_bits = mm_clear = bits_p0 = None
    gc.collect()
    os.replace(part_bits, path_bits)
    os.replace(part_clear, path_clear)
    outs = [path_bits, path_clear]
    for n, s in plan:
        f = os.path.join(d, "counters", f"{n}.npz")
        product.save_npz(f, {"clr": clr[n], "wet": wet[n], "key": np.array(n),
                             "p": np.array(-1)})
        outs.append(f)
    hp = os.path.join(d, "ocm_hist.npy")
    _save_npy(hp, hist)
    outs.append(hp)
    meta = {"format": V2_FORMAT, "year": int(year), "T": Ty, "region": [r0, r1, c0, c1],
            "keys": [n for n, _ in plan], "provenance": prov, "resumed_at": start or None,
            "outside_region": "C1 analog counts and P0 bits (declared)",
            "class_totals": {k: int(v) for k, v in zip(("clear", "thick", "thin", "shadow",
                                                         "nodata"), hist.sum(axis=0))}}
    mp = os.path.join(d, "ocm.json")
    write_json_lf(mp, meta)
    outs.append(mp)
    for p in (snap, prog_path):
        if os.path.exists(p):
            os.remove(p)
    return outs, {"T": Ty, "seconds": round(time.time() - t0, 1), "resumed_at": start,
                  "clear_share": float(hist[:, 0].sum() / max(hist[:, :4].sum(), 1))}


# ─────────────────────────────────────────────────────────────────────────────
#  dom (site): frozen at lock-A with a dom_version
# ─────────────────────────────────────────────────────────────────────────────

def close_lattice(mask, lattice):
    """Every pixel of each 20 m lattice block that holds a ``mask`` pixel
    (box blocks at phase (0, 0) without a lattice), so the 20 m indices at
    dom are computed from stored pixels only."""
    m = np.asarray(mask, bool)
    py, px = (0, 0) if lattice is None else (int(lattice[0]), int(lattice[1]))
    H, W = m.shape
    pads = [(py, (-(H + py)) % 2), (px, (-(W + px)) % 2)]
    mp = np.pad(m, pads, constant_values=False)
    Hb, Wb = mp.shape[0] // 2, mp.shape[1] // 2
    blk = mp.reshape(Hb, 2, Wb, 2).any(axis=(1, 3))
    return obs.expand20(blk, (py, px), (H, W)) | m


def current_dom(site2):
    """``dom/current.json`` (dom_version, frozen, ...) or None."""
    p = _p(site2, "dom", "current.json")
    return product.read_json(p) if os.path.exists(p) else None


def read_crosstab_p(root=V2_ROOT):
    """C3's frozen cut-off p from ``common/crosstab.json`` (``pilot
    crosstab``), or None before the cross-tab."""
    p = os.path.join(root, "common", CROSSTAB_JSON)
    if not os.path.exists(p):
        return None
    v = product.read_json(p).get("p")
    return None if v is None else int(v)


def v2_dom(site2, rec, use_c11, force=False):
    """dom = S0 ∪ dilate(U_rec over the A-mode curve settings of W3 {P0, C2,
    C11 (C3 when C11 is unavailable)}, 3 px), closed under the 20 m lattice
    blocks; dom_version = the first 16 hex digits of the sha256 of dom and
    its rule. A frozen dom is never replaced (:class:`DomFrozenError`)."""
    sup = _support(site2)
    H, W = int(rec["grid"]["H"]), int(rec["grid"]["W"])
    yrs = w3_years(rec)
    U = np.zeros((H, W), bool)
    terms = {}
    c = window_counters(site2, yrs, P0_KEY)
    terms[P0_KEY] = urec(*c)
    if use_c11:
        k = settings.P0.replace(cloud="C11").id
        c = window_counters(site2, yrs, k)
        if c is None:
            raise PackageError(f"{site2.site}: the C11 counters of {yrs} are missing")
        r0, r1, c0, c1 = (int(v) for v in sup["ocm_region"])
        u = np.zeros((H, W), bool)
        u[r0:r1, c0:c1] = urec(*c)[r0:r1, c0:c1]
        terms[k] = u
    else:
        p = read_crosstab_p(site2.root)
        k = settings.P0.replace(cloud="C3").id
        for pp in ((p,) if p is not None else C3_P_CANDIDATES):
            c = window_counters(site2, yrs, k, pp)
            if c is not None:
                terms[counter_name(k, pp)] = urec(*c)
    for u in terms.values():
        U |= u
    m = dilate_disk(U, DOM_UREC_DILATION_PX).ravel()
    m[sup["s0"]] = True
    lat = rec["lattice20"]["phase"]
    m = close_lattice(m.reshape(H, W), lat).ravel()
    dom = np.flatnonzero(m).astype(np.int64)
    rule = {"rule": "S0 ∪ dilate(U_rec(W3) of P0/C2 and C11 (C3 if C11 unavailable), 3 px), "
                    "closed under the 20 m lattice blocks",
            "terms": sorted(terms), "lattice20": lat, "s0_sha256":
            config.sha256_bytes(np.asarray(sup["s0"], "<i8").tobytes()),
            "has_d_truth": bool(sup["has_d_truth"]), "years": yrs}
    if len(terms) == 1:
        rule["note"] = ("neither C11 nor C3 counters (a source without the OCM bands and CLD, "
                        "e.g. a legacy cube): U_rec of P0 (C2 inside) only, declared")
    h = hashlib.sha256(dom.astype("<i8").tobytes())
    h.update(json.dumps(rule, sort_keys=True).encode("utf-8"))
    dv = h.hexdigest()[:16]
    cur = current_dom(site2)
    if cur is not None and cur.get("frozen") and cur["dom_version"] != dv and not force:
        raise DomFrozenError(f"{site2.site}: dom is frozen at {cur['dom_version']} (lock-A); "
                             f"the inputs now give {dv}. New years add shards for the frozen "
                             "dom; anything else needs the user's decision (--force)")
    if cur is not None and cur.get("frozen") and cur["dom_version"] == dv:
        log(f"[dom] {site2.site}: dom {dv} unchanged (frozen)")
    n_s0 = int(len(sup["s0"]))
    meta = {"format": V2_FORMAT, "dom_version": dv, **rule, "n_dom": int(len(dom)),
            "n_s0": n_s0, "ratio_dom_s0": round(len(dom) / max(n_s0, 1), 4),
            "urec_px": {k: int(v.sum()) for k, v in terms.items()},
            "n_tiles": int(-(-len(dom) // TILE))}
    out = _p(site2, "dom", dv, "dom.npz")
    product.save_npz(out, {"meta": np.array(json.dumps(meta, sort_keys=True,
                                                       default=_json_default)),
                           "dom": dom, "s0": np.asarray(sup["s0"], np.int64)})
    frozen = bool(cur and cur.get("frozen") and cur["dom_version"] == dv)
    history = sorted(set((cur or {}).get("frozen_history") or [])
                     | ({cur["dom_version"]} if cur and cur.get("frozen") else set()))
    cp = _p(site2, "dom", "current.json")
    write_json_lf(cp, {"dom_version": dv, "frozen": frozen,
                       "frozen_utc": cur.get("frozen_utc") if frozen else None,
                       "frozen_history": history,
                       "n_dom": int(len(dom)), "n_s0": n_s0,
                       "ratio_dom_s0": meta["ratio_dom_s0"]})
    log(f"[dom] {site2.site}: dom_version {dv}: {len(dom):,} px "
        f"({len(dom) / (H * W):.1%} of the frame; |dom|/|S0| = {meta['ratio_dom_s0']:.2f}); "
        f"terms {sorted(terms)}")
    return [out, cp], meta


#: the windows D_truth must be the union over before dom is frozen: every
#: Campaign-1 window an elevation unit is scored in (truth.ELEV_WINDOWS_1A)
FREEZE_TRUTH_WINDOWS = ("W3", "W1", "Y2023", "Y2024", "Y2025")


def freeze_dom(site2):
    """Freeze the current dom (lock-A; ``pilot dom --freeze``). Refuses at a
    site with elevation truth when dom was built without D_truth, when the
    elevation truth does not register every Campaign-1 elevation window
    (:data:`FREEZE_TRUTH_WINDOWS`: D_truth is their union) or when dom is
    older than the current D_truth."""
    cur = current_dom(site2)
    if cur is None:
        raise PackageError(f"{site2.site}: no dom yet (prepare --step dom)")
    z = product.load_npz(_p(site2, "dom", cur["dom_version"], "dom.npz"))
    meta = json.loads(str(z["meta"]))
    if site2.cfg.elevation_truth:
        if not meta.get("has_d_truth"):
            raise PackageError(f"{site2.site}: dom {cur['dom_version']} was built without "
                               f"D_truth (run truth {site2.site}, then prepare {site2.site} "
                               "--step support,dom: both rebuild when D_truth changes)")
        jp = os.path.join(site2.truth_dir, "truth_elev_v2.json")
        wins = product.read_json(jp).get("windows", []) if os.path.exists(jp) else []
        miss = [w for w in FREEZE_TRUTH_WINDOWS if w not in wins]
        if miss:
            raise PackageError(f"{site2.site}: the elevation truth registers {wins}, not "
                               f"{miss}: D_truth must be the union over every Campaign-1 "
                               f"window (run truth {site2.site} without --windows, then "
                               f"prepare {site2.site} --step support,dom)")
        sup = _support(site2)
        if config.sha256_bytes(np.asarray(sup["d_truth"], "<i8").tobytes()) != \
                _d_truth_id(site2):
            raise PackageError(f"{site2.site}: support.npz holds an older D_truth than the "
                               f"elevation truth: run prepare {site2.site} --step support,dom")
        if config.sha256_bytes(np.asarray(sup["s0"], "<i8").tobytes()) != meta["s0_sha256"]:
            raise PackageError(f"{site2.site}: dom {cur['dom_version']} was built on another "
                               f"S0: run prepare {site2.site} --step dom")
    cur.update(frozen=True, frozen_utc=product.utc_stamp())
    write_json_lf(_p(site2, "dom", "current.json"), cur)
    return cur


def load_dom(site2, dom_version=None):
    """``(dom_version, dom, meta)`` of the current (or given) dom."""
    if dom_version is None:
        cur = current_dom(site2)
        if cur is None:
            raise PackageError(f"{site2.site}: no dom (prepare --step dom)")
        dom_version = cur["dom_version"]
    z = product.load_npz(_p(site2, "dom", dom_version, "dom.npz"))
    return dom_version, z["dom"].astype(np.int64), json.loads(str(z["meta"]))


def dom_verdict(meta, n_scenes_w9=W9_SCENES):
    """pass 2's refusal rule (spec §2.10): |dom| <= 1.5 |S0| and the
    projected W9 package size."""
    gb = meta["n_dom"] * n_scenes_w9 * BYTES_PER_OBS / 1e9
    ok = meta["ratio_dom_s0"] <= DOM_MAX_OVER_S0
    return {"ratio_ok": bool(ok), "ratio": meta["ratio_dom_s0"],
            "projected_w9_gb": round(gb, 2), "budget_gb_all_sites": PACKAGE_W9_BUDGET_GB}


# ─────────────────────────────────────────────────────────────────────────────
#  pass 2 (per year, per dom_version): pixel-major tiles
# ─────────────────────────────────────────────────────────────────────────────

def _write_tiles(mm, sl, a, b, vals):
    """Write ``vals`` (n, b - a) of dom positions [a, b) into the tile
    memmap ``mm`` (n_tiles, T, TILE) at the shard scenes ``sl``."""
    for k in range(a // TILE, (b - 1) // TILE + 1):
        lo, hi = max(a, k * TILE), min(b, (k + 1) * TILE)
        mm[k, sl, lo - k * TILE:hi - k * TILE] = vals[:, lo - a:hi - a]


def v2_pass2(site2, rec, year, dv, dom, ocm_region=None, allow_large_dom=False,
             log_every=SCENE_STATS_LOG_EVERY):
    """Pass 2 of one year (spec §2.10, §13.4): the raw bands at dom written
    straight into per-band pixel-major tiles ``tile_<band>.npy`` of shape
    (n_tiles, T_Y, 4096) (int16 bands, uint8 SCL and CLD, the padding of the
    last tile at the fill), and ``cloudbits.npy`` (uint8: C3 at p = 20 and
    35, OmniCloudMask clear, inside the OCM region). Only the bands the
    source delivers are stored (a legacy cube: B03, B08, SCL; listed in
    ``shard.json``). Each scene's P0 V at dom must equal its pass-1 frame
    bits."""
    t0 = time.time()
    _, _, meta = load_dom(site2, dv)
    if not allow_large_dom and meta["ratio_dom_s0"] > DOM_MAX_OVER_S0:
        raise PackageError(f"{site2.site}: |dom|/|S0| = {meta['ratio_dom_s0']:.2f} > "
                           f"{DOM_MAX_OVER_S0}: pass 2 refused (pilot dom; the user decides)")
    qs = year_positions(rec, year)
    Ty = len(qs)
    P = len(dom)
    n_tiles = -(-P // TILE)
    d = _shard_dir(site2, dv, year)
    os.makedirs(d, exist_ok=True)
    clean_stale_tmps(d)
    src = open_source(site2, rec)
    H, W = src.grid.shape
    offs_all = np.asarray(rec["offsets"], np.int64)
    tbands = [b for b in TILE_BANDS if b in src.bands]
    names = tbands + ["cloudbits"]
    dtypes = dict(TILE_DTYPES, cloudbits=np.uint8)
    need = sum(memmap_bytes((n_tiles, Ty, TILE), dtypes[n]) for n in names)
    try:
        check_space(d, need, f"{site2.site}: pass 2 of {year} (dom {dv}, {P:,} px)")
    except PackageError:
        src.close()
        raise
    paths = {n: os.path.join(d, f"tile_{n}.npy") for n in names}
    tmps = {n: _tmp(p) for n, p in paths.items()}
    mms = {}
    try:
        for n in names:
            mms[n] = np.lib.format.open_memmap(tmps[n], mode="w+", dtype=dtypes[n],
                                               shape=(n_tiles, Ty, TILE))
            reserve_file(tmps[n])
    except BaseException as exc:
        src.close()
        mms.clear()
        _discard_after_failure(list(tmps.values()), exc)
        raise
    if P % TILE:
        for b in tbands:
            mms[b][-1, :, P % TILE:] = TILE_FILL[b]
    p1 = _year_dir(site2, year, "pass1")
    bits_p0 = np.load(os.path.join(p1, "bits_P0.npy"), mmap_mode="r")
    c3_ps = [p for p in C3_P_CANDIDATES if os.path.exists(os.path.join(p1, f"bits_C3p{p}.npy"))]
    ocm_clear = None
    if ocm_region is not None:
        oc = _year_dir(site2, year, "ocm", "ocm_clear.npy")
        if os.path.exists(oc):
            ocm_clear = np.load(oc, mmap_mode="r")
    rr, cc = dom // W, dom % W
    has_ocm = ocm_clear is not None
    if ocm_clear is not None:
        R0, R1, C0, C1 = (int(v) for v in ocm_region)
        in_reg = (rr >= R0) & (rr < R1) & (cc >= C0) & (cc < C1)
    p0 = settings.Setting.parse(P0_KEY)
    mm = None
    n_blocks = 0
    try:
        for blk in src.row_blocks(tbands, block_rows=site2.block_rows, positions=qs,
                                  pair20=False):
            loc = blk.positions - qs[0]
            sl = slice(int(loc[0]), int(loc[-1]) + 1)
            r0, r1 = blk.rows.start, blk.rows.stop
            a, b = np.searchsorted(dom, [r0 * W, r1 * W])
            if a == b:
                if r1 == H:
                    n_blocks += 1
                    log(_progress(f"pass 2 {year}", int(loc[-1]) + 1, Ty, t0, n_blocks))
                continue
            cols = dom[a:b] - r0 * W
            n = len(loc)
            for band in tbands:
                v = np.asarray(blk.data[band]).reshape(n, -1)[:, cols]
                _write_tiles(mms[band], sl, a, b, v)
            cb = np.zeros((n, b - a), np.uint8)
            for p in c3_ps:
                v = _neighbourhood_valid(src, blk.positions, r0, r1, "C3", p)
                cb |= (v.reshape(n, -1)[:, cols].astype(np.uint8)
                       * np.uint8(CLOUDBITS[f"C3p{p}"]))
            if ocm_clear is not None:
                ir = in_reg[a:b]
                if ir.any():
                    lo, hi = max(r0, R0), min(r1, R1)
                    blkc = _unpack_rows(ocm_clear[sl, lo - R0:hi - R0], C1 - C0)
                    ii = np.flatnonzero(ir)
                    vals = blkc[:, rr[a:b][ii] - lo, cc[a:b][ii] - C0]
                    cb[:, ii] |= (vals.astype(np.uint8) * np.uint8(CLOUDBITS["ocm_clear"])
                                  | np.uint8(CLOUDBITS["ocm_region"]))
            _write_tiles(mms["cloudbits"], sl, a, b, cb)
            cache = _ObsCache(blk.data, offs_all[blk.positions], None)
            _, Vp0, _ = cache.observe(p0)
            got = Vp0.reshape(n, -1)[:, cols]
            want = _unpack_rows(bits_p0[sl, r0:r1], W).reshape(n, -1)[:, cols]
            if not np.array_equal(got, want):
                raise PackageError(f"{site2.site} {year}: the P0 validity at dom differs from "
                                   "the pass-1 bits; the store changed between the passes")
            del cache
            if r1 == H:
                n_blocks += 1
                log(_progress(f"pass 2 {year}", int(loc[-1]) + 1, Ty, t0, n_blocks))
        for mm in mms.values():
            mm.flush()
    except BaseException as exc:
        src.close()
        del mm, mms, bits_p0, ocm_clear
        _discard_after_failure(list(tmps.values()), exc)
        raise
    src.close()
    del mm, mms, bits_p0, ocm_clear
    gc.collect()
    for n in names:
        os.replace(tmps[n], paths[n])
    sm = {"format": V2_FORMAT, "dom_version": dv, "year": int(year), "T": Ty, "P": P,
          "n_tiles": n_tiles, "tile": TILE, "bands": tbands,
          "dtypes": {k: np.dtype(v).name for k, v in dtypes.items()},
          "cloudbits": dict(CLOUDBITS), "c3_p": c3_ps, "ocm": has_ocm,
          "ocm_region": None if ocm_region is None else [int(v) for v in ocm_region],
          "dates": [rec["dates"][q] for q in qs]}
    sp = os.path.join(d, "shard.json")
    write_json_lf(sp, sm)
    return [paths[n] for n in names] + [sp], {"T": Ty, "P": P, "n_tiles": n_tiles,
                                              "seconds": round(time.time() - t0, 1)}


# ─────────────────────────────────────────────────────────────────────────────
#  the package as the methods see it (read side)
# ─────────────────────────────────────────────────────────────────────────────

class Blocks1D:
    """The 20 m lattice blocks of a set of frame pixels (sorted flat
    indices, closed under the blocks): ``inverse`` (pixel -> block),
    ``first`` (the block's first in-frame pixel, the one block_take20
    reads), ``counts``, ``exact`` (a measured lattice)."""

    def __init__(self, flat, W, lattice):
        from scipy import sparse
        flat = np.asarray(flat, np.int64)
        self.exact = lattice is not None
        py, px = (0, 0) if lattice is None else (int(lattice[0]), int(lattice[1]))
        Wb = (int(W) - 1 + px) // 2 + 1
        bid = ((flat // W) + py) // 2 * Wb + ((flat % W) + px) // 2
        _, first, inv = np.unique(bid, return_index=True, return_inverse=True)
        self.inverse = inv.astype(np.int64)
        self.first = first.astype(np.int64)
        self.n_blocks = len(first)
        self.counts = np.bincount(self.inverse, minlength=self.n_blocks).astype(np.float64)
        m = len(flat)
        self.M = sparse.csr_matrix((np.ones(m), (np.arange(m), self.inverse)),
                                   shape=(m, self.n_blocks))

    def mean(self, a):
        """block_mean20 on columns: float32 of the float64 block sum / count
        (a NaN in a block makes the block NaN)."""
        a = np.asarray(a, np.float32)
        lead = a.shape[:-1]
        a2 = a.reshape(-1, a.shape[-1]).astype(np.float64)
        s = np.asarray(self.M.T.dot(a2.T)).T
        return (s / self.counts).astype(np.float32).reshape(lead + (self.n_blocks,))

    def take(self, a):
        return np.asarray(a)[..., self.first]

    def expand(self, b):
        return np.asarray(b)[..., self.inverse]


def compute_index_cols(prepared, index, blocks):
    """``obs.compute_index`` on columns (..., m) whose 20 m blocks are
    ``blocks`` (:class:`Blocks1D`): bitwise the frame computation at those
    pixels (the 10 m sums of a block are exact integers)."""
    e = settings.INDEX[index]
    f = obs.INDEX_FUNCS[index]
    if not e.at20:
        return f({k: prepared[k] for k in e.bands10})
    blk = {k: blocks.mean(prepared[k]) for k in e.bands10}
    take = blocks.take if blocks.exact else blocks.mean
    blk.update({k: np.asarray(take(prepared[k]), np.float32) for k in e.bands20})
    return blocks.expand(f(blk))


def observe_cols(raw, setting, *, offset=0, blocks=None, params=None, threshold=None,
                 cloud_valid=None, cloud_params=None, fill=obs.FILL):
    """``obs.observe_setting`` on package columns: raw ``{band: (n, m)}`` of
    n scenes at m pixels (``blocks`` the pixels' 20 m blocks); returns
    ``Observation(X, Y, V, W)`` of shape (n, m). Neighbourhood cloud levels
    (C3, C5) and C11 need ``cloud_valid`` (the stored bits)."""
    s = settings.Setting.parse(setting)
    de = settings.DETECT[s.detect]
    p = dict(params) if params is not None else settings.index_params(s.index)
    t_k, s_k = float(p["t_k"]), float(p["s_k"])
    r3 = {b: np.asarray(v)[:, None, :] for b, v in raw.items()}
    scl8 = obs._as_scl8(r3["SCL"], fill)
    if cloud_valid is None:
        if s.pixel_rule in ("C3", "C5", "C11"):
            raise ValueError(f"{s.pixel_rule} at dom needs the stored cloud bits")
        cv = obs.pixel_valid(s.pixel_rule, scl8, None, None, cloud_params, fill)
    else:
        cv = np.asarray(cloud_valid, bool)[:, None, :]
    need = obs.spectral_bands(s)
    prep = obs.prepare(r3, need, offset, s.prep, fill)

    def X_of(i):
        e = settings.INDEX[i]
        if e.at20 and blocks is None:
            raise ValueError(f"{i} is computed at 20 m: pass the pixels' blocks")
        return compute_index_cols({b: prep[b] for b in e.bands}, i, blocks)

    X = X_of(s.index)
    V = cv & np.isfinite(X)
    for b in need:
        V &= ~np.isnan(prep[b])
    Y = X if (t_k == 0.0 and s_k == 1.0) else ((X - np.float32(t_k)) / np.float32(s_k)).astype(
        np.float32)
    if s.detect == "D1":
        Wt = obs.detect_d1(X, V, t_k)
    elif de.per_scene:
        if threshold is None:
            raise ValueError(f"{s.detect} needs the per-scene thresholds of the settings pass")
        Wt = obs.detect_threshold(X, V, np.asarray(threshold, np.float32))
    elif s.detect == "D4":
        Wt = obs.detect_d4(V, scl8)
    elif s.detect == "D5":
        Wt = obs.detect_d5(V, *[X if i == s.index else X_of(i) for i in de.indices],
                           votes=int(de.params["votes"]))
    else:
        raise ValueError(f"no detector implementation for {s.detect}")
    return obs.Observation(X[:, 0], Y[:, 0], V[:, 0], Wt[:, 0])


class V2Package:
    """A site's v2 package, read-only (spec §2.10, §13.4).

    Built progressively: after the geometry step it serves the truth module
    (``cfg``, ``shape``, ``grid``, ``full``, ``band_frame``, ``h0``, ``t``);
    after pass 2 it reads observations at dom.

    Attributes
    ----------
    site2, site, cfg, dir, record, dates, t, h0, rising, offsets, T,
    year_of, years, H, W, shape, lattice, pixel_m
    dom_version, dom, P, n_tiles (None before the dom step)
    """

    def __init__(self, site2, dom_version=None):
        self.site2 = site2
        self.site = site2.site
        self.cfg = site2.cfg
        self.res_m = int(round(site2.pixel_m))
        self.dir = site2.dir
        rec = _read_record(site2)
        self.record = rec
        self.dates = np.asarray(rec["dates"], str)
        self.t = pd.DatetimeIndex(np.asarray(rec["t_ns"], np.int64).view("M8[ns]"))
        self.T = len(self.dates)
        self.h0 = np.asarray(rec["h0"], np.float64)
        self.rising = np.asarray(rec["rising"], bool)
        self.offsets = np.asarray(rec["offsets"], np.int64)
        self.year_of = np.array([int(d[:4]) for d in self.dates], np.int64)
        self.years = [int(y) for y in rec["years"]]
        self.H, self.W = int(rec["grid"]["H"]), int(rec["grid"]["W"])
        self.shape = (self.H, self.W)
        lat = rec["lattice20"]["phase"]
        self.lattice = None if lat is None else tuple(int(v) for v in lat)
        self.pixel_m = site2.pixel_m
        self.fill = obs.FILL
        src = rec.get("source") or {}
        self.source = src.get("kind", "zarr")
        self.bands = tuple(b for b in TILE_BANDS if b in (src.get("bands") or TILE_BANDS))
        self.dom_version, self.dom, self.dom_meta = None, None, None
        cur = current_dom(site2)
        if dom_version is not None or cur is not None:
            self.dom_version, self.dom, self.dom_meta = load_dom(site2, dom_version)
        self.P = None if self.dom is None else int(len(self.dom))
        self.n_tiles = None if self.dom is None else -(-self.P // TILE)
        self._mm = {}
        self._domain_cache = {}
        self._provider = None

    def __repr__(self):
        return (f"<V2Package {self.site}: T = {self.T} ({self.years}), frame {self.H} x "
                f"{self.W}, dom {self.dom_version} ({self.P} px)>")

    def close(self):
        self._mm.clear()
        for k in ("full", "geometry", "support", "series1", "series10", "marks"):
            self.__dict__.pop(k, None)
        gc.collect()

    # -- record --------------------------------------------------------------
    def window(self, wid):
        """Global record positions of a window (spec §2.7)."""
        w = settings.window(wid)
        d = pd.DatetimeIndex(pd.to_datetime(self.dates))
        return np.flatnonzero((d >= pd.Timestamp(w.start)) & (d < pd.Timestamp(w.end))
                              ).astype(np.int64)

    def window_years(self, wid):
        """The window's years present in the record (all of them must be)."""
        w = settings.window(wid)
        miss = [y for y in w.years if y not in self.years]
        if miss:
            raise PackageError(f"{self.site}: window {w.id} needs the shard years {miss}")
        return list(w.years)

    # -- lazy pieces ---------------------------------------------------------
    @cached_property
    def geometry(self):
        return _geometry(self.site2)

    @cached_property
    def support(self):
        return _support(self.site2)

    @cached_property
    def full(self):
        """keep_full(P0, W3) as a domain.Domain (F mode)."""
        g = self.geometry
        keep = g["keep_full"].astype(np.int64)
        inter = np.zeros(self.H * self.W, bool)
        inter[keep] = True
        pos = None if self.dom is None else domain.positions_in(self.dom, keep)
        return domain.Domain("F", keep, pos, g["sea_full"], inter.reshape(self.shape),
                             s_km=g["s_km"], band_of=g["band_of"], centers=g["centers"],
                             edges=g["edges"], K=int(g["K"]), status=str(g["status"]),
                             n_sel=int(sum(len(self.window_years_safe(y))
                                           for y in g["years"].tolist())))

    def window_years_safe(self, year):
        return np.flatnonzero(self.year_of == int(year))

    @property
    def s_frame(self):
        return self.geometry["s_frame"]

    @property
    def band_frame(self):
        return self.geometry["band_frame"]

    @cached_property
    def grid(self):
        import pyproj
        from rasterio.transform import Affine
        xs = np.asarray(self.record["grid"]["xs"], np.float64)
        ys = np.asarray(self.record["grid"]["ys"], np.float64)
        dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
        wkt = self.record["grid"]["crs_wkt"]
        return dict(xs=xs, ys=ys, H=self.H, W=self.W, wkt=wkt, crs=pyproj.CRS.from_wkt(wkt),
                    transform=Affine(dx, 0.0, float(xs[0]) - dx / 2, 0.0, dy,
                                     float(ys[0]) - dy / 2))

    @property
    def s0(self):
        return np.asarray(self.support["s0"], np.int64)

    @property
    def omega(self):
        return np.asarray(self.geometry["omega"], np.int64)

    @cached_property
    def levels(self):
        """The exact level table at the record instants (``levels.npz``);
        beyond it, the live union provider. Refused when it was built on
        another record (its instants or its h0 differ from record.json)."""
        p = _p(self.site2, "levels.npz")
        if not os.path.exists(p):
            raise PackageError(f"{p} is missing: run prepare --step levels")
        tab = levels.LevelTable.load(p, provider=self.provider)
        if not np.array_equal(pd.DatetimeIndex(tab.t).asi8, self.t.asi8) or \
                not np.array_equal(np.asarray(tab.h0, np.float64).view(np.int64),
                                   self.h0.view(np.int64)):
            raise PackageError(f"{p} was built on another record (its instants or h_B differ "
                               f"from record.json): run prepare {self.site} --step levels")
        return tab

    #: the package files the truths depend on (spec §13.2: a truth is
    #: rebuilt when any of them changes)
    KEY_FILES = {"record": "record.json", "levels": "levels.npz",
                 "series": SERIES_V2, "marks": MARKS_V2, "geometry": "geometry.npz"}

    @cached_property
    def keys(self):
        """``{"record", "levels", "series", "marks", "geometry"}``: the sha256
        of each package file the truths read (None when absent), and
        ``package_key``, the sha256 of them all (the truth modules key on
        it, ``truth._package_key``)."""
        out = {}
        for k, name in self.KEY_FILES.items():
            p = _p(self.site2, name)
            out[k] = product.sha256_file(p) if os.path.exists(p) else None
        out["package_key"] = config.sha256_json({k: out[k] for k in sorted(self.KEY_FILES)})
        return out

    @cached_property
    def series1(self):
        """The 1-min h_B series ``(t_min, h)`` over the provider's period
        (``prepare --step series``; v8's grid and tag)."""
        p = _p(self.site2, SERIES_V2)
        if not os.path.exists(p):
            raise PackageError(f"{p} is missing: run prepare {self.site} --step series")
        return levels.load_series_1min(p, name=self.record["boundary"]["name"],
                                       period=boundary_period(self.site2))

    @cached_property
    def series10(self):
        """The 10-min h_B series ``(DatetimeIndex, h)`` (FROZEN's 2023-2025 grid)."""
        return levels.series_10min(*self.series1)

    @cached_property
    def marks(self):
        p = _p(self.site2, MARKS_V2)
        if not os.path.exists(p):
            raise PackageError(f"{p} is missing: run prepare {self.site} --step series")
        return levels.load_marks(p)

    @property
    def marks_lp(self):
        return self.marks["marks_lp"]

    @property
    def marks_raw(self):
        return self.marks["marks_raw"]

    def provider(self):
        """The live boundary provider (built once): the union provider, or
        the production one of a local legacy-cube package (record.json)."""
        if self._provider is None:
            self._provider = boundary_provider(self.site2)
        return self._provider

    @cached_property
    def settings_doc(self):
        p = _p(self.site2, "settings.json")
        return product.read_json(p) if os.path.exists(p) else None

    def thresholds(self, year):
        p = _year_dir(self.site2, year, "settings", "thresholds.npz")
        return product.load_npz(p) if os.path.exists(p) else None

    def index_params(self, index):
        """``{t_k, s_k}`` of an index from the settings pass (NDWI: (0, 1))."""
        if index == "ndwi":
            return settings.index_params("ndwi")
        doc = self.settings_doc
        if doc is None:
            raise PackageError(f"{self.site}: the settings pass has not run")
        e = doc["indices"][index]
        return {"t_k": float(e["t_k"]), "s_k": float(e["s_k"])}

    def scene_thresholds(self, detect, index, sel):
        """D2/D3 per-scene thresholds (X units) of record positions ``sel``."""
        out = np.empty(len(sel), np.float32)
        for y in np.unique(self.year_of[sel]):
            z = self.thresholds(int(y))
            if z is None:
                raise PackageError(f"{self.site} {y}: no thresholds (settings pass)")
            m = self.year_of[sel] == y
            local = np.asarray(sel)[m] - self.window_years_safe(y)[0]
            out[m] = z[f"{detect.lower()}_t_{index}"][local]
        return out

    # -- reading the tiles ----------------------------------------------------
    def _tile(self, year, name):
        k = (int(year), name)
        if k not in self._mm:
            p = _shard_dir(self.site2, self.dom_version, year, f"tile_{name}.npy")
            if not os.path.exists(p):
                raise PackageError(f"{p} is missing: run pass 2 of {year}")
            self._mm[k] = np.load(p, mmap_mode="r")
        return self._mm[k]

    def positions(self, flat):
        return domain.positions_in(self.dom, flat)

    def _sel(self, sel):
        sel = np.arange(self.T) if sel is None else np.asarray(sel)
        if sel.dtype == bool:
            sel = np.flatnonzero(sel)
        sel = sel.astype(np.int64)
        if sel.size and (sel[0] < 0 or sel[-1] >= self.T or np.any(np.diff(sel) <= 0)):
            raise ValueError("sel must be sorted unique record positions in [0, T)")
        return sel

    def read(self, names, sel=None, pos=None):
        """``{name: (n, m)}`` of record positions ``sel`` at dom positions
        ``pos`` (sorted; None = every dom pixel) from the tiles; names are
        bands or "cloudbits"."""
        if self.dom is None:
            raise PackageError(f"{self.site}: no dom yet")
        sel = self._sel(sel)
        pos = np.arange(self.P, dtype=np.int64) if pos is None else np.asarray(pos, np.int64)
        out = {}
        for name in ([names] if isinstance(names, str) else names):
            if name != "cloudbits" and name not in self.bands:
                raise PackageError(f"{self.site}: band {name} is not in this package (source "
                                   f"{self.source}: {', '.join(self.bands)})")
            dt = TILE_DTYPES.get(name, np.uint8)
            res = np.empty((len(sel), len(pos)), dt)
            for y in np.unique(self.year_of[sel]):
                m = np.flatnonzero(self.year_of[sel] == y)
                local = sel[m] - self.window_years_safe(y)[0]
                mm = self._tile(int(y), name)
                tk = pos // TILE
                for k in np.unique(tk):
                    j = np.flatnonzero(tk == k)
                    off = pos[j] - k * TILE
                    lo, hi = int(off[0]), int(off[-1]) + 1
                    blk = np.asarray(mm[k, local[0]:local[-1] + 1, lo:hi])
                    blk = blk[local - local[0]][:, off - lo]
                    res[np.ix_(m, j)] = blk
            out[name] = res
        return out

    def _closure(self, pos):
        """The block closure of dom positions ``pos``: ``(pos_c, idx)`` with
        ``pos_c`` the sorted positions of every pixel of their 20 m blocks
        and ``idx`` the places of ``pos`` in it."""
        if "_blk_sorted" not in self.__dict__:
            py, px = (0, 0) if self.lattice is None else self.lattice
            Wb = (self.W - 1 + px) // 2 + 1
            bid = ((self.dom // self.W) + py) // 2 * Wb + ((self.dom % self.W) + px) // 2
            order = np.argsort(bid, kind="stable")
            self.__dict__["_blk_sorted"] = (bid, bid[order], order)
        bid, sb, order = self.__dict__["_blk_sorted"]
        want = np.unique(bid[pos])
        lo = np.searchsorted(sb, want, "left")
        lens = np.searchsorted(sb, want, "right") - lo
        starts = np.repeat(lo - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
        pos_c = np.sort(order[starts + np.arange(int(lens.sum()))])
        idx = np.searchsorted(pos_c, pos)
        return pos_c.astype(np.int64), idx

    def c3_stored(self):
        """True when every shard year carries C3's cloud bits (the source has
        CLD; a legacy cube has not)."""
        if "_c3_stored" not in self.__dict__:
            ok = self.dom_version is not None
            for y in self.years:
                p = _shard_dir(self.site2, self.dom_version, y, "shard.json")
                ok &= os.path.exists(p) and bool(product.read_json(p).get("c3_p"))
            self.__dict__["_c3_stored"] = bool(ok)
        return self.__dict__["_c3_stored"]

    def ocm_ran(self):
        """True when every shard year carries the OmniCloudMask bits."""
        if "_ocm_ran" not in self.__dict__:
            ok = True
            for y in self.years:
                p = _shard_dir(self.site2, self.dom_version, y, "shard.json")
                ok &= os.path.exists(p) and bool(product.read_json(p).get("ocm"))
            self.__dict__["_ocm_ran"] = bool(ok)
        return self.__dict__["_ocm_ran"]

    def cloud_valid(self, level, cb, scl8, c3_p=None):
        """The stored pixel rule of a neighbourhood level / C11 at dom from
        the cloud bits (C11 outside the OCM region: C1, declared)."""
        if level == "C3":
            if not self.c3_stored():
                raise PackageError(f"{self.site}: C3 needs CLD, which this package's source "
                                   f"({self.source}) does not deliver")
            p = c3_p or read_crosstab_p(self.site2.root) or 20
            return (cb & CLOUDBITS[f"C3p{int(p)}"]) > 0
        if level == "C11":
            if not self.ocm_ran():
                raise PackageError(f"{self.site}: C11 needs the OCM step in every shard year "
                                   "(OmniCloudMask unavailable: C3 replaces C11, declared)")
            in_reg = (cb & CLOUDBITS["ocm_region"]) > 0
            clear = (cb & CLOUDBITS["ocm_clear"]) > 0
            c11 = obs.cloud_c11(scl8, np.where(clear, 0, 1))
            return np.where(in_reg, c11, obs.cloud_c1(scl8))
        raise ValueError(f"{level} has no stored bits")

    def observe(self, setting, sel=None, pos=None, params=None, c3_p=None):
        """``Observation(X, Y, V, W)`` (n, m) of ``setting`` on record
        positions ``sel`` at dom positions ``pos`` (spec §2.5)."""
        s = settings.Setting.parse(setting)
        sel = self._sel(sel)
        pos = np.arange(self.P, dtype=np.int64) if pos is None else np.asarray(pos, np.int64)
        need20 = settings.INDEX[s.index].at20 or any(
            settings.INDEX[i].at20 for i in settings.DETECT[s.detect].indices)
        if need20:
            pos_c, idx = self._closure(pos)
        else:
            pos_c, idx = pos, None
        bands = [b for b in TILE_BANDS if b in set(obs.spectral_bands(s)) | {"SCL"}]
        names = bands + (["cloudbits"] if s.pixel_rule in ("C3", "C11") else [])
        raw = self.read(names, sel, pos_c)
        cv = None
        if s.pixel_rule in ("C3", "C11"):
            cv = self.cloud_valid(s.pixel_rule, raw.pop("cloudbits"), raw["SCL"], c3_p)
        blocks = Blocks1D(self.dom[pos_c], self.W, self.lattice) if need20 else None
        prm = params if params is not None else (
            self.index_params(s.index) if s.index != "ndwi" else None)
        thr = (self.scene_thresholds(s.detect, s.index, sel)
               if settings.DETECT[s.detect].per_scene else None)
        o = observe_cols(raw, s, offset=self.offsets[sel], blocks=blocks, params=prm,
                         threshold=thr, cloud_valid=cv)
        if idx is not None:
            o = obs.Observation(*(a[:, idx] for a in o))
        return o

    def count(self, setting, sel=None, pos=None, chunk=TILE * 4):
        """Valid and wet counts (int32) per dom position over ``sel``."""
        sel = self._sel(sel)
        pos = np.arange(self.P, dtype=np.int64) if pos is None else np.asarray(pos, np.int64)
        clr = np.zeros(len(pos), np.int32)
        wet = np.zeros(len(pos), np.int32)
        for a in range(0, len(pos), int(chunk)):
            o = self.observe(setting, sel, pos[a:a + int(chunk)])
            clr[a:a + int(chunk)] = o.V.sum(axis=0, dtype=np.int32)
            wet[a:a + int(chunk)] = o.W.sum(axis=0, dtype=np.int32)
        return clr, wet

    # -- frame counters and bits -------------------------------------------
    def frame_counts(self, setting, wid):
        """Full-frame (clr, wet) int16 of a pixel setting over a whole window
        (the year counters), or None when a year lacks them."""
        s = settings.Setting.parse(setting)
        key = s.replace(cloud=s.pixel_rule, level="LB").id
        p = None
        if s.pixel_rule == "C3":
            p = read_crosstab_p(self.site2.root) or 20
        return window_counters(self.site2, self.window_years(wid), key, p)

    def bits_family(self, setting):
        """The frame bits that hold this setting's pixel rule: "P0" (C1 and
        the scene rules), "C11", "C3p<p>", or None."""
        s = settings.Setting.parse(setting)
        if s.pixel_rule == "C1":
            return "P0"
        if s.pixel_rule == "C11":
            return "C11"
        if s.pixel_rule == "C3":
            return f"C3p{read_crosstab_p(self.site2.root) or 20}"
        return None

    def _bits(self, year, family):
        sub = "ocm" if family == "C11" else "pass1"
        p = _year_dir(self.site2, year, sub, f"bits_{family}.npy")
        if not os.path.exists(p):
            return None
        return np.load(p, mmap_mode="r")

    def frame_clear_counts(self, family, sel):
        """Clear counts (int32 (H, W)) of the frame bits ``family`` over
        record positions ``sel``; None when a year has no such bits."""
        sel = self._sel(sel)
        acc = np.zeros((self.H, self.W), np.int32)
        for y in np.unique(self.year_of[sel]):
            mm = self._bits(int(y), family)
            if mm is None:
                return None
            local = sel[self.year_of[sel] == y] - self.window_years_safe(y)[0]
            for j in local:
                acc += _unpack_rows(mm[int(j)], self.W)
        return acc

    def rule_subset(self, rule, wid):
        """Global record positions of a window that pass a scene rule."""
        z = load_subsets(self.site2, wid)
        return np.asarray(z["positions"], np.int64)[z[f"rule_{rule}"]]


def union_provider(cfg):
    """The boundary provider over the union period 2017-2025 (spec §1.5):
    ``levels.union_provider`` when the levels module has it, else
    ``levels.make_provider`` with the union period."""
    up = getattr(levels, "union_provider", None)
    if up is not None:
        return up(cfg)
    return levels.make_provider(cfg, period=("2017-01-01", "2025-12-31"))


def load_v2(site2, dom_version=None):
    """The site's :class:`V2Package` (after at least the record step)."""
    return V2Package(site2, dom_version)


# ─────────────────────────────────────────────────────────────────────────────
#  settings pass (site): t_NIR, s_k, D2 and D3 per-scene thresholds
# ─────────────────────────────────────────────────────────────────────────────

def _hist_index(e, x):
    """Histogram values of an index (log for log_hist; finite only)."""
    x = np.asarray(x, np.float64)
    if e.log_hist:
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.log(x[x > 0])
    return x[np.isfinite(x)]


def _hist_fixed(e, x):
    """Fixed-range histogram of the settings pass (values outside the range
    in the end bins)."""
    lo, hi = e.hist_range
    v = np.clip(_hist_index(e, x), lo, hi)
    c, _ = np.histogram(v, bins=int(e.hist_bins), range=(lo, hi))
    return c.astype(np.int64)


def hist_quantile(counts, lo, hi, q):
    """Quantile ``q`` of a fixed-bin histogram (linear inside the bin)."""
    c = np.asarray(counts, np.float64)
    tot = c.sum()
    if tot <= 0:
        return float("nan")
    edges = np.linspace(lo, hi, len(c) + 1)
    cum = np.cumsum(c)
    k = int(np.searchsorted(cum, q * tot, side="left"))
    k = min(k, len(c) - 1)
    prev = cum[k - 1] if k else 0.0
    frac = (q * tot - prev) / c[k] if c[k] > 0 else 0.0
    return float(edges[k] + frac * (edges[k + 1] - edges[k]))


def planned_indices(keys):
    """The indices of the observation keys plus D5's three, sorted."""
    out = set()
    for k in keys:
        s = settings.Setting.parse(k)
        out.add(s.index)
        out |= set(settings.DETECT[s.detect].indices)
    return sorted(out)


def v2_settings(site2, pkg, keys, log_every=50):
    """The settings pass (spec §2.4-2.6), truth-blind, from the package.

    * per W3 year, per planned index: the fixed-bin histogram of X over the
      clear observations (C1, floor0) of keep_full(P0, W3); pooled: t_NIR =
      -Otsu (NIR's t_k), s_k = IQR_k / IQR_NDWI (WRI: quantiles of the log
      histogram);
    * per scene and index of the D2 keys: :func:`obs.d2_threshold` on Omega
      (streamed: the scene's range, then its 256-bin histogram);
    * per scene and index of the D3 keys: :func:`obs.d3_threshold` on the
      bounding box of dom plus 1 km, read from the stores.

    Writes ``settings.json`` and ``years/<Y>/settings/thresholds.npz``.
    """
    t0 = time.time()
    rec = pkg.record
    idx_all = planned_indices(keys)
    d2_idx = sorted({settings.Setting.parse(k).index for k in keys
                     if settings.Setting.parse(k).detect == "D2"})
    d3_idx = sorted({settings.Setting.parse(k).index for k in keys
                     if settings.Setting.parse(k).detect == "D3"})
    yrs = w3_years(rec)
    pos_k = pkg.positions(pkg.geometry["keep_full"].astype(np.int64))
    hist = {i: {} for i in idx_all}
    for y in yrs:
        sel = pkg.window_years_safe(y)
        acc = {i: np.zeros(int(settings.INDEX[i].hist_bins), np.int64) for i in idx_all}
        for a in range(0, len(pos_k), TILE * 8):
            pk = pos_k[a:a + TILE * 8]
            for i in idx_all:
                s = settings.Setting(index=i)
                o = pkg.observe(s, sel, pk, params={"t_k": 0.0, "s_k": 1.0})
                acc[i] += _hist_fixed(settings.INDEX[i], o.X[o.V])
        for i in idx_all:
            hist[i][str(y)] = acc[i]
        log(f"  [settings] {site2.site} {y}: index histograms of {len(idx_all)} indices on "
            f"{len(pos_k):,} keep_full px ({time.time() - t0:.0f} s, RSS {_rss_mb():.0f} MB)")
    doc = {"format": V2_FORMAT, "years": yrs, "indices": {}, "hist": {}}
    pooled = {i: sum(hist[i].values()) for i in idx_all}
    for i in idx_all:
        e = settings.INDEX[i]
        lo, hi = e.hist_range
        q1, q3 = (hist_quantile(pooled[i], lo, hi, q) for q in (0.25, 0.75))
        if e.log_hist:
            q1, q3 = float(np.exp(q1)), float(np.exp(q3))
        doc["indices"][i] = {"iqr": q3 - q1, "q25": q1, "q75": q3,
                             "n": int(pooled[i].sum())}
        doc["hist"][i] = {"range": [lo, hi], "bins": int(e.hist_bins),
                          "per_year_n": {y: int(c.sum()) for y, c in hist[i].items()}}
    iqr_ndwi = doc["indices"].get("ndwi", {}).get("iqr")
    for i in idx_all:
        e = settings.INDEX[i]
        ent = doc["indices"][i]
        if e.t_k is None:
            lo, hi = e.hist_range
            edges = np.linspace(lo, hi, int(e.hist_bins) + 1)
            t, eta, m = obs.otsu_hist(pooled[i], (edges[:-1] + edges[1:]) / 2.0)
            ent.update(t_k=float(t), t_nir=float(-t), otsu_eta=eta, otsu_minority=m)
        else:
            ent["t_k"] = float(e.t_k)
        if i == "ndwi":
            ent["s_k"] = 1.0
        elif iqr_ndwi and iqr_ndwi > 0 and ent["iqr"] > 0:
            ent["s_k"] = float(ent["iqr"] / iqr_ndwi)
        else:
            ent["s_k"] = float("nan")
    sp = _p(site2, "settings.json")
    write_json_lf(sp, doc)
    pkg.__dict__.pop("settings_doc", None)
    outs = [sp]
    omega_pos = pkg.positions(pkg.omega)
    d2p = dict(settings.DETECT["D2"].params)
    d3p = dict(settings.DETECT["D3"].params)
    reg = _bbox_region(_flat_mask(pkg.dom, pkg.shape), REGION_PAD_M / site2.pixel_m,
                       pkg.shape, pkg.lattice) if d3_idx else None
    src = open_source(site2, rec) if d3_idx else None
    try:
        for y in pkg.years:
            sel = pkg.window_years_safe(y)
            Ty = len(sel)
            arrays = {"dates": np.asarray(pkg.dates[sel])}
            for i in d2_idx:
                e = settings.INDEX[i]
                t_k = doc["indices"][i]["t_k"]
                mn = np.full(Ty, np.inf)
                mx = np.full(Ty, -np.inf)
                nn = np.zeros(Ty, np.int64)
                chunks = range(0, len(omega_pos), TILE * 8)
                for a in chunks:
                    o = pkg.observe(settings.Setting(index=i), sel, omega_pos[a:a + TILE * 8],
                                    params={"t_k": 0.0, "s_k": 1.0})
                    for j in range(Ty):
                        v = _hist_index(e, o.X[j][o.V[j]])
                        if v.size:
                            mn[j] = min(mn[j], v.min())
                            mx[j] = max(mx[j], v.max())
                            nn[j] += v.size
                counts = np.zeros((Ty, int(d2p["bins"])), np.int64)
                for a in chunks:
                    o = pkg.observe(settings.Setting(index=i), sel, omega_pos[a:a + TILE * 8],
                                    params={"t_k": 0.0, "s_k": 1.0})
                    for j in range(Ty):
                        if nn[j] < int(d2p["min_px"]) or not mn[j] < mx[j]:
                            continue
                        v = _hist_index(e, o.X[j][o.V[j]])
                        c, _ = np.histogram(v, bins=int(d2p["bins"]), range=(mn[j], mx[j]))
                        counts[j] += c
                t = np.full(Ty, t_k, np.float64)
                tot = np.full(Ty, np.nan)
                eta = np.zeros(Ty)
                mino = np.zeros(Ty)
                used = np.zeros(Ty, bool)
                for j in range(Ty):
                    if nn[j] < int(d2p["min_px"]) or not mn[j] < mx[j]:
                        continue
                    edges = np.linspace(mn[j], mx[j], int(d2p["bins"]) + 1)
                    tt, et, mi = obs.otsu_hist(counts[j], (edges[:-1] + edges[1:]) / 2.0)
                    tot[j] = float(np.exp(tt)) if e.log_hist else tt
                    eta[j], mino[j] = et, mi
                    if np.isfinite(tt) and et >= float(d2p["eta_min"]) and \
                            mi >= float(d2p["minority_min"]):
                        t[j], used[j] = tot[j], True
                arrays.update({f"d2_t_{i}": t.astype(np.float32), f"d2_t_otsu_{i}": tot,
                               f"d2_eta_{i}": eta, f"d2_minority_{i}": mino, f"d2_n_{i}": nn,
                               f"d2_used_{i}": used})
                log(f"  [settings D2 {y}] {i}: {Ty} scenes, Otsu used in {int(used.sum())} "
                    f"({time.time() - t0:.0f} s, RSS {_rss_mb():.0f} MB)")
            if d3_idx:
                r0, r1, c0, c1 = reg
                lat = pkg.lattice
                local = None if lat is None else ((lat[0] - r0) % 2, (lat[1] - c0) % 2)
                bands = sorted({b for i in d3_idx for b in settings.INDEX[i].bands} | {"SCL"},
                               key=TILE_BANDS.index)
                res = {i: {k: np.zeros(Ty, dt) for k, dt in
                           (("t", np.float32), ("t_otsu", np.float64), ("n_buffer", np.int64),
                            ("n_edges", np.int64), ("valid_share", np.float64),
                            ("used", bool), ("skipped", bool))} for i in d3_idx}
                t_d3 = time.time()
                for qblk in src.time_blocks(sel):
                    # one read of the region per stored time block, band by band
                    rawb = {}
                    for b in bands:
                        a = src.read_block(qblk, [b], slice(r0, r1))[b]
                        rawb[b] = np.ascontiguousarray(a[:, :, c0:c1])
                        del a
                    for k, q in enumerate(qblk):
                        j = int(np.searchsorted(sel, q))         # the scene's row in res
                        raw = {b: arr[k:k + 1] for b, arr in rawb.items()}
                        cache = _ObsCache(raw, int(pkg.offsets[q]), local)
                        for i in d3_idx:
                            t_k = doc["indices"][i]["t_k"]
                            X, V, _ = cache.observe(settings.Setting(index=i),
                                                    params={"t_k": t_k, "s_k": 1.0})
                            r = obs.d3_threshold(X[0], V[0], t_k, d3p,
                                                 log_hist=settings.INDEX[i].log_hist)
                            for kk in res[i]:
                                res[i][kk][j] = r[kk]
                    del rawb
                    log(_progress(f"settings D3 {y}", int(np.searchsorted(sel, qblk[-1])) + 1,
                                  Ty, t_d3))
                for i in d3_idx:
                    arrays.update({f"d3_{k}_{i}": v for k, v in res[i].items()})
            out = _year_dir(site2, y, "settings", "thresholds.npz")
            product.save_npz(out, arrays)
            outs.append(out)
    finally:
        if src is not None:
            src.close()
    log(f"[settings] {site2.site}: " + ", ".join(
        f"{i} t_k {doc['indices'][i]['t_k']:.4g} s_k {doc['indices'][i]['s_k']:.4g}"
        for i in idx_all) + f" ({time.time() - t0:.0f} s)")
    return outs, {"indices": {i: {k: doc["indices"][i][k] for k in ("t_k", "s_k")}
                              for i in idx_all}, "d2": d2_idx, "d3": d3_idx}


def v2_counters(site2, pkg, year, keys):
    """uint8 counters at dom (clr, wet over the year's scenes) of the keys
    that need the settings pass (D2, D3, NIR), from the package."""
    sel = pkg.window_years_safe(year)
    d = _shard_dir(site2, pkg.dom_version, year, "counters")
    os.makedirs(d, exist_ok=True)
    outs = []
    for k in dom_counter_plan(keys):
        clr, wet = pkg.count(k, sel)
        f = os.path.join(d, f"{k}.npz")
        product.save_npz(f, {"clr": clr.astype(np.uint8), "wet": wet.astype(np.uint8),
                             "key": np.array(k)})
        outs.append(f)
    return outs, {"keys": len(outs)}


def dom_counters(site2, dom_version, years, key):
    """Sum over ``years`` of the at-dom counters of a threshold key, or None."""
    clr = wet = None
    for y in years:
        f = _shard_dir(site2, dom_version, y, "counters", f"{key}.npz")
        if not os.path.exists(f):
            return None
        z = product.load_npz(f)
        clr = z["clr"].astype(np.int32) if clr is None else clr + z["clr"]
        wet = z["wet"].astype(np.int32) if wet is None else wet + z["wet"]
    return clr, wet


# ─────────────────────────────────────────────────────────────────────────────
#  MNAR table (site): masked rate x h_B decile x s band x SCL class x state
# ─────────────────────────────────────────────────────────────────────────────

def logistic_slope(n, k, x, iters=50):
    """Binomial GLM (logit link) of k successes in n trials on x by IRLS:
    ``(intercept, slope, se_slope)``; NaN when degenerate."""
    n = np.asarray(n, np.float64)
    k = np.asarray(k, np.float64)
    x = np.asarray(x, np.float64)
    ok = n > 0
    n, k, x = n[ok], k[ok], x[ok]
    if len(n) < 3 or np.ptp(x) == 0 or k.sum() in (0.0, n.sum()):
        return float("nan"), float("nan"), float("nan")
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    p0 = np.clip(k.sum() / n.sum(), 1e-6, 1 - 1e-6)
    beta[0] = np.log(p0 / (1 - p0))
    for _ in range(iters):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = n * mu * (1 - mu)
        if not np.all(np.isfinite(w)) or w.sum() <= 0:
            return float("nan"), float("nan"), float("nan")
        z = eta + (k - n * mu) / np.maximum(w, 1e-12)
        XtW = X.T * w
        try:
            new = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan"), float("nan")
        if np.max(np.abs(new - beta)) < 1e-10:
            beta = new
            break
        beta = new
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = n * mu * (1 - mu)
    try:
        cov = np.linalg.inv((X.T * w) @ X)
        se = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        se = float("nan")
    return float(beta[0]), float(beta[1]), se


MNAR_COLUMNS = ("window", "cloud", "h_decile", "s_band", "scl", "rising", "n", "n_masked",
                "rate")


def mnar_windows(pkg):
    """The windows whose years are all in the record: W3 (if complete) and
    the calendar years."""
    out = []
    for wid in ("W3", "W1", "W2") + tuple(f"Y{y}" for y in range(2017, 2026)):
        w = settings.window(wid)
        if all(y in pkg.years for y in w.years) and wid not in out:
            out.append(wid)
    return [w for w in out if not (w == "W1" and "Y2025" in out)]


def v2_mnar(site2, pkg, clouds=MNAR_CLOUDS):
    """The MNAR diagnostic (spec §2.2), truth-free: per window and pixel
    cloud level, the masked rate on keep_full(P0) against the h_B decile of
    the window's scenes, the 3-km s band, the SCL class and the tide state
    (rising), and the logistic slope of 'masked' on h_B (per metre, on scene
    aggregates). Masked = not valid under the level's pixel rule; scene
    rules (C2, C6, C8) share C1's pixel mask. Writes mnar.csv, mnar.json."""
    keep = pkg.geometry["keep_full"].astype(np.int64)
    pos_k = pkg.positions(keep)
    band = np.asarray(pkg.geometry["band_of"], np.int64)
    K = int(pkg.geometry["K"])
    nb = K + 1
    has_c11 = bool(pkg.dom_meta) and any("C11" in t for t in pkg.dom_meta.get("terms", []))
    clouds = [c for c in clouds if (c != "C11" or has_c11) and (c != "C3" or pkg.c3_stored())]
    rows = []
    fits = {}
    for wid in mnar_windows(pkg):
        sel = pkg.window(wid)
        h = pkg.h0[sel]
        qs_ = np.quantile(h, np.linspace(0, 1, MNAR_DECILES + 1))
        dec = np.clip(np.searchsorted(qs_[1:-1], h, side="right"), 0, MNAR_DECILES - 1)
        ris = pkg.rising[sel].astype(np.int64)
        shape = (MNAR_DECILES, nb, 13, 2)
        acc = {c: [np.zeros(shape, np.int64), np.zeros(shape, np.int64)] for c in clouds}
        per_scene = {c: [np.zeros(len(sel), np.int64), np.zeros(len(sel), np.int64)]
                     for c in clouds}
        for a in range(0, len(pos_k), TILE * 8):
            pk = pos_k[a:a + TILE * 8]
            bk = np.where(band[a:a + TILE * 8] < 0, K, band[a:a + TILE * 8])
            names = ["SCL", "cloudbits"] if pkg.dom is not None else ["SCL"]
            raw = pkg.read(names, sel, pk)
            scl8 = raw["SCL"]
            codes = _scl_codes13(scl8)
            base = (((dec[:, None] * nb + bk[None, :]) * 13 + codes) * 2 + ris[:, None])
            for c in clouds:
                if c in ("C3", "C11"):
                    v = pkg.cloud_valid(c, raw["cloudbits"], scl8)
                else:
                    v = obs.pixel_valid(c, scl8)
                masked = ~v
                acc[c][0] += np.bincount(base.ravel(), minlength=int(np.prod(shape))
                                         ).reshape(shape)
                acc[c][1] += np.bincount(base.ravel(), weights=masked.ravel(),
                                         minlength=int(np.prod(shape))).astype(np.int64
                                                                               ).reshape(shape)
                per_scene[c][0] += masked.shape[1]
                per_scene[c][1] += masked.sum(axis=1)
        fits[wid] = {}
        for c in clouds:
            n_, m_ = acc[c]
            if n_.sum() == 0:
                continue
            for idx in zip(*np.nonzero(n_)):
                dd, bb, ss, rr = (int(v) for v in idx)
                rows.append({"window": wid, "cloud": c, "h_decile": dd,
                             "s_band": bb if bb < K else -1, "scl": ss if ss < 12 else "nodata",
                             "rising": rr, "n": int(n_[idx]), "n_masked": int(m_[idx]),
                             "rate": float(m_[idx] / n_[idx])})
            b0, b1, se = logistic_slope(per_scene[c][0], per_scene[c][1], h)
            fits[wid][c] = {"intercept": b0, "slope_per_m": b1, "se": se,
                            "n_scenes": int(len(sel)), "n_obs": int(n_.sum()),
                            "masked_rate": float(m_.sum() / n_.sum())}
    cp = _p(site2, "mnar.csv")
    _write_csv_lf(cp, rows, MNAR_COLUMNS)
    jp = _p(site2, "mnar.json")
    write_json_lf(jp, {"format": V2_FORMAT, "clouds": clouds, "deciles": MNAR_DECILES,
                       "fits": fits, "note": ("masked = not valid under the pixel rule on "
                                              "keep_full(P0, W3); scene rules share C1's "
                                              "pixel mask; C11 outside the OCM region = C1")})
    return [cp, jp], {"windows": list(fits), "rows": len(rows)}


# ─────────────────────────────────────────────────────────────────────────────
#  subsets per window (site)
# ─────────────────────────────────────────────────────────────────────────────

def _scene_stats_all(site2, pkg):
    frames = [read_scene_stats_v2(site2, y) for y in pkg.years]
    st = pd.concat(frames, ignore_index=True)
    if not np.array_equal(st["q"].to_numpy(), np.arange(pkg.T)):
        raise PackageError("scene_stats.csv files are not one row per record scene in order")
    return st


def transition_zone(pkg, sel, scene_stats):
    """The C2 transition zone of the record positions ``sel`` (spec §2.2,
    ``stability.reference_and_clouds`` pass 1 on the production
    observation): wet/dry votes of the clean scenes (frame bad share <=
    0.05), stable at >= 0.95, transition dilated 10 iterations. Bool (H, W)."""
    from scipy.ndimage import binary_dilation
    p = settings.CLOUD["C2"].params
    clean = scene_stats["frame_bad"].to_numpy(float)[sel] <= float(p["clean_scene_max_bad"])
    wv = np.zeros((pkg.H, pkg.W), np.int64)
    dv = np.zeros((pkg.H, pkg.W), np.int64)
    for y in np.unique(pkg.year_of[sel]):
        mw = pkg._bits(int(y), "tzwet")
        md = pkg._bits(int(y), "tzdry")
        loc = sel[(pkg.year_of[sel] == y)]
        c = clean[pkg.year_of[sel] == y]
        for q in loc[c]:
            j = int(q - pkg.window_years_safe(y)[0])
            wv += _unpack_rows(mw[j], pkg.W)
            dv += _unpack_rows(md[j], pkg.W)
    votes = wv + dv
    votes[votes == 0] = 1
    st = float(p["stable_threshold"])
    stable = ((wv / votes) >= st) | ((dv / votes) >= st)
    tz = ~stable
    if int(p["transition_buffer_px"]) > 0:
        tz = binary_dilation(tz, iterations=int(p["transition_buffer_px"]))
    return tz


def rule_flags(pkg, sel, scene_stats, tz=None):
    """Keep flags of the scene rules C2, C6, C7, C8 for record positions
    ``sel`` (spec §2.2)."""
    st = scene_stats.iloc[sel]
    flags = {}
    if tz is None:
        tz = transition_zone(pkg, sel, scene_stats)
    n_t = max(int(tz.sum()), 1)
    tz_bad = np.zeros(len(sel))
    for k, q in enumerate(sel):
        y = int(pkg.year_of[q])
        j = int(q - pkg.window_years_safe(y)[0])
        bad = _unpack_rows(pkg._bits(y, "bad")[j], pkg.W)
        tz_bad[k] = float(np.count_nonzero(bad & tz)) / n_t
    p2 = settings.CLOUD["C2"].params
    flags["C2"] = np.asarray(obs.scene_c2(st["frame_bad"].to_numpy(float), tz_bad,
                                          float(p2["clean_scene_max_bad"]), float(p2["tz_max"])),
                             bool)
    from pyintertidal import granadeiro2021 as g21
    flags["C6"] = np.asarray(g21.scene_rule(st["g21_cloud"].to_numpy(float),
                                            st["g21_nodata"].to_numpy(float),
                                            np.ones(len(sel), bool), np.ones(len(sel), bool)),
                             bool)
    flags["C7"] = st["ard_good"].to_numpy(float) >= float(settings.CLOUD["C7"].params["min_good"])
    keep = pkg.geometry["keep_full"].astype(np.int64)
    if pkg.dom is not None and len(keep):
        pos = pkg.positions(keep)
        share = np.zeros(len(sel))
        n = np.zeros(len(sel))
        for a in range(0, len(pos), TILE * 8):
            s8 = pkg.read("SCL", sel, pos[a:a + TILE * 8])["SCL"]
            share += obs.cloud_c1(s8).sum(axis=1)
            n += s8.shape[1]
        flags["C8"] = share / np.maximum(n, 1) >= float(settings.CLOUD["C8"].params["min_share"])
    return flags, tz_bad


def v2_subsets(site2, pkg, windows=None):
    """``subsets_<window>.npz`` per window (spec §2.8): the window's global
    record positions, the rand permutations of the curve sets on it, the
    scene-rule keep flags (C2, C6, C7, C8) and the matched-N C1 draws of
    each rule (3 draws)."""
    st = _scene_stats_all(site2, pkg)
    outs = []
    info = {}
    for wid in (windows or mnar_windows(pkg)):
        sel = pkg.window(wid)
        T = len(sel)
        flags, tz_bad = rule_flags(pkg, sel, st)
        arrays = {"positions": sel, "T": np.array(T, np.int64), "tz_bad": tz_bad,
                  "window_num": np.array(settings.window_num(wid), np.int64)}
        for r, f in flags.items():
            arrays[f"rule_{r}"] = np.flatnonzero(f).astype(np.int64)
            n_rule = int(f.sum())
            for d, m in enumerate(settings.matched_subsets(site2.site, wid, np.arange(T),
                                                           n_rule, draws=3)
                                  if 0 < n_rule <= T else []):
                arrays[f"matched_{r}_{d}"] = m.astype(np.int64)
        for cs in settings.CURVES.values():
            if wid in cs.windows and cs.family == "rand":
                for N, R, _ in cs.grid:
                    if N == "all" or int(N) > T:
                        continue
                    kind, subs = settings.rand_subsets(site2.site, wid, T, N, R)
                    for r, sub in enumerate(subs):
                        arrays[f"rand_{int(N)}_{r}"] = sub.astype(np.int64)
        out = _p(site2, f"subsets_{wid}.npz")
        product.save_npz(out, arrays)
        outs.append(out)
        info[wid] = {"T": T, **{r: int(f.sum()) for r, f in flags.items()}}
        log(f"[subsets] {site2.site} {wid}: T = {T}; rules " + ", ".join(
            f"{r} {int(f.sum())}" for r, f in flags.items()))
    return outs, info


def load_subsets(site2, wid):
    p = _p(site2, f"subsets_{wid}.npz")
    if not os.path.exists(p):
        raise PackageError(f"{p} is missing: run prepare --step subsets")
    return product.load_npz(p)


# ─────────────────────────────────────────────────────────────────────────────
#  the level table (site) and the shard manifests (per year)
# ─────────────────────────────────────────────────────────────────────────────

def series_key(site2, rec):
    """What the 1-min h_B series depends on: the provider's name and period
    and the sha256 of every IOC chunk the record lists (its cache is matched
    on this key, not only on v8's name-and-period tag)."""
    return config.sha256_json({"name": rec["boundary"]["name"],
                               "period": list(boundary_period(site2)),
                               "ioc": [e.get("sha256") for e in rec.get("ioc_chunks", [])]})


def v2_series(site2, rec, provider):
    """The 1-min h_B series over the provider's period (v8's grid, chunks
    and tag; ``levels.boundary_series_1min``) and its marks (``marks_lp``
    on the 61-min running mean, ``marks_raw``; ``levels.build_marks``):
    ``boundary_1min.npz`` and ``marks.npz``, read by the lag truths (HW/LW,
    truth (b) on the 10-min series) and the range filter of the elevation
    truth. The cache is reused only under the same :func:`series_key`
    (``boundary_1min.key.json``): new or corrected IOC chunks rebuild it."""
    name = rec["boundary"]["name"]
    out = _p(site2, SERIES_V2)
    kp = out[:-4] + ".key.json"
    key = series_key(site2, rec)
    old = None
    if os.path.exists(kp):
        try:
            old = product.read_json(kp).get("key")
        except (OSError, ValueError):
            old = None
    if os.path.exists(out) and old != key:
        stale = f"{out[:-4]}.stale-{product.utc_stamp()}.npz"
        os.replace(out, stale)
        log(f"[series] {site2.site}: the cached 1-min h_B was built from other IOC chunks; "
            f"kept as {os.path.basename(stale)} and rebuilt")
    t_min, h, cached = levels.boundary_series_1min(provider, name, out,
                                                   period=boundary_period(site2))
    write_json_lf(kp, {"key": key, "name": name, "period": list(boundary_period(site2)),
                       "n_ioc_chunks": len(rec.get("ioc_chunks", []))})
    mk = levels.build_marks(t_min, h)
    mp = _p(site2, MARKS_V2)
    levels.save_marks(mp, **mk)
    log(f"[series] {site2.site}: 1-min h_B {'cached' if cached else 'built'} over "
        f"{boundary_period(site2)}: {len(h):,} samples ({int(np.isfinite(h).sum()):,} finite); "
        f"marks_lp {mk['marks_lp']['n_marks']:,}, marks_raw {mk['marks_raw']['n_marks']:,}")
    return [out, mp, kp], {"n": int(len(h)), "cached": bool(cached),
                           "n_marks_lp": int(mk["marks_lp"]["n_marks"])}


def v2_levels(site2, pkg, provider, chunk=None):
    """The exact level table at the record instants (v13 §2.7 on the v2
    record): integer tau in [-90, max(240, 180 (K_full - 1))]."""
    tab = levels.build_level_table(provider, pkg.t, config.tau_tab_max(int(pkg.geometry["K"])),
                                   chunk=int(chunk or levels.CHUNK), h0=pkg.h0)
    out = _p(site2, "levels.npz")
    tab.save(out)
    return [out], {"tau_tab_max": tab.tau_hi}


# ─────────────────────────────────────────────────────────────────────────────
#  the water-level sources (site): LB, LT, LM, LG at the record instants
# ─────────────────────────────────────────────────────────────────────────────

SOURCES_FORMAT = "validation-v2-level-sources-1"


def _cmems_root(site2, cmems_root=None):
    return cmems_root or os.path.join(site2.root, "common")


def lm_plan(site2, pkg_years):
    """``{window: {"dataset", "reason"}}`` of the Campaign-1 elevation
    windows whose years the record holds (``fetch.cmems.choose_dataset``)."""
    from experiments.benchmark.fetch import cmems
    out = {}
    for wid in LEVEL_WINDOWS:
        w = settings.window(wid)
        if not set(w.years) <= set(int(y) for y in pkg_years):
            continue
        dsid, why = cmems.choose_dataset(site2.site, w.start, w.end)
        out[wid] = {"dataset": dsid, "reason": why}
    return out


def _judge_series_ids(site2):
    """File ids of the judges' series of a site (LG reads them)."""
    from experiments.benchmark import judges
    try:
        stas = judges.stations(site2.site, (site2.cfg.extra or {}).get("stations_json"))
    except (OSError, KeyError, ValueError):
        return []
    return [_file_id(judges.series_path(st)) for st in stas]


def sources_inputs(site2, pkg_years, cmems_root=None, lt_kw=None):
    """The step inputs of 'sources' outside the package: the CMEMS manifests
    and stamp conventions of the datasets the windows need, the judge table
    and the judges' series (LG), and the tide models LT uses."""
    from experiments.benchmark.fetch import cmems
    from experiments.benchmark.fetch import tides as ft
    root = _cmems_root(site2, cmems_root)
    plan = lm_plan(site2, pkg_years)
    lm = {}
    for dsid in sorted({e["dataset"] for e in plan.values() if e["dataset"]}):
        d = cmems.site_cache_dir(site2.site, root, dsid)
        lm[dsid] = {"manifest": _small_sha(os.path.join(d, cmems.MANIFEST_NAME)),
                    "stamp": _small_sha(os.path.join(d, cmems.STAMP_SHIFT_NAME))}
    try:
        models = [list(m) for m in ((lt_kw or {}).get("models")
                                    or ft.lt_models((lt_kw or {}).get("directory")
                                                    or config.TIDE_DIR))]
    except Exception as e:                    # eo-tides missing: recorded, LT not computable
        models = f"{type(e).__name__}: {e}"
    lg = None
    if site2.site in settings.LEVEL["LG"].sites:
        lg = {"judges": _small_sha(os.path.join(site2.truth_dir, "judges.json")),
              "series": _judge_series_ids(site2)}
    return {"lm": lm, "lm_windows": {k: v["dataset"] for k, v in plan.items()},
            "lt_models": models, "lg": lg}


def _ds_tag(dsid):
    return re.sub(r"[^A-Za-z0-9.-]+", "_", str(dsid))


def v2_sources(site2, pkg, sources=("LB", "LT", "LM", "LG"), lt_kw=None, cmems_root=None,
               lm_fields=None, lg_kw=None):
    """The water-level sources at the record instants (spec §1.6; the
    ``levels --sources`` of §13.3), on the dom pixels.

    One ``levels_<source>.npz`` per source (``levels.SourceLevels``): LB (the
    table's tau = 0 row), LT (eo-tides 1 km nodes, nearest node; an ensemble
    without its BT26 ranking is saved NOT COMPUTABLE with its members, which
    ``levels.lt_with_ranking`` finishes later), LM (one file per CMEMS
    dataset the windows need, ``levels_LM_<dataset>.npz``; an hourly-mean
    dataset needs its measured stamp convention) and LG (the judge table's
    included judges at their s_g, centred series; Escalda, Wadden, Ems only,
    labelled upper bound). A source that cannot be computed is saved with
    its status and reason (NOT COMPUTABLE, NOT AVAILABLE, not applicable),
    never as zero. ``sources.json`` lists every file with its sha256 and,
    per Campaign-1 window, the LM dataset and file that serve it (None:
    NOT AVAILABLE, with the reason). The level-source units key on these
    shas; a level never enters a shard's package_key (it is a factor, §13.2).

    ``lt_kw``, ``lm_fields`` ({dataset: fields}) and ``lg_kw`` (e.g. the
    time-stamp conventions of synthetic judges) are injected by tests.
    """
    from experiments.benchmark import judges
    from experiments.benchmark.fetch import cmems
    t0 = time.time()
    site = site2.site
    t = pkg.t
    pix = np.asarray(pkg.dom, np.int64)
    s_km = np.asarray(pkg.s_frame, np.float64).ravel()[pix]
    tab = pkg.levels
    ref = np.asarray(tab.h0, np.float64)
    outs = []
    doc = {"format": SOURCES_FORMAT, "site": site, "T": int(pkg.T), "n_pix": int(len(pix)),
           "dom_version": pkg.dom_version, "files": {}, "lm_windows": {}}

    def save(src, tag=""):
        p = levels.source_path(site2.dir, src.source, tag)
        src.save(p)
        outs.append(p)
        doc["files"][os.path.basename(p)] = {
            "source": src.source, "status": src.status, "reason": src.reason,
            "unit_status": levels.UNIT_STATUS[src.status], "groups": int(src.G),
            "flags": src.flags, "sha256": product.sha256_file(p)}
        log(f"[sources] {site} {src.source}{' ' + tag if tag else ''}: {src.status}"
            + (f" ({src.reason})" if src.reason else f", {src.G} groups"))
        return os.path.basename(p)

    def failed(source, e, flags=None):
        return levels.SourceLevels.unavailable(source, site, t, levels.STATUS_NOT_COMPUTABLE,
                                               f"{type(e).__name__}: {e}", flags)

    if "LB" in sources:
        save(levels.level_lb(tab, site))
    if "LT" in sources:
        try:
            src = levels.level_lt(pkg.grid, t, pix, site, **(lt_kw or {}))
        except Exception as e:                # tide models missing, eo-tides error
            src = failed("LT", e)
        save(src)
    if "LM" in sources:
        root = _cmems_root(site2, cmems_root)
        plan = lm_plan(site2, pkg.years)
        files = {}
        for dsid in sorted({e["dataset"] for e in plan.values() if e["dataset"]}):
            d = cmems.site_cache_dir(site, root, dsid)
            man = cmems.read_manifest(d)
            flags = {"dataset": dsid, "degenerate": site in cmems.DEGENERATE}
            try:
                if lm_fields is not None and dsid in lm_fields:
                    fields = lm_fields[dsid]
                elif man is None:
                    raise FileNotFoundError(f"no CMEMS manifest in {d} (run fetch cmems)")
                elif man.get("status") == cmems.STATUS_OK:
                    fields = cmems.manifest_files(d, man)
                else:
                    fields = []
                shift = (cmems.read_stamp_shift(d) if cmems.DATASETS[dsid]["kind"] == "m"
                         else None)
                src = levels.level_lm(site, t, pix, s_km, pkg.grid, fields, ref, dsid,
                                      manifest=man, stamp_shift_min=shift)
            except Exception as e:
                src = failed("LM", e, flags)
            files[dsid] = save(src, tag=_ds_tag(dsid))
        for wid, e in plan.items():
            doc["lm_windows"][wid] = {"dataset": e["dataset"], "reason": e["reason"],
                                      "file": files.get(e["dataset"]),
                                      "status": (doc["files"][files[e["dataset"]]]["unit_status"]
                                                 if e["dataset"] else "NOT AVAILABLE")}
    if "LG" in sources:
        if site not in settings.LEVEL["LG"].sites:
            src = levels.level_lg(site, t, ref, [], pix, s_km)
        else:
            jt = os.path.join(site2.truth_dir, "judges.json")
            try:
                if not os.path.exists(jt):
                    raise FileNotFoundError(f"{jt} is missing: run truth {site} first (the "
                                            "judge table)")
                table = judges.load(jt, judges.grid_of(pkg))
                s_g = {r["code"]: float(r["s_g_km"]) for r in table
                       if r.get("included") and r.get("s_g_km") is not None}
                knots = levels.lg_knots(site, t, s_g,
                                        path=(site2.cfg.extra or {}).get("stations_json"),
                                        **(lg_kw or {}))
                src = levels.level_lg(site, t, ref, knots, pix, s_km)
            except Exception as e:
                src = failed("LG", e, {"upper_bound": True, "confirmatory": False})
        save(src)
    sp = _p(site2, SOURCES_JSON)
    doc["seconds"] = round(time.time() - t0, 1)
    write_json_lf(sp, doc)
    outs.append(sp)
    return outs, {k: v["status"] for k, v in doc["files"].items()}


def load_source(site2, source, window=None):
    """The ``levels.SourceLevels`` of a site's source (LM: the file that
    serves ``window``, from ``sources.json``)."""
    doc = product.read_json(_p(site2, SOURCES_JSON))
    if source == "LM":
        if window is None:
            raise ValueError("LM is per window: pass the window id")
        e = doc["lm_windows"].get(window)
        if e is None or e.get("file") is None:
            return levels.SourceLevels.unavailable(
                "LM", site2.site, pd.DatetimeIndex([]), levels.STATUS_NOT_AVAILABLE,
                (e or {}).get("reason") or f"no LM dataset for {window}")
        name = e["file"]
    else:
        name = os.path.basename(levels.source_path(site2.dir, source))
    p = _p(site2, name)
    if product.sha256_file(p) != doc["files"][name]["sha256"]:
        raise PackageError(f"{p} changed since the sources step (sha256)")
    return levels.SourceLevels.load(p)


#: the FROZEN sections of a shard's package_key (spec §13.2)
SHARD_KEY_SECTIONS = ("record", "package", "domain", "levels", "subsets")


def shard_key_payload(man):
    return {"record_year": man["record_year_sha256"], "files": {k: e["sha256"] for k, e in
                                                               man["files"].items()},
            "obs_core": man["obs_core"], "frozen": man["frozen"],
            "fingerprint": man["fingerprint"], "ioc": man["ioc"], "sensing": man["sensing"],
            "dom_version": man["dom_version"]}


def v2_manifest(site2, pkg, year):
    """``shards/<dv>/<Y>/manifest.json`` and the shard's package_key:
    sha256 of the year's record slice, every tile and the dom file, the obs
    core, the FROZEN record/package/domain/levels/subsets shas, the store
    fingerprint, the IOC chunks, the sensing-time file hash and the
    dom_version (no factor level enters it)."""
    dv = pkg.dom_version
    d = _shard_dir(site2, dv, year)
    rec = pkg.record
    sel = pkg.window_years_safe(year)
    rec_year = {"dates": [rec["dates"][q] for q in sel], "t_ns": [rec["t_ns"][q] for q in sel],
                "h0": [rec["h0"][q] for q in sel], "offsets": [rec["offsets"][q] for q in sel],
                "rising": [rec["rising"][q] for q in sel], "lattice20": rec["lattice20"]["phase"],
                "grid": rec["grid"]["sha256"]}
    files = {}
    shard = product.read_json(os.path.join(d, "shard.json"))
    for name in [b for b in TILE_BANDS if b in shard.get("bands", TILE_BANDS)] + ["cloudbits"]:
        p = os.path.join(d, f"tile_{name}.npy")
        files[f"tile_{name}.npy"] = product.file_entry(p)
    files["dom.npz"] = product.file_entry(_p(site2, "dom", dv, "dom.npz"))
    man = {"format": V2_FORMAT, "site": site2.site, "year": int(year), "dom_version": dv,
           "record_year_sha256": config.sha256_json(rec_year), "files": files,
           "obs_core": obs.core_sha256(),
           "frozen": config.frozen_shas(SHARD_KEY_SECTIONS),
           "fingerprint": (rec.get("fingerprints") or {}).get(str(year)),
           "ioc": [e.get("sha256") for e in rec.get("ioc_chunks", [])],
           "sensing": rec.get("sensing_sha256"), "git": product.git_state(),
           "created_utc": product.utc_stamp()}
    man["package_key"] = config.sha256_json(shard_key_payload(man))
    out = os.path.join(d, "manifest.json")
    write_json_lf(out, man)
    log(f"[manifest] {site2.site} {year} (dom {dv}): package_key {man['package_key']}")
    return [out], {"package_key": man["package_key"]}


def shard_keys(site2, dom_version=None, years=None):
    """``{year: package_key}`` of the shards with a manifest."""
    if dom_version is None:
        cur = current_dom(site2)
        dom_version = None if cur is None else cur["dom_version"]
    out = {}
    if dom_version is None:
        return out
    base = _p(site2, "shards", dom_version)
    for y in (years or (sorted(os.listdir(base)) if os.path.isdir(base) else [])):
        mp = os.path.join(base, str(y), "manifest.json")
        if os.path.exists(mp):
            man = product.read_json(mp)
            if config.sha256_json(shard_key_payload(man)) != man["package_key"]:
                raise PackageError(f"{mp}: its entries do not give its package_key")
            out[int(y)] = man["package_key"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  prepare (v2): steps, years, resume
# ─────────────────────────────────────────────────────────────────────────────

def _years_arg(rec, years):
    have = [int(y) for y in rec["years"]]
    if years is None:
        return have
    ys = [int(y) for y in years]
    bad = [y for y in ys if y not in have]
    if bad:
        raise PackageError(f"years {bad} are not in the record ({have}): the stores were "
                           "incomplete when it was built; rebuild it with --force --step "
                           "record once they are downloaded")
    return ys


def _d_truth_id(site2):
    """sha256 of the D_truth the support step reads ("none" without one)."""
    p = os.path.join(site2.truth_dir, TRUTH_ELEV_V2)
    if not os.path.exists(p):
        return "none"
    with np.load(p, allow_pickle=False) as z:
        if "d_truth_union" not in z.files:
            return "none"
        d = np.unique(np.asarray(z["d_truth_union"], np.int64))
    return config.sha256_bytes(d.astype("<i8").tobytes())


def support_inputs(site2):
    """The support step's inputs outside the package: D_truth (the v2
    elevation truth) and the judges' stations file (NB)."""
    path = site2.stations_json or (site2.cfg.extra or {}).get("stations_json",
                                                              config.STATIONS_JSON)
    return {"d_truth": _d_truth_id(site2) if site2.cfg.elevation_truth else "no truth",
            "stations": _small_sha(_abs(path)) if site2.cfg.judges else None}


def _predictor_id(predictor):
    if predictor is not None:
        return getattr(predictor, "description", None) or type(predictor).__name__
    import importlib.metadata as md
    try:
        return f"omnicloudmask {md.version('omnicloudmask')}"
    except md.PackageNotFoundError:
        return "omnicloudmask (not installed)"


def prepare_v2(site2, steps=None, years=None, force=False, provider=None, predictor=None,
               allow_large_dom=False, level_chunk=None, sources_kw=None):
    """Build or extend a site's v2 package (spec §2.10, §13.3 ``prepare``).

    Parameters
    ----------
    site2 : V2Site
    steps : sequence of str, optional
        Steps of :data:`V2_STEPS` (default all, in order); a step whose
        outputs are intact, whose upstream steps are the same and unchanged
        and whose other inputs (:class:`_V2Book`) did not change is skipped.
    years : sequence of int, optional
        The shard years of the per-year steps (default the record's).
    force : bool
        Rerun the listed steps (a frozen dom still refuses to change).
    provider : object with ``levels``, optional
        The boundary provider (default :func:`boundary_provider`: the union
        provider, or the production one for ``boundary="production"``, when
        the record or levels step runs).
    predictor : callable, optional
        OmniCloudMask stand-in (tests); default the installed package.
    allow_large_dom : bool
        Run pass 2 even when |dom| > 1.5 |S0| (the user's decision).
    sources_kw : dict, optional
        Passed to :func:`v2_sources` (tests inject tide and CMEMS fields).

    Step inputs (a change makes the step stale)
    -------------------------------------------
    record: the instants, the stores' consolidated metadata (or the cube),
    the frozen offsets, the IOC chunk files (:func:`record_inputs`); a
    request for OTHER instants than the record's is refused
    (:func:`check_record_instants`). pass1:<y>: the plan. support: D_truth
    and the stations file. ocm:<y>: pass1:<y>, the region, the plan and the
    predictor (not the support: a new D_truth does not rerun OmniCloudMask
    when the region is unchanged). dom: whether C11 is used, the OCM years
    and C3's cut-off. pass2:<dv>:<y>: dom, pass1:<y> and ocm:<y> once it
    exists. sources: the CMEMS manifests, the judge table, the tide models.

    Returns
    -------
    dict
        ``{step key: info}`` of the steps that ran.
    """
    steps = list(V2_STEPS if steps is None else steps)
    bad = [s for s in steps if s not in V2_STEPS + V2_EXTRA_STEPS]
    if bad:
        raise ValueError(f"unknown step(s) {bad}; expected {V2_STEPS + V2_EXTRA_STEPS}")
    os.makedirs(site2.dir, exist_ok=True)
    keys = site2.observation_keys()
    ran = {}
    with _Lock(site2.dir):
        book = _V2Book(site2)

        def provider_():
            nonlocal provider
            if provider is None:
                log(f"[package] building the {site2.site} boundary provider "
                    f"({site2.boundary}, {' to '.join(boundary_period(site2))})")
                provider = boundary_provider(site2)
            return provider

        def run(key, fn, upstream, *a, extra=None):
            ex = {} if extra is None else extra
            why = "forced" if force else book.why_stale(key, upstream, ex)
            if why is None:
                log(f"[package] {site2.site} step {key}: done")
                return False
            miss = [u for u in upstream if book.rec(u) is None]
            if miss:
                raise PackageError(f"step {key} needs {miss} first")
            t0 = time.time()
            log(f"[package] {site2.site} step {key} ... ({why})")
            outs, info = fn(*a)
            book.record(key, [_rel2(site2, o) for o in outs], upstream, info,
                        round(time.time() - t0, 1), extra=ex)
            ran[key] = info
            log(f"[package] {site2.site} step {key} done in {time.time() - t0:.1f} s "
                f"(RSS {_rss_mb():.0f} MB)")
            return True

        rec_path = _p(site2, "record.json")
        if os.path.exists(rec_path) and not (force and "record" in steps):
            rec0 = _read_record(site2)
            check_record_instants(site2, rec0)
            have = [int(y) for y in rec0.get("years_requested", rec0["years"])]
            if [int(y) for y in site2.years] != have:
                # --years selects the per-year steps of an existing record; it
                # never rebuilds the record over other years (that is --force
                # --step record)
                log(f"[package] {site2.site}: the record covers {have}; the years "
                    f"{list(site2.years)} select the per-year steps only")
                if years is None:
                    years = [int(y) for y in site2.years]
                site2 = _dc.replace(site2, years=tuple(have))
        if "record" in steps:
            def _rec():
                r = v2_record(site2, provider_())
                write_json_lf(rec_path, r)
                log(f"[record] {site2.site}: T = {r['T']} over {r['years']} (missing "
                    f"{r['missing_years']}); instants {r['instants']}"
                    f"{' (' + r['instants_path'] + ')' if r['instants_path'] else ''}; "
                    f"offsets {r['offset_source']}; lattice {r['lattice20']['phase']}")
                return [rec_path], {"T": r["T"], "years": r["years"],
                                    "instants": r["instants"]}
            run("record", _rec, [], extra=record_inputs(site2))
        elif os.path.exists(rec_path):
            why = book.why_stale("record", [], record_inputs(site2))
            if why:
                log(f"[package] {site2.site}: WARNING the record's inputs changed since it was "
                    f"built ({why}): run prepare {site2.site} --step record")
        if not os.path.exists(rec_path):
            return ran
        rec = _read_record(site2)
        kind = (rec.get("boundary") or {}).get("kind", "union")
        if kind != site2.boundary:
            raise PackageError(f"{site2.site}: record.json was built with the {kind!r} boundary "
                               f"provider, not {site2.boundary!r} (pass --boundary {kind}, or "
                               "rebuild the record with --force)")
        src_kind = (rec.get("source") or {}).get("kind", "zarr")
        if src_kind != site2.source:
            raise PackageError(f"{site2.site}: record.json was built from the {src_kind!r} "
                               f"source, not {site2.source!r}")
        ys = _years_arg(rec, years)
        w3 = w3_years(rec)
        if "series" in steps:
            run("series", lambda: v2_series(site2, rec, provider_()), ["record"],
                extra={"series_key": series_key(site2, rec)})
        unav = site2.unavailable_keys()
        if unav and "pass1" in steps:
            log(f"[package] {site2.site}: {len(unav)} planned observation key(s) need bands the "
                f"{site2.source} source does not deliver ({', '.join(site2.bands)}): "
                f"{sorted(unav)}")
        p1_plan = [n for n, _, _ in pass1_plan(keys)]
        if "pass1" in steps:
            for y in ys:
                run(f"pass1:{y}", lambda y=y: v2_pass1(site2, rec, y, keys), ["record"],
                    extra={"plan": p1_plan})
        if "geometry" in steps:
            run("geometry", lambda: v2_geometry(site2, rec), [f"pass1:{y}" for y in w3])
        if "support" in steps:
            run("support", lambda: v2_support(site2, rec),
                ["geometry"] + [f"pass1:{y}" for y in w3], extra=support_inputs(site2))
        ocm_ok = ocm_available(predictor)
        ocm_bands = set(settings.CLOUD["C11"].params["bands"]) <= set(site2.bands)
        if "ocm" in steps:
            if not ocm_bands:
                log(f"[ocm] {site2.site}: the {site2.source} source lacks the OmniCloudMask "
                    f"bands {list(settings.CLOUD['C11'].params['bands'])}: C11 is unavailable")
            elif not ocm_ok:
                log(f"[ocm] {site2.site}: OmniCloudMask is not installed: C11 is unavailable; "
                    "C3 replaces it in dom and G1' (declared)")
            elif book.rec("support") is None:
                raise PackageError("step ocm needs ['support'] first")
            else:
                region = [int(v) for v in _support(site2)["ocm_region"]]
                ocm_ex = {"region": region, "plan": [n for n, _ in ocm_plan(keys)],
                          "predictor": _predictor_id(predictor),
                          "card": config.sha256_json(dict(settings.CLOUD["C11"].params))}
                for y in ys:
                    run(f"ocm:{y}", lambda y=y: v2_ocm(site2, rec, y, keys, region, predictor),
                        [f"pass1:{y}"], extra=ocm_ex)
        ocm_years = [y for y in w3 if book.rec(f"ocm:{y}") is not None]
        use_c11 = bool(w3) and len(ocm_years) == len(w3)
        if "dom" in steps:
            ups = ["support"] + ([f"ocm:{y}" for y in w3] if use_c11 else [])
            run("dom", lambda: v2_dom(site2, rec, use_c11, force=False), ups,
                extra={"use_c11": use_c11, "ocm_years": ocm_years if use_c11 else [],
                       "crosstab_p": None if use_c11 else read_crosstab_p(site2.root)})
        if "levels" in steps and book.rec("geometry") is not None:
            # needs the record and K only (not dom): the lag truths read it
            def _levels():
                p0 = V2Package(site2)
                try:
                    return v2_levels(site2, p0, provider_(), level_chunk)
                finally:
                    p0.close()
            run("levels", _levels, ["record", "geometry"])
        cur = current_dom(site2)
        if cur is None:
            return ran
        dv = cur["dom_version"]
        if "pass2" in steps:
            _, dom, _ = load_dom(site2, dv)
            region = [int(v) for v in _support(site2)["ocm_region"]]
            for y in ys:
                has_ocm = book.rec(f"ocm:{y}") is not None
                ups = ["dom", f"pass1:{y}"] + ([f"ocm:{y}"] if has_ocm else [])
                run(f"pass2:{dv}:{y}", lambda y=y, o=has_ocm: v2_pass2(
                    site2, rec, y, dv, dom, region if o else None, allow_large_dom), ups,
                    extra={"ocm_region": region if has_ocm else None})
        pkg = None

        def pkg_():
            nonlocal pkg
            if pkg is None:
                pkg = V2Package(site2, dv)
            return pkg

        have2 = [y for y in rec["years"] if book.rec(f"pass2:{dv}:{y}") is not None]
        if "settings" in steps and have2:
            run("settings", lambda: v2_settings(site2, pkg_(), keys),
                ["geometry"] + [f"pass2:{dv}:{y}" for y in have2], extra={"keys": list(keys)})
        if "counters" in steps and book.rec("settings") is not None:
            for y in ys:
                if book.rec(f"pass2:{dv}:{y}") is not None:
                    run(f"counters:{dv}:{y}", lambda y=y: v2_counters(site2, pkg_(), y, keys),
                        ["settings", f"pass2:{dv}:{y}"],
                        extra={"keys": dom_counter_plan(keys)})
        if "mnar" in steps and have2:
            run("mnar", lambda: v2_mnar(site2, pkg_()),
                ["geometry"] + [f"pass2:{dv}:{y}" for y in have2])
        if "subsets" in steps and have2:
            run("subsets", lambda: v2_subsets(site2, pkg_()),
                ["geometry"] + [f"pass1:{y}" for y in rec["years"]]
                + [f"pass2:{dv}:{y}" for y in have2])
        if "sources" in steps:
            if book.rec("levels") is None:
                log(f"[sources] {site2.site}: waits for the levels step")
            else:
                skw = dict(sources_kw or {})
                run("sources", lambda: v2_sources(site2, pkg_(), **skw),
                    ["record", "geometry", "dom", "levels"],
                    extra=sources_inputs(site2, rec["years"], skw.get("cmems_root"),
                                         skw.get("lt_kw")))
        if "manifest" in steps:
            for y in ys:
                if book.rec(f"pass2:{dv}:{y}") is not None:
                    run(f"manifest:{dv}:{y}", lambda y=y: v2_manifest(site2, pkg_(), y),
                        ["record", "dom", f"pass2:{dv}:{y}"])
        if pkg is not None:
            pkg.close()
    return ran


def _dir_bytes(d):
    total = 0
    for dirpath, _, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def dom_versions(site2):
    """Every dom_version with stored files: ``{dv: {"current", "frozen",
    "dom_gb", "shards_gb"}}`` (``status`` prints it: superseded versions
    hold full copies of the shards)."""
    cur = current_dom(site2) or {}
    frozen = set(cur.get("frozen_history") or [])
    if cur.get("frozen"):
        frozen.add(cur.get("dom_version"))
    out = {}
    for sub in ("dom", "shards"):
        base = _p(site2, sub)
        if not os.path.isdir(base):
            continue
        for dv in sorted(os.listdir(base)):
            if not os.path.isdir(os.path.join(base, dv)):
                continue
            e = out.setdefault(dv, {"current": dv == cur.get("dom_version"),
                                    "frozen": dv in frozen, "dom_gb": 0.0, "shards_gb": 0.0})
            e[f"{sub}_gb"] = round(_dir_bytes(os.path.join(base, dv)) / 1e9, 3)
    return out


def prune_stale_shards(site2, dry_run=False):
    """Remove the dom and shard folders of every dom_version that is neither
    the current one nor a frozen one (``prepare --prune-stale-shards``), and
    their bookkeeping entries. Returns ``{dv: GB freed}``."""
    book = _V2Book(site2)
    out = {}
    for dv, e in dom_versions(site2).items():
        if e["current"] or e["frozen"]:
            continue
        out[dv] = round(e["dom_gb"] + e["shards_gb"], 3)
        if dry_run:
            continue
        for sub in ("dom", "shards"):
            d = _p(site2, sub, dv)
            if os.path.isdir(d):
                shutil.rmtree(d)
        for k in [k for k in book.doc["steps"] if f":{dv}:" in k]:
            del book.doc["steps"][k]
        product.write_json(book.path, book.doc)
        log(f"[package] {site2.site}: removed the superseded dom_version {dv} "
            f"({out[dv]:.2f} GB)")
    return out


def status_v2(site2):
    """Step states of a site's v2 package, its dom, its dom_versions (with
    their sizes) and shard keys (for the CLI)."""
    book = _V2Book(site2)
    out = {"dir": site2.dir, "steps": book.status(), "dom": current_dom(site2),
           "dom_versions": dom_versions(site2)}
    try:
        out["shard_keys"] = shard_keys(site2)
    except PackageError as e:
        out["shard_keys"] = f"invalid: {e}"
    rp = _p(site2, "record.json")
    if os.path.exists(rp):
        rec = product.read_json(rp)
        out["record"] = {"T": rec.get("T"), "years": rec.get("years"),
                         "instants": rec.get("instants"),
                         "instants_path": rec.get("instants_path"),
                         "offset_source": rec.get("offset_source"),
                         "boundary_coverage": (rec.get("boundary_coverage") or {}).get("windows")}
    return out
