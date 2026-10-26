# -*- coding: utf-8 -*-
"""v13 benchmark — units, products, statuses, atomic IO and keys (§1.8, §2.11).

Question
--------
How is one run of one method on one record subset described, stored and
recognised as done, so that a campaign of thousands of units can be killed
and resumed at any point and a later code or plan edit invalidates exactly
the units it changes?

Pre-registered rules
--------------------
* A :class:`Unit` is (site, res_m, group, variant, family, mode, N, rep,
  sens) plus its record subset ``sel`` (sorted record positions) and
  ``subset_key = sha256(sel.astype("<i8").tobytes())``.
* A :class:`Product` is one method's output on the site support S, with a
  status in :data:`STATUSES`. ``ok``/``empty``/``not_computable`` are final
  (:data:`DONE_STATUSES`); ``r_failed``, ``budget`` and ``error`` are rerun.
* Keys (§2.11):
  - ``package_key`` = sha256 of exactly: the sha256 of every keyed package
    file (all but ``manifest.json`` and ``judges.json``, which depends only
    on the score-side judges section), of every input (cube, extraction,
    overpass file, IOC chunks), ``obs_sha`` and the FROZEN shas of
    ``record``, ``package``, ``obs``, ``domain``, ``levels``, ``subsets``.
    Never the mask rule, the index mode, a method, a scoring parameter or
    git;
  - ``config_key`` = sha256(adapter CONFIG, FROZEN[method section], mode,
    sens, variant, the group's BLAS threads);
  - ``code_key`` = sha256 of the adapter module, ``obs.py``, ``domain.py``,
    ``levels.py`` and the adapter's declared dependencies (pyintertidal
    modules, the R script), each with CRLF normalised to LF;
  - ``inputs_key`` = sha256(package_key, subset_key, method, variant,
    config_key, code_key). A unit is skipped when its run.json has a final
    status and the same inputs_key.
* A unit is written to ``<dir>.tmp-<host>-<pid>/``; an existing ``<dir>``
  (a failed or stale attempt) is renamed to ``<dir>.old-<utc>`` and the
  temporary directory is published with ``os.replace``. Nothing is deleted.
* File hashes of ``.npz`` files cover the member names and the
  decompressed ``.npy`` bytes, not the zip container, whose entries carry
  the write time; every ``.npz`` this package writes is also byte-stable
  (:func:`save_npz` fixes the entry times).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import shutil
import socket
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from experiments.benchmark import config

STATUSES = ("ok", "empty", "not_computable", "r_failed", "budget", "error")
#: final statuses: the unit is not rerun while its inputs_key is unchanged
DONE_STATUSES = ("ok", "empty", "not_computable")
FAMILIES = ("rand", "win", "rule-tz", "rule-g21", "rule-ard")
MODES = ("A", "F")
LAG_KINDS = ("linear", "phase")
PRODUCT_FORMAT = "v13-product-1"
#: the per-pixel arrays of a product on the support S, with their dtypes
ARRAY_DTYPES = {"z": np.float32, "z_raw": np.float32, "conf": np.float32,
                "clear_obs": np.int16, "lag": np.float32}
#: package files left out of the package key (manifest.json is the key's
#: container; judges.json follows the score-side judges section)
PACKAGE_UNKEYED_FILES = ("manifest.json", "judges.json")
#: the FROZEN sections behind the package key
PACKAGE_KEY_SECTIONS = config.PACKAGE_SECTIONS
#: the harness modules every adapter's code key includes
CORE_CODE_FILES = ("experiments/benchmark/obs.py", "experiments/benchmark/domain.py",
                   "experiments/benchmark/levels.py")
#: the Granadeiro R side, for its code key
R_SCRIPT = "pyintertidal/r/granadeiro_fit.R"


def log(msg):
    print(msg, flush=True)


class NotComputable(Exception):
    """The method's own rule stops it (status ``not_computable``).

    Parameters
    ----------
    stage : str
        Where it stopped (``record_floor``, ``no_intertidal``,
        ``empty_interval``, ``preflight``, ``no_valid_pixel``,
        ``scenes_lt_8``, ``harmonic_all_bands``, ...).
    reason : str or dict, optional
        Human-readable reason; a mapping is taken as ``diag``.
    diag : dict, optional
        Numbers behind the stop (saved in run.json).
    """

    def __init__(self, stage, reason=None, diag=None):
        if isinstance(reason, Mapping) and diag is None:
            reason, diag = None, dict(reason)
        self.stage = str(stage)
        self.reason = None if reason is None else str(reason)
        self.diag = dict(diag or {})
        super().__init__(f"{self.stage}: {self.reason}" if self.reason else self.stage)


# ─────────────────────────────────────────────────────────────────────────────
#  hashing and atomic IO
# ─────────────────────────────────────────────────────────────────────────────

def utc_stamp():
    """UTC time stamp for file names, e.g. 20261008T101500123456Z."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _tmp_name(path):
    return f"{path}.tmp-{socket.gethostname()}-{os.getpid()}"


def sha256_file(path, block=1 << 20):
    """Content sha256 of a file.

    For ``.npz`` files: the sorted member names and each member's
    decompressed bytes (zip entry times and compression do not enter, so
    re-writing identical arrays gives the same hash). For anything else: the
    raw bytes.
    """
    h = hashlib.sha256()
    if str(path).endswith(".npz"):
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                info = zf.getinfo(name)
                h.update(name.encode("utf-8") + b"\0"
                         + int(info.file_size).to_bytes(8, "little"))
                with zf.open(name) as f:
                    for chunk in iter(lambda: f.read(block), b""):
                        h.update(chunk)
        return h.hexdigest()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entry(path, sha=True):
    """Manifest entry of a file: sha256 (content, see :func:`sha256_file`),
    size and mtime_ns (the cheap check of ``verify_manifest(full=False)``)."""
    st = os.stat(path)
    out = {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    if sha:
        out["sha256"] = sha256_file(path)
    return out


_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def save_npz(path, arrays, compress=True):
    """Write ``arrays`` ({name: array}) as an ``.npz``, atomically and
    byte-stably (fixed entry times, insertion order), readable by np.load.

    Object arrays are refused (np.load(allow_pickle=False) everywhere).
    """
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = _tmp_name(path)
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    try:
        with zipfile.ZipFile(tmp, mode="w", compression=method, allowZip64=True) as zf:
            for name, val in arrays.items():
                a = np.asanyarray(val)
                if a.dtype.hasobject:
                    raise TypeError(f"{name}: object arrays are not stored")
                info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
                info.compress_type = method
                info.external_attr = 0o644 << 16
                with zf.open(info, "w", force_zip64=True) as f:
                    np.lib.format.write_array(f, np.asarray(a), allow_pickle=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_npz(path):
    """All members of an ``.npz`` as ``{name: ndarray}`` (no pickles)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Mapping):
        return dict(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, (_dt.datetime, _dt.date)):
        return o.isoformat()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def write_json(path, obj, indent=1):
    """Write JSON atomically (temporary file + os.replace). NaN is written
    as ``NaN`` (Python's json reads it back), numpy values as Python ones."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = _tmp_name(path)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, default=_json_default, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_text(path, text):
    """Write a text file atomically."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = _tmp_name(path)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
#  keys
# ─────────────────────────────────────────────────────────────────────────────

def _positions(sel):
    a = np.asarray(sel)
    if a.dtype == bool:
        raise TypeError("pass record positions (sorted integers), not a mask")
    if a.ndim != 1:
        raise ValueError("sel must be 1-D record positions")
    if a.size and a.dtype.kind not in "iu":
        if a.dtype.kind == "f" and np.all(a == np.floor(a)):
            a = a.astype(np.int64)
        else:
            raise TypeError(f"sel must hold integer positions, not {a.dtype}")
    a = a.astype("<i8")
    if a.size and (a[0] < 0 or np.any(np.diff(a) <= 0)):
        raise ValueError("sel must be sorted, unique, non-negative record positions")
    return a


def subset_key(sel):
    """``sha256(sel.astype("<i8").tobytes())`` of sorted record positions."""
    return config.sha256_bytes(_positions(sel).tobytes())


def manifest_payload(manifest):
    """The exact entries the package key hashes (see :func:`package_key`)."""
    try:
        files = {name: e["sha256"] for name, e in manifest["files"].items()
                 if name not in PACKAGE_UNKEYED_FILES}
        inputs = {name: e["sha256"] for name, e in manifest["inputs"].items()}
        frozen = {s: manifest["frozen"][s] for s in PACKAGE_KEY_SECTIONS}
        obs_sha = manifest["obs_sha"]
    except KeyError as e:
        raise KeyError(f"manifest lacks {e} (needs files, inputs, obs_sha and the "
                       f"FROZEN shas of {PACKAGE_KEY_SECTIONS})") from None
    return {"files": files, "inputs": inputs, "obs_sha": obs_sha, "frozen": frozen}


def package_key(manifest):
    """The package key of a manifest (§2.11).

    Parameters
    ----------
    manifest : dict
        ``{"files": {name: {"sha256", ...}}, "inputs": {name: {"sha256",
        ...}}, "obs_sha": str, "frozen": {section: sha, ...}, ...}``; any
        other entry (git, sizes, times, extra FROZEN shas) is provenance and
        does not enter the key.
    """
    return config.sha256_json(manifest_payload(manifest))


def config_key(cfg, section, mode, sens, variant, group=None):
    """sha256 of the adapter CONFIG, FROZEN[section], the unit's mode, sens
    and variant, and (with ``group``) the group's BLAS thread count.

    ``section`` is the method's FROZEN section (``config.METHOD_SECTION``),
    or None for a group without one (the as-published loaders).
    """
    if section is not None and section not in config.FROZEN:
        raise KeyError(f"unknown FROZEN section {section!r}")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    frozen = {} if section is None else {section: config.FROZEN[section]}
    payload = {"config": cfg, "frozen": frozen,
               "mode": mode, "sens": dict(sens or {}), "variant": variant or ""}
    if group is not None:
        payload["blas_threads"] = config.FROZEN["threads"]["blas"][group]
    return config.sha256_json(payload)


def pyintertidal_files(*modules):
    """Repository paths of pyintertidal modules, for :func:`code_key`."""
    return tuple(f"pyintertidal/{m}.py" for m in modules)


def _rel(path):
    p = os.path.abspath(path if os.path.isabs(path) else os.path.join(config.ROOT, path))
    try:
        return os.path.relpath(p, config.ROOT).replace("\\", "/"), p
    except ValueError:                      # another drive on Windows
        return p.replace("\\", "/"), p


def code_key(adapter_file, deps=()):
    """sha256 over (path, LF-normalised source sha) of the adapter module,
    :data:`CORE_CODE_FILES` and ``deps`` (repository-relative or absolute)."""
    entries = {}
    for f in (adapter_file, *CORE_CODE_FILES, *deps):
        rel, absolute = _rel(f)
        if not os.path.exists(absolute):
            raise FileNotFoundError(f"code_key: {rel} does not exist")
        entries[rel] = config.sha256_source(absolute)
    return config.sha256_json(entries)


def _package_key_of(pkg):
    if isinstance(pkg, str):
        return pkg
    keys = getattr(pkg, "keys", None)
    if isinstance(keys, Mapping) and "package_key" in keys:
        return keys["package_key"]
    if isinstance(pkg, Mapping) and "package_key" in pkg:
        return pkg["package_key"]
    raise TypeError("pkg must be a package_key string or carry keys['package_key']")


def inputs_key(pkg, unit, method, variant, config_key, code_key):
    """sha256(package_key, subset_key, method, variant, config_key, code_key).

    ``pkg`` is a CommonPackage (its ``keys["package_key"]``) or the key;
    ``unit`` a :class:`Unit` or its subset_key.
    """
    sk = unit if isinstance(unit, str) else unit.subset_key
    return config.sha256_json({"package_key": _package_key_of(pkg), "subset_key": sk,
                               "method": str(method), "variant": variant or "",
                               "config_key": config_key, "code_key": code_key})


# ─────────────────────────────────────────────────────────────────────────────
#  units and products
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Unit:
    """One run of one adapter group on one record subset."""

    site: str
    res_m: int
    group: str
    variant: str
    family: str          # "rand" | "win" | "rule-tz" | "rule-g21" | "rule-ard"
    mode: str            # "A" | "F"
    N: Union[int, str]   # int or "all"
    rep: int
    sens: dict           # {"mask": ..., "index": ..., "floor": ...}; {} = controlled
    sel: np.ndarray      # sorted record positions
    subset_key: str = ""

    def __post_init__(self):
        config.check_res(self.site, self.res_m)
        self.res_m = int(self.res_m)
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}; expected one of {FAMILIES}")
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}; expected one of {MODES}")
        if self.N != config.N_ALL:
            if isinstance(self.N, bool) or int(self.N) != self.N or int(self.N) < 1:
                raise ValueError(f"N must be a positive integer or 'all', got {self.N!r}")
            self.N = int(self.N)
        if int(self.rep) != self.rep or int(self.rep) < 0:
            raise ValueError(f"rep must be a non-negative integer, got {self.rep!r}")
        self.rep = int(self.rep)
        self.variant = self.variant or ""
        self.sens = {str(k): v for k, v in dict(self.sens or {}).items()}
        self.sel = _positions(self.sel).astype(np.int64)
        key = subset_key(self.sel)
        if self.subset_key and self.subset_key != key:
            raise ValueError("subset_key does not match sel")
        self.subset_key = key

    def __eq__(self, other):
        # the generated __eq__ would compare the sel arrays elementwise and
        # raise; the description and the subset key identify a unit
        if not isinstance(other, Unit):
            return NotImplemented
        return self.to_json() == other.to_json()

    __hash__ = None

    @property
    def n(self):
        return int(len(self.sel))

    @property
    def label(self):
        v = f"[{self.variant}]" if self.variant else ""
        s = f" {senstag(self.sens)}" if self.sens else ""
        return (f"{self.site} {self.res_m}m {self.group}{v} {self.family}-{self.mode} "
                f"N={self.N} r{self.rep}{s}")

    def to_json(self):
        return {"site": self.site, "res_m": self.res_m, "group": self.group,
                "variant": self.variant, "family": self.family, "mode": self.mode,
                "N": self.N, "rep": self.rep, "sens": self.sens,
                "subset_key": self.subset_key, "n_sel": self.n}

    @classmethod
    def from_json(cls, d, sel):
        return cls(site=d["site"], res_m=d["res_m"], group=d["group"],
                   variant=d.get("variant", ""), family=d["family"], mode=d["mode"],
                   N=d["N"], rep=d["rep"], sens=d.get("sens", {}), sel=sel,
                   subset_key=d.get("subset_key", ""))


@dataclass(eq=False)
class Product:
    """One method's output for one unit, on the site support S (§5.1).

    ``z`` carries the method's own quality rule, ``z_raw`` the values before
    it, ``conf`` the method's own confidence (higher = more confident),
    ``clear_obs`` the valid observations behind each pixel and ``lag`` the
    applied lag (min; NaN where the method defines none). The per-pixel
    arrays are cast to :data:`ARRAY_DTYPES` on construction, so a loaded
    product equals the saved one.
    """

    method: str
    variant: str
    unit: Unit
    status: str
    stage: Optional[str] = None
    reason: Optional[str] = None
    z: Optional[np.ndarray] = None
    z_raw: Optional[np.ndarray] = None
    conf: Optional[np.ndarray] = None
    clear_obs: Optional[np.ndarray] = None
    lag: Optional[np.ndarray] = None
    lag_kind: str = "linear"
    method_bands: Optional[dict] = None
    fallback: Optional[str] = None
    domain: dict = field(default_factory=dict)
    diag: dict = field(default_factory=dict)
    keys: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    peak_rss_mb: float = float("nan")
    smoke: bool = False
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}; expected one of {STATUSES}")
        if self.lag_kind not in LAG_KINDS:
            raise ValueError(f"unknown lag_kind {self.lag_kind!r}; expected one of {LAG_KINDS}")
        self.variant = self.variant or ""
        n = None
        for name, dtype in ARRAY_DTYPES.items():
            a = getattr(self, name)
            if a is None:
                continue
            a = np.asarray(a)
            if a.ndim != 1:
                raise ValueError(f"{name} must be 1-D on the support S, got {a.shape}")
            if n is not None and len(a) != n:
                raise ValueError(f"{name} has {len(a)} values, other arrays {n}")
            n = len(a)
            if np.issubdtype(np.dtype(dtype), np.integer) and a.size:
                if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
                    raise ValueError(f"{name} must be finite integers")
                info = np.iinfo(dtype)
                if a.min() < info.min or a.max() > info.max:
                    raise ValueError(f"{name} does not fit {np.dtype(dtype)}")
            setattr(self, name, a.astype(dtype, copy=False))

    @property
    def n_support(self):
        for name in ARRAY_DTYPES:
            a = getattr(self, name)
            if a is not None:
                return int(len(a))
        return 0


def senstag(sens):
    """Directory tag of a sensitivity setting: sorted ``key-value`` pairs
    joined by '+', e.g. {"mask": "blocklist", "floor": 80} ->
    "floor-80+mask-blocklist"; "" for the controlled setting."""
    def clean(x):
        return "".join(c if (c.isalnum() or c in "._-") else "-" for c in str(x))
    return "+".join(f"{clean(k)}-{clean(v)}" for k, v in sorted(dict(sens or {}).items()))


def unit_dir(root, unit, method, variant):
    """``<root>/runs/<site>/<method>[__<variant>]/<family>-<mode>/N<NNN>_r<RR>[__<senstag>]``."""
    m = f"{method}__{variant}" if variant else str(method)
    n = "Nall" if unit.N == config.N_ALL else f"N{int(unit.N):03d}"
    leaf = f"{n}_r{int(unit.rep):02d}"
    if unit.sens:
        leaf += f"__{senstag(unit.sens)}"
    return "/".join([str(root).rstrip("/\\").replace("\\", "/"), "runs", unit.site, m,
                     f"{unit.family}-{unit.mode}", leaf])


def _split_arrays(obj, prefix, arrays):
    """JSON-ready copy of ``obj`` with every ndarray moved to ``arrays``
    (member ``<prefix>.<i>``) and replaced by {"__npz__": member}."""
    if isinstance(obj, np.ndarray):
        name = f"{prefix}.{len([k for k in arrays if k.startswith(prefix + '.')])}"
        arrays[name] = obj
        return {"__npz__": name}
    if isinstance(obj, Mapping):
        return {str(k): _split_arrays(v, prefix, arrays) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_split_arrays(v, prefix, arrays) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _join_arrays(obj, arrays):
    if isinstance(obj, dict):
        if set(obj) == {"__npz__"}:
            return arrays[obj["__npz__"]]
        return {k: _join_arrays(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_join_arrays(v, arrays) for v in obj]
    return obj


def save(product, out_dir, log_text=None, files=None):
    """Publish a product directory atomically (§7.4).

    Writes ``product.npz`` (the per-pixel arrays, ``unit_sel`` and every
    array found in ``method_bands`` or ``diag``), ``run.json`` and, when
    given, ``log.txt`` into ``<out_dir>.tmp-<host>-<pid>/``; moves ``files``
    ({name: existing path}, e.g. full-frame rasters) into it; renames an
    existing ``out_dir`` to ``<out_dir>.old-<utc>``; then publishes the
    temporary directory with ``os.replace``. Returns ``out_dir``.
    """
    out_dir = os.path.normpath(os.fspath(out_dir))
    os.makedirs(os.path.dirname(os.path.abspath(out_dir)), exist_ok=True)
    tmp = _tmp_name(out_dir)
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    arrays = {name: getattr(product, name) for name in ARRAY_DTYPES
              if getattr(product, name) is not None}
    arrays["unit_sel"] = product.unit.sel
    side = {}
    run = {
        "format": PRODUCT_FORMAT,
        "method": product.method, "variant": product.variant,
        "status": product.status, "stage": product.stage, "reason": product.reason,
        "lag_kind": product.lag_kind, "fallback": product.fallback,
        "n_support": product.n_support,
        "unit": product.unit.to_json(),
        "method_bands": (None if product.method_bands is None
                         else _split_arrays(product.method_bands, "method_bands", side)),
        "domain": _split_arrays(product.domain, "domain", side),
        "diag": _split_arrays(product.diag, "diag", side),
        "keys": dict(product.keys), "timings": dict(product.timings),
        "peak_rss_mb": product.peak_rss_mb, "smoke": bool(product.smoke),
        "provenance": _split_arrays(product.provenance, "provenance", side),
        "arrays": sorted(arrays),
        "saved_utc": utc_stamp(),
    }
    arrays.update(side)
    save_npz(os.path.join(tmp, "product.npz"), arrays, compress=True)
    write_json(os.path.join(tmp, "run.json"), run)
    if log_text is not None:
        write_text(os.path.join(tmp, "log.txt"), log_text)
    for name, src in (files or {}).items():
        if os.path.isabs(name) or ".." in name.replace("\\", "/").split("/"):
            raise ValueError(f"extra file name {name!r} must be relative to the unit")
        dst = os.path.join(tmp, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    if os.path.exists(out_dir):
        old = f"{out_dir}.old-{utc_stamp()}"
        os.replace(out_dir, old)
        log(f"[product] previous attempt kept as {old}")
    os.replace(tmp, out_dir)
    return out_dir


def read_run(out_dir):
    """The unit's run.json, or None when absent or unreadable."""
    p = os.path.join(out_dir, "run.json")
    if not os.path.exists(p):
        return None
    try:
        return read_json(p)
    except (OSError, ValueError):
        return None


def is_done(out_dir, inputs_key_):
    """True when ``out_dir`` holds a final product with this inputs_key."""
    run = read_run(out_dir)
    return (run is not None and run.get("status") in DONE_STATUSES
            and run.get("keys", {}).get("inputs_key") == inputs_key_)


def iter_runs(root, site=None):
    """Yield ``(unit directory, run.json dict)`` of every published unit under
    ``<root>/runs[/<site>]``; temporary (``.tmp-``) and superseded
    (``.old-``) attempts are skipped."""
    import glob

    base = os.path.join(root, "runs", site) if site else os.path.join(root, "runs")
    for p in sorted(glob.glob(os.path.join(base, "**", "run.json"), recursive=True)):
        d = os.path.dirname(p)
        parts = d.replace("\\", "/").split("/")
        if any(".tmp-" in s or ".old-" in s for s in parts):
            continue
        run = read_run(d)
        if run is not None:
            yield d.replace("\\", "/"), run


def load(out_dir):
    """The :class:`Product` saved in ``out_dir``."""
    run = read_json(os.path.join(out_dir, "run.json"))
    if run.get("format") != PRODUCT_FORMAT:
        raise ValueError(f"{out_dir}: unknown product format {run.get('format')!r}")
    arrays = load_npz(os.path.join(out_dir, "product.npz"))
    unit = Unit.from_json(run["unit"], arrays["unit_sel"])
    mb = run.get("method_bands")
    return Product(
        method=run["method"], variant=run.get("variant", ""), unit=unit,
        status=run["status"], stage=run.get("stage"), reason=run.get("reason"),
        **{name: arrays.get(name) for name in ARRAY_DTYPES},
        lag_kind=run.get("lag_kind", "linear"),
        method_bands=None if mb is None else _join_arrays(mb, arrays),
        fallback=run.get("fallback"),
        domain=_join_arrays(run.get("domain", {}), arrays),
        diag=_join_arrays(run.get("diag", {}), arrays),
        keys=run.get("keys", {}), timings=run.get("timings", {}),
        peak_rss_mb=run.get("peak_rss_mb", float("nan")),
        smoke=bool(run.get("smoke", False)),
        provenance=_join_arrays(run.get("provenance", {}), arrays))


# ─────────────────────────────────────────────────────────────────────────────
#  provenance (recorded, never part of a key)
# ─────────────────────────────────────────────────────────────────────────────

#: the repository paths whose state a run records (git_state)
CODE_PATHS = ("experiments", "pyintertidal", "intertidal", "tests",
              "requirements-cluster.txt")


def git_state(root=None):
    """``{"head": sha or None, "dirty": bool or None}`` of the repository:
    ``dirty`` means a file of the code paths (:data:`CODE_PATHS`) is
    modified or untracked (the output and data trees are not listed: they
    are huge and the listing could time out); a failed or slow status gives
    ``dirty: None`` but keeps HEAD."""
    import subprocess

    root = root or config.ROOT
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "dirty": None}
    try:                                      # a slow status keeps HEAD
        st = subprocess.run(["git", "status", "--porcelain", "--", *CODE_PATHS], cwd=root,
                            capture_output=True, text=True, timeout=60)
        dirty = bool(st.stdout.strip()) if st.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return {"head": head, "dirty": dirty}


def runtime_provenance():
    """Interpreter, library versions, thread settings and host of this run."""
    import importlib
    import platform
    import sys

    out = {"python": sys.version.split()[0], "platform": platform.platform(),
           "host": socket.gethostname()}
    for mod in ("numpy", "scipy", "pandas", "xarray", "pyTMD", "eo_tides"):
        try:
            out[mod] = getattr(importlib.import_module(mod), "__version__", "?")
        except Exception:                   # absent or broken: recorded as such
            out[mod] = None
    out["threads"] = {k: os.environ.get(k) for k in
                      ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    return out
