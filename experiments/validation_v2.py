# -*- coding: utf-8 -*-
"""Validation v2 — factorial validation of MAREA: the CLI (docs/validation_v2_spec.md §13.3).

Question
--------
Which command does the user run on the server for each stage of
validation v2, and what does each one read, write and refuse? This file is
the only entry point of validation v2 and the only module that changes
directory (to the repository root).

usage (from the repository root; on the server after ``source env.sh``):

  python -X utf8 -m experiments.validation_v2 fetch ioc|judges|stac-meta|sensing|cmems|tides|emodnet|pnoa-meta|vak-meta|models [--sites ...] [--dry-run]
  python -X utf8 -m experiments.validation_v2 inputs [--sites ...] [--cubes-root DIR] [--res 10|20] [--sensing-times FILE]
  python -X utf8 -m experiments.validation_v2 check-env [--sites ...] [--res 10|20] [--network]
  python -X utf8 -m experiments.validation_v2 prepare escalda [--years 2023-2025] [--step pass1,ocm,...] [--res 10|20] [--sensing-times FILE]
  python -X utf8 -m experiments.validation_v2 truth escalda [--part elev,lag] [--res 10|20] [--sensing-times FILE]
  python -X utf8 -m experiments.validation_v2 pilot crosstab|dom [--sites ...] [--freeze] [--res 10|20]
  python -X utf8 -m experiments.validation_v2 plan --design <name>|tier1a|tier1b|all [--sites ...]
  python -X utf8 -m experiments.validation_v2 status [--sites ...] [--res 10|20]
  python -X utf8 -m experiments.validation_v2 diag time [--sites ...]          (OPTIONAL)
  python -X utf8 -m experiments.validation_v2 extract ferrol --out SCRATCH     (local checks)

Scene instants (user decision of 2026-10-14)
--------------------------------------------
Every command uses the catalogue / product-name time of each scene (the
datatake start of ``overpass_times.json``), exactly as every production
result, without warning or flag. The solar-angle sensing-time correction is
an OPTIONAL DIAGNOSTIC: ``diag time`` (``gate time`` remains only as an
alias that says so; ``fetch sensing`` fetches its STAC items). It is never
a gate, no other command requires it, and it is no factor, setting or row
of any design. A run reads a ``sensing_times.json`` only when given
``--sensing-times <file>`` (a file, or the diagnostic's output folder
holding ``<site>/sensing_times.json``).

Resolutions
-----------
``--res 10`` (default): the per-year Zarr stores and the output root
``results/validation_v2`` (the reported numbers). ``--res 20``: the legacy
20 m netCDF cubes (B03, B08, SCL) of escalda, wadden and ems with the
production boundary provider over 2023-2025, under ``results/
validation_v2_20m``: local software checks only, never reported numbers.
``VALIDATION_V2_ROOT`` overrides the output root (tests, scratch runs);
``VALIDATION_V2_SPACE_MARGIN_GB`` lowers the 20 GB free-space margin of the
passes for local scratch checks on a small disk (never on the server).

Milestone V1 delivers fetch, inputs, check-env, prepare, truth, pilot
crosstab and pilot dom, plan and status (and the optional ``diag time``).
The other commands of §13.3 (the gates, control, timing, lock, run, pilot
reps/units/attain, score, report) are in the parser with their arguments and
exit with "available from milestone V2/V3/..." (exit 2).
``experiments.v13_benchmark`` stays the v13 entry point; this CLI shares its
thread and check helpers.

Pre-registered rules this file enforces (§13.3, §13.7, §13.8)
-------------------------------------------------------------
* OMP/OPENBLAS/MKL threads are set before numpy is imported (4 unless
  ``--blas-threads``); every command runs from the repository root.
* Every command line is appended to ``<root>/logs/commands.jsonl`` with git
  HEAD, host, exit code and duration.
* ``prepare`` builds the v2 year shards (``package.prepare_v2``); pass 2 is
  refused while |dom| > 1.5 |S0| unless the user passes
  ``--allow-large-dom`` after ``pilot dom``. Order on the server: ``fetch
  ioc`` (exit 0 only when the union cache is ready), ``inputs --sites all``
  (one process), ``prepare <site> --step record,pass1,geometry,levels``,
  ``truth <site>`` (D_truth enters S0; the judge table; the lag truths),
  ``prepare <site> --step support,ocm,dom``, ``pilot dom`` (the size check,
  before any tile is written), then ``prepare <site>`` (pass 2, settings,
  counters, MNAR, subsets, the level sources LB/LT/LM/LG, manifests).
* A step is redone when its outputs changed, when it would read other
  upstream steps than it read, when one of them was rebuilt, or when its
  other inputs changed (instants, store metadata, offsets, IOC chunks,
  D_truth, the OCM region, CMEMS manifests, the judge table, ...); a run
  asking for OTHER instants than the record's is refused. Every pass checks
  the free disk space before writing (20 GB margin), and SIGTERM/SIGHUP
  clean up like Ctrl-C; leftovers of killed runs are removed at the next
  run of the step.
* ``pilot crosstab`` applies the frozen C3 rule (p = 35 if the median CLD
  over any anchor class exceeds 20 % in more than 10 % of the scenes at any
  site, else 20) and writes ``common/crosstab.json``; ``pilot dom`` prints
  |S0|, |dom|, |U_rec| and |keep| per setting and window, the W9 projection
  and the refusal verdict; ``--freeze`` freezes dom (the user's approval).
* ``plan`` writes ``plans/<design>.csv`` and prints the Tier 1A' projection
  against the 10-day cap with the pre-ordered cuts.
* No command reads a truth before its lock (the scoring commands are V2+).
"""
import argparse
import importlib
import json
import os
import socket
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: the output root per resolution (spec §13.1); ``VALIDATION_V2_ROOT``
#: overrides it (tests, scratch runs)
V2_ROOT = "results/validation_v2"
V2_ROOTS = {10: V2_ROOT, 20: "results/validation_v2_20m"}
RES_CHOICES = (10, 20)
DEFAULT_THREADS = 4
FETCH_WHATS = ("ioc", "judges", "stac-meta", "sensing", "cmems", "tides", "emodnet",
               "pnoa-meta", "vak-meta", "models")
PILOT_WHATS = ("crosstab", "dom", "reps", "units", "attain")
DIAG_WHATS = ("time",)
#: the milestone of every command (or sub-command) not in V1 (spec §15)
LATER = {
    "gate": "V2 (G-zarr, G-offset, G-record, G-keep, G-marea, G-lagtruth, G-level, G-ocm); "
            "V3 (G-nidem, G-nidem-code, G-dea, G-flood2topo, G-costa, G-frameview, G-score); "
            "V4 (G-granadeiro, G-bijagos)",
    "control": "V2 (lag worlds); V3 (elevation methods)",
    "timing": "V2",
    "lock": "V2",
    "run": "V2 (lag plans); V3 (elevation plans); V4 (Granadeiro)",
    "score": "V2 (--part lag); V3 (elev, curves, timing)",
    "report": "V6",
    "pilot reps": "V2", "pilot attain": "V2", "pilot units": "V3",
}
#: the package shards of Campaign 1 (W3) at the five sites (spec §14.4 and the
#: server review of 2026-10-14: 17 B per stored observation, 110-160 GB), plus
#: the pass-1 bits and the OCM bits; check-env warns below it
DISK_W3_GB = 180.0
#: the user decision of 2026-10-14, printed by 'diag time' and its alias
DIAG_NOTE = ("OPTIONAL diagnostic (user decision 2026-10-14): every run uses the catalogue / "
             "product-name times (datatake start); this measures how far the site sensing "
             "time is from them, gates nothing, and its sensing_times.json is read only when "
             "a run is given --sensing-times")
GATE_TIME_NOTE = "'gate time' is now the optional 'diag time': " + DIAG_NOTE
#: C3's mechanical cut-off rule (spec §2.2): written before the cross-tab runs
CROSSTAB_RULE = {"p_default": 20, "p_raised": 35, "median_cld_pct": 20.0,
                 "share_of_scenes": 0.10, "min_anchor_px": 100,
                 "anchor_classes": ["permanent water (wf >= 0.98)",
                                    "dry flats at low tide (keep_full, wf <= 0.3, h_B < q25)",
                                    "turbid channel water (EMODnet depth >= 2 m below LAT)"],
                 "control_class": "stable land (wf <= 0.02)",
                 "pixels": "anchor pixels inside Omega that SCL calls clear {4, 5, 6, 7}, CLD "
                           "not fill (declared: real clouds would trigger the rule anywhere)"}


def log(msg):
    print(msg, flush=True)


def _res(args):
    """The grid resolution of a command (10 unless ``--res``)."""
    return int(getattr(args, "res", None) or 10)


def _root(res=10):
    """The output root: ``VALIDATION_V2_ROOT``, else the resolution's
    (``results/validation_v2`` at 10 m, ``results/validation_v2_20m`` at 20 m)."""
    env = os.environ.get("VALIDATION_V2_ROOT")
    if env:
        return env
    return V2_ROOTS.get(int(res), f"results/validation_v2_{int(res)}m")


def _v13():
    from experiments import v13_benchmark as v13
    return v13


def _split(text):
    return _v13()._split(text)


def _years(text):
    """'2023-2025' or '2023,2025' -> [2023, 2024, 2025]."""
    if not text:
        return None
    out = []
    for part in _split(text):
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            out += list(range(a, b + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _sites(text, res=10):
    """Site names of a ``--sites`` value at a resolution ("all": every site
    with a cube at that resolution)."""
    from experiments.benchmark import config
    have = [s for s in config.SITES if s in config.RES_SITES[int(res)]]
    names = _split(text) or ["all"]
    out = []
    for s in names:
        if s == "all":
            out += have
        elif s in have:
            out.append(s)
        elif s in config.SITES:
            raise ValueError(f"{s} has no {res} m input (sites at {res} m: {have})")
        else:
            raise ValueError(f"unknown site {s!r}; expected one of {config.SITES} or all")
    return list(dict.fromkeys(out))


def _module(name):
    """``experiments.benchmark.<name>`` or None with a message when it is not
    in this build (parallel V1 work)."""
    try:
        return importlib.import_module(f"experiments.benchmark.{name}")
    except ModuleNotFoundError as e:
        if e.name and e.name.endswith(name.split(".")[-1]):
            log(f"[v2] experiments/benchmark/{name.replace('.', '/')}.py is not in this build")
            return None
        raise


def _sensing_file(site, args, n_sites=1):
    """The explicit ``--sensing-times`` of a site (a file, or a folder with
    ``<site>/sensing_times.json``), or None: the catalogue times. A FILE
    must be the site's (its ``site`` field); a command over several sites
    needs the folder form."""
    p = getattr(args, "sensing_times", None)
    if not p:
        return None
    if os.path.isdir(p):
        f = os.path.join(p, site, "sensing_times.json")
        if not os.path.isfile(f):
            raise ValueError(f"--sensing-times {p}: no {site}/sensing_times.json in it")
        return f
    if not os.path.isfile(p):
        raise ValueError(f"--sensing-times {p}: no such file")
    try:
        with open(p, encoding="utf-8") as fh:
            doc_site = (json.load(fh) or {}).get("site")
    except (OSError, ValueError, AttributeError):
        doc_site = None
    if doc_site is not None and doc_site != site:
        raise ValueError(f"--sensing-times {p} is the file of {doc_site!r}, not {site!r}")
    if doc_site is None and n_sites > 1:
        raise ValueError(f"--sensing-times {p} names no site: give the folder form "
                         "(<dir>/<site>/sensing_times.json) for a command over several sites")
    return p


def _site2(site, args, n_sites=1):
    from experiments.benchmark import package
    kw = {}
    st = _sensing_file(site, args, n_sites)
    if st:
        kw["instants"] = st
    if getattr(args, "block_rows", None):
        kw["block_rows"] = int(args.block_rows)
    if getattr(args, "cache_mb", None):
        kw["cache_mb"] = float(args.cache_mb)
    if getattr(args, "boundary", None):
        kw["boundary"] = args.boundary
    if getattr(args, "allow_provisional_offsets", False):
        kw["allow_provisional_offsets"] = True
    yrs = _years(getattr(args, "years", None))
    if yrs:
        kw["years"] = tuple(yrs)
    res = _res(args)
    return package.v2_site(site, root=_root(res), cubes_root=getattr(args, "cubes_root", None),
                           res=res, **kw)


def _even_rows(text):
    """argparse type of --block-rows: a positive even number (a 20 m block
    must not straddle two row blocks)."""
    v = int(text)
    if v <= 0 or v % 2:
        raise argparse.ArgumentTypeError(f"{v} is not a positive even number of rows")
    return v


def _instants_line(rec):
    """The record's instants, for the logs: catalogue (default) or the file."""
    src = rec.get("instants")
    path = rec.get("instants_path")
    return f"{src}" + (f" ({path})" if path else "")


def _log_command(args, rc, seconds):
    try:
        from experiments.benchmark import product
        logs = os.path.join(_root(_res(args)), "logs")
        os.makedirs(logs, exist_ok=True)
        rec = {"utc": product.utc_stamp(), "host": socket.gethostname(), "pid": os.getpid(),
               "command": args.command, "argv": sys.argv[1:], "exit": rc,
               "seconds": round(seconds, 1), "git": product.git_state()}
        with open(os.path.join(logs, "commands.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:                       # provenance must never fail a command
        log(f"[v2] could not log the command: {e}")


def _write_json(path, obj):
    from experiments.benchmark import product
    product.write_json(path, obj)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_ioc(args):
    """The boundary gauges 2017-2025 (§1.5). Exit 1 while the union cache is
    not ready (a year missing, or failing the coverage rule): h_B cannot be
    built and no prepare may run."""
    m = _module("fetch.ioc")
    if m is None:
        return 2
    if args.dry_run:
        for c in m.UNION_CODES:
            log(f"[fetch ioc] {c}: years to download {m.missing_years(c, m.UNION_CACHE)}")
        ready, problems = m.union_ready(m.UNION_CODES, m.UNION_CACHE)
        log(f"[fetch ioc] union cache {'READY' if ready else 'NOT READY'}: {problems}")
        return 0
    res = m.fetch_union(accept_gaps=bool(args.accept_gaps))
    for c, r in res["codes"].items():
        yrs = r.get("years", {})
        cnt = ", ".join(f"{y}: {e['samples']} samples, {len(e['days'])} months"
                        + ("" if e["ok"] else f" THIN {e['thin']}") for y, e in yrs.items())
        log(f"[fetch ioc] {c}: {cnt}")
        for key in ("incomplete", "accepted_gaps", "production_gaps", "missing"):
            if r.get(key):
                log(f"[fetch ioc] {c}: {key} {r[key]}")
    log(f"[fetch ioc] union cache {'READY' if res['ready'] else 'NOT READY'}"
        + ("" if res["ready"] else f": {res['problems']}") + f" -> "
        f"{os.path.join(m.UNION_CACHE, m.MANIFEST_NAME)}")
    return 0 if res["ready"] else 1


def _fetch_judges(args):
    m = _module("fetch.judges")
    if m is None:
        return 2
    res = m.run(_sites(args.sites), out_dir=os.path.join(_root(10), "common"),
                execute=bool(args.execute))
    log(f"[fetch judges] {res}")
    return 0


def _fetch_sensing(args):
    st = _module("sensing_time")
    if st is None:
        return 2
    log(f"[fetch sensing] the STAC items of the {DIAG_NOTE}")
    argv = ["fetch", "--sites", *_sites(args.sites)]
    if args.cubes_root:
        argv += ["--cubes-root", args.cubes_root]
    if args.refresh:
        argv.append("--refresh")
    return st.main(argv)


def _fetch_stac_meta(args):
    m = _module("fetch.stac_meta")
    if m is None:
        log("[fetch stac-meta] the scene_meta.json sidecar builder (fetch/stac_meta.py) is "
            "not in this build; inputs.scene_meta_from_items converts raw STAC features")
        return 2
    return m.run(_sites(args.sites), root=args.cubes_root, refresh=bool(args.refresh)) or 0


def _cmems_windows(m, args):
    """The Campaign-1 windows of ``fetch cmems`` (restricted by --years)."""
    from experiments.benchmark import settings
    yrs = _years(getattr(args, "years", None))
    wins = list(m.CAMPAIGN1_WINDOWS)
    if yrs:
        wins = [w for w in wins if set(settings.window(w).years) <= set(yrs)]
    return wins


def _fetch_cmems(args):
    """LM and L16: per site and Campaign-1 window the CMEMS dataset that
    covers it (none: NOT AVAILABLE there, declared), one folder and manifest
    per dataset under <root>/common/<site>/cmems/, and the stamp convention
    of the hourly-mean IBI reanalysis measured on the overlap month. Exit 0
    when every needed dataset is fetched, 3 without credentials (LM NOT
    COMPUTABLE), 1 when a fetch failed."""
    m = _module("fetch.cmems")
    if m is None:
        return 2
    cred = m.check_credentials()
    log(f"[fetch cmems] credentials: {cred}")
    wins = _cmems_windows(m, args)
    root = os.path.join(_root(10), "common")
    sites = _sites(args.sites)
    for site in sites:
        plan = m.plan_site(site, wins)
        for wid, e in plan["windows"].items():
            log(f"[fetch cmems] {site} {wid}: "
                + (e["dataset"] if e["dataset"] else f"NOT AVAILABLE ({e['reason']})"))
        log(f"[fetch cmems] {site}: would fetch {plan['fetch'] or 'nothing'}; stamp "
            f"conventions to measure {plan['stamp'] or 'none'}")
    if args.dry_run:
        log(f"[fetch cmems] dry run (public catalogue, no credentials): "
            f"{json.dumps(m.dry_run(), default=str)[:4000]}")
        return 0
    status = cred.get("status") if isinstance(cred, dict) else cred
    if status not in ("ok", True):
        log("[fetch cmems] credentials missing: every LM unit is NOT COMPUTABLE until the "
            "Copernicus Marine account works (.config/copernicusmarine/"
            ".copernicusmarine-credentials); nothing was requested")
        return 3
    from experiments import zarr_cubes as zc
    from experiments.benchmark import inputs
    bad, cred_bad = [], []
    for site in sites:
        try:
            ds = zc.open_cube(site, "2017-01-01", "2026-01-01", root=args.cubes_root,
                              bands=["SCL"], missing="skip")
            grid = inputs.grid_of(ds)
            ds.close()
            res = m.fetch_campaign(site, grid, wins, root=root)
            for dsid, e in res["datasets"].items():
                log(f"[fetch cmems] {site} {dsid} {e['years']}: {e['status']}"
                    + (f" ({e['reason']})" if e.get("reason") else "") + f" -> {e['dir']}")
            if res["status"] == m.STATUS_CREDENTIALS:
                cred_bad.append(site)
            elif res["status"] != m.STATUS_OK:
                bad.append(site)
        except Exception as e:
            bad.append(site)
            log(traceback.format_exc().rstrip())
            log(f"[fetch cmems] {site}: FAILED ({type(e).__name__}: {e})")
    if bad:
        log(f"[fetch cmems] FAILED at {bad}: rerun 'fetch cmems' (cached files are kept)")
        return 1
    if cred_bad:
        log(f"[fetch cmems] the credentials were rejected at {cred_bad}: LM NOT COMPUTABLE")
        return 3
    return 0


def _fetch_tides(args):
    m = _module("fetch.tides")
    if m is None:
        return 2
    inst = m.installed_models()
    lt = m.lt_models(available=inst)
    log(f"[fetch tides] installed tide models: {inst}")
    log(f"[fetch tides] LT models: {lt}; rule: {m.lt_rule(lt)}")
    return 0


def _fetch_pnoa(args):
    m = _module("fetch.pnoa_meta")
    if m is None:
        return 2
    log(f"[fetch pnoa-meta] -> "
        f"{m.build(out=os.path.join(_root(10), 'common', 'pnoa_meta.json'))}")
    return 0


def _fetch_vak(args):
    m = _module("fetch.vak_meta")
    if m is None:
        return 2
    from experiments.benchmark import config
    for site in _sites(args.sites):
        if "vaklodingen" in config.ELEVATION_TRUTH.get(site, ()):
            log(f"[fetch vak-meta] {site} -> "
                f"{m.build(site, out_dir=os.path.join(_root(10), 'common', 'vak_meta'))}")
    return 0


def _fetch_emodnet(args):
    m = _module("fetch.emodnet")
    if m is None:
        log("[fetch emodnet] the EMODnet bathymetry fetch (fetch/emodnet.py: channel anchors "
            "for the cross-tab, L18 depths) is not in this build")
        return 2
    return m.run(_sites(args.sites)) or 0


FETCHERS = {"ioc": _fetch_ioc, "judges": _fetch_judges, "stac-meta": _fetch_stac_meta,
            "sensing": _fetch_sensing, "cmems": _fetch_cmems, "tides": _fetch_tides,
            "models": _fetch_tides, "emodnet": _fetch_emodnet, "pnoa-meta": _fetch_pnoa,
            "vak-meta": _fetch_vak}


def cmd_fetch(args):
    log(f"[fetch {args.what}] network downloads into the gitignored data folders (§13.3)")
    try:
        return FETCHERS[args.what](args)
    except Exception as e:
        log(traceback.format_exc().rstrip())
        log(f"[fetch {args.what}] FAILED ({type(e).__name__}: {e})")
        return 1


# ─────────────────────────────────────────────────────────────────────────────
#  inputs and check-env
# ─────────────────────────────────────────────────────────────────────────────

def cmd_inputs(args):
    """The §1.11 checks: the Zarr stores at 10 m, the legacy netCDF cube at
    20 m (``--res 20``); catalogue instants unless ``--sensing-times``."""
    from experiments.benchmark import config, inputs
    res = _res(args)
    report = os.path.join(_root(res), "common", "inputs_report.json")
    bad = []
    sites = _sites(args.sites, res)
    for site in sites:
        t0 = time.time()
        try:
            inst = _sensing_file(site, args, len(sites)) or inputs.CATALOGUE
            if res != 10:
                cfg = config.site_config(site, res)
                rep = inputs.legacy_inputs_report(site, cfg.cube, config.resolve(cfg.overpass),
                                                  res, instants=inst, scan=not args.no_scan,
                                                  full_fingerprint=bool(args.full_fingerprint),
                                                  out=report, cache_mb=args.cache_mb)
            else:
                legacy = config.site_config(site, 10).cube
                legacy = legacy if os.path.exists(legacy) else None
                rtk = None
                if site == "villaviciosa":
                    from pyintertidal import rtk as _rtk
                    rtk = _rtk.DEFAULT_GRID if os.path.exists(_rtk.DEFAULT_GRID) else None
                rep = inputs.inputs_report(site, root=args.cubes_root, legacy=legacy,
                                           rtk_grid=rtk, instants=inst, scan=not args.no_scan,
                                           full_fingerprint=bool(args.full_fingerprint),
                                           out=report, cache_mb=args.cache_mb)
        except Exception as e:
            bad.append(site)
            log(traceback.format_exc().rstrip())
            log(f"[inputs] {site}: FAILED ({type(e).__name__}: {e})")
            continue
        for p in rep.get("problems", []):
            log(f"[inputs] {site}: PROBLEM {p}")
        for w in rep.get("warnings", []):
            log(f"[inputs] {site}: warning {w}")
        r = rep.get("record", {})
        log(f"[inputs] {site}: {rep.get('source', {}).get('kind')} source, T = {r.get('T')} "
            f"({r.get('instant_source')} instants), grid {rep.get('grid', {}).get('H')} x "
            f"{rep.get('grid', {}).get('W')}, offsets {rep.get('offset_counts')}")
        log(f"[inputs] {site}: {'ok' if rep.get('ok') else 'NOT OK'} "
            f"({time.time() - t0:.0f} s) -> {report}")
        if not rep.get("ok"):
            bad.append(site)
    return 1 if bad else 0


def _version(dist):
    import importlib.metadata as md
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def check_env(sites, cubes_root=None, rscript=None, network=False, res=10):
    """The §13.3 environment checks; returns the v13 ``Checks``. At 10 m the
    Zarr stores and their catalogue times; at 20 m the legacy netCDF cubes
    and their catalogue times (local runs)."""
    import platform
    import shutil

    v13 = _v13()
    ck = v13.Checks()
    ck.info("python", f"{sys.version.split()[0]} on {platform.platform()}")
    ck.info("resolution", f"{res} m: " + ("Zarr stores (reported numbers)" if res == 10 else
                                          "legacy netCDF cubes (local software checks only)"))
    for dist, need in (("numpy", True), ("pandas", True), ("xarray", True), ("zarr", True),
                       ("numcodecs", True), ("scipy", True), ("scikit-image", True),
                       ("pyproj", True), ("rasterio", True), ("eo-tides", True),
                       ("pyTMD", True), ("copernicusmarine", False), ("torch", False),
                       ("torchvision", False), ("timm", False),
                       ("segmentation-models-pytorch", False),
                       ("omnicloudmask", False), ("psutil", False), ("threadpoolctl", False),
                       ("pyarrow", False)):
        v = _version(dist)
        if v:
            ck.ok(dist, v)
        elif need:
            ck.fail(dist, "not installed")
        else:
            ck.warn(dist, "not installed")
    if _version("omnicloudmask"):              # torch is imported only when C11 can run
        try:
            from experiments.benchmark import obs, package
            prov = package.ocm_provenance()
            import torch
            import torchvision.ops
            torchvision.ops.nms(torch.rand(3, 4), torch.rand(3), 0.5)   # CPU/CUDA build mismatch
            import omnicloudmask  # noqa: F401  (timm and segmentation-models-pytorch)
            ck.ok("OmniCloudMask (C11)", f"{prov['version']}, licence {prov['licence']}, "
                                         f"libraries {prov.get('libraries')}, torch threads "
                                         f"{prov.get('torch_threads')}")
            ck.info("torch device", "cuda available" if torch.cuda.is_available()
                    else "CPU only")
            md = obs.ocm_model_dir()
            w = sorted(f for f in os.listdir(md)) if os.path.isdir(md) else []
            (ck.ok if w else ck.warn)("OmniCloudMask weights", f"{md}: {w}" if w else
                                      f"{md} is empty: the ocm step downloads them from "
                                      "Hugging Face on first use (or run get_models first)")
        except Exception as e:
            ck.fail("OmniCloudMask (C11)", f"{type(e).__name__}: {e} (install torch and "
                    "torchvision together from the CPU index, requirements-cluster.txt)")
    else:
        ck.warn("OmniCloudMask (C11)", "not installed: C3 replaces C11 in dom and G1' "
                                       "(declared); install CPU torch, then "
                                       "requirements-cluster.txt")
    v13._check_r(ck, rscript or os.environ.get("RSCRIPT") or None)
    try:
        from experiments.benchmark.fetch import tides as ft
        ck.info("tide models installed", ft.installed_models())
    except Exception as e:
        ck.warn("tide models", f"{type(e).__name__}: {e}")
    try:
        from experiments.benchmark.fetch import cmems as fc
        ck.info("CMEMS credentials (LM)", fc.check_credentials())
    except Exception as e:
        ck.warn("CMEMS credentials (LM)", f"{type(e).__name__}: {e}")
    ck.info("scene instants", "catalogue / product-name times (datatake start), the default "
                              "of every run; a sensing_times.json only with --sensing-times "
                              "(the optional 'diag time', user decision 2026-10-14)")
    from experiments import zarr_cubes as zc
    from experiments.benchmark import config
    for site in sites:
        if res == 10:
            yrs = zc.list_years(site, cubes_root)
            (ck.ok if yrs else ck.warn)(f"{site} Zarr stores", yrs or "none")
            if yrs:
                from experiments.benchmark import extract
                ex = [y for y in yrs if extract.extract_info(site, y, cubes_root)]
                if ex:
                    ck.warn(f"{site} Zarr stores", f"{ex} are local EXTRACTS (synthetic bands): "
                                                   "software checks only")
                op = zc.overpass_path(site, cubes_root)
                (ck.ok if op.is_file() else ck.fail)(f"{site} catalogue times",
                                                     str(op) if op.is_file()
                                                     else f"{op} missing")
            sd = zc.site_dir(site, cubes_root)
            sm = sd / "scene_meta.json"
            (ck.ok if sm.is_file() else ck.warn)(f"{site} scene_meta.json",
                                                 "present" if sm.is_file() else "missing")
            cube = config.site_config(site, 10).cube
            ck.info(f"{site} legacy 10 m cube (reproduction gates)",
                    cube if os.path.exists(cube) else f"{cube} not here")
        else:
            cfg = config.site_config(site, res)
            (ck.ok if os.path.exists(cfg.cube) else ck.fail)(
                f"{site} legacy {res} m cube", cfg.cube if os.path.exists(cfg.cube)
                else f"{cfg.cube} missing")
            lab = config.resolve(cfg.overpass)
            (ck.ok if lab and os.path.exists(lab) else ck.fail)(
                f"{site} catalogue times", lab if lab and os.path.exists(lab)
                else f"{cfg.overpass} missing")
            boundary_ok = True
            if cfg.boundary.get("kind") == "make_boundary":
                from experiments.benchmark import levels
                try:
                    chunks = levels.ioc_chunks(cfg)
                    boundary_ok = bool(chunks)
                    ck.ok(f"{site} production IOC chunks (2023-2025)", f"{len(chunks)} files")
                except Exception as e:
                    boundary_ok = False
                    ck.fail(f"{site} production IOC chunks", f"{type(e).__name__}: {e}")
            if not boundary_ok:
                ck.warn(f"{site} boundary", "h_B cannot be built for the local run")
    dea = os.environ.get("VENV_DEA") or ("venv-dea" if os.path.isdir("venv-dea") else None)
    (ck.ok if dea else ck.warn)("venv-dea (E2, V3)", dea or "not found")
    oct_ = shutil.which("octave")
    (ck.ok if oct_ else ck.warn)("Octave (G-flood2topo, V3)", oct_ or "not found")
    try:
        import psutil
        vm = psutil.virtual_memory()
        ck.info("RAM", f"{vm.total / 2**30:.1f} GB total, {vm.available / 2**30:.1f} GB free")
    except ImportError:
        pass
    try:
        from threadpoolctl import threadpool_info
        ck.info("BLAS", [(i.get("internal_api"), i.get("num_threads"))
                         for i in threadpool_info()])
    except ImportError:
        pass
    from experiments.benchmark import inputs, package
    out_root = _root(res)
    paths = [p for p in (ROOT, str(zc.cubes_root(cubes_root)), out_root,
                         os.environ.get("TMPDIR", "")) if p and os.path.exists(p)]
    try:
        sample = None
        if res == 10:
            for site in sites:
                sample = inputs.disk_sample_dir(site, cubes_root)
                if sample:
                    break
        dk = inputs.disk_check(paths, read_from=sample)
        ck.info("disk", dk)
        d = out_root
        while not os.path.isdir(d) and os.path.dirname(os.path.abspath(d)) != \
                os.path.abspath(d):
            d = os.path.dirname(os.path.abspath(d))
        free = shutil.disk_usage(d).free / 1e9
        margin = package.space_margin_gb()
        need = margin + (DISK_W3_GB if res == 10 else 0.0)
        msg = (f"{free:.0f} GB free under {out_root}; Campaign 1 needs about {DISK_W3_GB:.0f} "
               f"GB of package shards at W3 (5 sites) plus the {margin:.0f} GB "
               "margin every pass keeps (passes refuse to start below it)")
        if free < margin:
            ck.fail("disk space (output root)", msg)
        elif free < need:
            ck.warn("disk space (output root)", msg)
        else:
            ck.ok("disk space (output root)", msg)
    except Exception as e:
        ck.warn("disk", f"{type(e).__name__}: {e}")
    if network:
        try:
            import requests
            from pyintertidal import net
            net.use_system_certificates()
            r = requests.get("https://stac.dataspace.copernicus.eu/v1/collections/"
                             "sentinel-2-l2a", timeout=60)
            r.raise_for_status()
            text = json.dumps(r.json())
            has = [k for k in ("CLP", "CLM") if f'"{k}' in text or f"_{k}" in text]
            (ck.ok if has else ck.info)("s2cloudless CLP/CLM in CDSE L2A (C12)",
                                        has or "not exposed")
        except Exception as e:
            ck.warn("CDSE collection metadata", f"{type(e).__name__}: {e}")
    return ck


def cmd_check_env(args):
    from experiments.benchmark import product
    res = _res(args)
    ck = check_env(_sites(args.sites, res), args.cubes_root, args.rscript, args.network, res)
    path = os.path.join(_root(res), "logs", f"check_env_{socket.gethostname()}_{res}m.json")
    try:
        product.write_json(path, {"utc": product.utc_stamp(), "res_m": res, "items": ck.items,
                                  "fail": ck.count("FAIL"), "warn": ck.count("WARN")})
    except OSError as e:
        log(f"[check-env] could not write {path}: {e}")
    log(f"[check-env] {ck.count('FAIL')} FAIL, {ck.count('WARN')} WARN, {ck.count('OK')} OK "
        f"-> {path}")
    return 1 if ck.count("FAIL") else 0


# ─────────────────────────────────────────────────────────────────────────────
#  diag time (optional) and its old name 'gate time'
# ─────────────────────────────────────────────────────────────────────────────

def cmd_diag(args):
    """The OPTIONAL sensing-time diagnostic (``sensing_time.py``): it gates
    nothing and no command requires it (user decision of 2026-10-14)."""
    st = _module("sensing_time")
    if st is None:
        return 2
    log(f"[diag time] {DIAG_NOTE}")
    argv = ["diag", "--sites", *_sites(args.sites)]
    if args.cubes_root:
        argv += ["--cubes-root", args.cubes_root]
    for flag in ("no_fetch", "refresh", "all_rows", "copy_to_cubes"):
        if getattr(args, flag, False):
            argv.append("--" + flag.replace("_", "-"))
    return st.main(argv)


def cmd_gate(args):
    if args.name == "time":
        log(f"[gate time] {GATE_TIME_NOTE}")
        return cmd_diag(args)
    return _later(f"gate {args.name}", LATER["gate"])


# ─────────────────────────────────────────────────────────────────────────────
#  prepare, truth, status
# ─────────────────────────────────────────────────────────────────────────────

def cmd_prepare(args):
    from experiments.benchmark import package
    steps = _split(args.step) or None
    if steps:
        alias = {"tiles": "pass2", "levels-sources": "sources"}   # pass 2 writes the tiles
        steps = [alias.get(s, s) for s in steps]
        bad = [s for s in steps if s not in package.V2_STEPS + package.V2_EXTRA_STEPS]
        if bad:
            log(f"[prepare] unknown step(s) {bad}; steps: "
                f"{package.V2_STEPS + package.V2_EXTRA_STEPS}")
            return 2
    failed = []
    sites = _sites(args.site, _res(args))
    for site in sites:
        t0 = time.time()
        try:
            s2 = _site2(site, args, len(sites))
            req = package.instants_identity(s2)
            asked = _instants_line({"instants": req["source"], "instants_path": req["path"]})
            log(f"[prepare] {site}: {s2.source} source, {s2.pixel_m:g} m, boundary "
                f"{s2.boundary}, instants requested {asked} -> {s2.dir}")
            if args.prune_stale_shards:
                freed = package.prune_stale_shards(s2)
                log(f"[prepare] {site}: superseded dom_versions removed {freed or 'none'}")
                if not steps:
                    continue
            ran = package.prepare_v2(s2, steps=steps, years=_years(args.years),
                                     force=bool(args.force),
                                     allow_large_dom=bool(args.allow_large_dom))
            log(f"[prepare] {site}: ran {list(ran) or 'nothing (all done)'} "
                f"({time.time() - t0:.0f} s)")
            st = package.status_v2(s2)
            if st.get("record"):
                r = st["record"]
                log(f"[prepare] {site}: record T = {r['T']} over {r['years']}, instants "
                    f"{_instants_line(r)}, offsets {r['offset_source']}; EOT20-only share "
                    f"of h_B per window {r.get('boundary_coverage')}")
            log(f"[prepare] {site}: dom {st['dom']}; shard keys {st['shard_keys']}")
        except Exception as e:
            failed.append(site)
            log(traceback.format_exc().rstrip())
            log(f"[prepare] {site}: FAILED ({type(e).__name__}: {e})")
    return 1 if failed else 0


def cmd_truth(args):
    """The v2 truths (spec §1.8-1.9, §6.2) on the v2 package after its
    geometry step: the 1-min h_B series and marks (``prepare --step series``,
    run here when missing), the judge table on the v2 record (truth/<site>/
    judges.json), elevation (all epochs, cell support, range filter from the
    1-min h_B) and the lag truths per window (they read the level table:
    ``prepare --step levels``)."""
    from experiments.benchmark import judges, package
    truth = _module("truth")
    if truth is None:
        return 2
    parts = _split(args.part) or ["elev", "lag"]
    windows = _split(args.windows) or None
    if windows:
        log(f"[truth] --windows {windows}: a partial truth (the defaults are every Campaign-1 "
            f"window: elevation {list(truth.ELEV_WINDOWS_1A)}, lag {list(truth.LAG_WINDOWS_1A)}"
            "); pilot dom --freeze refuses a D_truth that lacks one")
    kw = {"windows": tuple(windows)} if windows else {}
    failed = []
    sites = _sites(args.site, _res(args))
    for site in sites:
        t0 = time.time()
        try:
            s2 = _site2(site, args, len(sites))
            pkg = package.V2Package(s2)
            package.check_record_instants(s2, pkg.record)
            kind = (pkg.record.get("boundary") or {}).get("kind", s2.boundary)
            if kind != s2.boundary:
                raise package.PackageError(f"the package was built with boundary {kind!r} "
                                           f"(pass --boundary {kind})")
            log(f"[truth] {site}: record T = {pkg.T} over {pkg.years}, instants "
                f"{_instants_line(pkg.record)}")
            os.makedirs(s2.truth_dir, exist_ok=True)
            # the 1-min h_B series and its marks (resumable package step)
            package.prepare_v2(s2, steps=["series"], provider=pkg.provider())
            book = package._V2Book(s2)
            for step in ("geometry", "series"):
                why = book.why_stale(step)
                if why:
                    raise package.PackageError(f"the package step {step} is not done ({why}): "
                                               f"run prepare {site} --step {step}")
            jt = f"{s2.truth_dir}/judges.json"
            if s2.cfg.judges:
                judges.build(pkg, path=jt, provider=pkg.provider())
            if "elev" in parts:
                if not s2.cfg.elevation_truth:
                    log(f"[truth] {site}: no elevation truth at this site (lag only)")
                else:
                    _, h1 = pkg.series1
                    log(f"[truth] {site}: elevation -> "
                        f"{truth.build_elevation_truth_v2(pkg, h1=h1, **kw)}")
            if "lag" in parts:
                if not s2.cfg.judges:
                    log(f"[truth] {site}: no judges (lags descriptive)")
                else:
                    why = book.why_stale("levels")
                    if why:
                        raise package.PackageError(
                            f"the level table is not done ({why}): run prepare {site} --step "
                            "levels before the lag truths")
                    log(f"[truth] {site}: lag -> {truth.build_lag_truth_v2(pkg, **kw)}")
            log(f"[truth] {site}: done in {time.time() - t0:.0f} s")
            pkg.close()
        except Exception as e:
            failed.append(site)
            log(traceback.format_exc().rstrip())
            log(f"[truth] {site}: FAILED ({type(e).__name__}: {e})")
    return 1 if failed else 0


def cmd_status(args):
    from experiments.benchmark import package
    sites = _sites(args.sites, _res(args))
    for site in sites:
        s2 = _site2(site, args, len(sites))
        st = package.status_v2(s2) if os.path.isdir(s2.dir) else {"steps": {}, "dom": None}
        steps = st["steps"]
        done = sum(v == "done" for v in steps.values())
        log(f"[status] {site}: {done}/{len(steps)} steps done; stale "
            f"{[k for k, v in steps.items() if v != 'done']}; dom {st.get('dom')}; "
            f"shard keys {st.get('shard_keys')}")
        if st.get("record"):
            r = st["record"]
            log(f"[status] {site}: record T = {r['T']} over {r['years']}, instants "
                f"{_instants_line(r)}, offsets {r['offset_source']}")
        for dv, e in (st.get("dom_versions") or {}).items():
            tag = "current" if e["current"] else ("frozen" if e["frozen"] else "SUPERSEDED")
            if e["current"] and e["frozen"]:
                tag = "current, frozen"
            hint = "  <- prepare --prune-stale-shards removes it" if tag == "SUPERSEDED" else ""
            log(f"[status] {site}: dom_version {dv} ({tag}): dom {e['dom_gb']:.2f} GB, "
                f"shards {e['shards_gb']:.2f} GB{hint}")
    log("[status] runs, claims and completeness per design: available from milestone V2")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
#  pilot crosstab and pilot dom
# ─────────────────────────────────────────────────────────────────────────────

def _cld_quantiles(counts, qs=(0.01, 0.1, 0.5, 0.9, 0.99)):
    """Quantiles of CLD (0..100) from per-value counts (no fill)."""
    import numpy as np
    c = np.asarray(counts, np.float64)[:101]
    tot = c.sum()
    if tot <= 0:
        return None
    cum = np.cumsum(c) / tot
    out = {f"q{int(q * 100):02d}": int(np.searchsorted(cum, q)) for q in qs}
    nz = np.flatnonzero(c)
    out.update(min=int(nz[0]), max=int(nz[-1]), n=int(tot))
    return out


def crosstab_site(site2):
    """The truth-free SCL x CLD cross-tab of one site (spec §2.2): per year
    and processing baseline the CLD quantiles of each SCL class, and the
    per-scene median CLD of the anchor classes (``"no CLD"`` for a source
    without it, e.g. a legacy cube)."""
    import numpy as np
    from experiments import zarr_cubes as zc
    from experiments.benchmark import inputs, obs, package
    pkg = package.V2Package(site2)
    base = {}
    sm = zc.site_dir(site2.site, site2.cubes_root) / inputs.SCENE_META_NAME
    if site2.source == "zarr" and sm.is_file():
        meta = inputs.read_scene_meta(sm)
        base = {d: ",".join(v.get("processing_baselines", []) or ["unknown"])
                for d, v in meta.get("dates", {}).items()}
    has_cld = "CLD" in pkg.bands
    out = {"site": site2.site, "source": site2.source, "has_cld": has_cld, "years": {},
           "anchors": {}}
    for y in pkg.years:
        xt = np.load(package._year_dir(site2, y, "pass1", "crosstab.npy"))
        dates = pkg.dates[pkg.window_years_safe(y)]
        groups = {}
        for j, d in enumerate(dates):
            groups.setdefault(base.get(str(d), "unknown"), []).append(j)
        out["years"][str(y)] = {}
        for g, idx in groups.items():
            tab = xt[idx].sum(axis=0)
            out["years"][str(y)][g] = {
                "n_scenes": len(idx),
                "scl": {str(c): _cld_quantiles(tab[c]) for c in range(12) if tab[c].sum()},
                "scl_px": {str(c): int(tab[c].sum()) for c in range(12) if tab[c].sum()},
                "scl_nodata": _cld_quantiles(tab[12])}
    g = pkg.geometry
    wf = np.asarray(g["wf"], np.float64).ravel()
    omega = np.asarray(g["omega"], np.int64)
    keep = np.asarray(g["keep_full"], np.int64)
    w3 = pkg.window("W3") if all(y in pkg.years for y in (2023, 2024, 2025)) else \
        np.arange(pkg.T)
    low = pkg.h0[w3] < np.quantile(pkg.h0[w3], 0.25)
    classes = {"permanent_water": omega[wf[omega] >= 0.98],
               "stable_land": omega[wf[omega] <= 0.02],
               "dry_flats_low_tide": keep[wf[keep] <= 0.3]}
    chan = os.path.join(site2.dir, "channel_anchors.npy")
    if os.path.exists(chan):
        classes["turbid_channel"] = np.intersect1d(np.load(chan), pkg.dom)
    for name, flat in classes.items():
        if not len(flat):
            out["anchors"][name] = {"n_px": 0}
            continue
        if not has_cld:
            out["anchors"][name] = {"n_px": int(len(flat)), "n_scenes": 0,
                                    "status": f"no CLD in the {site2.source} source"}
            continue
        pos = pkg.positions(flat)
        sel = w3[low] if name == "dry_flats_low_tide" else w3
        raw = pkg.read(["SCL", "CLD"], sel, pos)
        med = []
        for j in range(len(sel)):
            ok = obs.cloud_c1(raw["SCL"][j]) & (raw["CLD"][j] != obs.CLD_NODATA_U8)
            if ok.sum() >= CROSSTAB_RULE["min_anchor_px"]:
                med.append(float(np.median(raw["CLD"][j][ok])))
        med = np.asarray(med)
        share = float((med > CROSSTAB_RULE["median_cld_pct"]).mean()) if med.size else None
        out["anchors"][name] = {"n_px": int(len(flat)), "n_scenes": int(med.size),
                                "median_of_medians": float(np.median(med)) if med.size else None,
                                "share_scenes_median_gt_20": share}
    if "turbid_channel" not in classes:
        out["anchors"]["turbid_channel"] = {"n_px": 0, "status": "not available (EMODnet "
                                            "channel anchors not fetched; declared)"}
    out["n_scenes"] = int(pkg.T)
    pkg.close()
    return out


def cmd_pilot(args):
    if args.what in ("reps", "units", "attain"):
        return _later(f"pilot {args.what}", LATER[f"pilot {args.what}"])
    if args.what == "crosstab":
        return pilot_crosstab(args)
    return pilot_dom(args)


def pilot_crosstab(args):
    from experiments.benchmark import config, package
    res = _res(args)
    sites = _sites(args.sites, res)
    res_ = {}
    for site in sites:
        try:
            res_[site] = crosstab_site(_site2(site, args))
        except Exception as e:
            log(traceback.format_exc().rstrip())
            log(f"[pilot crosstab] {site}: FAILED ({type(e).__name__}: {e})")
            res_[site] = {"error": f"{type(e).__name__}: {e}"}
    raise_p = False
    for site, r in res_.items():
        for name in ("permanent_water", "dry_flats_low_tide", "turbid_channel"):
            sh = (r.get("anchors", {}).get(name) or {}).get("share_scenes_median_gt_20")
            if sh is not None and sh > CROSSTAB_RULE["share_of_scenes"]:
                raise_p = True
                log(f"[pilot crosstab] {site} {name}: {sh:.1%} of scenes with median CLD > 20 %")
    with_cld = [s for s, r in res_.items() if r.get("has_cld")]
    if with_cld:
        p = CROSSTAB_RULE["p_raised"] if raise_p else CROSSTAB_RULE["p_default"]
    else:
        p = None
        log("[pilot crosstab] no site has CLD (legacy sources): p is not decided")
    complete = set(sites) == set(config.SITES) and not any("error" in r for r in res_.values())
    lacking = {}
    for site, r in res_.items():
        if "error" in r:
            continue
        miss = [name for name in ("permanent_water", "dry_flats_low_tide", "turbid_channel")
                if not (r.get("anchors", {}).get(name) or {}).get("n_scenes")]
        if miss:
            lacking[site] = miss
    if lacking:
        log(f"[pilot crosstab] anchor classes not evaluated {lacking}: p stays PROVISIONAL "
            "(the turbid channel class needs the EMODnet channel anchors, fetch emodnet)")
    doc = {"format": "validation-v2-crosstab/1", "rule": CROSSTAB_RULE, "p": p,
           "provisional": bool(not complete or lacking), "anchor_classes_missing": lacking,
           "res_m": res, "sites": res_}
    path = os.path.join(_root(res), "common", package.CROSSTAB_JSON)
    if not complete and os.path.exists(path):
        log(f"[pilot crosstab] only {sites}: {path} is kept; this partial result -> "
            f"{path[:-5]}_partial.json")
        path = path[:-5] + "_partial.json"
    _write_json(path, doc)
    for site, r in res_.items():
        for name, a in r.get("anchors", {}).items():
            log(f"[pilot crosstab] {site} {name}: {a}")
    why = "; ".join(w for w, c in (("not every site", not complete),
                                   ("anchor classes missing", bool(lacking))) if c)
    log(f"[pilot crosstab] C3 cut-off p = {p}{' %' if p is not None else ''}"
        f"{f' (PROVISIONAL: {why})' if why else ''} -> {path}; the user freezes p [V1]")
    failed = [s for s, r in res_.items() if "error" in r]
    return 1 if failed else 0


def pilot_dom(args):
    import numpy as np

    from experiments.benchmark import domain, package, settings
    res = _res(args)
    rows = {}
    total_gb = 0.0
    for site in _sites(args.sites, res):
        s2 = _site2(site, args)
        try:
            pkg = package.V2Package(s2)
        except package.PackageError as e:
            log(f"[pilot dom] {site}: {e}")
            continue
        if pkg.dom is None:
            log(f"[pilot dom] {site}: no dom yet (prepare --step dom)")
            continue
        meta = pkg.dom_meta
        v = package.dom_verdict(meta)
        total_gb += v["projected_w9_gb"]
        r = {"dom_version": pkg.dom_version, "n_s0": meta["n_s0"], "n_dom": meta["n_dom"],
             "ratio": meta["ratio_dom_s0"], "has_d_truth": meta.get("has_d_truth"),
             "terms": meta.get("terms"), "n_keep_full": int(len(pkg.full.keep)),
             "verdict": v, "settings": {}}
        dom_set = np.zeros(pkg.H * pkg.W, bool)
        dom_set[pkg.dom] = True
        for key in sorted(set(s2.observation_keys()) | {settings.P0.id}):
            for wid in ("W3", "Y2023", "Y2024", "Y2025"):
                try:
                    c = pkg.frame_counts(key, wid)
                except package.PackageError:
                    c = None
                if c is None:
                    continue
                u = package.urec(*c)
                inter, _ = domain.classify(*c)
                r["settings"].setdefault(key, {})[wid] = {
                    "n_urec": int(u.sum()), "n_keep": int(inter.sum()),
                    "n_keep_outside_dom": int((inter.ravel() & ~dom_set).sum())}
        rows[site] = r
        log(f"[pilot dom] {site}: |S0| = {r['n_s0']:,}, |dom| = {r['n_dom']:,} "
            f"(ratio {r['ratio']:.2f}{' > 1.5: pass 2 REFUSED' if not v['ratio_ok'] else ''}), "
            f"|keep_full| = {r['n_keep_full']:,}, W9 package ~{v['projected_w9_gb']:.1f} GB, "
            f"D_truth {'in' if r['has_d_truth'] else 'MISSING'}; U_rec terms {r['terms']}")
        for key, per in r["settings"].items():
            w = per.get("W3")
            if w:
                log(f"[pilot dom]   {key:24} W3: |U_rec| {w['n_urec']:,}, |keep| {w['n_keep']:,}"
                    f" ({w['n_keep_outside_dom']:,} outside dom)")
        if args.freeze:
            if not v["ratio_ok"] and not args.force:
                log(f"[pilot dom] {site}: not frozen (|dom| > 1.5 |S0|; --force to override)")
            else:
                try:
                    log(f"[pilot dom] {site}: dom FROZEN: {package.freeze_dom(s2)}")
                except package.PackageError as e:
                    log(f"[pilot dom] {site}: not frozen ({e})")
        pkg.close()
    budget_ok = total_gb <= package.PACKAGE_W9_BUDGET_GB
    log(f"[pilot dom] projected W9 package {total_gb:.1f} GB of the "
        f"{package.PACKAGE_W9_BUDGET_GB:.0f} GB budget: {'ok' if budget_ok else 'OVER BUDGET'}")
    _write_json(os.path.join(_root(res), "common", "pilot_dom.json"),
                {"sites": rows, "projected_w9_gb": round(total_gb, 2), "res_m": res,
                 "budget_gb": package.PACKAGE_W9_BUDGET_GB, "budget_ok": budget_ok})
    refused = [s for s, r in rows.items() if not r["verdict"]["ratio_ok"]]
    return 1 if refused or not budget_ok else 0


# ─────────────────────────────────────────────────────────────────────────────
#  plan
# ─────────────────────────────────────────────────────────────────────────────

def cmd_plan(args):
    from experiments.benchmark import designs
    names = _split(args.design)
    expand = {"tier1a": designs.QUEUE_1A, "tier1b": designs.QUEUE_1B,
              "all": designs.QUEUE_1A + designs.QUEUE_1B}
    todo = []
    for n in names:
        todo += list(expand.get(n, (n,)))
    bad = [n for n in todo if n not in designs.DESIGNS]
    if bad:
        log(f"[plan] unknown design(s) {bad}; designs: {tuple(designs.DESIGNS)} or "
            f"tier1a, tier1b, all")
        return 2
    sites = _sites(args.sites) if args.sites else None
    for n in dict.fromkeys(todo):
        path, s = designs.write_plan(n, root=_root(), sites=sites)
        log(f"[plan] {n}: {s['n_rows']} rows ({s['n_planned']} planned, {s['n_shared']} shared, "
            f"{s['n_not_applicable']} not applicable, {s['n_reuse']} reused); "
            f"{s['cheap_wh']} wh, Granadeiro {s['granadeiro_h']} h, max RAM "
            f"{s['max_ram_gb']} GB, read {s['read_gb']} GB -> {path}")
    p = designs.projection(sites=sites, cap_days=args.cap_days, workers=args.workers,
                           wh_per_day=args.wh_per_day)
    log(f"[plan] Tier 1A' projection ({p['workers']} workers, utilisation "
        f"{p['utilisation']}, cap {p['cap_days']} days):")
    for st in p["steps"]:
        log(f"[plan]   cut {st['step']}: {st['cut']:<62} {st['cheap_wh']:>7} wh, Granadeiro "
            f"{st['granadeiro_h']:>5} h -> {st['days']:>5} days ({st['days_with_contingency']} "
            f"with 20 % contingency) {'FITS' if st['fits'] else ''}")
    for item, e in p["never_cut_check"].items():
        state = {True: "ok", False: "VIOLATED", None: "n/a"}[e["ok"]]
        log(f"[plan] never cut: {item}: {state} ({e['detail']})")
    if p["fits"]:
        log(f"[plan] fits after cut step {p['applied']}")
    elif p["within_cap"]:
        log(f"[plan] within the cap after cut step {p['applied']}, but a never-cut item is "
            "VIOLATED: the user decides (e.g. the AMM15 archive depth for LM, §1.6)")
    else:
        log("[plan] does NOT fit even after every cut: the user decides (one-week fallback, "
            "§5.14)")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
#  extract (local software checks)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_extract(args):
    """A small Zarr store from a legacy 10 m netCDF cube (real B03, B08, SCL;
    synthetic, flagged B02, B04, B8A, B11, B12, CLD) under a scratch root,
    for local runs of the Zarr path (``extract.py``)."""
    from experiments.benchmark import extract
    rows, cols = (int(v) for v in str(args.size).lower().split("x"))
    origin = tuple(int(v) for v in _split(args.origin)) if args.origin else None
    desc = extract.legacy_extract(args.site, args.out, cube=args.cube, year=args.year,
                                  n_dates=args.dates, size=(rows, cols), origin=origin,
                                  force=bool(args.force))
    log(f"[extract] {args.site}: {desc['store']} (window {desc['window']}, {desc['n_dates']} "
        f"dates, lattice {desc['lattice20']}); use --cubes-root {args.out}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
#  later milestones
# ─────────────────────────────────────────────────────────────────────────────

def _later(what, ms):
    log(f"validation_v2 {what}: available from milestone {ms}; this build is V1: fetch, "
        f"inputs, check-env, prepare, truth, pilot crosstab|dom, plan, status (and the "
        f"optional diag time)")
    return 2


def cmd_later(args):
    return _later(args.command, LATER[args.command])


# ─────────────────────────────────────────────────────────────────────────────
#  parser and main
# ─────────────────────────────────────────────────────────────────────────────

def _common(sp, sites=True, res=True):
    if sites:
        sp.add_argument("--sites", default="all", help="site, comma list or all")
    sp.add_argument("--cubes-root", default=None, help="Zarr cubes root (default "
                                                      "ZARR_CUBES_ROOT or cubes/)")
    sp.add_argument("--blas-threads", type=int, default=None)
    if res:
        sp.add_argument("--res", type=int, default=10, choices=RES_CHOICES,
                        help="10: Zarr stores (reported numbers); 20: legacy 20 m netCDF "
                             "cubes (local software checks; root results/validation_v2_20m)")


def _sensing_arg(sp):
    sp.add_argument("--sensing-times", default=None, metavar="FILE",
                    help="use this sensing_times.json (or the diag folder holding "
                         "<site>/sensing_times.json) instead of the catalogue times; "
                         "optional, never needed (user decision 2026-10-14)")


def build_parser():
    p = argparse.ArgumentParser(prog="python -X utf8 -m experiments.validation_v2",
                                description="Validation v2 of MAREA "
                                            "(docs/validation_v2_spec.md). Scene instants: "
                                            "the catalogue times; 'diag time' is optional.")
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    sp = sub.add_parser("fetch", help="network downloads (IOC, judges, CMEMS, tides, ...; "
                                      "'sensing' only feeds the optional diag time)")
    sp.add_argument("what", choices=FETCH_WHATS)
    _common(sp, res=False)
    sp.add_argument("--years", default=None, help="2023-2025 (cmems)")
    sp.add_argument("--dry-run", action="store_true", help="ioc: missing years; cmems: dataset "
                                                          "ids, resolution, archive depth")
    sp.add_argument("--execute", action="store_true", help="judges: run the download scripts")
    sp.add_argument("--refresh", action="store_true")
    sp.add_argument("--accept-gaps", action="store_true",
                    help="ioc: keep downloaded years that fail the coverage rule (a real "
                         "gauge gap, after checking the counts; recorded in the manifest)")
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("inputs", help="the §1.11 data checks -> common/inputs_report.json")
    _common(sp)
    _sensing_arg(sp)
    sp.add_argument("--no-scan", action="store_true")
    sp.add_argument("--full-fingerprint", action="store_true")
    sp.add_argument("--cache-mb", type=float, default=128.0,
                    help="block cache of the scan (each chunk is read once)")
    sp.set_defaults(func=cmd_inputs)

    sp = sub.add_parser("check-env", help="versions, R, tide models, OmniCloudMask, CMEMS, "
                                          "inputs of the resolution, disk, RAM, BLAS")
    _common(sp)
    sp.add_argument("--rscript", default=None)
    sp.add_argument("--network", action="store_true", help="query the CDSE collection "
                                                           "metadata (CLP/CLM)")
    sp.set_defaults(func=cmd_check_env)

    sp = sub.add_parser("diag", help="OPTIONAL diagnostics: 'diag time' compares the site "
                                     "sensing time (solar angles) with the catalogue time; "
                                     "it gates nothing")
    sp.add_argument("what", choices=DIAG_WHATS)
    _common(sp, res=False)
    sp.add_argument("--no-fetch", action="store_true")
    sp.add_argument("--refresh", action="store_true")
    sp.add_argument("--all-rows", action="store_true")
    sp.add_argument("--copy-to-cubes", action="store_true",
                    help="also copy sensing_times.json beside the stores (still unused "
                         "unless a run is given --sensing-times)")
    sp.set_defaults(func=cmd_diag)

    sp = sub.add_parser("gate", help="the gates of V2-V4; 'gate time' is only an alias of the "
                                     "optional 'diag time'")
    sp.add_argument("name", help="zarr | offset | record | keep | marea | lagtruth | level | "
                                 "ocm | nidem | dea | ... | all (time: alias of diag time)")
    _common(sp, res=False)
    sp.add_argument("--no-fetch", action="store_true")
    sp.add_argument("--refresh", action="store_true")
    sp.add_argument("--all-rows", action="store_true")
    sp.add_argument("--copy-to-cubes", action="store_true")
    sp.set_defaults(func=cmd_gate)

    sp = sub.add_parser("prepare", help="build or extend the package year shards (resumable)")
    sp.add_argument("site", help="site, comma list or all")
    _common(sp, sites=False)
    _sensing_arg(sp)
    sp.add_argument("--years", default=None, help="e.g. 2023-2025 (default the record's)")
    sp.add_argument("--step", default=None, help="comma list of " + ",".join(
        ("record", "pass1", "geometry", "support", "ocm", "dom", "pass2", "settings",
         "counters", "mnar", "subsets", "levels", "sources", "manifest")) + " (tiles = pass2; "
        "sources = the water-level sources LB/LT/LM/LG, after truth); series (the 1-min h_B "
        "and marks; truth runs it when missing)")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--boundary", default=None, choices=("union", "production"),
                    help="h_B provider (default: union at 10 m, production at 20 m)")
    sp.add_argument("--allow-large-dom", action="store_true",
                    help="run pass 2 although |dom| > 1.5 |S0| (the user's decision)")
    sp.add_argument("--allow-provisional-offsets", action="store_true",
                    help="build a Zarr record without the inputs report's frozen per-scene "
                         "offsets (recorded in record.json; never for the reported runs)")
    sp.add_argument("--prune-stale-shards", action="store_true",
                    help="remove the dom and shard folders of superseded dom_versions (never "
                         "the current or a frozen one); alone, it only prunes")
    sp.add_argument("--block-rows", type=_even_rows, default=None,
                    help="rows per pass block (even; default 256, the chunk rows)")
    sp.add_argument("--cache-mb", type=float, default=None,
                    help="decoded-chunk cache (default 1024; 4096 on the server)")
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("truth", help="v2 judge table, elevation and lag truths")
    sp.add_argument("site", help="site, comma list or all")
    _common(sp, sites=False)
    _sensing_arg(sp)               # must name the record's instants (refused otherwise)
    sp.add_argument("--part", default="elev,lag")
    sp.add_argument("--windows", default=None, help="e.g. W3,Y2023,Y2024,Y2025")
    sp.add_argument("--boundary", default=None, choices=("union", "production"))
    sp.set_defaults(func=cmd_truth)

    sp = sub.add_parser("pilot", help="crosstab | dom (V1); reps | attain (V2); units (V3)")
    sp.add_argument("what", choices=PILOT_WHATS)
    _common(sp)
    sp.add_argument("--freeze", action="store_true", help="dom: freeze the current dom "
                                                          "(the user's approval, lock-A)")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_pilot)

    sp = sub.add_parser("plan", help="write plans/<design>.csv; the 10-day projection")
    sp.add_argument("--design", required=True, help="design name(s), tier1a, tier1b or all")
    sp.add_argument("--sites", default=None)
    sp.add_argument("--cap-days", type=float, default=10.0)
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--wh-per-day", type=float, default=None,
                    help="measured cheap throughput (pilot)")
    sp.add_argument("--blas-threads", type=int, default=None)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("status", help="package steps, dom and shard keys per site")
    _common(sp)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("extract", help="a small Zarr extract of a legacy 10 m cube under a "
                                        "scratch root (local software checks only)")
    sp.add_argument("site")
    sp.add_argument("--out", required=True, help="a scratch cubes root (never cubes/)")
    sp.add_argument("--cube", default=None, help="the legacy 10 m netCDF (default the site's)")
    sp.add_argument("--year", type=int, default=2023)
    sp.add_argument("--dates", type=int, default=40)
    sp.add_argument("--size", default="300x400", help="rows x cols (10 m pixels)")
    sp.add_argument("--origin", default=None, help="r0,c0 (default: the window with the most "
                                                   "intertidal pixels of the production mask)")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--blas-threads", type=int, default=None)
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser("control", help="positive control on the synthetic store (V2/V3)")
    sp.add_argument("--method", default=None)
    sp.add_argument("--world", default="gating", choices=("gating", "misspec"))
    sp.add_argument("--kind", default=None, help="lag | elev")
    sp.set_defaults(func=cmd_later)
    sp = sub.add_parser("timing", help="dedicated timing runs on a quiet machine (V2)")
    sp.set_defaults(func=cmd_later)
    sp = sub.add_parser("lock", help="write LOCK.json (V2)")
    sp.add_argument("--phase", required=True, choices=("A", "B"))
    sp.set_defaults(func=cmd_later)
    sp = sub.add_parser("run", help="the multi-plan, instance-major unit queue (V2+)")
    sp.add_argument("--plans", required=True, help="d1,d2,... in priority order")
    sp.add_argument("--worker-id", default="1")
    sp.add_argument("--threads", type=int, default=None)
    sp.add_argument("--steal-stale", action="store_true")
    sp.set_defaults(func=cmd_later)
    sp = sub.add_parser("score", help="sufficient statistics and score tables (V2/V3)")
    sp.add_argument("--part", default="all", choices=("lag", "elev", "curves", "timing", "all"))
    sp.add_argument("--blocks", action="store_true")
    sp.set_defaults(func=cmd_later)
    sp = sub.add_parser("report", help="tables, figures, numbers.json, report.html (V6)")
    sp.set_defaults(func=cmd_later)
    return p


def _interrupt_on_signals():
    """SIGTERM and SIGHUP (kill, tmux kill-server, a reboot, a closed
    terminal) raise KeyboardInterrupt, so the cleanup of the temporary files,
    the command log and the final line run as for Ctrl-C. Installed by
    :func:`main` only (no import-time side effect); returns the previous
    handlers."""
    import signal

    def handler(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    old = {}
    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            old[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):           # not the main thread, or not supported
            pass
    return old


#: what the output root's .gitignore keeps out of git: the packages, truths,
#: products and claims (100+ GB); the small reports, plans and logs stay
#: committable
OUTPUT_GITIGNORE = ("# validation v2 (written by the CLI): the heavy outputs are never "
                    "committed\ncommon/*/\ntruth/\nruns/\nclaims/\n*.npz\n*.npy\n*.tmp-*\n"
                    "*.partial\n*.partial.*\n")


def _ignore_output_root(root):
    """A .gitignore inside the output root (as download_cube_zarr does for
    cubes/): the package shards, truths and products are never swept into a
    commit by ``git add -A`` or ``git stash -u``."""
    try:
        os.makedirs(root, exist_ok=True)
        p = os.path.join(root, ".gitignore")
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(OUTPUT_GITIGNORE)
    except OSError as e:
        log(f"[v2] could not write {root}/.gitignore: {e}")


def main(argv=None):
    """Parse, set the BLAS threads (before numpy), chdir to the repository
    root and run the command. Returns the exit code."""
    args = build_parser().parse_args(argv)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    v13 = _v13()
    n = getattr(args, "blas_threads", None) or DEFAULT_THREADS
    v13.set_thread_env(n, "--blas-threads" if getattr(args, "blas_threads", None)
                       else "default")
    old = _interrupt_on_signals()
    t0 = time.time()
    log(f"[v2] {args.command} on {socket.gethostname()} (pid {os.getpid()}), "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    _ignore_output_root(_root(_res(args)))
    try:
        rc = args.func(args)
    except KeyboardInterrupt as e:
        log(f"[v2] {args.command} interrupted ({e or 'Ctrl-C'})")
        rc = 130
    except ValueError as e:
        log(f"[v2] {args.command}: {e}")
        rc = 2
    finally:
        import signal
        for sig, h in old.items():
            try:
                signal.signal(sig, h)
            except (ValueError, OSError):
                pass
    if rc != 2:
        _log_command(args, rc, time.time() - t0)
    log(f"[v2] {args.command} finished in {time.time() - t0:.0f} s (exit {rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
