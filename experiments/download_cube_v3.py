# -*- coding: utf-8 -*-
"""Download the cubes of validation v3 (the Cantabrian estuaries) on the 10 m grid.

The cube of experiments.download_cube and of the Villaviciosa 10-year cube,
the process of ``pyintertidal.cube.SentinelCube.ensure``:
``load_collection(SENTINEL2_L2A, B03 B08 SCL, max_cloud_cover 100)`` ->
``resample_spatial(10 m, near)`` -> ``save_result(netCDF)``, one openEO batch
job per site and year on the Copernicus Data Space, each kept as the part
``cubes_v3/<site>/<year>.nc``. Validation v3
(experiments/validation_v3_cantabrian_2026-10-29.md) centres its windows on the
LiDAR flights (2016 at Foz, 2017 and 2024 at Urdaibai, 2020 at the Eo), so the
parts are fetched in the order of the queue on the command line: the years
nearest the truths first.

Why not experiments.download_cube_zarr: its stores are registered against the
site's reference netCDF cube, or a store registered against one, and these
sites have neither; a netCDF delivery carries its own coordinates.

Credits. The run stops at the first job the backend refuses for credits (HTTP
402) and says how to resume. A part's job id is written to
``cubes_v3/<site>/<year>.job.json`` BEFORE the job is started, so a later run
with the same queue skips the finished parts and reattaches the job of a part
not yet downloaded instead of paying for it again. A job that ended in error is
replaced by a new one only with ``--retry-failed``.

``--expect-account``: the account's e-mail is printed, and the run stops before
any job is looked up or submitted if it is not this one (OPENEO_CONFIG_HOME
chooses the refresh token).

The overpass times of the dates of each part are added to
``products_<site>_10m/overpass_times.json`` from the catalogue, as
experiments.download_cube does: the products read that file, not the catalogue.

``--merge SITE Y0-Y1``: the parts of those years into
``ndwi_cube_<site>_<y0>-<y1>_10m.nc`` (streamed, as download_cube.merge_parts,
but the parts are kept). ``--status``: what each part of the queue is, from the
files only. ``--dry-run``: the queue, the boxes and a credit estimate, no
network.

    OPENEO_CONFIG_HOME=$PWD/.local/openeo_fer python -m experiments.download_cube_v3 \\
        urdaibai:2020 urdaibai:2023 eo:2020 foz:2020 --expect-account f.garcia@ieo.csic.es
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402

#: lon/lat boxes, drawn on 2026-10-29 from the head of the tide to the open sea,
#: with 1-4 km of margin and the sea on the north edge, where
#: pyintertidal.geometry finds the mouth. From OpenStreetMap: the bay polygons
#: of the Ría de Foz (43.544-43.575 N, 7.267-7.236 W; its ZEPA to 43.532 N,
#: 7.219 W) and of the Ría de Ribadeo / Eo (43.472-43.557 N, 7.064-7.005 W); the
#: Urdaibai channel (43.342-43.398 N, 2.693-2.649 W) and the Oka to Gernika
#: (43.31 N). Wider than the registry polygons, which leave part of the estuary
#: out (Urdaibai's stops at 43.35 N, north of Gernika and of the 2023 sheet
#: 0062) or do not exist (Eo). None: the registry polygon's box (the maps-only
#: sites).
BOXES = {
    "urdaibai": {"west": -2.735, "south": 43.300, "east": -2.625, "north": 43.435},
    "foz": {"west": -7.300, "south": 43.525, "east": -7.215, "north": 43.600},
    "eo": {"west": -7.095, "south": 43.440, "east": -6.985, "north": 43.580},
    "santander": None,
    "santona": None,
}
PART_ROOT = "cubes_v3"
COLLECTION = "SENTINEL2_L2A"
BANDS = ("B03", "B08", "SCL")
RESOLUTION_M = 10
#: as experiments.download_cube.JOB_OPTIONS
JOB_OPTIONS = {"executor-memory": "4G", "executor-memoryOverhead": "4G"}
#: credits per million output pixels per year of the 3-band yearly jobs on
#: Fer's account, 2026-10-26/27 (Westerschelde, Wadden, Ems; costs over the
#: box's 10 m pixels): the range of the --dry-run estimate
CREDITS_PER_MPX_YEAR = (35.0, 110.0)
POLL_S = 120
CAPACITY_WAIT_S = 600


class CreditsExhausted(RuntimeError):
    """The backend refused a job for credits: the run stops."""


class AccountMismatch(RuntimeError):
    """--expect-account: the connection is another account's."""


def utcnow():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def box(site):
    b = BOXES[site]
    if b is not None:
        return dict(b)
    import pyintertidal as pit

    return dict(pit.sites.get(site).bbox)


def part_path(site, year):
    return os.path.join(PART_ROOT, site, f"{year}.nc")


def ledger_path(site, year):
    return os.path.join(PART_ROOT, site, f"{year}.job.json")


def merged_path(site, y0, y1):
    return f"ndwi_cube_{site}_{y0}-{y1}_10m.nc"


def overpass_path(site):
    return f"products_{site}_10m/overpass_times.json"


def year_window(year):
    """openEO's end-exclusive temporal extent of one calendar year."""
    return f"{year}-01-01", f"{year + 1}-01-01"


def parse_years(text):
    """``2020`` or ``2019-2021`` -> [years], ascending."""
    m = re.fullmatch(r"(\d{4})(?:-(\d{4}))?", text.strip())
    if not m:
        raise ValueError(f"not a year or a range: {text!r}")
    y0, y1 = int(m.group(1)), int(m.group(2) or m.group(1))
    if y1 < y0 or y0 < 2015 or y1 > dt.date.today().year:
        raise ValueError(f"years outside Sentinel-2 L2A: {text!r}")
    return list(range(y0, y1 + 1))


def parse_queue(items):
    """``["urdaibai:2020", "foz:2019-2021"]`` -> [(site, year)] in that order, once each."""
    out = []
    for it in items:
        site, sep, years = it.partition(":")
        if not sep or site not in BOXES:
            raise ValueError(f"{it!r}: expected <site>:<year or y0-y1>, site one of {sorted(BOXES)}")
        for y in parse_years(years):
            if (site, y) not in out:
                out.append((site, y))
    return out


def box_px(b, res=RESOLUTION_M):
    """Approximate 10 m pixels of a lon/lat box (rows, cols)."""
    lat = math.radians((b["south"] + b["north"]) / 2)
    rows = (b["north"] - b["south"]) * 111_320.0 / res
    cols = (b["east"] - b["west"]) * 111_320.0 * math.cos(lat) / res
    return int(round(rows)), int(round(cols))


def write_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def load_ledger(site, year):
    path = ledger_path(site, year)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"site": site, "year": year, "bbox": box(site), "temporal_extent": list(year_window(year)),
            "collection": COLLECTION, "bands": list(BANDS), "resolution_m": RESOLUTION_M, "jobs": []}


def part_info(path):
    """``{"dates", "shape", "res"}`` of a complete part, or None (missing or unreadable)."""
    if not os.path.isfile(path):
        return None
    import xarray as xr

    try:
        with xr.open_dataset(path) as ds:
            if any(b not in ds for b in BANDS):
                return None
            t_dim = [d for d in ds["B03"].dims if d not in ("x", "y")][0]
            dates = [str(np.datetime64(v, "D")) for v in ds[t_dim].values]
            res = float(abs(ds["x"].values[1] - ds["x"].values[0]))
            return {"dates": dates, "shape": [int(ds.sizes["y"]), int(ds.sizes["x"])], "res": res}
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  the backend
# ─────────────────────────────────────────────────────────────────────────────

def _status_code(exc):
    return getattr(exc, "http_status_code", None)


def _credit_refusal(exc):
    msg = f"{getattr(exc, 'code', '')} {exc}"
    return _status_code(exc) == 402 or re.search(r"credit|payment|insufficient funds", msg, re.I)


def _capacity_refusal(exc):
    msg = f"{getattr(exc, 'code', '')} {exc}"
    return _status_code(exc) in (429, 502, 503, 504) or re.search(
        r"concurren|too many|rate.?limit|capacity|try again", msg, re.I)


def check_account(conn, expected, log):
    from experiments.download_cube_zarr import account_email

    email = account_email(conn.describe_account())
    log(f"openEO account: {email or '(no e-mail)'}; expected {expected}")
    if not email or email.strip().lower() != expected.strip().lower():
        raise AccountMismatch(f"the connection is not {expected}'s (OPENEO_CONFIG_HOME selects the "
                              f"refresh token); nothing started")
    return email


def create_job(conn, site, year, job_options):
    from openeo.rest import OpenEoApiPlainError

    cube = conn.load_collection(COLLECTION, spatial_extent=box(site),
                                temporal_extent=list(year_window(year)), bands=list(BANDS),
                                max_cloud_cover=100)
    cube = cube.resample_spatial(resolution=RESOLUTION_M, method="near")
    try:
        return cube.save_result(format="netCDF").create_job(
            title=f"marea_v3_{site}_{year}", job_options=job_options)
    except OpenEoApiPlainError as exc:
        if _credit_refusal(exc):
            raise CreditsExhausted(f"job creation refused: {exc}") from exc
        raise


def start_job(job, log, max_wait_s):
    from openeo.rest import OpenEoApiPlainError

    t0 = time.time()
    while True:
        try:
            job.start()
            return
        except OpenEoApiPlainError as exc:
            if _credit_refusal(exc):
                raise CreditsExhausted(f"job {job.job_id} refused at start: {exc}") from exc
            if not _capacity_refusal(exc) or time.time() - t0 > max_wait_s:
                raise
            log(f"job {job.job_id} not started ({str(exc)[:200]}); retrying in "
                f"{CAPACITY_WAIT_S // 60} min")
            time.sleep(CAPACITY_WAIT_S)


def wait_job(job, log, max_wait_s):
    """Poll until the job leaves created / queued / running; its description."""
    t0, last = time.time(), None
    while True:
        try:
            info = job.describe()
        except Exception as exc:  # noqa: BLE001 - a network blip: poll again
            log(f"job {job.job_id}: status unknown ({type(exc).__name__}: {str(exc)[:150]})")
            info = {"status": last or "running"}
        status = info.get("status")
        if status in ("finished", "error", "canceled"):
            return info
        if status != last or int(time.time() - t0) % 1800 < POLL_S:
            log(f"job {job.job_id}: {status} after {dt.timedelta(seconds=int(time.time() - t0))}")
        last = status
        if time.time() - t0 > max_wait_s:
            raise TimeoutError(f"job {job.job_id} still {status} after {max_wait_s / 3600:.0f} h; it "
                               f"keeps running: rerun to reattach it")
        time.sleep(POLL_S)


def download(job, part, log):
    """The job's netCDF asset to ``part``, through a temporary file, checked."""
    assets = [a for a in job.get_results().get_assets() if str(a.name).endswith(".nc")]
    if len(assets) != 1:
        raise RuntimeError(f"job {job.job_id}: expected one netCDF asset, got "
                           f"{[a.name for a in job.get_results().get_assets()]}")
    tmp = part + ".download"
    assets[0].download(tmp)
    info = part_info(tmp)
    if info is None or not info["dates"]:
        raise RuntimeError(f"job {job.job_id}: the downloaded {tmp} has no {'/'.join(BANDS)} or no "
                           f"date (kept for inspection)")
    if abs(info["res"] - RESOLUTION_M) > 1e-6:
        raise RuntimeError(f"job {job.job_id}: grid of {info['res']} m, not {RESOLUTION_M} m (kept as {tmp})")
    os.replace(tmp, part)
    log(f"-> {part}: {len(info['dates'])} dates, {info['shape'][0]} x {info['shape'][1]} px, "
        f"{os.path.getsize(part) / 1e6:.0f} MB")
    return info


def update_overpass(site, year, dates, log):
    """Add the overpass times of ``dates`` to the site's overpass file (the catalogue)."""
    import pandas as pd

    import pyintertidal as pit

    path = overpass_path(site)
    have = json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else {}
    missing = [d for d in dates if d not in have]
    if not missing:
        return []
    if missing:
        pit.net.use_system_certificates()
        fresh = pit.overpass.get_overpass_times(box(site), (f"{year}-01-01", f"{year}-12-31"),
                                                verbose=False)
        for d in missing:
            if d in fresh:
                have[d] = pd.Timestamp(fresh[d]).to_pydatetime().isoformat()
        write_json(dict(sorted(have.items())), path)
    still = [d for d in dates if d not in have]
    log(f"overpass times: {len(dates) - len(still)} of {len(dates)} dates in {path}"
        + (f"; STILL MISSING {still}" if still else ""))
    return still


def fetch(conn, site, year, a, log):
    """One part: skipped when complete; else its recorded job reattached, or a
    new job created (its id recorded first), waited for and downloaded."""
    from openeo.rest import OpenEoApiPlainError

    part = part_path(site, year)
    have = part_info(part)
    if have is not None:
        log("already downloaded")
        if update_overpass(site, year, have["dates"], log):   # a catalogue outage last time
            return "failed"
        return "cached"
    led = load_ledger(site, year)
    job, status = None, None
    if led["jobs"]:
        rec = led["jobs"][-1]
        try:
            job = conn.job(rec["job_id"])
            status = job.status()
        except OpenEoApiPlainError as exc:
            if _status_code(exc) != 404:
                raise
            status = "gone"
        rec["status"] = status
        write_json(led, ledger_path(site, year))
        if status in ("error", "canceled", "gone"):
            if status == "error" and not a.retry_failed:
                log(f"job {rec['job_id']} ended in error; not paid again without --retry-failed")
                return "failed"
            job = None
        else:
            log(f"reattached job {rec['job_id']} ({status})")
    if job is None:
        job = create_job(conn, site, year, a.job_options)
        led["jobs"].append({"job_id": job.job_id, "title": f"marea_v3_{site}_{year}",
                            "created_utc": utcnow(), "status": "created", "account": a.account,
                            "job_options": a.job_options})
        write_json(led, ledger_path(site, year))
        status = "created"
        log(f"job {job.job_id} created")
    if status == "created":
        start_job(job, log, a.max_wait_h * 3600)
    info = wait_job(job, log, a.max_wait_h * 3600)
    rec = led["jobs"][-1]
    rec.update(status=info.get("status"), costs=info.get("costs"), usage=info.get("usage"),
               updated_utc=utcnow())
    write_json(led, ledger_path(site, year))
    log(f"job {job.job_id}: {info.get('status')}, {info.get('costs')} credits")
    if info.get("status") != "finished":
        try:
            for e in job.logs(level="error")[:20]:
                log(f"  backend error: {str(e.get('message', e))[:300]}")
        except Exception:  # noqa: BLE001 - the logs are a courtesy
            pass
        return "failed"
    got = download(job, part, log)
    led.update(part=part, dates=len(got["dates"]), shape=got["shape"], bytes=os.path.getsize(part),
               first=got["dates"][0], last=got["dates"][-1], downloaded_utc=utcnow())
    write_json(led, ledger_path(site, year))
    update_overpass(site, year, got["dates"], log)
    return "downloaded"


# ─────────────────────────────────────────────────────────────────────────────
#  merge, status, dry run
# ─────────────────────────────────────────────────────────────────────────────

def merge(site, years, log=print):
    """The parts of ``years`` into one cube, as download_cube.merge_parts (same
    encoding, grids identical, dates strictly increasing), the parts kept."""
    import xarray as xr

    parts = [part_path(site, y) for y in years]
    bad = [p for p in parts if part_info(p) is None]
    if bad:
        raise FileNotFoundError(f"parts missing or incomplete: {bad}")
    out = merged_path(site, years[0], years[-1])
    dss = [xr.open_dataset(p, chunks={}) for p in parts]
    tdim = "t" if "t" in dss[0].dims else "time"
    for p, d in zip(parts[1:], dss[1:]):
        assert np.array_equal(d["x"].values, dss[0]["x"].values), f"{p}: x grid differs"
        assert np.array_equal(d["y"].values, dss[0]["y"].values), f"{p}: y grid differs"
    ds = xr.concat(dss, dim=tdim, data_vars="minimal", coords="minimal", compat="override")
    t = ds[tdim].values
    assert np.all(np.diff(t) > np.timedelta64(0, "ns")), "dates not strictly increasing across years"
    enc = {}
    for v in BANDS:
        e = dss[0][v].encoding
        enc[v] = {k: e[k] for k in ("dtype", "zlib", "complevel", "shuffle", "chunksizes", "_FillValue")
                  if k in e}
    te = dss[0][tdim].encoding
    enc[tdim] = {k: te[k] for k in ("units", "calendar", "dtype") if k in te}
    tmp = out + ".merge"
    ds.chunk({tdim: 1}).to_netcdf(tmp, encoding=enc)
    for d in dss:
        d.close()
    with xr.open_dataset(tmp) as chk:
        assert chk.sizes[tdim] == len(t), "merged cube lost scenes"
    os.replace(tmp, out)
    log(f"[{site}] merged {years[0]}-{years[-1]} -> {out} ({len(t)} scenes; parts kept)")
    return out


def status(queue, log=print):
    total = 0.0
    for site, year in queue:
        info = part_info(part_path(site, year))
        led = load_ledger(site, year)
        costs = sum(float(j.get("costs") or 0) for j in led["jobs"])
        total += costs
        if info is not None:
            what = f"downloaded, {len(info['dates'])} dates, {info['shape'][0]} x {info['shape'][1]} px"
        elif led["jobs"]:
            j = led["jobs"][-1]
            what = f"job {j['job_id']} {j.get('status')} (as last recorded)"
        else:
            what = "not started"
        log(f"{site:10s} {year}  {what}  [{costs:g} credits]")
    log(f"credits recorded in these ledgers: {total:g}")


def dry_run(queue, log=print):
    lo = hi = 0.0
    for site, year in queue:
        b = box(site)
        r, c = box_px(b)
        mpx = r * c / 1e6
        lo += mpx * CREDITS_PER_MPX_YEAR[0]
        hi += mpx * CREDITS_PER_MPX_YEAR[1]
        log(f"{site:10s} {year}  {year_window(year)}  box {b}  ~{r} x {c} px  "
            f"~{mpx * CREDITS_PER_MPX_YEAR[0]:.0f}-{mpx * CREDITS_PER_MPX_YEAR[1]:.0f} credits  "
            f"-> {part_path(site, year)}")
    log(f"{len(queue)} parts, ~{lo:.0f}-{hi:.0f} credits (3-band rate of 2026-10-26/27)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue", nargs="*", help=f"<site>:<year or y0-y1>, in order; sites {sorted(BOXES)}")
    ap.add_argument("--expect-account", default=None, metavar="E-MAIL",
                    help="stop before any job unless the openEO account is this one")
    ap.add_argument("--retry-failed", action="store_true",
                    help="replace a part's job that ended in error by a new one (credits again)")
    ap.add_argument("--job-options", default=None, help=f"JSON (default {json.dumps(JOB_OPTIONS)})")
    ap.add_argument("--max-wait-h", type=float, default=24.0,
                    help="give up waiting for one job after this long (it keeps running)")
    ap.add_argument("--merge", nargs=2, metavar=("SITE", "Y0-Y1"), default=None,
                    help="merge the parts of those years (no network)")
    ap.add_argument("--status", action="store_true", help="the state of each part of the queue")
    ap.add_argument("--dry-run", action="store_true", help="the queue and a credit estimate")
    a = ap.parse_args(argv)
    a.job_options = json.loads(a.job_options) if a.job_options else dict(JOB_OPTIONS)
    if a.merge:
        merge(a.merge[0], parse_years(a.merge[1]))
        return 0
    queue = parse_queue(a.queue)
    if not queue:
        ap.error("an empty queue")
    if a.status:
        status(queue)
        return 0
    if a.dry_run:
        dry_run(queue)
        return 0

    import pyintertidal as pit

    def say(site, year):
        return lambda m: print(f"{utcnow()} [{site} {year}] {m}", flush=True)

    pit.net.use_system_certificates()
    conn = pit.scenes.connect(interactive=False)
    a.account = check_account(conn, a.expect_account, print) if a.expect_account else None
    done = {}
    for site, year in queue:
        log = say(site, year)
        try:
            done[(site, year)] = fetch(conn, site, year, a, log)
        except CreditsExhausted as exc:
            log(f"STOP, credits: {exc}. Rerun the same command when the credits renew; finished parts "
                f"are skipped and recorded jobs reattached")
            done[(site, year)] = "credits"
            break
        except Exception as exc:  # noqa: BLE001 - one part's failure does not stop the queue
            log(f"FAILED: {type(exc).__name__}: {exc}")
            done[(site, year)] = "failed"
    for (site, year), r in done.items():
        print(f"{site:10s} {year}  {r}")
    left = [f"{s}:{y}" for s, y in queue if (s, y) not in done]
    if left:
        print(f"not reached: {' '.join(left)}")
    return 0 if all(r in ("cached", "downloaded") for r in done.values()) and not left else 1


if __name__ == "__main__":
    sys.exit(main())
