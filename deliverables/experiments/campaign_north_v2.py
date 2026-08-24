"""MAREA north-coast campaign: submit everything, harvest, process, purge.

The naive runner (v1) downloaded and processed one cell at a time, which
takes days: the local link is the bottleneck. The key fact this version
exploits is that openEO batch jobs run SERVER-side — so every cell can be
submitted at once, Copernicus prepares the cubes in parallel under its own
account limits, and the local machine only harvests finished jobs and
processes them.

The four moving parts, in the order a cell experiences them:

1. submit    build the openEO process graph and start the job; never wait.
2. harvest   poll the pending jobs; download each finished cube
             (sequentially: one connection — bandwidth is shared and two
             concurrent downloads caused the OOM incident in the record).
3. process   a small pool of subprocesses runs
             ``pyintertidal.marea.reconstruct`` on downloaded cubes —
             the same single function every cell gets, no per-site tuning.
4. purge     a successfully processed cube is deleted (130 cubes would
             not fit on this disk); ``--keep-cubes`` disables that.

Everything is resumable: the manifest (``runs/campaign_jobs.json``) records
each cell's job id and state, and a finished cell is recognised by its
``runs/<cell>/marea/result.json``. Kill the process at any point — power
cut, suspend, Ctrl-C — and rerunning the same command continues where it
stopped, resubmitting nothing that is already queued, downloaded or done.

Usage:

    python -m experiments.campaign_north_v2 [options]

Options (all have sensible defaults; no environment variables are read):

    --cells PATH     cell definitions (default north_coast_cells.json)
    --workers N      processing subprocesses (default 4)
    --keep-cubes     do not delete cubes after successful processing
    --years A B      temporal extent (default 2023 2025, inclusive)
"""
import argparse
import json
import os
import subprocess
import sys
import time

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
os.chdir(repo_root)

log_path = os.path.join("runs", "campaign.log")
manifest_path = os.path.join("runs", "campaign_jobs.json")

# Measured on this account, 2026-08-20: the 31st concurrent job is refused
# with "[400] ConcurrentJobLimit". After hitting it we pause submissions
# for a couple of minutes instead of hammering the API.
submit_pause_after_limit_s = 120

# The backend also rate-limits bursts with HTTP 429; poll gently.
poll_interval_s = 45


def log(message):
    """Append one timestamped line to the campaign log and to stdout."""
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="MAREA campaign runner (see module docstring)")
    p.add_argument("--cells", default="north_coast_cells.json",
                   help="JSON list of {name, km2, polygon} cells")
    p.add_argument("--workers", type=int, default=4,
                   help="processing subprocesses (RAM-bound: each peaks "
                        "at 1-3 GB on a large cell)")
    p.add_argument("--keep-cubes", action="store_true",
                   help="keep downloaded cubes instead of purging them "
                        "after a successful reconstruction")
    p.add_argument("--years", nargs=2, type=int, default=[2023, 2025],
                   metavar=("FIRST", "LAST"),
                   help="inclusive year range of imagery to request")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("runs", exist_ok=True)

    # SSL first (corporate-TLS Windows machines fail otherwise), then the
    # authenticated connection. interactive=False -> refresh-token only,
    # so the campaign can run headless and survive relaunches.
    from pyintertidal.net import use_system_certificates
    use_system_certificates()
    from pyintertidal.scenes import connect
    from pyintertidal.aoi import AOI

    cells = {c["name"]: c for c in
             json.load(open(args.cells, encoding="utf-8"))}
    manifest = (json.load(open(manifest_path, encoding="utf-8"))
                if os.path.exists(manifest_path) else {})
    connection = connect(interactive=False)
    log(f"CAMPAIGN v2: {len(cells)} cells, {args.workers} workers, "
        f"purge={'no' if args.keep_cubes else 'yes'}")

    def save_manifest():
        json.dump(manifest, open(manifest_path, "w", encoding="utf-8"),
                  indent=1)

    def is_done(name):
        """A cell is finished when its result file exists — the manifest
        alone is not trusted, so deleting a result forces a re-run."""
        return os.path.exists(f"runs/{name}/marea/result.json")

    def cube_path(name):
        return f"ndwi_cube_{name}.nc"

    # ── 1. SUBMIT ────────────────────────────────────────────────────────
    # Queue of cells that still need a server job. Drained continuously
    # from the main loop: every harvested job frees one of the ~30 slots.
    to_submit = [name for name in cells
                 if not is_done(name)
                 and not os.path.exists(cube_path(name))
                 and manifest.get(name, {}).get("estado")
                 not in ("enviado", "descargado")]
    last_limit_hit = 0.0

    def try_submit_next():
        """Submit at most one cell; called every loop iteration."""
        nonlocal last_limit_hit
        if not to_submit:
            return
        if time.time() - last_limit_hit < submit_pause_after_limit_s:
            return                       # the account was full a moment ago
        name = to_submit[0]
        for attempt in range(4):
            try:
                aoi = AOI.from_polygon(cells[name]["polygon"], name=name)
                # The polygon defines the cell; the request itself uses its
                # bounding box (masks confine the analysis later, and small
                # gaps between neighbouring polygons stay covered).
                cube = connection.load_collection(
                    "SENTINEL2_L2A", spatial_extent=aoi.bbox,
                    temporal_extent=[f"{args.years[0]}-01-01",
                                     f"{args.years[1]}-12-31"],
                    bands=["B03", "B08", "SCL"], max_cloud_cover=100)
                try:
                    # B03/B08 are native 10 m but SCL is 20 m; resampling
                    # server-side keeps the three bands on one grid.
                    cube = cube.resample_spatial(resolution=10,
                                                 method="near")
                except Exception:
                    pass                 # older backends lack the process
                job = cube.save_result(format="netCDF").create_job(
                    title=f"marea_{name}")
                job.start()
                manifest[name] = {"job": job.job_id, "estado": "enviado"}
                save_manifest()
                log(f"SUBMITTED {name} -> {job.job_id} "
                    f"({len(to_submit) - 1} left to queue)")
                to_submit.pop(0)
                time.sleep(2.0)          # be gentle: burst submits draw 429s
                return
            except Exception as e:
                message = str(e)[:160]
                if "ConcurrentJobLimit" in message:
                    last_limit_hit = time.time()
                    return               # no slot free; retried later
                if "429" in message:
                    time.sleep(20 * (attempt + 1))
                    continue             # rate-limited: back off and retry
                log(f"ERROR submit {name}: {message}")
                to_submit.pop(0)         # a broken cell must not block
                return

    # Fill the account up to its concurrent limit before harvesting.
    for _ in range(len(to_submit)):
        before = len(to_submit)
        try_submit_next()
        if len(to_submit) == before:     # limit reached (or repeated error)
            break

    # ── 2-4. HARVEST, PROCESS, PURGE ─────────────────────────────────────
    workers = {}                         # cell name -> Popen

    def reap_finished_workers():
        """Collect finished subprocesses; log the verdict; purge cubes."""
        for name in [n for n, p in workers.items() if p.poll() is not None]:
            return_code = workers.pop(name).returncode
            verdict = "?"
            result_file = f"runs/{name}/marea/result.json"
            if os.path.exists(result_file):
                try:
                    result = json.load(open(result_file, encoding="utf-8"))
                    verdict = result.get("estado", "?")
                    if verdict == "ok":
                        verdict += (" OPERATOR" if result.get("con_operador")
                                    else " identity")
                except Exception:
                    pass
            log(f"DONE {name}: rc={return_code} {verdict}")
            if (return_code == 0 and verdict.startswith("ok")
                    and not args.keep_cubes):
                try:
                    os.remove(cube_path(name))
                    log(f"PURGED cube {name}")
                except OSError:
                    pass

    def start_processing(name):
        """Hand one downloaded cube to a reconstruction subprocess.

        A subprocess rather than a thread: reconstruct is NumPy-heavy and
        holds the cube in memory, so isolation both parallelises it and
        makes one cell's crash harmless to the campaign.
        """
        out_dir = f"runs/{name}/marea"
        code = (f"from pyintertidal import marea; "
                f"marea.reconstruct({cube_path(name)!r}, {out_dir!r}, "
                f"name={name!r})")
        workers[name] = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log(f"PROCESSING {name}")

    # Cubes already on disk (from an interrupted run) skip the queue.
    waiting_local = [name for name in cells
                     if os.path.exists(cube_path(name))
                     and not is_done(name)]

    while True:
        reap_finished_workers()
        try_submit_next()
        while waiting_local and len(workers) < args.workers:
            start_processing(waiting_local.pop(0))

        pending = [name for name, entry in manifest.items()
                   if entry.get("estado") == "enviado" and not is_done(name)]
        if (not pending and not waiting_local and not workers
                and not to_submit):
            break                        # nothing left anywhere: finished

        for name in pending:
            if len(workers) >= args.workers:
                break                    # no point downloading faster than
                                         # we can process
            try:
                job = connection.job(manifest[name]["job"])
                status = job.status()
            except Exception as e:
                # Network blip or expired session: reconnect once and let
                # the next iteration retry the whole pending list.
                log(f"ERROR status {name}: {str(e)[:120]}")
                try:
                    connection = connect(interactive=False)
                except Exception:
                    time.sleep(60)
                break
            if status == "finished":
                try:
                    log(f"DOWNLOADING {name}...")
                    job.get_results().download_file(cube_path(name))
                    manifest[name]["estado"] = "descargado"
                    save_manifest()
                    start_processing(name)
                except Exception as e:
                    log(f"ERROR download {name}: {str(e)[:160]}")
            elif status in ("error", "canceled"):
                log(f"JOB FAILED {name}: {status}")
                manifest[name]["estado"] = f"job_{status}"
                save_manifest()

        time.sleep(poll_interval_s)

    log("CAMPAIGN v2 COMPLETE")


if __name__ == "__main__":
    main()
