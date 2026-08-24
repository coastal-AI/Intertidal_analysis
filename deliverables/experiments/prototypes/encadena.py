"""Wait for the study notebook, then run the full campaign.

Not launched in parallel: this machine has 7.4 GB of RAM, 1.3 GB of it free
right now, and a single HSR fit peaks at 2.9 GB. Two of these at once would
swap, and a swapping fit is not slow — it is a fit that may never finish.

The smoke test is two real cells, downloads and all. That is the step the
plan calls for before committing 116 openEO jobs, and it is the only way to
learn what a cell actually costs and whether an empty one is reported as
`empty` rather than crashing.
"""
import io
import os
import subprocess
import sys
import time

import nbformat

# This script prints text read back from another process's log, which can
# contain replacement characters. On a cp1252 console that kills the print —
# and with it the campaign, hours before anyone notices. pyintertidal fixes
# its own streams on import, but this runs before that import.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="backslashreplace")
    except Exception:
        pass

ROOT = r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
LOG = os.path.join(SC, "ejecuta_nb_log.txt")
CAMPAIGN = os.path.join(ROOT, "campaign_north_coast.ipynb")
os.chdir(ROOT)

# ── wait for the study notebook to write itself back ─────────────────────
print("waiting for the study notebook to finish…", flush=True)
deadline = time.time() + 3 * 3600
while time.time() < deadline:
    try:
        text = open(LOG, encoding="utf-8", errors="replace").read()
    except OSError:
        text = ""
    if "written back after" in text:
        tail = [l for l in text.splitlines() if l.strip()][-6:]
        print("study notebook finished:", flush=True)
        for line in tail:
            print("   ", line[:110], flush=True)
        break
    time.sleep(30)
else:
    print("timed out waiting; not starting the campaign", flush=True)
    sys.exit(1)

# Give the kernel a moment to release its memory before the next one starts.
time.sleep(20)

# ── run the campaign notebook over every cell ────────────────────────────
print("\nstarting the FULL north-coast campaign (116 cells)", flush=True)
nb = nbformat.read(io.open(CAMPAIGN, encoding="utf-8"), as_version=4)
for c in nb.cells:
    if c.cell_type == "code" and "MAX_CELLS" in c.source:
        # Full campaign. The first cells still act as the smoke test: the
        # manifest is written after each one and a failure isolates that cell
        # instead of stopping the run, so a bad configuration shows up in the
        # first two results rather than after eighty downloads.
        break

from nbclient import NotebookClient

t0 = time.time()
client = NotebookClient(nb, timeout=72 * 3600, kernel_name="python3",
                        resources={"metadata": {"path": ROOT}},
                        allow_errors=True)
try:
    client.execute()
finally:
    out = os.path.join(ROOT, "runs", "campaign_run.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    nbformat.write(nb, io.open(out, "w", encoding="utf-8"))
    print(f"\ncampaign notebook written to {out} "
          f"after {(time.time() - t0) / 60:.1f} min", flush=True)

for i, c in enumerate(nb.cells):
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            print(f"  cell {i}: {o.get('ename')}: "
                  f"{(o.get('evalue') or '')[:200]}", flush=True)
        elif o.get("output_type") == "stream":
            for line in "".join(o["text"]).strip().splitlines():
                if any(k in line for k in ("cells ·", "SMOKE", "done", "empty",
                                           "failed", "openEO job", "evicted")):
                    print("   ", line[:120], flush=True)
