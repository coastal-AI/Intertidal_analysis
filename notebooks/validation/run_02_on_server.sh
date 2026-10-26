#!/usr/bin/env bash
# Runs 02_lag_vs_gauges_site.ipynb for every site (one after the other, the site passed in
# NB_SITE), then 02b_lag_summary.ipynb.
#
#   bash notebooks/validation/run_02_on_server.sh                    # the five sites
#   bash notebooks/validation/run_02_on_server.sh ferrol escalda     # some of them
#
# Executed copies: results/lag_nb/02_<site>.executed.ipynb and 02b_summary.executed.ipynb (+ .html
# when nbconvert is installed); the per-site CSVs and figures go to the same folder (NB_OUT). Logs
# (the cells' printed output as it comes) and STATUS: logs/v2/claude/nb02/ (NB_LOG). The notebooks
# are read from notebooks/validation/ (NB_DIR). A site that fails is recorded in STATUS and the
# next one still runs. Light on a shared server: 2 BLAS threads, nice 10.
set -u
cd "$(dirname "$0")/../.." || exit 1
source env.sh > /dev/null

SITES=${*:-ferrol escalda ems wadden villaviciosa}
NB=${NB_DIR:-notebooks/validation}
OUT=${NB_OUT:-results/lag_nb}
LOG=${NB_LOG:-logs/v2/claude/nb02}
export NB_OUT=$OUT                                   # the notebooks write there
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$OUT" "$LOG"
STATUS=$LOG/STATUS
echo "started $(date -Is) on $(hostname): sites $SITES, commit $(git rev-parse --short HEAD)" > "$STATUS"

# run_nb <notebook> <executed copy> <log>: every cell in a fresh kernel from the repository root,
# the printed output streamed to the log; the executed copy is saved even when a cell fails
run_nb() {
    nice -n 10 "$PY" -X utf8 - "$1" "$2" > "$3" 2>&1 <<'EOF'
import sys, time
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


class LoggedClient(NotebookClient):
    def output(self, outs, msg, display_id, cell_index):
        if msg["msg_type"] == "stream":
            print(msg["content"]["text"], end="", flush=True)
        return super().output(outs, msg, display_id, cell_index)


src, dst = sys.argv[1], sys.argv[2]
nb = nbformat.read(src, as_version=4)
t0, ok = time.time(), True
try:
    LoggedClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
except CellExecutionError as e:
    ok = False
    print(e)
finally:
    nbformat.write(nb, dst)
try:
    from nbconvert import HTMLExporter
    body, _ = HTMLExporter().from_notebook_node(nb)
    open(dst.replace(".ipynb", ".html"), "w", encoding="utf-8").write(body)
except ImportError:
    print("nbconvert not installed: no HTML")
print(f"\n{'OK' if ok else 'FAILED'} in {(time.time() - t0) / 60:.1f} min -> {dst}")
sys.exit(0 if ok else 1)
EOF
}

for site in $SITES; do
    t0=$(date +%s)
    echo "$(date -Is) $site running" >> "$STATUS"
    if NB_SITE=$site run_nb "$NB/02_lag_vs_gauges_site.ipynb" "$OUT/02_$site.executed.ipynb" \
            "$LOG/02_$site.log"; then
        echo "$(date -Is) $site ok ($(( ($(date +%s) - t0) / 60 )) min)" >> "$STATUS"
    else
        echo "$(date -Is) $site FAILED ($(( ($(date +%s) - t0) / 60 )) min), see $LOG/02_$site.log" >> "$STATUS"
    fi
done

if run_nb "$NB/02b_lag_summary.ipynb" "$OUT/02b_summary.executed.ipynb" "$LOG/02b_summary.log"; then
    echo "$(date -Is) summary ok" >> "$STATUS"
else
    echo "$(date -Is) summary FAILED, see $LOG/02b_summary.log" >> "$STATUS"
fi
echo "finished $(date -Is)" >> "$STATUS"
