#!/usr/bin/env bash
# Runs 03_marea_vs_nidem_scene_count_site.ipynb for the four sites (one after
# the other, the site passed in NB_SITE), then 03b_marea_vs_nidem_summary.ipynb.
#
#   bash notebooks/validation/run_03_on_server.sh                # the four sites
#   bash notebooks/validation/run_03_on_server.sh escalda ems    # some of them
#
# Executed copies: results/scene_count_curves_nb/03_<site>.executed.ipynb
# (+ .html when nbconvert is installed). Logs and STATUS: logs/v2/claude/nb03/.
# A site that fails is recorded in STATUS and the next one still runs.
cd "$(dirname "$0")/../.."
source env.sh > /dev/null
set -u
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2   # as the server run of v14

SITES=${*:-villaviciosa escalda wadden ems}
OUT=results/scene_count_curves_nb
LOG=logs/v2/claude/nb03
mkdir -p "$OUT" "$LOG"
STATUS=$LOG/STATUS
echo "started $(date -Is) on $(hostname), sites: $SITES" > "$STATUS"

run_nb() {   # notebook, executed copy: nbclient from the repository root, the environment inherited
    "$PY" -X utf8 - "$1" "$2" <<'EOF'
import sys
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

src, dst = sys.argv[1], sys.argv[2]
nb = nbformat.read(src, as_version=4)
failed = None
try:
    NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
except CellExecutionError as e:          # the outputs up to the failing cell are kept
    failed = e
nbformat.write(nb, dst)
try:
    from nbconvert import HTMLExporter
    body, _ = HTMLExporter().from_notebook_node(nb)
    open(dst.replace(".ipynb", ".html"), "w", encoding="utf-8").write(body)
except ImportError:
    print("nbconvert not installed: no HTML")
if failed is not None:
    print(str(failed)[-3000:])
    sys.exit(1)
EOF
}

for site in $SITES; do
    t0=$(date +%s)
    echo "$(date -Is) $site running" >> "$STATUS"
    if NB_SITE=$site run_nb notebooks/validation/03_marea_vs_nidem_scene_count_site.ipynb \
            "$OUT/03_$site.executed.ipynb" > "$LOG/03_$site.log" 2>&1; then
        echo "$(date -Is) $site ok ($(( ($(date +%s) - t0) / 60 )) min)" >> "$STATUS"
    else
        echo "$(date -Is) $site FAILED ($(( ($(date +%s) - t0) / 60 )) min), see $LOG/03_$site.log" >> "$STATUS"
    fi
done

if run_nb notebooks/validation/03b_marea_vs_nidem_summary.ipynb "$OUT/03b_summary.executed.ipynb" \
        > "$LOG/03b_summary.log" 2>&1; then
    echo "$(date -Is) summary ok" >> "$STATUS"
else
    echo "$(date -Is) summary FAILED, see $LOG/03b_summary.log" >> "$STATUS"
fi
echo "finished $(date -Is)" >> "$STATUS"
