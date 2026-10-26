#!/usr/bin/env bash
#
# ARCHIVED RUN SCRIPT (header added 2026-10-26; nothing below it was changed)
# ---------------------------------------------------------------------------
# What: the exact script that ran the NIDEM scene-count curves of wadden and
#   ems to completion (experiments/v15_nidem_scene_curves.py run <site>).
# Where: the lab server nserver1, in /media/HDD/jalias/Intertidal_analysis,
#   from its copy results/scene_count_curves/logs_v15/run_v15.sh, started with
#   the sites "wadden ems" on 2026-10-26T08:32:21Z (STATUS: "BEGIN v15 fix
#   wadden ems"); both sites ended with exit 0 at 09:18:34Z (ems) and
#   09:21:40Z (wadden), status complete (53/53 and 17/17 rows).
# Commit: git 82121c9 (82121c935bdc81df0d70b5a9f9c4fb911c46cc02); the script
#   checks v14 sha256 ef0f9a2e7b4f... and v15 sha256 370ebf8117b1..., the
#   v15 committed in 82121c9 (its own comment below still says
#   "uncommitted ... 2026-10-20" from an earlier use).
# Not covered: the NIDEM rows of villaviciosa and escalda, and the first 26
#   wadden and 13 ems rows, come from the run of 2026-10-20 (git 8b4737d) with
#   an earlier, uncommitted v15 (sha256 6d682cbaf089...) and an earlier version
#   of this script; the 2026-10-26 run resumed those CSVs and appended the rest.
# Source of this copy: results/scene_count_curves/server_2026-10-26/logs/run_v15.sh
#   (sha256 7a77dfd8969d0cb5a2050038a126c148641f38163650e29fa1fb6c3227371f31); line 1 is its
#   shebang, the lines after this header are its lines 2 onward, unchanged.
# Notebook: notebooks/validation/03_scene_count_curves_marea_nidem.ipynb.
# ---------------------------------------------------------------------------
#
# NIDEM on the same random scene subsets as the MAREA scene-count curves
# (experiments/v15_nidem_scene_curves.py, uncommitted, copied from the laptop;
# sha256 checked). The sites given (default: all four) run in parallel, their
# cube block reads serialised by a lock file; every gate runs before any
# curve row. Each site's CSV resumes: a finished site redoes only phase 1
# and its gates (with the full-frame audit) and leaves its CSV untouched; an
# unfinished one appends the missing rows. Outputs only
# results/scene_count_curves/<site>_nidem_all.{csv,_settings.json,.run.lock}
# and this folder (logs <site>_nidem_all.fix.log, STATUS). Nothing is deleted.
# Run by Claude with the user's permission, 2026-10-20.
#
# Copy to results/scene_count_curves/logs_v15/run_v15.sh and start it with
#   tmux new-session -d -s v15 'bash results/scene_count_curves/logs_v15/run_v15.sh [site ...]'
set -u
cd /media/HDD/jalias/Intertidal_analysis || exit 1
source env.sh >/dev/null 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=results/scene_count_curves/logs_v15
mkdir -p "$LOG"
st() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG/STATUS"; }
want14=ef0f9a2e7b4f9b02f78d19fb22bed07a3b91ed5e8a4a59615e8333cfd4de5caf
want15=370ebf8117b196b0c964a64b14ef2f4df705d9778bab8bfb1f98b07dc44c1475
[ "$(sha256sum experiments/v14_scene_count_curves.py | cut -d' ' -f1)" = "$want14" ] || { st "STOPPED: v14 sha256"; exit 1; }
[ "$(sha256sum experiments/v15_nidem_scene_curves.py | cut -d' ' -f1)" = "$want15" ] || { st "STOPPED: v15 sha256"; exit 1; }
SITES=("$@")
[ "${#SITES[@]}" -gt 0 ] || SITES=(villaviciosa escalda wadden ems)
declare -A W=([villaviciosa]=2 [escalda]=2 [wadden]=4 [ems]=3)
st "BEGIN v15 fix ${SITES[*]} (v15 ${want15:0:12}, git $(git log --oneline -1 | cut -c1-50))"
for s in "${SITES[@]}"; do
  [ -n "${W[$s]:-}" ] || { st "SKIPPED unknown site $s"; continue; }
  ( echo "===== $(date -u +%FT%TZ) run_v15.sh, v15 ${want15:0:12} =====" >> "$LOG/${s}_nidem_all.fix.log"
    "$PY" -X utf8 -m experiments.v15_nidem_scene_curves run "$s" --workers "${W[$s]}" \
        --io-lock "$LOG/cube_read.lock" --max-minutes 100 >> "$LOG/${s}_nidem_all.fix.log" 2>&1
    st "END $s (exit $?)" ) &
done
wait
st "DONE v15 fix ${SITES[*]}"
for s in "${SITES[@]}"; do
  grep -hE "\] gate [a-z0-9_]+: FAILED|CRASHED|Traceback|\] (complete|stopped_by_time|incomplete):" \
      "$LOG/${s}_nidem_all.fix.log" 2>/dev/null | tail -n 3 | cut -c1-300 | tee -a "$LOG/STATUS"
done
