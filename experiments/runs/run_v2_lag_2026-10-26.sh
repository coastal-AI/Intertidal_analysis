#!/usr/bin/env bash
# >>> archive header
# Archived run script (this header was added when the script was archived;
# without the lines from '>>> archive header' to '<<< archive header', the
# file is byte for byte the script that ran).
#   what  : validation v2, Stage L, the lag benchmark: fetch stac-meta, emodnet
#           and the tide models; the positive control; plan --design lag; 9
#           workers; score --pre-lock --show-unvalidated (W3 stays BLINDED)
#   where : nserver1, /media/HDD/jalias/Intertidal_analysis
#   when  : 2026-10-26, 08:32:21 to 09:41:19 UTC (results/validation_v2/logs/STATUS_lag)
#   commit: 82121c9, the HEAD during the run (clean; scores/prelock_ledger.json);
#           the comment below names d800fc9, the commit that added Stage L
#   exact : this is the exact script used; sha256 of the original file: 2e9d2da2b6d590d5f3f36b26d28d4f6db5be22119189081a8aedc852d5c99a70
#   viewer: notebooks/validation/02_lag_benchmark.ipynb (built by
#           notebooks/validation/_build_02_lag_benchmark.py) re-checks that hash
# <<< archive header
# Validation v2, Stage L (the lag benchmark, commit d800fc9) on the V1
# packages: free inputs (STAC orbits, EMODnet depths, tide-model check), the
# positive control, the plan, the workers (partial-year sites, Wadden, then
# Villaviciosa) and the pre-lock lag table. Nothing is frozen (lock-A is the
# user's decision); W3 stays blinded in the pre-lock score.
# Run by Claude with the user's permission, 2026-10-26.
set -u -o pipefail
cd /media/HDD/jalias/Intertidal_analysis || exit 1
source env.sh >/dev/null 2>&1
export PYTHONUNBUFFERED=1
unset VALIDATION_V2_ROOT VALIDATION_V2_SPACE_MARGIN_GB
L=results/validation_v2/logs
mkdir -p "$L"
st() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L/STATUS_lag"; }
V2() { "$PY" -X utf8 -m experiments.validation_v2 "$@"; }
st "BEGIN lag benchmark (git $(git log --oneline -1 | cut -c1-60))"
V2 status --sites villaviciosa,wadden,ems,escalda,ferrol > "$L/status_before.log" 2>&1
V2 fetch stac-meta --sites escalda,wadden,ems,ferrol,villaviciosa > "$L/fetch_stac_meta.log" 2>&1 || st "fetch stac-meta failed (L01v groups by date phase)"
V2 fetch emodnet --sites escalda,wadden,ems,ferrol > "$L/fetch_emodnet.log" 2>&1 || st "fetch emodnet failed (L18 not computable)"
V2 fetch tides > "$L/fetch_tides.log" 2>&1 || st "fetch tides reported a problem"
if V2 control --kind lag --world gating --scenario both > "$L/control.log" 2>&1; then st "OK control"; else st "control FAILED (rows print as 'implementation not validated')"; fi
V2 plan --design lag > "$L/plan.log" 2>&1 || { st "STOPPED: plan failed"; exit 1; }
st "OK plan"
run_w() { local k=$1 sites=$2; V2 run --plans lag --worker-id "$k" --threads 4 --boot-threads 1 --ram-margin-gb 16 --sites "$sites" > "$L/lag_w$k.log" 2>&1; st "END worker $k ($sites, exit $?)"; }
for k in 1 2 3; do run_w $k escalda,ems,ferrol & done
for k in 4 5; do run_w $k wadden & done
wait
st "partial sites and first Wadden pass done"
for k in 6 7; do run_w $k wadden & done
for k in 8 9; do run_w $k villaviciosa & done
wait
V2 status --sites escalda,wadden,ems,ferrol,villaviciosa > "$L/status_after.log" 2>&1
if V2 score --pre-lock --show-unvalidated > "$L/score_lag.log" 2>&1; then st "OK score"; else st "score FAILED (see $L/score_lag.log)"; fi
st "DONE lag benchmark"
