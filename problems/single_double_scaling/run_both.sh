#!/usr/bin/env bash
#
# run_both.sh
# Run the single- and double-precision builds one after the other on the GPU,
# so each run has the GPU to itself (clean timing for scaling_diagnostics.py).
#
# Each run is started through its own build directory's run_file.sh, so outputs
# land in the usual places:
#   build_gpu_single/tcgm_single/
#   build-gpu/tcgm_double/
#
# Usage (from this directory): ./run_both.sh [input.ini] [extra idefix options...]
#   input.ini      parameter file (default: ./idefix.ini)
#   extra options  passed to BOTH runs, e.g. -maxcycles 100
#
# Environment:
#   FOREGROUND=1   run in the terminal instead of in the background
#   FORCE=1        overwrite existing outputs in both run folders
#   ORDER="double single"   change the run order (default: single first)
#
# Examples:
#   ./run_both.sh                                         # full runs, in background
#   FOREGROUND=1 ./run_both.sh idefix.ini -maxcycles 100  # quick smoke test
#
# Stop: kill the PID in run_both.pid (stops the queue), then touch <run dir>/stop
#       to make the idefix run currently in progress write a dump and exit.
#

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

INPUT="${1:-idefix.ini}"
if [ $# -ge 1 ]; then shift; fi
EXTRA_ARGS=("$@")

if [ ! -f "$INPUT" ]; then
    echo "Error: input file $INPUT not found"
    exit 1
fi
INPUT_ABS="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"

ORDER="${ORDER:-single double}"

# precision -> build directory and output folder (must match run_file.sh)
build_dir() { case "$1" in single) echo "$HERE/build_gpu_single";; double) echo "$HERE/build-gpu";; esac; }
out_dir()   { case "$1" in single) echo "$(build_dir single)/tcgm_single";; double) echo "$(build_dir double)/tcgm_double";; esac; }

# ---------------------------------------------------------------------------
# Check everything up front, so the second run does not fail hours later
# ---------------------------------------------------------------------------
for p in $ORDER; do
    b="$(build_dir "$p")"
    if [ -z "$b" ]; then
        echo "Error: unknown precision '$p' in ORDER (use single / double)"
        exit 1
    fi
    if [ ! -x "$b/idefix" ] || [ ! -x "$b/run_file.sh" ]; then
        echo "Error: $b needs an idefix executable and an executable run_file.sh"
        exit 1
    fi
    if [ "${FORCE:-0}" != "1" ] && compgen -G "$(out_dir "$p")/data.*.vtk" > /dev/null; then
        echo "Error: outputs already exist in $(out_dir "$p")"
        echo "Move them away (e.g. mv $(out_dir "$p") $(out_dir "$p")_shared_gpu) or set FORCE=1."
        exit 1
    fi
done

# Warn if something else is already using the GPU (would spoil the timing)
if command -v nvidia-smi > /dev/null; then
    BUSY="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)"
    if [ -n "$BUSY" ]; then
        echo "Warning: other processes are using the GPU:"
        echo "$BUSY"
        echo ""
    fi
fi

# ---------------------------------------------------------------------------
# Re-launch this script in the background unless asked to stay in the terminal
# ---------------------------------------------------------------------------
if [ "${FOREGROUND:-0}" != "1" ] && [ "${_RUN_BOTH_CHILD:-0}" != "1" ]; then
    _RUN_BOTH_CHILD=1 nohup "$0" "$INPUT_ABS" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
        > "$HERE/run_both.log" 2>&1 &
    PID=$!
    echo "$PID" > "$HERE/run_both.pid"
    echo "Started in background, PID $PID (saved in run_both.pid)"
    echo "  Order          : $ORDER"
    echo "  Follow progress: tail -f $HERE/run_both.log"
    echo "  Per-run logs   : <build dir>/tcgm_<precision>/run.log"
    exit 0
fi

# ---------------------------------------------------------------------------
# Run the builds one after the other
# ---------------------------------------------------------------------------
for p in $ORDER; do
    echo "=================================================================="
    echo "[$(date '+%F %T')] Starting $p precision run"
    echo "=================================================================="
    START=$(date +%s)
    # FOREGROUND=1 makes run_file.sh wait for idefix to finish before returning
    FOREGROUND=1 "$(build_dir "$p")/run_file.sh" "$INPUT_ABS" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" > /dev/null
    echo "[$(date '+%F %T')] $p run finished in $(( $(date +%s) - START )) s"
    echo "  log: $(out_dir "$p")/run.log"
done

echo ""
echo "[$(date '+%F %T')] Both runs done. Next: python3 scaling_diagnostics.py"
