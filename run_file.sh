#!/usr/bin/env bash
#
# run_file.sh
# Copy this script into a build directory (e.g. problems/RunawayBH/build-gpu)
# and run it from there. Outputs are written in RUN_DIR/OUTPUT_FILENAME
# (RUN_DIR = the directory holding this script and the idefix executable).
#
# Usage: ./run_file.sh [input.ini] [extra idefix options...]
#   input.ini      parameter file (default: ../idefix.ini)
#   extra options  passed straight to idefix, e.g. -maxcycles 100, -restart, -profile
#
# Environment:
#   FOREGROUND=1   run in the terminal instead of in the background
#   FORCE=1        start a fresh run even if outputs already exist (overwrites them)
#
# Output folder: edit OUTPUT_FILENAME below for each run.
#
# Examples:
#   ./run_file.sh                              # fresh run with ../idefix.ini
#   FOREGROUND=1 ./run_file.sh ../idefix.ini -maxcycles 100   # smoke test
#   ./run_file.sh ../idefix.ini -restart       # resume from the last dump
#

set -euo pipefail

# ---------------------------------------------------------------------------
# All outputs (vtk, dumps, logs, copy of the .ini) go to RUN_DIR/OUTPUT_FILENAME
# ---------------------------------------------------------------------------
OUTPUT_FILENAME="wind_bondi"

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN_DIR"

INPUT="${1:-../idefix.ini}"
if [ $# -ge 1 ]; then shift; fi
EXTRA_ARGS=("$@")

OUT_DIR="$RUN_DIR/$OUTPUT_FILENAME"

if [ ! -x ./idefix ]; then
    echo "Error: no idefix executable in $RUN_DIR"
    echo "Build it first: make -j 8"
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "Error: input file $INPUT not found"
    exit 1
fi
INPUT_ABS="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"

# idefix writes its outputs to the directory it is started from
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
cd "$OUT_DIR"

# Refuse to overwrite an existing run unless restarting or forced
IS_RESTART=0
for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
    [ "$arg" = "-restart" ] && IS_RESTART=1
done
if [ "$IS_RESTART" -eq 0 ] && [ "${FORCE:-0}" != "1" ] \
   && compgen -G "data.*.vtk" > /dev/null; then
    echo "Error: outputs (data.*.vtk) already exist in $OUT_DIR"
    echo "Move them away, resume with -restart, or set FORCE=1 to overwrite."
    exit 1
fi

# Keep a copy of the parameters used, next to the data they produced.
# On restart, keep the existing copy so it still matches the original run.
RUN_INI="idefix_run.ini"
if [ "$IS_RESTART" -eq 0 ] || [ ! -f "$RUN_INI" ]; then
    cp "$INPUT_ABS" "$RUN_INI"
fi

CMD=("$RUN_DIR/idefix" -i "$RUN_INI" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
echo "Outputs in : $OUT_DIR"
echo "Command    : ${CMD[*]}"

if [ "${FOREGROUND:-0}" = "1" ]; then
    "${CMD[@]}" 2>&1 | tee run.log
else
    nohup "${CMD[@]}" > run.log 2>&1 &
    PID=$!
    echo "$PID" > run.pid
    echo ""
    echo "Started in background, PID $PID (saved in run.pid)"
    echo "  Follow the log : tail -f $OUT_DIR/run.log"
    echo "  Stop cleanly   : touch $OUT_DIR/stop   (writes a restart dump)"
    echo "  Resume later   : ./run_file.sh $INPUT -restart   (same OUTPUT_FILENAME)"
fi
