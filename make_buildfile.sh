#!/usr/bin/env bash
# Configure (cmake only, no make) an Idefix build for a problem directory.
# Usage: ./make_buildfile.sh <problem_name>     (a folder under problems/, e.g. RunawayBH)
#        then: cd problems/<problem_name>/build-gpu && make -j 8
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <problem_name>   (a folder under problems/)" >&2
  exit 1
fi

IDEFIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROB_DIR="${IDEFIX_DIR}/problems/$1"
if [ ! -d "${PROB_DIR}" ]; then
  echo "Error: ${PROB_DIR} does not exist" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# GPU build (NVIDIA RTX 4000 Ada, compute capability 8.9)
# ---------------------------------------------------------------------------
BUILD_DIR="${PROB_DIR}/build-gpu"
mkdir -p "${BUILD_DIR}"
cmake -S "${IDEFIX_DIR}" -B "${BUILD_DIR}" \
  -DIdefix_PROBLEM_DIR="${PROB_DIR}" \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_ADA89=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# ---------------------------------------------------------------------------
# CPU build (serial, for debugging). Uncomment to use.
# ---------------------------------------------------------------------------
# BUILD_DIR="${PROB_DIR}/build-cpu"
# mkdir -p "${BUILD_DIR}"
# cmake -S "${IDEFIX_DIR}" -B "${BUILD_DIR}" \
#   -DIdefix_PROBLEM_DIR="${PROB_DIR}" \
#   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "Configured in ${BUILD_DIR}. Next: cd ${BUILD_DIR} && make -j 8"
