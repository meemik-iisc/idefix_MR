#!/usr/bin/env bash
# Configure (cmake only, no make) a SINGLE-PRECISION Idefix GPU build for a problem directory.
# Usage: ./make_build_single.sh <problem_name>     (a folder under problems/, e.g. tcgm_new)
#        then: cd problems/<problem_name>/build_gpu_single && make -j 8
#
# Why: the RTX 4000 Ada runs FP64 at 1/64 of its FP32 rate, so single precision can be
# much faster. Validate against a double-precision run (build-gpu) on a short test first.
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
# GPU build, single precision (NVIDIA RTX 4000 Ada, compute capability 8.9)
# ---------------------------------------------------------------------------
BUILD_DIR="${PROB_DIR}/build_gpu_single"
mkdir -p "${BUILD_DIR}"
cmake -S "${IDEFIX_DIR}" -B "${BUILD_DIR}" \
  -DIdefix_PROBLEM_DIR="${PROB_DIR}" \
  -DIdefix_PRECISION=Single \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_ADA89=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "Configured (single precision) in ${BUILD_DIR}. Next: cd ${BUILD_DIR} && make -j 8"
