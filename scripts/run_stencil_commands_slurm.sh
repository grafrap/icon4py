#!/bin/bash
#SBATCH --job-name=icon4py_stencils
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --uenv=icon/25.2:v3
#SBATCH --view=default

WORKDIR=/scratch/mch/rgraf/icon4py
cd "${WORKDIR}"

# The uenv from SBATCH loads CUDA, but we need to ensure all paths are set
echo "=== Initial CUDA environment check ==="
which nvcc && nvcc --version || echo "WARNING: nvcc not found initially"

# If uenv didn't set CUDA paths, we may need to find them
if ! which nvcc > /dev/null 2>&1; then
    # Fallback: try to find CUDA in /user-environment
    CUDA_FOUND=$(find /user-environment -name nvcc -type f 2>/dev/null | head -1)
    if [ -n "$CUDA_FOUND" ]; then
        CUDA_ROOT=$(dirname "$(dirname "$CUDA_FOUND")")
        export CUDA_HOME=${CUDA_ROOT}
        export CUDA_PATH=${CUDA_ROOT}
        export PATH="${CUDA_ROOT}/bin:${PATH}"
        export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${CUDA_ROOT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
        echo "Found CUDA at: $CUDA_ROOT"
    fi
fi

echo "=== After setup ==="
which nvcc && echo "nvcc: $(which nvcc) — $(nvcc --version | grep release)" || echo "ERROR: nvcc still not found"

# 3. GT4PY and Environment Settings
export GT4PY_BACKEND_DEVICE_TYPE=cuda
# Cache lives at ${WORKDIR}/.gt4py_cache (default when GT4PY_BUILD_CACHE_DIR=WORKDIR).
# Keep consistent with run_stencil_commands.sh which also clears ${repo_root}/.gt4py_cache.
export GT4PY_BUILD_CACHE_DIR="${WORKDIR}"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
# Clear stale cache before running (run_stencil_commands.sh also does this, but belt+suspenders).
rm -rf "${WORKDIR}/.gt4py_cache" 2>/dev/null || true
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE=0
export USE_STRUCTURED_BACKEND=1
export PYTHONOPTIMIZE=1
export MPICH_GPU_SUPPORT_ENABLED=1

# CuPy settings for batch job GPU initialization
export CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# DEBUG: Enable synchronous CUDA error reporting
export CUDA_LAUNCH_BLOCKING=1

# help cmake find the right compiler
export CC=gcc
export CXX=g++

# Save the uenv cmake (the working one) before venv activation shadows it
UENV_CMAKE_DIR=$(dirname "$(which cmake 2>/dev/null)")

source ${WORKDIR}/.venv/bin/activate

# Re-prepend the uenv cmake so it wins over the broken pip cmake in the venv
[ -n "$UENV_CMAKE_DIR" ] && export PATH=${UENV_CMAKE_DIR}:$PATH

# Verify CUDA is available after venv activation
echo "=== Environment Check ==="
echo "nvcc: $(which nvcc) — $(nvcc --version | grep release)"
echo "cmake: $(which cmake) — $(cmake --version | head -1)"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $(echo $LD_LIBRARY_PATH | cut -d: -f1-3)..."
echo "========================="

# GPU tests must run sequentially — the GPU is in exclusive-process compute mode
# (cudaErrorDevicesUnavailable when multiple processes compete for the same device).
# DaCe compilation still uses all allocated CPUs internally via cmake/nvcc.
export JOBS=1
bash scripts/run_stencil_commands.sh "$@"