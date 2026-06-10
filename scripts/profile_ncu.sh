#!/bin/bash
#SBATCH --job-name=icon4py_ncu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --output=slurm/ncu_%j.out
#SBATCH --error=slurm/ncu_%j.err
#SBATCH --uenv=icon/25.2:v3
#SBATCH --view=default
#SBATCH --account=cwd01

WORKDIR=/capstor/scratch/cscs/rgraf/icon4py
GRID=/capstor/scratch/cscs/rgraf/grid_generator/parallelogram_grid_516.nc
K_LEVELS=50
NCU_OUT="${WORKDIR}/ncu_out.ncu-rep"

cd "${WORKDIR}"
mkdir -p slurm

# --- CUDA setup (mirrors run_stencil_commands_slurm.sh) ---
if ! which nvcc > /dev/null 2>&1; then
    CUDA_FOUND=$(find /user-environment -name nvcc -type f 2>/dev/null | head -1)
    if [ -n "$CUDA_FOUND" ]; then
        CUDA_ROOT=$(dirname "$(dirname "$CUDA_FOUND")")
        export CUDA_HOME=${CUDA_ROOT}
        export CUDA_PATH=${CUDA_ROOT}
        export PATH="${CUDA_ROOT}/bin:${PATH}"
        export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${CUDA_ROOT}/targets/sbsa-linux/lib:${LD_LIBRARY_PATH}"
    fi
fi

NVHPC_ATM=$(find /user-environment -name "libnvhpcatm.so" -path "*/compilers/lib/*" 2>/dev/null | head -1)
[ -n "$NVHPC_ATM" ] && export LD_LIBRARY_PATH="$(dirname "$NVHPC_ATM"):${LD_LIBRARY_PATH}"

# --- Environment ---
export GT4PY_BACKEND_DEVICE_TYPE=cuda
export GT4PY_BUILD_CACHE_DIR="${WORKDIR}"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_TRANSLATOR_MESH="${GRID}"
export USE_STRUCTURED_BACKEND=1
export PYTHONOPTIMIZE=1
export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1
export CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export CUPY_CACHE_DIR="${WORKDIR}/.cupy_cache"
mkdir -p "${CUPY_CACHE_DIR}"
export CC=gcc
export CXX=g++

UENV_CMAKE_DIR=$(dirname "$(which cmake 2>/dev/null)")
source "${WORKDIR}/.venv/bin/activate"
[ -n "$UENV_CMAKE_DIR" ] && export PATH="${UENV_CMAKE_DIR}:${PATH}"

echo "=== NCU profiling: rbf_nabla4 structured, grid=${GRID}, K=${K_LEVELS} ==="
echo "Report will be written to: ${NCU_OUT}"

# --- First pass: run without ncu to compile and warm up the cache ---
echo "=== Warm-up run (compile + cache) ==="
VENV_PYTHON="${WORKDIR}/.venv/bin/python3"
"${VENV_PYTHON}" -m pytest -q \
    model/atmosphere/diffusion/tests/diffusion/stencil_tests/test_rbf_nabla4.py \
    --backend=dace_gpu \
    --grid "${GRID}:${K_LEVELS}" \
    --maxfail=1 -s 2>&1 | tail -20

# --- Second pass: ncu profile (kernels already compiled, no JIT overhead) ---
# Use sections instead of raw metric names: sections are architecture-aware and
# resolve the correct counter names for CC 9.0 (GH200/Hopper) automatically.
echo "=== NCU profiling pass ==="

ncu \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section Occupancy \
    --kernel-name "regex:map_2[23]_fieldop" \
    --launch-skip 0 \
    --launch-count 3 \
    --target-processes all \
    --export "${NCU_OUT}" \
    "${VENV_PYTHON}" -m pytest -q \
        model/atmosphere/diffusion/tests/diffusion/stencil_tests/test_rbf_nabla4.py \
        --backend=dace_gpu \
        --grid "${GRID}:${K_LEVELS}" \
        --maxfail=1 -s 2>&1 | tee "${WORKDIR}/ncu_stdout.txt"

echo "=== NCU done. Import with: ==="
echo "  ncu --import ${NCU_OUT} --page details"
