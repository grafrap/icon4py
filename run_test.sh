#!/bin/bash
#SBATCH --job-name=icon4py_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --uenv=icon/25.2:v3
#SBATCH --view=default

WORKDIR=/scratch/mch/rgraf/icon4py

cd ${WORKDIR}
export LD_LIBRARY_PATH=/user-environment/linux-sles15-zen3/gcc-13.2.0/cuda-12.6.0-zio6z46edr7vbelm3rt4r3qidbrlurf6/targets/x86_64-linux/lib/:${LD_LIBRARY_PATH}
export GT4PY_BUILD_CACHE_DIR=${WORKDIR}/gt_cache
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE=0
export USE_STRUCTURED_BACKEND=1
export PYTHONOPTIMIZE=1

# help cmake find the right compiler
export CC=gcc
export CXX=g++

# Save the uenv cmake (the working one) before venv activation shadows it
UENV_CMAKE_DIR=$(dirname "$(which cmake 2>/dev/null)")

source ${WORKDIR}/.venv/bin/activate

# Re-prepend the uenv cmake so it wins over the broken pip cmake in the venv
[ -n "$UENV_CMAKE_DIR" ] && export PATH=${UENV_CMAKE_DIR}:$PATH

echo "cmake: $(which cmake) — $(cmake --version | head -1)"

pytest -q model/atmosphere/dycore/tests/dycore/stencil_tests/test_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl.py \
    --backend=gtfn_cpu \
    --grid /scratch/mch/rgraf/grid_generator/parallelogram_grid.nc:5 \
    --maxfail=1 \
    -s
