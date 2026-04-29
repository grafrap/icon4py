#!/usr/bin/env bash

set -u -o pipefail

#
# Run structured vs. unstructured backend comparison for the standalone driver.
#
# This script:
#   1. Sets up the Python virtual environment
#   2. Cleans stale cache entries
#   3. Runs the driver with --compare flag to test both backends
#   4. Displays the comparison results
#

usage() {
  cat <<'EOF'
Usage: run_comparison.sh [options]

Options:
  -g, --grid PATH           Path to the grid file (default: ../grid-generator/parallelogram_grid.nc)
  -b, --backend BACKEND     GT4Py backend (default: gtfn_cpu)
  -o, --output PATH         Output directory (default: ./output)
  -l, --log-level LEVEL     Logging level: debug, info, warning, error, critical (default: info)
  --no-cache-clean          Skip cache cleanup (not recommended)
  --unstructured-only       Run unstructured backend only (no comparison)
  --structured-only         Run structured backend only (no comparison)
  -h, --help                Show this help message

Examples:
  # Run full comparison with default settings
  ./scripts/run_comparison.sh

  # Run with custom grid path
  ./scripts/run_comparison.sh --grid /path/to/grid.nc

  # Run unstructured backend only
  ./scripts/run_comparison.sh --unstructured-only
EOF
}

#
# Configuration
#
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
grid_path="${repo_root}/../grid-generator/parallelogram_grid.nc"
backend="gtfn_cpu"
output_dir="${repo_root}/output"
log_level="info"
do_compare=true
clean_cache=true
unstructured_only=false
structured_only=false

#
# Parse command-line arguments
#
while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--grid)
      grid_path="${2:?Missing value for --grid}"
      shift 2
      ;;
    -b|--backend)
      backend="${2:?Missing value for --backend}"
      shift 2
      ;;
    -o|--output)
      output_dir="${2:?Missing value for --output}"
      shift 2
      ;;
    -l|--log-level)
      log_level="${2:?Missing value for --log-level}"
      shift 2
      ;;
    --no-cache-clean)
      clean_cache=false
      shift
      ;;
    --unstructured-only)
      unstructured_only=true
      shift
      ;;
    --structured-only)
      structured_only=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "ERROR: Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      echo "ERROR: Unexpected positional argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

#
# Validate configuration
#
if [[ ! -f "${grid_path}" ]]; then
  echo "ERROR: Grid file not found: ${grid_path}" >&2
  exit 1
fi

if [[ ! -d "${repo_root}" ]]; then
  echo "ERROR: Repository root not found: ${repo_root}" >&2
  exit 1
fi

#
# Activate virtual environment
#
echo "========================================================================"
echo "Activating Python virtual environment..."
echo "========================================================================"

venv_path="${repo_root}/.venv"
if [[ ! -d "${venv_path}" ]]; then
  echo "ERROR: Virtual environment not found at ${venv_path}" >&2
  echo "Please create it first with: python3 -m venv ${venv_path}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${venv_path}/bin/activate"
echo "✓ Virtual environment activated"

#
# Clean stale cache entries (if requested)
#
if [[ "${clean_cache}" == "true" ]]; then
  echo ""
  echo "========================================================================"
  echo "Cleaning stale cache entries..."
  echo "========================================================================"
  
  python3 << 'CLEANUP_EOF'
import pickle
import glob
import os
import sys

cache_dir = ".gt4py_cache/gtfn_cache"
stencils_to_clean = ["copy_field", "scale_k", "setup_fields_for_initial_step"]
deleted_count = 0

if not os.path.exists(cache_dir):
    print(f"Cache directory not found: {cache_dir}")
    sys.exit(0)

for pkl_file in sorted(glob.glob(f"{cache_dir}/*.pkl")):
    try:
        with open(pkl_file, "rb") as fp:
            obj = pickle.load(fp)
        obj_str = str(obj)[:500]
        
        if any(stencil in obj_str for stencil in stencils_to_clean):
            # Delete all associated files (base + extensions)
            base_name = pkl_file.replace(".pkl", "")
            for file_to_delete in glob.glob(f"{base_name}*"):
                os.remove(file_to_delete)
                deleted_count += 1
                print(f"  Deleted: {os.path.basename(file_to_delete)}")
    except Exception as e:
        # Silently skip files that can't be read
        pass

print(f"✓ Cache cleanup complete ({deleted_count} files deleted)")
CLEANUP_EOF
fi

#
# Prepare output directory
#
mkdir -p "${output_dir}"
cd "${repo_root}"

#
# Run the comparison
#
echo ""
echo "========================================================================"
echo "Running standalone driver comparison"
echo "========================================================================"
echo "Grid file:      ${grid_path}"
echo "Backend:        ${backend}"
echo "Output dir:     ${output_dir}"
echo "Log level:      ${log_level}"
echo "========================================================================"
echo ""

# Determine which mode to run
if [[ "${unstructured_only}" == "true" ]]; then
  echo "Mode: Unstructured backend only"
  export USE_STRUCTURED_BACKEND=0
  export PYTHONOPTIMIZE=0
  
  python3 model/standalone_driver/src/icon4py/model/standalone_driver/main.py \
    --grid-file-path "${grid_path}" \
    --icon4py-backend "${backend}" \
    --output-path "${output_dir}" \
    --log-level "${log_level}" \
    > "${output_dir}/unstructured_run.log" 2>&1
  
  exit_code=$?
  
elif [[ "${structured_only}" == "true" ]]; then
  echo "Mode: Structured backend only"
  export USE_STRUCTURED_BACKEND=1
  export PYTHONOPTIMIZE=0
  
  python3 model/standalone_driver/src/icon4py/model/standalone_driver/main.py \
    --grid-file-path "${grid_path}" \
    --icon4py-backend "${backend}" \
    --output-path "${output_dir}" \
    --log-level "${log_level}" \
    > "${output_dir}/structured_run.log" 2>&1
  exit_code=$?
  
else
  echo "Mode: Comparison (structured vs. unstructured)"
  
  # Create temporary output directories for each backend
  temp_output="${output_dir}/temp_comparison"
  mkdir -p "${temp_output}"
  
  struct_output="${temp_output}/structured"
  unstruct_output="${temp_output}/unstructured"
  mkdir -p "${struct_output}" "${unstruct_output}"
  
  echo ""
  echo "Running structured backend in separate process..."
  USE_STRUCTURED_BACKEND=1 PYTHONOPTIMIZE=0 python3 model/standalone_driver/src/icon4py/model/standalone_driver/main.py \
    --grid-file-path "${grid_path}" \
    --icon4py-backend "${backend}" \
    --output-path "${struct_output}" \
    --log-level "${log_level}" \
    > "${output_dir}/structured_run.log" 2>&1
  
  struct_exit=$?
  
  if [[ ${struct_exit} -eq 0 ]]; then
    echo "✓ Structured backend completed successfully"
    
    # Clean cache between runs (critical!)
    # Only delete the three stencils that were pre-cleaned before structured run
    echo ""
    echo "Cleaning cache between runs..."
    python3 << 'CLEANUP_BETWEEN'
import pickle
import glob
import os
import sys

cache_dir = ".gt4py_cache/gtfn_cache"
stencils_to_clean = ["copy_field", "scale_k", "setup_fields_for_initial_step"]
deleted_count = 0

if not os.path.exists(cache_dir):
    print(f"Cache directory not found: {cache_dir}")
    sys.exit(0)

for pkl_file in sorted(glob.glob(f"{cache_dir}/*.pkl")):
    try:
        with open(pkl_file, "rb") as fp:
            obj = pickle.load(fp)
        obj_str = str(obj)[:500]
        
        if any(stencil in obj_str for stencil in stencils_to_clean):
            # Delete all associated files (base + extensions)
            base_name = pkl_file.replace(".pkl", "")
            for file_to_delete in glob.glob(f"{base_name}*"):
                os.remove(file_to_delete)
                deleted_count += 1
    except Exception as e:
        # Silently skip files that can't be read
        pass

print(f"✓ Cache between runs cleaned ({deleted_count} files deleted)")
CLEANUP_BETWEEN
    
    echo ""
    echo "Running unstructured backend in separate process..."
    USE_STRUCTURED_BACKEND=0 PYTHONOPTIMIZE=0 python3 model/standalone_driver/src/icon4py/model/standalone_driver/main.py \
      --grid-file-path "${grid_path}" \
      --icon4py-backend "${backend}" \
      --output-path "${unstruct_output}" \
      --log-level "${log_level}" \
      > "${output_dir}/unstructured_run.log" 2>&1
    
    unstruct_exit=$?
  else
    echo "✗ Structured backend failed, skipping unstructured run"
    unstruct_exit=1
  fi
  
  if [[ ${struct_exit} -eq 0 && ${unstruct_exit} -eq 0 ]]; then
    echo "✓ Both backends completed successfully"
    
    # Compare the saved arrays
    echo ""
    echo "Comparing prognostic fields..."
    python3 << 'COMPARE_SCRIPT'
import numpy as np
import sys
from pathlib import Path

struct_dir = Path("output/temp_comparison/structured")
unstruct_dir = Path("output/temp_comparison/unstructured")
output_dir = Path("output")

fields = ["vn", "w", "theta_v", "exner", "rho"]
results = []

print("\n" + "="*70)
print("Field Comparison: Structured vs Unstructured Backend")
print("="*70 + "\n")

all_match = True
for field in fields:
    struct_file = struct_dir / f"{field}.npy"
    unstruct_file = unstruct_dir / f"{field}.npy"
    
    if not struct_file.exists():
        print(f"ERROR: {field} not found in structured output")
        all_match = False
        continue
    
    if not unstruct_file.exists():
        print(f"ERROR: {field} not found in unstructured output")
        all_match = False
        continue
    
    struct_data = np.load(struct_file)
    unstruct_data = np.load(unstruct_file)
    
    # Compare with tolerance
    close = np.isclose(struct_data, unstruct_data, rtol=1e-5, atol=1e-8)
    pct_match = close.mean() * 100
    max_abs_diff = np.abs(struct_data - unstruct_data).max()
    max_rel_diff = (np.abs(struct_data - unstruct_data) / (np.abs(unstruct_data) + 1e-30)).max()
    
    line = (
        f"{field:12s}: {pct_match:6.2f}% match  "
        f"max_abs={max_abs_diff:.3e}  max_rel={max_rel_diff:.3e}"
    )
    print(line)
    results.append(line)
    
    if pct_match < 99.0:
        all_match = False

# Write results to file
compare_file = output_dir / "compare_out.txt"
with open(compare_file, "w") as f:
    f.write("Field Comparison: Structured vs Unstructured Backend\n")
    f.write("="*70 + "\n\n")
    for line in results:
        f.write(line + "\n")
    f.write("\n" + "="*70 + "\n")
    if all_match:
        f.write("✓ All fields match within tolerance (99% or higher)\n")
    else:
        f.write("⚠ Some fields show differences - check max_abs and max_rel values\n")

print("\n" + "="*70)
if all_match:
    print("✓ All fields match within tolerance (99% or higher)")
    sys.exit(0)
else:
    print("⚠ Some fields show differences - check max_abs and max_rel values")
    sys.exit(1)
COMPARE_SCRIPT
    
    exit_code=$?
  else
    if [[ ${struct_exit} -ne 0 ]]; then
      echo "✗ Structured backend failed with exit code ${struct_exit}"
      echo "  See ${output_dir}/structured_run.log for details"
    fi
    if [[ ${unstruct_exit} -ne 0 ]]; then
      echo "✗ Unstructured backend failed with exit code ${unstruct_exit}"
      echo "  See ${output_dir}/unstructured_run.log for details"
    fi
    exit_code=1
  fi
fi

#
# Display results
#
echo ""
echo "========================================================================"
if [[ ${exit_code} -eq 0 ]]; then
  echo "✓ Comparison completed successfully - fields match!"
else
  echo "✗ Comparison or one/both runs failed"
fi
echo "========================================================================"

# Show log file locations
echo ""
echo "Run Logs:"
if [[ -f "${output_dir}/structured_run.log" ]]; then
  echo "  Structured:   ${output_dir}/structured_run.log"
fi
if [[ -f "${output_dir}/unstructured_run.log" ]]; then
  echo "  Unstructured: ${output_dir}/unstructured_run.log"
fi

# Show comparison results if available
if [[ -f "${output_dir}/compare_out.txt" ]]; then
  echo ""
  echo "Detailed Comparison Results:"
  echo "------------------------------------------------------------------------"
  cat "${output_dir}/compare_out.txt"
  echo "------------------------------------------------------------------------"
fi

echo ""
echo "Array files saved in:"
if [[ -d "${output_dir}/temp_comparison/structured" ]]; then
  echo "  Structured:   ${output_dir}/temp_comparison/structured/"
fi
if [[ -d "${output_dir}/temp_comparison/unstructured" ]]; then
  echo "  Unstructured: ${output_dir}/temp_comparison/unstructured/"
fi

exit ${exit_code}
