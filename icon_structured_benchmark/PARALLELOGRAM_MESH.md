# Running `icon_structured_benchmark` on the Parallelogram Mesh

`icon_structured_benchmark/` (GridTools C++ + pybind, driven by
`run_filtered_torus_grid_int_nabla4.py`) originally only supported the periodic **torus**
grid. This document records what was needed to get the **nabla4** benchmark running on the
project's own (non-periodic) **parallelogram mesh** instead, on GPU.

## Why the torus driver doesn't work as-is

The torus driver relies on three properties that only hold for the torus:

1. **No boundary**: `edges == 3 * vertices` exactly. The parallelogram has real boundaries,
   so per-kolor edge counts are **ragged**: `nx*(ny+1) + (nx+1)*ny + nx*ny != 3*n_vertices`
   (e.g. for the 512 grid: `512*513 + 513*512 + 512*512 = 787456 != 3*263169`).
2. **Uniform, already-structured ordering**: the torus's natural vertex/edge ordering from
   the grid generator is already row-major, so the driver can filter the GridManager's
   tables directly.
3. **Periodicity**: wraparound neighbor accesses are always in-bounds.

The parallelogram's NetCDF vertex ordering is **not** row-major, and it has genuine
boundaries, so none of the above hold.

## New driver: `run_filtered_parallelogram_grid_nabla4.py`

A new driver (copy-and-adapt of the torus one) builds the structured (i, j, kolor) tables
**analytically**, in the exact frame the `*_structured_torus_*_halo` GPU kernels hard-code
(vertex flat id `= j*nx + i` with `i` the fast axis; per-color E2C2V slot offsets; E2ECV
`= slot*(nx*ny*3) + color*(nx*ny) + anchor`), then **verifies every interior (i, j, color)
formula against the real mesh topology**:

- `rowmajor_to_mesh = np.lexsort((x, y))` builds the row-major → mesh-vertex-id permutation
  from vertex coordinates.
- `edge_pairs = {frozenset((v0, v1)): edge_id for each real mesh edge}` is a lookup table
  from unordered vertex-id pairs to the real edge id.
- For every interior (i, j, color), the analytic formula's endpoint pair is looked up in
  `edge_pairs` to confirm the formula actually matches a real edge in the mesh — this is
  the topology check printed as `"topology verified against mesh"`.

Benchmark data itself is synthetic/random (as in the torus driver); only the **connectivity
tables** need to be real. Interior-only compute (`halo=2`, same as the torus-halo variant)
means the kernels never read past the real boundary, which plays the same role periodicity
plays for the torus.

Scope of this first version: **nabla4 only, GPU backends only** (`gtfn_gpu`, `gpu_kloop`,
`gpu_naive`).

## Build issues

Building the `sizet`/GPU pybind extension (`build_gpu_sizet.sh`, an sbatch script,
`debug` partition, `--uenv=icon/25.2:v3`) hit two environment problems on a fresh node:

1. **CMake found the wrong Python.** The uenv's system Python (no `nanobind`) was picked up
   instead of the project's `.venv`. Fixed by pinning both the executable and the search
   strategy:
   ```
   -DPython_EXECUTABLE=/capstor/scratch/cscs/rgraf/icon4py/.venv/bin/python
   -DPython_FIND_VIRTUALENV=ONLY
   ```
2. **The venv's `nanobind` install was broken** — the package's `cmake/` directory
   (containing `nanobind-config.cmake` etc.) was missing, so `find_package(nanobind)`
   failed even with the right Python. `pip install nanobind` failed on the login node with
   an SSL certificate error. Fixed by fetching the wheel with `curl` and unzipping just the
   package contents directly into site-packages:
   ```bash
   cd .venv/lib/python3.10/site-packages && unzip -o -q nanobind.whl 'nanobind/*'
   ```
   This restored the missing headers and CMake config files without needing `pip`/network
   access from the compute node.

After both fixes, `build_gpu_sizet.sh` completed cleanly (`ptxas info: 0 bytes spill
stores/loads`, `.so` artifacts produced).

## Runtime issues

1. **`srun` needs an explicit account.** `ERROR: you must specify a project account
   (-A <account>)` — add `-A cwd01` (same account `build_gpu_sizet.sh` uses).
2. **Launch scripts must live on the shared filesystem, not local `/tmp`.** A script written
   to the harness's scratchpad (local `/tmp` on the login node) is invisible to the compute
   node `srun` lands on (`No such file or directory`, exit 127). Write launch scripts under
   the project directory (`/capstor/scratch/...`) instead.
3. **CuPy blocks implicit array conversion — a real bug in the driver's sanity check.**
   `run_sanity_checks`'s `gtfn_gpu` branch did:
   ```python
   compare_ndarrays(np.asarray(z_gtfn), ...)
   ```
   where `z_gtfn` is a CuPy-backed `gt4py.storage` array (`backend="gt:gpu"`). CuPy
   deliberately raises `TypeError: Implicit conversion to a NumPy array is not allowed` on
   `np.asarray(cupy_array)` — the caller must do the host transfer explicitly. Fixed with:
   ```python
   import cupy as cp
   compare_ndarrays(cp.asnumpy(z_gtfn), ...)
   ```
   (`cp.asnumpy` is a no-op passthrough for already-NumPy input, so it's safe generically.)
   This only manifests on the `gtfn_gpu` check path — the other backends' pybind calls
   (`icon_benchmark.nabla4_validate_*`) already return plain host-side arrays.
4. **Python stdout buffering hides a genuinely long-running benchmark as a "silent hang".**
   Without a TTY, Python block-buffers stdout; if a run is killed by a `srun` time limit
   before it exits/flushes, the log file can show **zero output** even though earlier prints
   already happened — this looked like a hang before any progress, but was just buffering.
   Always run with `python3 -u` (or set `PYTHONUNBUFFERED=1`) for any background/timed run
   so partial progress is visible in the log if it gets killed.
5. **The actual timing/benchmark loop (as opposed to the sanity checks) has no built-in
   progress output**, so a long-but-legitimate benchmark call is indistinguishable from a
   real hang without instrumentation. Added a `_timed()` wrapper (and matching prints around
   the `gtfn_gpu` benchmark call) around each `icon_benchmark.nabla4_*_benchmark_*` call in
   `run_benchmarks()`, printing `Running <label> ...` / `Finished <label> in <x>s wall` per
   call.
6. **The debug partition caps job time at 30 minutes** (`sinfo` shows `TIMELIMIT 30:00`),
   which is enough for sanity checks but not necessarily for the full 101-repetition
   benchmark sweep at 512×512×K=80. Use the `normal` partition (no such cap, longer queue
   wait) for the actual timed benchmark run.

## Current status

- **26×26 mesh** (`parallelogram_grid_26.nc`, K=5, `--backend gpu_kloop`): sanity checks
  pass — `unstructured gpu kloop`, `structured gpu kloop`, `structured gpu kloop vertical`.
- **512×512 mesh** (`parallelogram_grid_512.nc`, K=80, `--backend all_gpu`): **all 7**
  sanity checks pass — `gtfn_gpu`, `unstructured gpu_kloop` (+vertical),
  `structured gpu_kloop` (+vertical), `unstructured gpu_naive` (+vertical),
  `structured gpu_naive` (+vertical). This is the correctness-critical result: the
  structured torus-halo kernels match the unstructured GridTools reference on the real
  parallelogram mesh at full production scale.
- The timed benchmark sweep (101 repetitions + 10 dry runs, for the median runtime numbers)
  is a separate, longer-running step — not a correctness question — and is still being
  timed out on the `normal` partition.

## How to run

```bash
# Build (once per environment/branch), sbatch, debug partition:
sbatch icon_structured_benchmark/build_gpu_sizet.sh

# Sanity check + benchmark, 26 mesh (fast, debug partition is fine):
srun -A cwd01 --partition=debug --time=00:25:00 --gres=gpu:1 \
  --uenv=icon/25.2:v3 --view=default bash -c '
    cd icon_structured_benchmark
    source /capstor/scratch/cscs/rgraf/icon4py/.venv/bin/activate
    export PYTHONPATH=build_gpu_sizet:$PYTHONPATH
    python3 -u run_filtered_parallelogram_grid_nabla4.py \
      /capstor/scratch/cscs/rgraf/grid_generator/parallelogram_grid_26.nc \
      --klevels 5 --backend gpu_kloop --sanity-checks --repetitions 101 --dry-run
  '

# Sanity check + benchmark, full 512 mesh (use normal partition, no 30-min cap):
srun -A cwd01 --partition=normal --time=01:00:00 --gres=gpu:1 \
  --uenv=icon/25.2:v3 --view=default \
  bash icon_structured_benchmark/run_512_gpu.sh
```

`run_512_gpu.sh` (checked in) wraps the CUDA/venv environment setup plus the `-u` (unbuffered)
Python invocation with `--backend all_gpu --klevels 80 --sanity-checks --repetitions 101
--dry-run`, writing results to `results/output_parallelogram_512_gpu.json`.

## `strided_gpu_kloop_inlined` K-loop tuning (514 mesh, K=120) — both levers are dead ends

Investigated whether `run_gpu_kloop_nabla4_interpolate_inlined_structured` (the fused
nabla4+interpolate kloop kernel, `strided_gpu_kloop_inlined` in the violin plots — currently
the fastest kernel in the whole comparison at ~0.89ms) could be pushed further by tuning its
K-loop, the same way `DACE_KTILE` won ~16% on the GT4Py/DaCe side (see `Opt.md` Optimization
19). Two build-time macros were added to `include/nabla4_interpolate_structured_inlined.hpp`
(default off, zero effect unless set): `KLOOP_KDIM_CT` (bakes K=120 as a compile-time constant
instead of the runtime `KDim` parameter — required for any real unroll, since nvcc can't prove
a runtime-bounded loop's trip count) and `KLOOP_BLOCKDIM_Z_CT` / `KLOOP_UNROLL_N` (the two
levers actually swept). **Both came back negative — this kernel is already at its practical
floor for K-loop restructuring.**

### Lever 1: `KLOOP_BLOCKDIM_Z_CT` (split K across more threads per `(i,j)`) — rejected

Swept Z = 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20 (`sweep_kloop_inlined_ktile.sh`). Result:

| Z | median | outcome |
|---|---|---|
| 1 | 0.892 ms | baseline |
| 2 | 7.90 ms | 8.9× slower |
| 3 | 12.2 ms | 13.7× slower |
| 4 | 15.0 ms | 16.8× slower |
| ≥5 | — | **correctness failure** |

Root cause: this kernel precomputes its (expensive, several-global-read) per-`(i,j)` setup
— V2E/E2C2V neighbor indices, `primal_normal_vert` reads ×24, `ptr_coeff` reads ×12, etc. —
**once per thread**, before the K-loop. Increasing `blockDim.z` doesn't share that precompute
across the Z threads working on the same `(i,j)`; it **duplicates** it Z times (each
`threadIdx.z` independently redoes the whole setup), so the cost scales with Z while the
K-work per thread only shrinks by the same factor — a strictly bad trade once precompute is
non-trivial. Z≥5 additionally exceeds the 1024-threads/block hardware limit at
`blockDim={32,8,Z}` (`256·Z`), producing an invalid launch that silently yields garbage output
rather than a caught error — confirmed as the exact cause of the correctness failures.

This is also *why* `strided_gpu_naive_inlined` (2.8–3.6ms) is 3–4× slower than
`strided_gpu_kloop_inlined` despite looking superficially similar: naive's launch is
`grid.z = ceil(KDim/blockDim.z)` (`run_gpu_naive_helper`), i.e. **120 separate grid blocks
along z** even at `blockDim.z=1` — 120 independent threads per `(i,j)`, one per K-level, each
redoing the full precompute. kloop's `blockDim.z=1` case is structurally different: `grid.z`
is fixed at 1, and a single thread per `(i,j)` loops over all 120 K-levels in software, so
precompute happens exactly once. naive is effectively the Z→120 extreme of the same failure
mode this sweep found devastating at Z=2–4.

### Lever 2: `KLOOP_UNROLL_N` (partial `#pragma unroll` on the per-thread K-loop) — inert

With `KLOOP_BLOCKDIM_Z_CT=1` fixed (the only viable value per above), swept the unroll factor
N = 0 (bare `#pragma unroll`, full 120-way unroll), 1 (`#pragma unroll 1`, no unrolling), 2, 3,
4, 5, 6, 8, 10 (`sweep_kloop_inlined_unroll.sh`; N=12/20 didn't finish inside the 1h job
budget). All results land within a **2.9% band, 0.8925–0.9181 ms, no trend with N**.

Confirmed via `cuobjdump --dump-sass` that this is a real null result, not a measurement
artifact or the compiler ignoring the pragma: extracted SASS instruction counts for the exact
kernel scale cleanly with N (N=1: 1175 lines → N=2: 1575 → N=4: 2023 → N=0/full: 3159 — a
2.7× spread), while register counts stay flat (236–242) and runtime stays flat. The kernel is
Long-Scoreboard-stall-dominated (confirmed earlier via NCU — see the `strided_gpu_naive_inlined`
analysis above/`Opt.md`), i.e. bound by global-memory latency, not by instruction-level
parallelism/issue rate — so changing how much of the K-loop is unrolled at compile time
changes code size but not the wall-clock bottleneck.

**Practical implications**: `~0.89ms` is very likely close to `strided_gpu_kloop_inlined`'s
real floor without a more invasive change (e.g. restructuring the redundant `u_vert`/`v_vert`/
`primal_normal_vert` reads across V2E neighbors, not the loop/thread structure). Both macros
are left in `nabla4_interpolate_structured_inlined.hpp`, default-off
(`KLOOP_KDIM_CT=0` disables the whole compile-time path, restoring the original
runtime-KDim/grid-stride loop byte-for-byte), documented inline at their definitions and at
the K-loop itself. Do not re-sweep `KLOOP_BLOCKDIM_Z_CT` above 1 — it is a confirmed,
mechanistically-understood dead end, not just an unswept region.
