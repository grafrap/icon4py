# Optimization Notes: GPU Memory Layout and Warp Alignment

## Root Cause of Poor GPU Performance

DaCe's CUDA thread assignment maps alphabetically-LAST GT4Py horizontal dim → `threadIdx.x` (the warp lane dimension). With the original ordering `IDim < JDim < Kolor`, **Kolor** becomes the warp dimension. Kolor has only 3 values (one per edge kolor), so:

- Per-kolor split SetAts have Kolor range `[k, k+1)` = 1 value → `blockDim.x = min(32, 1) = 1`
- Total threads per block = `(1, 8, 1) = 8` — catastrophically small (a full warp = 32)
- Warp threads (Kolor) access Kolor stride ≈ ni × nj ≈ 260,000 in DaCe transients → completely uncoalesced

Separately, DaCe transients use **Fortran-order** by default: the alphabetically-FIRST dim (IDim) gets stride 1. So IDim already has stride 1 in transients — but it's in `threadIdx.z` (1 thread/block), wasted.

## Fix: Two Changes

### Change 1 — `unit_strides_dim=["IDim"]` (gt4py)

Added `unit_strides_dim` parameter to `gt_auto_optimize` and `_gt_auto_configure_maps_and_strides` in `auto_optimize.py`. When provided, it calls `gt_set_iteration_order(unit_strides_dim=...)` to make IDim the **innermost map parameter** → `threadIdx.x` (warp) on GPU.

In `translation.py`, `structured_opt_args` now uses:
```python
"unit_strides_dim": [common.Dimension("IDim")],
```
instead of `"unit_strides_kind": VERTICAL`.

**Effect**: IDim (~510 values per SetAt) becomes the warp → `blockDim.x = min(32, 510) = 32` → 256 threads/block (32× improvement). IDim_stride = 1 in transients (Fortran-order, unchanged) → coalesced transient access.

### Change 2 — Fortran-order numpy arrays (gt4py translator.py)

Added `order='F'` to all `np.zeros`/`np.full` calls that create structured field arrays in `translator.py`:
- `pack_edge_field` / `pack_edge_field_to_structured`
- `pack_vertex_field` / `pack_vertex_field_to_structured`
- `pack_cell_field`
- `pack_sparse_local_field_to_structured`
- `apply_sparse_pack_mapping`
- `pack_c2e2co_field`

With Fortran-order, numpy axis 0 (IDim dimension) has stride 1 — matching DaCe's Fortran-order transients. When DaCe reads the numpy strides at runtime, it receives IDim_stride = 1 for I/O arrays. The warp (IDim) now accesses stride-1 in **both** transients and I/O arrays → fully coalesced.

Array shapes unchanged (`(ni, nj, n_kolor, nk)`) — only memory order changes. FieldType dimension declarations in `cartesian_interceptor.py` unchanged since GT4Py uses alphabetical ordering (`[IDim, JDim, Kolor, K]`) which matches numpy axis 0→IDim regardless of C/F order.

## Expected Performance Gain

- `blockDim` changes from `(1, 8, 1)` → `(32, 8, 1)` (visible in `.gt4py_cache/.../cuda/*.cu`)
- Transient index: IDim has coefficient 1 (unchanged); IDim now also in `threadIdx.x`
- I/O stride: IDim_stride = 1 (from Fortran numpy) vs `nj × n_kolor × nk` before
- `apply_diffusion_to_vn` exec baseline: 0.02508 s (from `output/big_baseline`)

## Previous Attempt: Color-First Layout (Reverted)

Branch `grafrap_opt` (commits `0bab491ed`, `166e97b6c`, `554c76979`) tried `Kolor → Color` (C < I alphabetically) + `IDim → ZDim` (Z > C, last), making ZDim the warp. This was slower because:

1. Pack/unpack used Python loops (`for c in range(n_color)`) instead of vectorized numpy → pack/unpack time dominated
2. DaCe transients had Color=stride-1 (Fortran, Color was first) but warp=ZDim → ZDim_stride = ni × nj = large → uncoalesced transients
3. Cross-kolor cache distance: with Color first, same (i,j) across kolors are 200 MB apart

Benchmark (512×512, K=50): 2–4× slower than Kolor-last baseline.

## Files Changed

| File | Change |
|---|---|
| `../gt4py/src/gt4py/next/program_processors/runners/dace/transformations/auto_optimize.py` | Add `unit_strides_dim` parameter to `gt_auto_optimize` + `_gt_auto_configure_maps_and_strides` |
| `../gt4py/src/gt4py/next/program_processors/runners/dace/workflow/translation.py` | `unit_strides_dim=["IDim"]` in `structured_opt_args` + fallback path |
| `../gt4py/src/gt4py/next/modules/translator.py` | `order='F'` in all structured pack array allocations |
