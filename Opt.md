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

## Benchmark Results (Opt 1 only, 512×512 grid, K=50, dace_gpu)

| Run | Median | vs Unstructured |
|-----|--------|----------------|
| Unstructured (512×512) | 1.363 ms | 1.000× |
| Structured warp-fix (512) | 2.355 ms | +72.8% |

---

# Optimization 2: Origin-Shift (Write Domain Pointer Alignment)

## Problem: Partial Cache-Line Writes at Domain Boundary

After Opt 1, the first write per JDim row starts at IDim = `i_lo` (= 2, the first interior edge index). In Fortran-order with IDim stride = 1, this means:

- Thread 0 of warp 0 writes to buffer offset `2` (not 0)
- Cache line 0 covers offsets `0..15` (128 bytes / 8 bytes per double = 16 doubles)
- Only 14 of 16 doubles in cache line 0 are written → **partial write** at the start
- Symmetric partial write at the end of the last warp per row
- ≈ 4 partial cache-line writes per row × 512 rows × 3 kolors × 50 K ≈ 300K per call

## Fix: Compact Packing + IR Domain Shift

Instead of packing to a full `(ni, nj, 3, nk)` array starting at IDim=0, pack to a compact `(ni−shift_i, nj−shift_j, 3, nk)` array where:
- IDim=i_lo data is placed at compact position 0 → **first write at `ptr[0]`** (cache-aligned)
- The IR domain is simultaneously shifted from `[i_lo, i_hi)` to `[0, i_hi−i_lo)` so the kernel matches

**Edge fields**: compact packing, DaCe `range_0=0`, domain `[0, ni−shift_i)`.

**Vertex fields** (E2C2V reads with di ∈ {−1,0,+1}): packed to a padded array `(1+ni_v−shift_i, 1+nj_v−shift_j, 1, nk)` with 1 extra row/col before the domain start. Passed with `origin={IDim: 1, JDim: 1}` → DaCe `range_0=−1` → access `ptr[i+1]` → accessing i=−1 from domain start goes to `ptr[0]` (within allocation, not OOB on GPU).

**Sparse fields** (primal_normal_vert_v1/v2): compactly sliced with `struct_np[shift_i:, shift_j:]`.

**Unpack**: boundary edges (below `horizontal_start`) preserved via `unstruct_np = orig_np.copy()` before overwriting interior edges.

**IR changes** in `structured_backend_passes.py` (4 sites, all subtract `shift_i/j`):
- `_per_kolor_domain`: write domain bounds
- `_build_edge_validity_masked_expr`: per-kolor validity mask bounds
- `_mapping_based_threshold_condition`: threshold params (e.g. `start_2nd_nudge_line_idx_e`)
- `_rewrite_edge_range_threshold`: range threshold bounds

## Files Changed (Opt 2)

| File | Change |
|---|---|
| `../gt4py/src/gt4py/next/modules/translator.py` | Add `pack_edge_field_compact`, `pack_vertex_field_padded` |
| `../gt4py/src/gt4py/next/modules/cartesian_interceptor.py` | Shift computation in `_get_or_compile`; compact/padded dispatch in `_pack_argument`; shifted unpack in `_unpack_to_buffer`; `horizontal_start` extraction moved before packing loop |
| `../gt4py/src/gt4py/next/iterator/transforms/structured_backend_passes.py` | Subtract `shift_i/j` at 4 coordinate-embedding sites |

## Benchmark Results (Opt 1 + 2, 512 and 516 grids, K=50, dace_gpu)

| Run | Median | StdDev | vs Unstr-516 |
|-----|--------|--------|-------------|
| Unstructured (516×516) | 1.316 ms | 0.050 ms | 1.000× |
| Structured warp-fix (512) | 2.355 ms | 0.198 ms | +78.9% |
| Structured shift (512) | 2.304 ms | 0.168 ms | +75.1% |
| **Structured shift (516)** | **1.836 ms** | **0.019 ms** | **+39.5%** |

Origin shift on 512 gives only 2% improvement (partial last warp still present). On 516 (aligned grid): **22% faster than warp-fix baseline**, stddev drops 10×.

---

# Optimization 3: Grid Size Alignment (N ≡ 4 mod 32)

## Why Grid Size Matters

For the write domain width to be a multiple of 32 (one or more full warps, no partial last warp per row), we need:

```
write domain width = N − 4   (2 halo layers on each side)
(N − 4) % 32 == 0  →  N ≡ 4 (mod 32)
→ N = 36, 68, ..., 516, 548, ...
```

For `N=512`: width = 508 = 15×32 + 28 → last warp has 28/32 active threads  
For `N=516`: width = 512 = 16×32 → **zero waste**, every warp full

Kolor 1 has `ni = N+1` → width = N−3, which cannot simultaneously be a multiple of 32 alongside kolors 0 and 2. At N=516, kolor 1 has width=513 = 16×32+1 (1 partial warp per row — negligible).

**Nudging zone** (threshold `start_2nd_nudge_line_idx_e`): domain width = N−8, so `N=516` gives width=508 (still 28/32 in last warp). To align nudging zone too: N ≡ 8 (mod 32) — conflicts with N ≡ 4, so perfect alignment of both simultaneously is impossible.

**Recommended grid sizes**: N = 36 (debug/test), N = 516 (production). For timing benchmarks always use N = 516.

---

# Optimization 4: IDim Stride Padding (Row Cache-Line Alignment) — Planned

## Problem: Adjacent JDim Rows Share Cache Lines

After Opt 2+3, writes start at `ptr[0]` and the write domain is 512 elements (full warps). However, the compact IDim array size is `ni_compact = ni_kolor − shift_i = 516 − 2 = 514`. This is the JDim stride (elements between the start of consecutive rows).

- Row j starts at byte offset `j × 514 × 8 = j × 4112 bytes`
- `4112 / 128 = 32.125` → NOT a cache-line boundary
- Row j and row j+1 share a cache line at the boundary → **GPU cannot process rows fully independently**; adjacent rows may cause cache-line contention

## Fix: Pad IDim to Next Multiple of 32

In `pack_edge_field_compact`, allocate `ceil(514/32)×32 = 544` in IDim (instead of 514). The extra `544−514=30` elements are zero-padded and never written (DaCe domain ends at 512). The JDim stride becomes 544 = 17×32, and each row starts at a 32-element (256-byte = 2 cache-line) boundary.

```
N=516: ni_compact=514 → padded_ni=544
Row j: offset = j × 544 × 8 = j × 4352 bytes
4352 / 128 = 34  ✓  cache-line aligned
```

DaCe strides are **runtime parameters** (not compile-time constants), so this change requires no recompilation. Same compiled SDFG works for any stride value.

The same padding applies to `pack_vertex_field_padded` and the sparse field compact slice in `_pack_argument`.

## Files to Change (Opt 4)

| File | Change |
|---|---|
| `../gt4py/src/gt4py/next/modules/translator.py` | Add `_STRIDE_PAD=32`; round `ni` up in `pack_edge_field_compact` and `pack_vertex_field_padded` |
| `../gt4py/src/gt4py/next/modules/cartesian_interceptor.py` | Pad sparse field compact slice IDim to multiple of 32 |
