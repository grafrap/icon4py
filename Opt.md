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

---

# Optimization 5: concat_where SetAt-Split (diffusion_vn temp+copy elimination)

## Context

Applies to `apply_diffusion_to_vn` with `limited_area=True` (the only relevant case for a
limited-area mesh). After the `limited_area` compile-time fold (27 → 12 kernels), the native
DaCe `concat_where` lowering produces, per kolor (×3):

| Map | Range | Output | Formula |
|-----|-------|--------|---------|
| `map_boundary` | FULL kolor domain (~513×512) | `gtir_tmp_X` (temp) | boundary: `vn + fac_bdydiff_v·area_edge·z_nabla2_e` |
| `map_nudging` | nudging zone (~508×507) | `vn` directly | nabla4 (`_apply_nabla2_and_nabla4_to_vn`, E2C2V) |
| `copy_X` ×2 | frame bands → `vn` | `vn[band]` | identity copy from `gtir_tmp_X` |

= **12 kernels** + 3 large temps (`gtir_tmp_3/54/104`, ~513×512×50 dbl ≈ 105 MB each →
~630 MB write+read-back traffic per call).

## Why DaCe Produces This

DaCe lowers `concat_where(cond, true, false)` by computing `false` over the **full** output
domain into a transient, computing `true` over the conditioned sub-domain, then copying the
false sub-regions back to the output. The `MapSplitter`/`GT4PyMapBufferElimination` passes that
would normally restrict and fold this live in the `gt_auto_optimize` *splitting block*, which
the structured backend disables (`disable_splitting=True`, because `propagate_memlets_sdfg`
collapses single-element `Kolor:[k,k+1)` ranges). **Confirmed empirically**: loading the saved
SDFG and applying `MapSplitter` + `gt_split_access_nodes` + `GT4PyMapBufferElimination`
standalone does **not** fire — the temp has 1 producer + 2 *overlapping* band-reads (they share
the frame corner), which blocks the split/eliminate chain. So copy removal is not achievable
post-hoc on this branch.

## Three Approaches Compared (516×516, K=50, dace_gpu, DACE_OPT_EXPERIMENT=FD)

| Approach | Kernels | Copy maps | Temps | Exec (median) |
|---|---|---|---|---|
| Native concat_where lowering | 12 | 6 | 3 (~630 MB/call) | 2.413 ms |
| `if_` full fusion (`transform_to_as_fieldop`) | 3 | 0 | 0 | **2.954 ms** ❌ |
| **SetAt-split (chosen)** | 15 | 0 | 0 | **1.879 ms** ✅ |

- **`if_` full fusion** (`concat_where.transform_to_as_fieldop`, single-field): collapses both
  branches into one `if_`-based `as_fieldop` → 1 map/kolor. Correct, but **22% slower** than
  native: the expensive nabla4 is now evaluated over the *whole* domain (incl. boundary frame)
  plus branch divergence. Kept as an **opt-in** flag (`GT4PY_CONCAT_WHERE_AS_FIELDOP=1`) for
  stencils whose branches are cheap; OFF by default.
- **SetAt-split** (chosen, default ON): keeps the expensive interior restricted to the nudging
  zone and writes the cheap boundary only over the thin frame, all direct-to-`vn`. **22% faster
  than native, 36% faster than `if_`**, despite 15 launches — it eliminates the ~630 MB/call
  temp traffic. Disable with `GT4PY_DISABLE_CONCAT_WHERE_SETAT_SPLIT=1`.

## The SetAt-Split (`ConcatWhereSetAtSplitter`)

The structured backend emits, per kolor, a SetAt of the form

```
SetAt(vn, (λ(b) → concat_where(c1, b, concat_where(c2, b, … INTERIOR)))(BOUNDARY), D_full)
```

where `b` (= BOUNDARY, full-domain cheap formula) is the shared true branch of every nested
`concat_where`, the conditions `ci` are single-axis half-space `cartesian_domain`s on
IDim/JDim (the Kolor ones are vestigial within a single-kolor domain), and INTERIOR is the
nudging `as_fieldop`. The pass (`structured_backend_passes.py`, run in `apply_common_transforms`
just before the final `infer_program`) rewrites it into direct-write SetAts:

```
SetAt(vn, INTERIOR, interior_rect)            # nudging zone, nabla4 restricted
SetAt(vn, BOUNDARY, frame_rect_i)  × ≤4       # cheap boundary, frame only — no temp, no copy
```

`interior_rect` is derived by narrowing `D_full`'s I/J range with the half-space conditions
(`IDim:[-∞,a)` → interior starts at `a`; `IDim:[a,∞)` → interior ends at `a`). The frame is the
4 non-overlapping rectangles of `D_full \ interior_rect` (empty ones dropped).

**Result: 1 interior + 4 frame = 5 SetAts/kolor × 3 = 15 maps, no temp, no copy.**

### Safety guards (only fires on the diffusion-style horizontal-threshold pattern)
- Both branches must be applied `as_fieldop` (rejects the per-kolor edge-validity mask pattern
  whose else-branch is the bare `vn` passthrough — splitting that would make the interior a
  `vn = vn` no-op).
- Every condition must be a clean single-axis half-space; bails otherwise.
- At least one IDim/JDim narrowing must occur; Kolor conditions must be vestigial; vertical
  (KDim) thresholds bail → left to the native lowering.

## Files Changed (Opt 5)

| File | Change |
|---|---|
| `../gt4py/.../iterator/transforms/structured_backend_passes.py` | New `ConcatWhereSetAtSplitter` pass + helpers (`_match_threshold_concat_where`, `_half_space_condition`, `_domain_ranges_map`, `_retarget_as_fieldop_domain`) |
| `../gt4py/.../iterator/transforms/cart_unroll.py` | Re-export `ConcatWhereSetAtSplitter` |
| `../gt4py/.../iterator/transforms/pass_manager.py` | Call splitter in `apply_common_transforms` before final `infer_program` (structured, default ON); add opt-in single-field `transform_to_as_fieldop` `if_` path (default OFF) |
| `../gt4py/.../iterator/transforms/concat_where/transform_to_as_fieldop.py` | `only_single_field` flag (skip tuple-output concat_where) |

## Env Flags

| Flag | Effect |
|---|---|
| `GT4PY_DISABLE_CONCAT_WHERE_SETAT_SPLIT=1` | Disable the SetAt-split → native concat_where lowering (12 kernels) |
| `GT4PY_CONCAT_WHERE_AS_FIELDOP=1` | Opt-in `if_` full fusion for single-field concat_where (3 kernels; only for cheap-branch stencils) |

---

# Optimization 6: rbf_nabla4 composed-reduction — fusion is NOT beneficial (keep 2 kernels)

## Context

`rbf_nabla4` is a v2e2c2v composition: `u_vert/v_vert → [E2C2V nabla4] → z_nabla4_e2 (edge
field) → [V2E reduction] → (u_out, v_out)`. The structured DaCe backend lowers it to **2
kernels**: one materialises the edge intermediate `z_nabla4_e2` once, the second does the V2E
reduction reading it at 6 neighbour edges per vertex (×2 tuple outputs).

The intermediate stays a separate map because it is passed to the outer reduction through a
lambda wrapper (`(λ(__ct_el_6,__ct_el_7) → nabla4_as_fieldop(...))(...)`), so it is not a bare
applied `as_fieldop` and `FuseAsFieldOp._arg_inline_predicate` skips it. (No exception is
raised; it is simply ineligible.)

## Measured: forcing the fusion is a ~9× regression

Forcing the fusion (`InlineLambdas(force_inline_lambda_args=True, opcount_preserving=False)` +
a final `FuseAsFieldOp`) does collapse it to **1 kernel** and stays correct, but the inlined
kernel recomputes the entire nabla4 — itself an E2C2V (4-neighbour) reduction — at all 6 V2E
slots × 2 outputs. Measured on 512×512, K=50, dace_gpu, DACE_OPT_EXPERIMENT=FD:

| Form | Kernels | exec (median) |
|---|---|---|
| **Materialised (default)** | 2 | **2.25 ms** |
| Forced fusion | 1 | **21.0 ms** (~9× slower) |

## Conclusion

The 2-kernel materialised form is **optimal** — materialising the shared edge intermediate
once is far cheaper than recomputing the nested E2C2V nabla4 per V2E slot. This is the same
recompute-vs-materialise lesson as Optimization 5 (where the `if_` full fusion lost to the
SetAt-split), here amplified by the nested reduction. **No code change shipped** — the forcing
path was an experiment, measured, and reverted (a one-paragraph note remains at the
corresponding spot in `pass_manager.py`). Fewer kernels is not a goal in itself when it trades
a materialised buffer for redundant compute of an expensive nested reduction.

---

# Optimization 7: rbf_nabla4_direct — single-kernel via V2E2C2V composed connectivity

## Context

Optimization 6 showed that forced IR fusion of `rbf_nabla4` (2 kernels: E2C2V edge kernel +
V2E vertex kernel) causes a ~9× regression by recomputing the full nabla4 at all 6 V2E slots.
The insight missing from that approach: the composed V2E×E2C2V traversal on a parallelogram
mesh reaches only **7 unique vertex positions** (center + 6 compass directions). If we express
this as a 7-slot `V2E2C2V` connectivity, the structured backend can lower each slot to a plain
`(IDim, JDim)` shift — no `concat_where` branching, no intermediate buffer, no recompute.

## Design

**New connectivity** (`dimension.py`, `map_dict.py`):
```python
V2E2C2VDim = gtx.Dimension("V2E2C2V", gtx.DimensionKind.LOCAL)
V2E2C2V = gtx.FieldOffset("V2E2C2V", source=VertexDim, target=(VertexDim, V2E2C2VDim))
```
7 slots, all "shift" kind, Kolor=0 (the V2E Kolor delta and E2C2V branch delta cancel):

| Slot | Vertex | di | dj |
|------|--------|----|----|
| 0 | center |  0 |  0 |
| 1 | N      |  0 | +1 |
| 2 | NW     | -1 | +1 |
| 3 | E      | +1 |  0 |
| 4 | SE     | +1 | -1 |
| 5 | S      |  0 | -1 |
| 6 | W      | -1 |  0 |

**24→7 slot lookup** (V2E slot × E2C2V slot → V2E2C2V slot):

| V2E \ E2C2V | 0 | 1 | 2 | 3 |
|-------------|---|---|---|---|
| **[0]**     | 0 | 1 | 2 | 3 |
| **[1]**     | 0 | 3 | 1 | 4 |
| **[2]**     | 0 | 4 | 3 | 5 |
| **[3]**     | 5 | 0 | 6 | 4 |
| **[4]**     | 6 | 0 | 2 | 5 |
| **[5]**     | 2 | 0 | 1 | 6 |

**New stencil** (`rbf_nabla4_direct.py`): vertex-domain field operator that reads each of the 7
unique vertex values once, then uses the lookup table to build the 6 per-edge nabla4 scalars,
then accumulates with ptr_coeff_1/2. No vpfloat intermediate; nabla4 stays in wpfloat.

## Structured backend prediction

Every `u_wp(V2E2C2V[i])` → pure `(IDim=di, JDim=dj)` shift via map_dict — no `concat_where`.
Every `primal_normal_vert_v1[E2C2VDim(j)](V2E[s])` → V2E "shift" kind → also a direct shift.
Expected result: **1 kernel**, 7 unique vertex reads per u/v (vs 24 naive), no temp buffer.

## Status

Implementation complete (2026-06-11):
- `dimension.py`: `V2E2C2VDim` + `V2E2C2V` added
- `map_dict.py`: 7 "shift" entries added
- `rbf_nabla4_direct.py`: new stencil created

**Pending**: unstructured correctness test + structured kernel count + timing comparison.
Build the unstructured `V2E2C2V` connectivity table from `vertex_to_ij` + offsets_7 and add
`"V2E2C2V": gtx.NeighborTableOffsetProvider(v2e2c2v, VertexDim, VertexDim, 7)` to the test's
offset_provider. After that: sbatch small-grid sweep → compare kernel count and exec time
against the 2-kernel baseline (2.25 ms at 512×512, K=50).

---

## Alternative (future): IR-level inlining + DCE + CSE + prefactor gathering

Instead of hand-writing `rbf_nabla4_direct`, one could automate this entirely at the IR level:

1. **Inline** `_calculate_nabla4` into `_rbf_nabla4` — eliminate the lambda boundary
2. **DCE** — remove any dead intermediate nodes
3. **CSE** — detect equal shift patterns (the 7 unique vertex offsets appear in all 24 terms)
4. **Prefactor gathering** — for each unique shift, accumulate all prefactors (ptr_coeff × pnv) into a single multiplication

This would produce the same 7-unique-vertex, 1-kernel result *automatically*, without manual
stencil rewriting. The key insight is that CSE exposes the shared subexpressions that the
hand-coded form makes explicit. Not yet implemented; this is the general form of the
optimization and could also benefit other stencils with composed connectivity reductions.

---

# TODO: Full Re-benchmark of DACE_OPT_EXPERIMENT

The `DACE_OPT_EXPERIMENT=FD` default was chosen based on benchmarks run before Opt 2 (origin-shift/compact packing) and Opt 3 (grid-size alignment) were implemented. Those optimizations changed the memory layout and array strides significantly — in particular, compact packing makes the write domain start at `ptr[0]` and the IDim size is now `ni_kolor − shift_i` rather than the full `ni`. The relative benefit of loop transformations like JDim blocking (`D`) and scan-loop unrolling (`F`) may have changed.

**Action required**: Re-run the full `DACE_OPT_EXPERIMENT` sweep on a 516×516 grid (K=50) after the current code state (Opt 1+2+3, plus any Opt 4/5 that get implemented) to identify the best default. Experiments to include:

| Experiment key | Description |
|---|---|
| `none` | Pure auto_optimize baseline (no extras) |
| `F` | scan_loop_unrolling only |
| `D` | JDim blocking only (blocking_size=8) |
| `FD` | Current default |
| `I` | Multi-consumer transient inlining only (new, Opt 5 candidate) |
| `FI` | Unrolling + inlining |
| `DI` | Blocking + inlining |
| `FDI` | All three |

Run all 10 stencils for each variant. Use `scripts/compare_timing_results.py` to compare against the `none` baseline. Re-record the winning combination as the new `DACE_OPT_EXPERIMENT` default in `translation.py`.
