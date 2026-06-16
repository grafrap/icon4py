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

# Optimization 8: ComposedShiftInliner — automatic chained-connectivity shift composition

## What It Does

`ComposedShiftInliner` is a new IR pass (in `structured_backend_passes.py`, runs after the fusion loop in `apply_common_transforms`) that automatically fuses a two-level connectivity chain into direct composed shifts, eliminating an intermediate materialized field.

**Target pattern**:
```
vertex_out ← as_fieldop(λ(ptr_coeff, edge_intermediate, ...) →
  Σ_s ptr_coeff_s × shift(V2E_s)(edge_intermediate), vertex_domain
)(ptr_coeff, as_fieldop(λ(src) → concat_where(K[k,∞), shift_k(src), ...), edge_domain)(src), ...)
```

When `edge_intermediate` is a **simple per-kolor shifted passthrough** — i.e., each kolor branch is just `as_fieldop(λ(it) → deref(shift(di,dj,dk)(it)))(source_field)` — the outer V2E shifts and inner per-kolor shifts can be **composed arithmetically**:

```
composed(di, dj, dk) = (outer_V2E_di + inner_kolor_k_di,
                        outer_V2E_dj + inner_kolor_k_dj,
                        outer_V2E_dk + inner_kolor_k_dk)
```

The composed kolor (`outer_dk + inner_dk`) must land within the outer output domain's valid kolor range (validity guard). After composition and CSE on unique shifts, the intermediate `as_fieldop` is eliminated and the outer kernel reads `source_field` directly.

**Example**: V2E∘E2C2V with dk always cancelling to 0 → 24 (6×4) raw compositions → 7 unique (IDim, JDim) vertex→vertex shifts.

## Where the Pass Lives

- Registration: `apply_common_transforms` in `pass_manager.py`, **after the fusion loop** (after `FuseAsFieldOp` + `InlineLambdas` create `__iasfop_N` synthetic names)
- Also registered in `apply_fieldview_transforms` (DaCe path) after its fusion loop
- Gated on `USE_STRUCTURED_BACKEND=1`
- Exported from `cart_unroll.py` alongside the other structured passes

## Two Bugs Fixed to Make It Work in Production

**Bug 1 — Ordering**: Originally placed after `CartesianReductionUnroller` (too early). The `__iasfop_N` synthetic intermediates are created by `FuseAsFieldOp` in the fusion loop, which runs ~50 passes later. Running the inliner before the fusion loop means the intermediate doesn't exist yet — the pattern never matches.

**Bug 2 — Condition format**: `_extract_kolor_branch_shifts` only handled `K[k,k+1)` (exact/forward) conditions produced by `_build_field_concat_where_from_branches`. After `canonicalize_domain_argument` + `FuseAsFieldOp` + `InlineLambdas`, the conditions change to `K[k,∞)` (threshold/inverse) with `ir.InfinityLiteral.POSITIVE` as the upper bound. Added `_kolor_lo_from_cond()` static method to detect the infinity upper bound, and rewrote `_extract_kolor_branch_shifts` to try the exact format first, then the threshold format.

## Why It Does Not Fire on rbf_nabla4

Looking at the actual production IR for `rbf_nabla4`, `__iasfop_216` = `z_nabla4_e2` is **not** a simple per-kolor shift passthrough. It is the full `_calculate_nabla4` formula: 4 E2C2V reads of u_vert/v_vert cast to wpfloat, each multiplied by `primal_normal_vert_v1/v2`, combined with `z_nabla2_e`, `inv_vert_vert_length`, `inv_primal_edge_length`. This is wrapped in a `λ(__ct_el_6, __ct_el_7) → nabla4_asfop(...)` closure (not a bare applied `as_fieldop`), so `ComposedShiftInliner._find_eligible_inner_arg` returns None — the pass is a no-op.

## When It Does Fire

The pass fires when the inner intermediate is purely a per-kolor shift remapping — no arithmetic, just `deref(shift(di,dj,dk)(src))`. Example stencil pattern:
```python
# Pass 1: remap u_vert to edge domain (pure shift lookup, no formula)
edge_u = u_vert(E2C2V[0])          # just shifts u_vert, no multiplication

# Pass 2: accumulate across V2E
out = Σ_s coeff_s * edge_u(V2E[s])  # V2E reads the shift-passthrough intermediate
```

No current stencil in the suite has this exact shape (all edge intermediates have real arithmetic), so the pass is currently a no-op for all 10 stencils. It is useful for future stencils or if the inner passthrough shape emerges after IR simplification.

## Unit Tests

File: `../gt4py/tests/next_tests/unit_tests/iterator_tests/transforms_tests/test_composed_shift_inliner.py`

31 tests covering:
- `_extract_kolor_branch_shifts` with exact `K[k,k+1)` format (4 tests)
- `_extract_kolor_branch_shifts` with threshold `K[k,∞)` format (5 tests, including `_kolor_lo_from_cond`)
- V2E∘E2C2V 2-kolor fusion (4 tests)
- E2C∘C2E edge fusion (4 tests)
- C2E∘E2C cell fusion (3 tests)
- V2E∘E2C2V 3-kolor fusion (2 tests) — models the rbf_nabla4 composed shift pattern
- Threshold-format 3-kolor V2E∘E2C2V fusion (3 tests) — tests the production IR condition format
- Threshold-format E2C∘C2E fusion (2 tests)
- No-op cases: single outer shift, same kolor domain, invalid composed kolor (3 tests)

All 31 tests pass on the cluster.

## Files Changed

| File | Change |
|---|---|
| `../gt4py/src/gt4py/next/iterator/transforms/structured_backend_passes.py` | `_kolor_lo_from_cond` static method; rewrote `_extract_kolor_branch_shifts` to handle both `K[k,k+1)` and `K[k,∞)` |
| `../gt4py/src/gt4py/next/iterator/transforms/pass_manager.py` | Moved `ComposedShiftInliner.apply()` from after `CartesianReductionUnroller` to after the fusion loop, in both `apply_common_transforms` and `apply_fieldview_transforms` |
| `../gt4py/tests/next_tests/unit_tests/iterator_tests/transforms_tests/test_composed_shift_inliner.py` | New file: 31 unit tests |

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

# Optimization 9: rbf_nabla4 auto-opt + GPU-knob re-benchmark (2026-06-13)

Re-ran the full `DACE_OPT_EXPERIMENT` sweep plus a new GPU-knob sweep on **rbf_nabla4 only**
(@ 512×512, K=50, dace_gpu). All numbers are the **GT4Py Timer Report median** (GPU
exec-only; structured additionally pays pack/unpack on top). The target to beat is the
**unstructured** run of the same stencil.

## Baseline to beat

| Run | Median |
|---|---|
| **unstructured** (USE_STRUCTURED_BACKEND=0) | **0.863 ms** |

The structured backend is still **~13% behind unstructured** on exec time for this stencil,
even after the best tuning below.

## DACE_OPT_EXPERIMENT sweep (2-kernel form, CSI off)

| Experiment | Median | Note |
|---|---|---|
| `FD` (current default) | 1.003 ms | JDim block + scan unroll |
| `G` reuse_transients | 1.005 ms | **no longer catastrophic** (was 19 ms class) |
| `F` scan unroll | 1.006 ms | |
| `D` JDim block | 1.006 ms | |
| `none` opt_v2 | 1.015 ms | |
| `E` fuse_tasklets | 1.022 ms | **no longer catastrophic** |
| `K` KDim block | 2.440 ms | regresses — K is unit-stride, no cross-iter reuse |
| `FK` | 2.496 ms | regresses |
| `I` inline transients | 17.723 ms | **now catastrophic** (recomputes edge field) |

Takeaway: the dataflow knobs (none/F/D/FD/E/G) all cluster at ~1.00–1.02 ms (within noise).
K-blocking hurts (K is already the contiguous innermost dim — tiling it only adds overhead).
`I` inlines the materialized edge intermediate → recompute → blows up, same trap as CSI fusion.

## GPU-knob sweep (new letters, applied on top of FD)

New `DACE_OPT_EXPERIMENT` letters wired in `translation.py`:

| Letter | Knob | Value env (default) |
|---|---|---|
| `B` | `gpu_block_size` | `DACE_GPU_BLOCK_SIZE` ("64,4,1") |
| `R` | `gpu_maxnreg` | `DACE_GPU_MAXNREG` ("64") |
| `L` | `gpu_launch_factor` (clears fixed launch_bounds) | `DACE_GPU_LAUNCH_FACTOR` ("2") |
| `P` | `make_persistent=True` | — |
| `M` | `demote_fields` | `DACE_DEMOTE_FIELDS` (comma-separated names) |
| `N` | `blocking_only_if_independent_nodes=False` | — |

Note: `B` keeps the fixed `gpu_launch_bounds="256,8"`, so block-size products must stay ≤256.

| Config | Median | vs FD |
|---|---|---|
| **FDBR** (block 64,4,1 + maxnreg 64) | **0.974 ms** | **−4.1%** ✓ best structured |
| `FDB` (block 64,4,1) | 0.976 ms | −3.9% ✓ |
| `FDB` (block 128,2,1) | 0.981 ms | −3.4% ✓ |
| `FDR` (maxnreg 64) | 1.010 ms | ~0 |
| `FDL` (launch factor 2) | 1.012 ms | ~0 |
| `FD` reference | 1.016 ms | — |
| `FDR` (maxnreg 128) | 1.043 ms | +2.7% |
| `FDN` (blocking everywhere) | 1.460 ms | +44% ✗ |
| `FDP` (make_persistent) | 24.253 ms | catastrophic ✗ |

**Only `gpu_block_size` helps**: reshaping the GPU block from the default `(32,8,1)` to
`(64,4,1)` is a clean ~4% win (1.016 → 0.976 ms); `maxnreg=64` on top nudges to 0.974 ms.
`make_persistent` is catastrophic; `launch_factor`/`maxnreg`/`blocking-everywhere` don't help.
Best structured (0.974 ms) still trails unstructured (0.863 ms) by ~13% — the remainder is
structural, not reachable by these knobs.

### `demote_fields` (M) — investigated, inapplicable to rbf_nabla4

`demote_fields` flips a **non-transient** field to `transient=True` so the optimizer can
eliminate it; it warns and skips anything already transient. Dumping the SDFG arrays
(`DACE_DUMP_ARRAYS=1`) for rbf_nabla4 @ 512 shows **11 non-transient** arrays — all genuine
program I/O (`u_vert_out`, `v_vert_out`, `u_vert`, `v_vert`, `z_nabla2_e`, `ptr_coeff_1/2`,
`primal_normal_vert_v1/v2`, `inv_vert_vert_length`, `inv_primal_edge_length`) — and **129
transients** (`gtir_tmp_*`), which already include the materialized edge nabla4 field. So the
one field we'd want to demote is **already a transient** (fully eligible for fusion/elimination),
and the only demotable fields are real I/O (demoting an output discards the result). The 2-kernel
split is **not** a missing-transient-flag problem — it's inherent: the V2E reduce reads 6 distinct
edge neighbors per vertex, so the edge map and vertex map can't fuse without the recompute penalty
(= the ~4 ms CSI/`I`/direct path). `demote_fields` is therefore a no-op for this stencil. The `M`
letter + `DACE_DUMP_ARRAYS=1` diagnostic remain wired for stencils that *do* expose a non-transient
scratch field.

### Blocking dimension (1D vs 2D) — only J blocks by default; 2D doesn't help

`LoopBlocking` is single-dimension (tiles one map param into outer `coarse:N:B` + inner
sequential loop, hoisting that-dim-*independent* nodes to once-per-block). The `FD` default
blocks **only JDim**; IDim is the warp/thread dim (`unit_strides_dim`), K is the unrolled scan
loop, Kolor is trivial. Added multi-dim support: `blocking_dim` may now be a list (LoopBlocking
applied once per dim in `auto_optimize.py`), driven by `DACE_BLOCKING_DIMS` (e.g. "IDim,JDim").

Sweep (rbf_nabla4 @ 512): block-J(ref) 1.025, block-I 1.004, block-I+J 1.005 (sizes 4/8/16 all
~1.005), block-I+J+`gpu_block_size` 0.989 ms. **2D blocking is within noise of 1D and *worse*
than the 1D `FDB`/`FDBR` best (0.974–0.976)**. Reason: nabla4's body depends on both I- and
J-neighbor reads, so there are almost no dimension-independent nodes to hoist on any axis — the
blocking *dimension* is not the lever for this stencil. Capability kept for future stencils with
real per-axis reuse.

### GPU grid-dim remap (iteration order) — works, but slower

`unit_strides_dim` controls only the map iteration order (→ GPU grid-dim assignment), not the
memory strides (those are `gt_change_strides(HORIZONTAL)` → IDim stride-1, independent). Exposed
via `DACE_UNIT_STRIDES_DIMS` (default "IDim"). Setting "IDim,JDim" reorders the grid so
**JDim→blockIdx.y** and **K&Kolor→blockIdx.z** (verified in the generated `.cu`), vs the default
**K→blockIdx.y**. Result: **1.58 ms (FD) / 1.59 ms (FDB)** — ~58% *slower* than the K→y default.
Keeping K on the thread `y`-dimension (processing several K-levels per block, K contiguous after
stride setup) coalesces better than putting large-stride JDim on threads and serializing K across
`blockIdx.z`. Default stays "IDim"; knob kept for experimentation.

### `cudaFuncSetAttribute` — inapplicable (no shared memory)

The generated kernels use **0 shared memory** (`grep -c __shared__` = 0). So
`cudaFuncAttributePreferredSharedMemoryCarveout` is a no-op (the driver already gives a
zero-shared kernel max L1), and `cudaFuncAttributeMaxDynamicSharedMemorySize` is irrelevant
(nothing to size). `cudaFuncSetAttribute` only becomes useful **if** shared-memory tiling is
implemented (to request the large shared allocation for a staged edge field) — i.e. it's a
prerequisite for that structural fix, not an independent lever.

### Sequential-K (2.5D vertical loop) — implemented, works, but regresses without hoisting

NCU on the fastest config (FDB): dominant kernel `map_22` (862 µs, ~80% of runtime) is
**latency-bound but near-optimal** — achieved occupancy 86.6% (32 regs/thread, on the
max-occupancy plateau), SM Busy 66%, Mem Busy 65% / Max BW 57%, L1 53% / L2 73%, top pipe
ALU/ADU (address arithmetic). No saturated unit; occupancy is not the lever.

Implemented an opt-in `DACE_SEQUENTIAL_K=1` transform (`_sequentialize_k_dimension` in
`translation.py`): after the GPU transform it uses `dace.transformation.helpers.extract_map_dims`
to strip K out of the top-level GPU map into a nested `Sequential` map → GPU grid over
[IDim,JDim,Kolor] + per-thread `for K` loop. **Verified**: the `.cu` shows a real
`for (i_K=0; i_K<50; ++i_K)` loop, Kolor moves to blockIdx.y, results match reference.

**But it's ~85% slower** (seqK_FD 1.812, seqK_FDB 1.902 ms vs ~1.0 / 0.976). Reason: the K-loop
body still re-evaluates the full `concat_where` boundary ternaries **and** all address
arithmetic *every* K iteration — the K-independent work was **not hoisted** out of the loop
(DaCe does no automatic LICM across the new sequential map, and the boundary ternaries live
inside the tasklet so map-restructuring can't lift them). So we lost K-parallelism (which was
hiding latency) without reducing per-element work. The transform is kept (gated, off by default)
because it's a correct prerequisite — the 2.5D win only materializes once paired with
K-invariant hoisting + the `concat_where` domain-split (interior vs boundary).

## ComposedShiftInliner update (supersedes Opt 8 "does not fire")

The Opt 8 note that CSI is a no-op for rbf_nabla4 is **out of date**. CSI now fires for
rbf_nabla4 after three fixes: (1) `_resolve_inner_asfop` beta-reduces the
`λ(__ct_el_6,__ct_el_7) → nabla4_asfop(...)` closure to reach the inner edge `as_fieldop`;
(2) sparse-local args (`primal_normal_vert_v1/v2`, read via `list_get`) are passed raw and
shifted **in-body** (`list_get(n, ·⟪shift⟫(field))`) instead of bailing; (3) `infer_program`
gets `allow_uninferred=True` + `prune_empty_concat_where` at the structured infer sites, and
`prune_empty_concat_where` now also prunes `DomainAccessDescriptor.NEVER` branches. CSI then
produces a **single kernel with 0 Kolor concat_where**, structurally equal to
`rbf_nabla4_direct`.

**But fusion is a pessimization at scale** (confirms Opt 6): CSI 1-kernel = **4.35 ms** vs
2-kernel = **1.00 ms** at 512×50, because the E2C2V nabla4 is recomputed at all 6 V2E slots.
The hand-written `rbf_nabla4_direct` is **4.02 ms** — i.e. the single-kernel approach itself
is ~4× slower, not a CSI deficiency (the earlier "2× speedup" for the direct version was a
vertical-levels measurement artifact). CSI is therefore **gated off by default**, opt-in via
`GT4PY_ENABLE_CSI=1`.

## Files changed (Opt 9)

| File | Change |
|---|---|
| `../gt4py/.../dace/workflow/translation.py` | New experiment letters `B/R/L/P/M/N` + `K`; `DACE_BLOCKING_SIZE` override |
| `../gt4py/.../iterator/transforms/pass_manager.py` | CSI gated opt-in (`GT4PY_ENABLE_CSI=1`); `allow_uninferred=True` + prune at 3 structured infer sites |
| `../gt4py/.../iterator/transforms/structured_backend_passes.py` | CSI: `_resolve_inner_asfop` beta-reduce fix; sparse-arg in-body shift handling |
| `../gt4py/.../iterator/transforms/prune_empty_concat_where.py` | Prune `DomainAccessDescriptor.NEVER` branches |

Command files: `commands_stencils_nabla_512_optsweep.txt`, `commands_stencils_nabla_512_gpusweep.txt`.
Outputs: `output/csi_512_optsweep/`, `output/csi_512_gpusweep/`.

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

---

# Optimization 10: Fix DaCe CopyND Scalar Local-Memory Spill

## Problem

NCU profiling of `rbf_nabla4_direct` (512×512, K=50) shows:
- **L1TEX Local Load**: 1.04/32 bytes/sector (3% efficiency, 61% speedup potential)
- **L2 Local Store**: 1.04/32 bytes/sector (3% efficiency, 70% speedup potential)
- **Long Scoreboard stalls**: 88.5% of warp cycles stalled on L1TEX (local memory)

Root cause: 108 `dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(src, &gtir_tmp_N, 1)` calls in the kernel. The `&gtir_tmp_N` passes the address of a local variable as a `T*` function argument, forcing the compiler to allocate `gtir_tmp_N` in CUDA **local memory** (per-thread addressable stack) rather than a register. CUDA local memory for 256 threads per block is stored non-interleaved → 1 element per 32-byte sector = 3% efficiency.

Note: global memory accesses show 28–30/32 bytes/sector (89–94% efficient) — F-order arrays with IDim in `threadIdx.x` ARE correctly coalesced. The bottleneck is purely local memory.

## Fix: Direct Assignment in DaCe cpu.py

File: `/capstor/scratch/cscs/rgraf/icon4py/.venv/lib/python3.10/site-packages/dace/codegen/targets/cpu.py`

In `_emit_copy()` (line ~862), when:
- `nc == True` (no write-conflict)
- `memlet.wcr is None` (plain copy)
- `dst_expr.startswith('&')` (scalar destination, DaCe DefinedType.Scalar)
- `dynshape == 0` (static shape, confirmed by CopyND template `<..., 1>`)
- `copy_shape == [1]` (one element)

Emit `{var} = *({src_ptr});` instead of `CopyND(..., &{var}, 1)`. Direct assignment never takes the address of `{var}` → compiler keeps it in a register → no local memory spill.

This fix flows to GPU kernels via: `cuda._emit_copy` (storage mismatch) → `self._cpu_codegen.copy_memory()` → `cpu._emit_copy()`.

## Secondary Issue: K in threadIdx.y → Redundant 2D Field Loads

With `threadIdx.y = K` (8 threads) and K-stride ~6.3 MB, all 8 K-threads simultaneously load the same K-independent values (primal_normal_vert, ptr_coeff). Fix via `DACE_SEQUENTIAL_K=1` (already implemented).

## Benchmark Commands

```bash
# All 4 variants in commands_stencils_nabla_512_copyndfix.txt:
#   direct_FD:      rbf_nabla4_direct with CopyND fix   (target: < 4.47ms baseline)
#   twokernel_FD:   rbf_nabla4 (2-kernel) with fix      (target: < 1.45ms current best)
#   direct_seqK_F:  direct + seqK + scan_loop_unrolling
#   direct_seqK_FD: direct + seqK + JDim blocking + scan

rm -rf /scratch/mch/rgraf/icon4py/.gt4py_cache
sbatch scripts/santis_run_stencil_commands_slurm.sh \
  -c commands_stencils_nabla_512_copyndfix.txt \
  -o output/copyndfix_test/
```

Expected: CopyND calls in generated `.cu` replaced by `gtir_tmp_N = *(ptr + offset);`
NCU after fix: L1TEX Local Load → near 32/32 bytes/sector; Long Scoreboard stalls eliminated.

**Result (output/copyndfix_test): zero runtime improvement.** The fix correctly eliminated all
108 CopyND calls in the generated `.cu`, but exec time was unchanged (two-kernel: 1.007 ms before
and after). Root cause: the register spilling was NOT caused by CopyND address-taking. The
compiler was already spilling to local memory due to `__launch_bounds__(256, 8)` forcing only 32
registers per thread — CopyND variables were a symptom, not the cause. See Optimization 11 for
the actual fix.

---

# Optimization 11: GPU Launch Bounds — Eliminate Register Spill on Large Kernels

## Root Cause

NCU profiling of `rbf_nabla4_direct` (512×512, K=50, default `DACE_OPT_EXPERIMENT=FD`) shows:
- **83.6% of warp stalls** from local memory (Long Scoreboard stalls on L1TEX)
- **L1TEX Local Load**: 1.04/32 bytes/sector (3% efficiency)
- **L2 Local Store**: 1.04/32 bytes/sector (3% efficiency)

The D-blocking pass (blocking_dim=JDim, blocking_size=8) creates a sequential inner JDim loop of
8 iterations per thread. Each iteration introduces ~30 local double temporaries → ~237 doubles
total per thread → 474 half-registers needed. DaCe hardcoded `__launch_bounds__(256, 8)` on every
GPU map: `min_blocks=8` → only `65536 regs / (8 blocks × 256 threads) = 32 regs/thread` allocated
by the compiler. With 474 needed and 32 available, **nearly all temporaries spill to CUDA local
memory** (per-thread stack, non-interleaved across warp → 1 element per 32-byte sector = 3%
efficiency). The single fused kernel (produced by CSI or written by hand as `rbf_nabla4_direct`)
is particularly exposed because it has the full V2E×E2C2V computation in one kernel body.

## Fix: `DACE_GPU_LAUNCH_BOUNDS` env var

Added `DACE_GPU_LAUNCH_BOUNDS` support to `translation.py`. The value is passed as the
`gpu_launch_bounds` parameter to `gt_auto_optimize` (and the fallback `gt_gpu_transformation`
path). Default stays `"256, 8"` to avoid breaking existing stencils; override per-experiment:

```python
# translation.py (structured_opt_args):
_launch_bounds = _os.environ.get("DACE_GPU_LAUNCH_BOUNDS", "256, 8")
structured_opt_args = {
    ...
    "gpu_launch_bounds": _launch_bounds,
}
```

| Value | min_blocks | regs/thread | Effect |
|---|---|---|---|
| `"256, 8"` (default) | 8 | 32 | Heavy spill — only viable for small kernels |
| `"256, 2"` | 2 | 128 | **Sweet spot**: enough regs to avoid worst spill, enough occupancy |
| `"256, 1"` | 1 | 255 | Too low occupancy (1 block/SM) → worse latency hiding |
| `"0"` | — | compiler decides | Compiler chose same occupancy as lb11 → 0.925 ms |

## Benchmark Results (512×512, K=50, dace_gpu, `DACE_OPT_EXPERIMENT=FD`)

All numbers are the **GT4Py Timer Report median**. All structured runs used 512×512 grid.

| Config | Stencil | Median | vs Unstructured |
|---|---|---|---|
| Unstructured dace_gpu | rbf_nabla4 | **0.845 ms** | baseline |
| Structured FD, lb82 (default) | rbf_nabla4_direct | 4.030 ms | 4.8× slower |
| Structured FD, lb21 | rbf_nabla4_direct | **0.650 ms** | **23% faster** ✓ |
| Structured FD, lb11 | rbf_nabla4_direct | 0.925 ms | 9.5% slower |
| Structured FD, lb0 (compiler) | rbf_nabla4_direct | 0.925 ms | 9.5% slower |
| Structured FD, lb82 (default) | rbf_nabla4 (2-kernel) | 1.007 ms | 19% slower |
| Structured FD, lb21 | rbf_nabla4 (2-kernel) | 1.014 ms | 20% slower |
| CSI + FD, lb82 (no lb change) | rbf_nabla4 (CSI-fused) | 4.350 ms | 5.1× slower |
| **CSI + FD, lb21** | **rbf_nabla4 (CSI-fused)** | **0.669 ms** | **21% faster** ✓ |

Outputs: `output/launchbounds_test/`, `output/csi_lb21/`.

## Key Findings

**lb21 ("256, 2") is the sweet spot**: 2 blocks/SM → 128 regs/thread → enough registers
to eliminate the worst spilling (from 32 regs) while keeping enough occupancy (2 warps/SM
block-slot = 16 active warps) for latency hiding. lb11 and lb0 both land at 0.925 ms because
dropping to 1 block/SM loses too much latency hiding even though registers increase.

**The two-kernel is unaffected**: The FD default two-kernel (1.007 ms) stays ~1.007 ms with
lb21, because each of its two smaller kernels fits in 32 regs/thread without catastrophic
spilling. The launch bounds lever only matters for large single-kernel stencils.

**CSI + lb21 beats unstructured**: `GT4PY_ENABLE_CSI=1 DACE_GPU_LAUNCH_BOUNDS="256, 2"` on the
standard `rbf_nabla4` stencil gives 0.669 ms — 21% faster than the unstructured baseline (0.845
ms) — using no hand-written stencil. The auto-fused + lb21 result is within 3% of the
hand-written `rbf_nabla4_direct` with lb21 (0.650 ms), confirming that CSI produces a
structurally equivalent single kernel.

**Why direct kernel with lb21 is slightly faster than CSI + lb21 (0.650 vs 0.669 ms)**: minor
differences in kernel structure from the fusion pass (extra lambda indirections in the CSI IR
path).

## Command and Output

```bash
# Launchbounds sweep (direct kernel only):
sbatch scripts/santis_run_stencil_commands_slurm.sh \
  -c commands_stencils_nabla_512_launchbounds.txt \
  -o output/launchbounds_test/

# CSI + lb21 vs unstructured head-to-head:
sbatch scripts/santis_run_stencil_commands_slurm.sh \
  -c commands_stencils_nabla_csi_lb21.txt \
  -o output/csi_lb21/
```

## Files Changed

| File | Change |
|---|---|
| `../gt4py/src/gt4py/next/program_processors/runners/dace/workflow/translation.py` | `DACE_GPU_LAUNCH_BOUNDS` env var; pass `gpu_launch_bounds` to `gt_auto_optimize` and fallback `gt_gpu_transformation` |
| `commands_stencils_nabla_512_launchbounds.txt` | 5-experiment sweep: lb82/lb21/lb11/lb0 on direct, lb21 on two-kernel |
| `commands_stencils_nabla_csi_lb21.txt` | 3-way head-to-head: unstructured / two-kernel FD / CSI+FD+lb21 |
