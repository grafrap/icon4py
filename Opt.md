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

---

# Optimization 12: __maxnreg__ and Sequential-K sweep (2026-06-17)

## Goal

After establishing CSI+FD+lb21 = 0.669 ms (Opt 11), explore whether:
1. `__maxnreg__(N)` (CUDA 12.5+ attribute, no occupancy constraint) beats lb21 for large kernels
2. Removing D-blocking (F-only) changes the spill pattern significantly
3. Sequential K (structurally closest to reference `gpu_kloop`) helps

The reference kernel (`gpu_kloop_nabla4_interpolate_inlined_structured`) uses:
- Sequential K with `blockDim.z=2` stride, K-independent pre-load of 72 doubles into register arrays
- `__launch_bounds__(256)` only (no min_blocks)
- 7 unique u_vert reads per K-thread (NVCC CSE on `v2e2c2v_compressed`)
- ~96 K-independent loads done ONCE before K-loop

Our DaCe CSI kernel structural gaps (all post Opt11):
- 416 `double gtir_tmp_*` locals vs ~78 in reference
- 48 u_vert reads (2×24 because nabla4 recomputed for both outputs — no shared intermediate)
- K-independent loads (`primal_normal`, `ptr_coeff`) re-read per K-thread (L2 absorbs at K=50)
- lb21 gives 128 regs/thread → 64 doubles in registers, ~352 doubles spill

See `cuda_comparison/COMPARISON.md` for the full side-by-side analysis.

## Sweep: 8 experiments (job 933329, `output/csi_maxnreg_sweep/`)

| Label | DACE_OPT_EXPERIMENT | Launch control | Notes |
|---|---|---|---|
| csi_FD_lb21 | FD | `LAUNCH_BOUNDS="256, 2"` | Previous best — baseline verify |
| csi_F_lb21 | F | `LAUNCH_BOUNDS="256, 2"` | No D-blocking |
| csi_FD_lb10 | FD | `LAUNCH_BOUNDS="256"` | Reference-style (no min_blocks) |
| csi_FD_maxnreg62 | FDR | `MAXNREG=62` | Colleague tip — drops lb |
| csi_F_maxnreg62 | FR | `MAXNREG=62` | No D-blocking + maxnreg |
| csi_F_maxnreg128 | FR | `MAXNREG=128` | Same budget as lb21 without occupancy constraint |
| csi_seqK_F_lb21 | F | `LAUNCH_BOUNDS="256, 2"` + `SEQUENTIAL_K=1` | Sequential K, no K-hoisting in DaCe |
| csi_seqK_F_maxnreg62 | FR | `MAXNREG=62` + `SEQUENTIAL_K=1` | Closest to reference structure |
| unstruct | — (USE_STRUCTURED_BACKEND=0) | — | Unstructured baseline |

## Structural findings from CUDA analysis (other cluster, pre-Opt10)

See `cuda_comparison/COMPARISON.md` for full details. Key results:

- **FD == F** (cuda1 == cuda2): D-blocking is a **no-op** for this vertex stencil. JDim blocking doesn't create an inner sequential loop here — the gridDim stays at `(16,7,509)` with one J-block per thread.
- **FDR == FR** (cuda4 == cuda5): same no-op conclusion for maxnreg experiments
- **416 gtir_tmp** in ALL variants — CSI expansion alone (not D-blocking)
- **seqK (cuda7) is structurally broken** for vertex stencils: Kolor=[0,1) but seqK puts Kolor in threadIdx.y (8 slots) → 87.5% idle threads. K-independent loads also NOT hoisted before sequential K-loop.
- **maxnreg(62)** provides 62 regs/thread (vs lb21's 128 regs) — worse than lb21 for 416-var kernel. **maxnreg(128)** matches lb21's budget without occupancy constraint → potentially marginal improvement.

## Results (GT4Py Timer medians, 30 runs each)

### Santis cluster (A100, post-Opt10 DaCe)

| Config | Median | vs Unstructured | Notes |
|---|---|---|---|
| **csi_FD_lb21** | **0.656 ms** | **−22.5%** | Previous best (Opt 11) — confirmed |
| csi_F_lb21 | 0.655 ms | −22.7% | **Identical kernel** (FD==F confirmed) |
| csi_FD_lb10 | 0.965 ms | +14.0% | Worse — compiler picks lower occupancy |
| csi_FD_maxnreg62 | 1.009 ms | +19.1% | Worse — 62 regs too few for 416-var kernel |
| csi_F_maxnreg62 | 1.010 ms | +19.2% | **Identical to #4** (FDR==FR confirmed) |
| **csi_F_maxnreg128** | **0.655 ms** | **−22.6%** | **Ties lb21** — same effective budget |
| csi_seqK_F_lb21 | 1.141 ms | +34.7% | 87.5% Kolor thread waste + no LICM |
| csi_seqK_F_maxnreg62 | 2.677 ms | +216% | seqK waste + extreme spill |
| **unstruct (baseline)** | **0.847 ms** | baseline | |

### Other GPU cluster (pre-Opt10 DaCe, ~63% slower GPU)

| Config | Median | vs Unstructured |
|---|---|---|
| csi_FD_lb21 / csi_F_lb21 | 1.088 ms | −27.1% |
| csi_F_maxnreg128 | 1.088 ms | −27.1% |
| csi_FD_lb10 | 1.649 ms | +10.5% |
| csi_FD_maxnreg62 / csi_F_maxnreg62 | 1.753 ms | +17.5% |
| csi_seqK_F_lb21 | 1.983 ms | +32.9% |
| csi_seqK_F_maxnreg62 | 5.512 ms | +269% |
| unstruct | 1.492 ms | baseline |

## Key Findings

1. **lb21 = maxnreg(128)**: Both achieve −27% vs unstructured. The compiler at 128 max regs naturally
   selects ≥2 blocks/SM occupancy, so the explicit `min_blocks=2` in lb21 is redundant for this kernel.
   Either can be used; lb21 is more predictable.

2. **lb10 is worse than lb21**: Without `min_blocks=2`, the compiler picks lower occupancy (possibly 1 block/SM
   or higher regs that don't help), resulting in 52% slower than lb21.

3. **maxnreg(62) is significantly worse**: 62 regs/thread → more spill than lb21's 128 regs. For a 416-var
   kernel, the register budget matters more than removing the occupancy constraint. Colleague's tip confirmed
   not beneficial here.

4. **seqK is harmful for single-Kolor vertex stencils**: The seqK pass moves K out of the GPU grid and puts
   Kolor into threadIdx.y (8 threads). Since Kolor=[0,1) has only 1 valid value, 7/8 threads are idle.
   Additionally, K-independent reads (primal_normal, inv_*) are NOT hoisted before the K-loop → re-read 50×.

5. **D-blocking is a no-op**: FD==F and FDR==FR produce identical kernels (0-diff). D-blocking
   (`blocking_dim=JDim`) has no effect on this vertex stencil's GPU grid or variable count.

**Conclusion**: lb21 (`__launch_bounds__(256, 2)`) remains optimal. No improvement found from maxnreg or seqK.
The 416 gtir_tmp register spill is the fundamental bottleneck — addressing it would require DaCe-level CSE
or a restructured IR that materializes z_nabla4_e2[6] as an explicit intermediate.

## Files Created

| File | Purpose |
|---|---|
| `commands_stencils_nabla_csi_sweep.txt` | 8-experiment sweep command file |
| `cuda_comparison/rbf_nabla4_existing_cache.cu` | Saved DaCe kernel from cache (CSI+FD+lb21, Jun 16 2026) |
| `cuda_comparison/COMPARISON.md` | Full structural comparison: DaCe kernel vs reference `gpu_kloop` |

---

# Optimization 13: CSI Let-Bindings — Share nabla4 Intermediate Between Tuple Outputs

## Root Cause

The CSI-fused `rbf_nabla4` kernel has **416 `gtir_tmp_N` double variables** (vs ~78 in the
reference). The main contributor: the 6 nabla4 sub-expressions (one per V2E edge) are each
computed TWICE — once in the u_vert_out sum and once in the v_vert_out sum. This gives
2 × 6 × ~20 sub-expression nodes = ~240 variables just for nabla4, instead of the 6 × ~20 = 120
that sharing would give. The reference kernel avoids this via `z_nabla4_e2_wp[6]`, computed once
and reused for both outputs.

Confirmed from IR analysis (ir_out.txt lines 3288–3388): the SAME nabla4 expressions appear
identically in both the u_out and v_out sections of the CSI lambda body.

## Fix: Let-Bindings in `_try_inline_intermediate`

**File**: `../gt4py/src/gt4py/next/iterator/transforms/structured_backend_passes.py`

When `len(all_shifts) > len(unique_outer_shifts)` (i.e. some intermediate shifts appear more
than once in the outer body — the tuple-output case), `_try_inline_intermediate` now:

1. Creates `__csi_inner_k` SymRef placeholders for each unique shift
2. Substitutes with SymRefs (not full expressions) in the outer body
3. Wraps the body in nested let-bindings: `(λ(__csi_inner_k) → …)(nabla4_k_expr)` for each k

DaCe's `visit_FunCall` in `gtir_dataflow.py` already handles `cpm.is_let` nodes by computing
each argument ONCE into a shared transient and mapping the param to that transient in
`symbol_map` — so both u_out and v_out use the same scalar transient, not duplicated reads.

`InlineLambdas(opcount_preserving=True)` preserves the let-bindings since each `__csi_inner_k`
is used in ≥2 places (u_out and v_out). The DaCe pipeline has no post-CSI InlineLambdas anyway.

Single-output stencils: `len(all_shifts) == len(unique_outer_shifts)` → else branch (no change).

## Expected Impact (pending benchmark)

| Metric | Before | After |
|---|---|---|
| `gtir_tmp_N` doubles | ~416 | ~250–280 |
| nabla4 sub-expressions | 6 × ~20 × 2 = ~240 | 6 × ~20 = ~120 |
| u_vert global reads | 48 (7 unique × ~7×) | ~24 (7 unique × ~3×) |
| Spill with lb21 (128 regs) | ~352 | ~180–210 |
| Kernel count | 1 | 1 (unchanged) |

## Files Changed

| File | Change |
|---|---|
| `../gt4py/src/gt4py/next/iterator/transforms/structured_backend_passes.py` | `_try_inline_intermediate`: conditional let-binding when sharing occurs |
| `../gt4py/tests/.../test_composed_shift_inliner.py` | `TestLetBindingForTupleOutput`: 6 new tests verifying let-binding structure |

## Verification Command

```bash
# After submitting to cluster:
export PYTHONOPTIMIZE=1 && export USE_STRUCTURED_BACKEND=1 && export OMP_NUM_THREADS=1 \
  && export GT4PY_TRANSLATOR_MESH=.../parallelogram_grid_512.nc \
  && export DACE_OPT_EXPERIMENT=FD && export GT4PY_ENABLE_CSI=1 \
  && export DACE_GPU_LAUNCH_BOUNDS="256, 2" \
  && pytest -q test_rbf_nabla4.py --backend=dace_gpu --grid .../512.nc:50 --maxfail=1 -s
# Check: grep -c "double gtir_tmp_" <kernel>.cu  → should be ~250-280 (was 416)
# Check: grep -c "double __tlet_arg0 = u_vert\[" <kernel>.cu → should be ~24 (was 48)
```

---

# Investigation: opcount_preserving=False at pass_manager.py line 558

## What the opcount_preserving does

`InlineLambdas(opcount_preserving=False, force_inline_lambda_args=True)` at line 558 of
`apply_common_transforms` is a **destructive flatten** pass that inlines ALL lambda-application
let-bindings, even those where params are used ≥2 times. This is required for BOTH the GTFN
and DaCe backends (neither can handle remaining `_cs_*` let-bindings in the output IR).

## Why it is needed

After the fusion loop (lines ~471–509) and CSI (line 523), CSE at line 526 extracts repeated
subexpressions INSIDE `as_fieldop` stencil bodies as `_cs_*` let-bindings with ref_count ≥ 2.
These include:
- `primal_normal_vert_v1_*` — `list_get(n, deref(shift(k)(sparse_param)))` results reused across V2E edges
- `_cs_N` — arithmetic sub-expressions appearing in both u_out and v_out nabla4 terms

`InlineLambdas(opcount_preserving=True)` at line 530 preserves them. Then line 558 destroys
them. Without line 558, both GTFN and DaCe fail (DaCe: 98.4% mismatch).

## Why DaCe fails with preserved _cs_* lets

The DaCe `is_let` handler (gtir_dataflow.py line 1955) handles simple scalar lets correctly
(proven by `repro_dace_let.py` reproducer). The issue is specific to the `list_get`-derived
lets (`primal_normal_vert_v1_` etc.). When `_visit_list_get` creates a `ValueExpr` for a sparse
field slot, using that `ValueExpr` multiple times in a let-binding context likely causes issues
in the DaCe SDFG (wrong connections or wrong data flow for the list-type transient).

## The fundamental blocker

All IR-level nabla4-sharing approaches are blocked by this constraint: any let-binding that
survives to line 558 gets destroyed. Moving CSI after line 558 makes the let-bindings survive
but still fails with 98.4% (likely infer_domain or SDFG construction issue for that placement).

## Current approach: SDFG-level (gated, dead end)

`gt_merge_identical_scalar_reads`: correctly identifies 28 (small grid) / 154 (512×512) redundant
reads in the SDFG, but DaCe re-materializes during codegen → no variable count reduction →
+2.7% slower (0.674 vs 0.656 ms). Gated behind `GT4PY_ENABLE_MISR=1` (off by default).

---

# Optimization 14: Fix SeqK Dimension Ordering (I→J→Kolor, K sequential)

## Problem

`DACE_SEQUENTIAL_K=1` experiment was 85% slower (1.983ms vs 0.656ms) because
`_sequentialize_k_dimension` stripped K from the GPU map but left Kolor in threadIdx.y=8 slots
when Kolor=[0,1) has only 1 valid value → 87.5% thread waste.

## Fix

Changed default `DACE_UNIT_STRIDES_DIMS` from `"IDim"` to `"IDim,JDim,Kolor"` in
`translation.py`. With `unit_strides_dim=["IDim","JDim","Kolor"]`, `gt_set_iteration_order`
produces map param order `[K, Kolor, JDim, IDim]`. After seqK strips K:
- IDim → threadIdx.x (32, warp) ✓
- JDim → threadIdx.y (8) ✓
- Kolor → threadIdx.z (1 for vertex, zero waste) ✓

Matches reference kernel structure: block=(32,4,2), grid=(I/32,J/4,1), K sequential.

## Results (job 938296, pending)

| Config | Previous (broken ordering) | Expected (fixed ordering) |
|---|---|---|
| parallel-K baseline | 0.656 ms | ~0.656 ms (no regression) |
| seqK + lb21 | 1.983 ms (+202%) | TBD |
| seqK + maxnreg(62) | 2.677 ms (+307%) | TBD |
| seqK + maxnreg(128) | N/A | TBD |

## Complete Opt 14 Results (dimorder sweep, job 938296)

| Config | Median | vs old parallel-K 0.656ms |
|---|---|---|
| parallel-K with IDim,JDim,Kolor ordering | 1.475ms | +124.9% **regression** |
| **seqK + F + lb21, fixed ordering** | **0.685ms** | **+4.5%** |
| seqK + F + maxnreg(62), fixed | 2.982ms | +354.5% |
| **seqK + F + maxnreg(128), fixed** | **0.658ms** | **+0.4% ≈ ties!** |

**Key findings:**
- IDim,JDim,Kolor ordering is WRONG for parallel-K (K loses its efficient y-thread placement)
- IDim,JDim,Kolor ordering with seqK ELIMINATES the 87.5% Kolor thread waste
- **seqK+maxnreg128 = 0.658ms ≈ parallel-K lb21 = 0.656ms** — essentially tied!
- seqK is now viable; with K-invariant hoisting (LICM) it could beat parallel-K

**Implementation**: `translation.py` now auto-selects `unit_strides_dim`:
- Default (parallel-K): `"IDim"` → params [JDim,Kolor,K,IDim]
- With `DACE_SEQUENTIAL_K=1`: `"IDim,JDim,Kolor"` → params [K,Kolor,JDim,IDim], K stripped

## New best configuration

**seqK + maxnreg(128) + CSI** = 0.658ms = effectively same as parallel-K lb21 (0.656ms).
SeqK opens the door to K-invariant hoisting (LICM) which would be the next optimization:
with 50 K-levels, each thread loops 50 times over K-independent primal_normal/ptr_coeff
reads → hoisting would save 49/50 = 98% of those reads from the K-loop.

---

# Optimization 15: LICM (K-invariant hoisting) with element-count threshold

## Problem

The map-entry-connector LICM (Optimization 14 follow-up) hoisted 6 K-invariant reads before
the sequential K-loop: ptr_coeff_1/2 (6 doubles each), primal_normal_vert_v1/v2 (48 doubles each),
inv_vert_vert_length (12 doubles), inv_primal_edge_length (12 doubles) — **132 doubles total**.
This caused a **4.8% regression** vs seqK-no-LICM (0.718ms vs 0.685ms) because storing 132 doubles
in per-thread Register arrays exceeded the lb21 budget (128 regs/thread → L2 spilling).

## Fix: `DACE_LICM_MAX_ELEMS` threshold in `licm_sequential_k.py`

Added env-var `DACE_LICM_MAX_ELEMS` (default **12**) to skip hoisting any K-invariant subset
with more elements than the threshold. This keeps only arrays that fit in registers:

| Array | n_elems | threshold=12 | threshold=6 |
|-------|---------|-------------|-------------|
| ptr_coeff_1 / ptr_coeff_2 | 6 each | ✅ hoist | ✅ hoist |
| inv_vert_vert_length / inv_primal_edge_length | 12 each | ✅ hoist | ❌ skip |
| primal_normal_vert_v1 / primal_normal_vert_v2 | 48 each | ❌ skip | ❌ skip |

## Results (512-grid, K=50)

| Config | Hoisted | Median |
|--------|---------|--------|
| seqK + no LICM (Opt 14 baseline) | 0 arrays | 0.685ms |
| seqK + LICM all (threshold=∞) | 6 arrays, 132 doubles | 0.718ms **−4.8%** |
| **seqK + LICM threshold=12** | **4 arrays, 36 doubles** | **0.645ms +6.2%** |
| seqK + LICM threshold=6 | 2 arrays, 12 doubles | 0.646ms +5.9% |
| parallel-K + FD + lb21 (previous best) | — | 0.672ms |
| unstructured baseline | — | 0.848ms |

**Best result: seqK + LICM(threshold=12) = 0.645ms** — beats parallel-K by 4.1% and
unstructured by 31.6%.

Threshold=12 and threshold=6 are nearly identical, suggesting the `inv_*` arrays (12 elems,
the 2×2×3 neighbourhood block) don't add meaningful register pressure but also don't help
much. The primary gain comes from `ptr_coeff_1/2` (6 V2E slot values each).

## CUDA Code Verified

`__tmp0[6]` (ptr_coeff_1) and `__tmp1[6]` (ptr_coeff_2) are loaded via `CopyND` before the
K-loop, then accessed inside as `*(__tmp0 + slot)` — confirmed correct hoisting.

## Remaining Register Pressure Issues (CUDA analysis, for future work)

Profiling the threshold=12 CUDA kernel reveals three remaining issues:

### 1. 426 computation intermediates (`gtir_tmp_*`) inside K-loop — primary register pressure
The fully-unrolled CSI+F stencil declares 426 `double gtir_tmp_N` scalars inside the K-loop
body. With 128 regs/thread (lb21), the vast majority must spill to L2 local memory. These are
unavoidable for the current computation structure (fully-unrolled E2C2V × V2E × slots).

### 2. 99 K-invariant direct pointer reads of `primal_normal_vert_v1/v2` inside K-loop
CSI generates `gtir_tmp_N = *(primal_normal_vert_v1 + IDim+JDim+Kolor_offset)` where the
offset has NO K term. These bypass LICM (which works at the map-entry-connector level, not
direct pointer level). They repeat identically every K-iteration. After the first K-iteration,
they hit L1 cache, so the global-memory cost is small — but the address arithmetic
(4× multiply-add per read × 99 reads × 50 K-iters) is pure ALU waste.

### 3. 10 K-invariant `__map_fusion_gtir_tmp_*[1]` CopyND reads of `inv_*` inside K-loop
CSI's `FuseAsFieldOp` generates 10 single-element CopyND reads of `inv_primal_edge_length`
and `inv_vert_vert_length` at specific neighbor positions (e.g., [i-1, j, kolor+2]), declared
as `double[1]` arrays INSIDE the K-loop. These are also K-invariant but bypass LICM.
Additionally, `double[1]` inhibits register allocation compared to plain `double`.

### Future optimization directions

| # | Optimization | Impact | Complexity |
|---|---|---|---|
| A | **Inner-scope LICM**: scan K-loop tasklets for `*(array+K-free-offset)` and hoist to pre-K scalars | Eliminates 99+10=109 K-repeated reads; saves ~5341 redundant accesses/thread/call | Medium |
| B | **`double[1]` → `double`** for `__map_fusion_gtir_tmp_*` via DaCe `ArrayToScalar` | Enables register allocation for 10 size-1 arrays | Low |
| C | **Pre-factor geometry**: split `primal_normal * u_vert` into K-invariant factor + K-dep dot product | Could reduce 426 intermediates significantly; requires stencil restructuring | High |

## Files changed (Opt 15)

| File | Change |
|------|--------|
| `../gt4py/.../transformations/licm_sequential_k.py` | `DACE_LICM_MAX_ELEMS` threshold; debug skip messages; array transient with local subsets |
| `../gt4py/.../workflow/translation.py` | `DACE_LICM_DISABLE=1`, `DACE_SEQK_DEBUG=1`, `DACE_LICM_MAX_ELEMS` env-var wiring |

---

# Optimization 16: Pre-factored nabla4 Stencil (Option C)

## Problem

Despite LICM(threshold=12) hoisting 4 arrays (36 doubles), the K-loop still had **426
`gtir_tmp_*` scalars** causing register spilling. The root cause: CSI generates
`primal_normal_vert_v1/v2` reads as direct pointer dereferences `*(pn_v1 + offset)` inside
the K-loop body — these bypass the connector-level LICM (which only sees IN-edges to the
Sequential map boundary).

## Approach: Pre-compute combined geometry weights as separate EdgeDim fields

Restructure `_calculate_nabla4` to pre-compute K-invariant combined weights before the K-loop:
```
W_v1[slot] = 4 * pn_v1[slot] * inv_pel^2   (tang slots 0,1)
           = 4 * pn_v1[slot] * inv_vvl^2   (norm slots 2,3)
W_v2[slot]: same with pn_v2
W_z        = -8 * (inv_vvl^2 + inv_pel^2)
```
The K-loop then reduces to: `Σ_s (W_v1[s]*u(E2C2V[s]) + W_v2[s]*v(E2C2V[s])) + W_z*z`

## Attempts and Findings

### v1: Weights precomputed in numpy (outside GT4Py program)
- **0.539ms** (512-grid, K=50) — best result for this approach
- LICM skips weights (48 elements = 2×2×3×4 spatial neighbourhood) — weights are sparse
  fields `[EdgeDim, E2C2VDim]` that expand to 48-element neighbourhood blocks in structured
- Timing does NOT include weight precomputation (done in numpy fixture)
- **Practical validity**: primal_normal/inv_length are grid constants updated once per grid,
  not per timestep — precomputing weights at initialization is architecturally sound

### v2: Weights computed inside GT4Py program (two-step: weight kernel + K-loop)
- **0.680ms** (512-grid, K=50) — SLOWER than seqK+LICM(12) = 0.645ms
- Root cause: **4 GPU kernels** (per-kolor split creates 3 weight kernels + 1 K-loop)
- K-loop has 236 intermediates (better than 426 but still > 128 → spilling)
- Multi-kernel launch overhead outweighs the savings

## Key Insight: Why pre-factoring alone is insufficient

The V2E × E2C2V chain (6 V2E neighbors × 4 E2C2V slots = 24 vertex reads) inherently
generates ~240 computation intermediates regardless of weight pre-factoring, because the
products `weight[s] × u_vert(E2C2V[s])` still generate one intermediate per multiplication
per K-iteration. Pre-factoring reduces intermediates from 426 → 236, but not below 128.

## Result Summary

| Config | Median | vs unstr | vs parallel-K | Kernels |
|--------|--------|----------|---------------|---------|
| seqK + LICM(12) | 0.645ms | +31% | +4% | 1 |
| v2 prefactored (in-program weights) | 0.680ms | +25% | — | **4** |
| v1 prefactored (numpy precomputed) | **0.539ms** | **+57%** | **+24%** | 1 |

## Files created (Opt 16)

| File | Role |
|------|------|
| `model/atmosphere/diffusion/src/.../stencils/calculate_nabla4_prefactored.py` | Simplified K-loop FO with pre-computed weight fields `[EdgeDim, E2C2VDim]` |
| `model/atmosphere/diffusion/src/.../stencils/rbf_nabla4_prefactored.py` | Wraps prefactored nabla4 + RBF interpolation |
| `model/atmosphere/diffusion/src/.../stencils/rbf_nabla4_prefactored_v2.py` | Two-step program: weight computation (EdgeDim) + K-loop (VertexDim×K) |
| `model/atmosphere/diffusion/tests/.../test_rbf_nabla4_prefactored.py` | Test for v1 (numpy weight precomputation) |
| `model/atmosphere/diffusion/tests/.../test_rbf_nabla4_prefactored_v2.py` | Test for v2 (in-program weight computation) |

---

# Optimization 17: Per-element LICM for large neighbourhood arrays

## Problem

LICM(threshold=12) skipped `primal_normal_vert_v1/v2` (n_elems=48) to avoid register pressure
from 48-element Register arrays. But this left 99 K-invariant pointer reads of these arrays
INSIDE the K-loop (generated by CSI as `*(pn_v1 + offset)` patterns). These 99 reads per
K-iteration × 49 wasted reads = 4851 extra memory accesses per thread per call.

## Fix: Per-element fallback in `_hoist_from_sequential_map`

When a large array (n_elems > `DACE_LICM_MAX_ELEMS`) is encountered, instead of skipping,
the function groups the Sequential map's OUT-edges by their specific accessed element position
and creates **one Register scalar per unique position**. Each scalar has n_elems=1.

Key difference from the original threshold approach:
- **Before (threshold=12)**: `primal_normal_vert_v1` → 1 × 48-element Register array → L2 spill risk
- **After (per-element)**: `primal_normal_vert_v1` → 48 × 1-element Register scalars → all in registers

The compiler can keep 48 individual `double` scalars in registers far more easily than one
`double[48]` local array, because scalar allocation is per-variable while arrays require
contiguous memory that may exceed register file capacity.

## Result

- **52 K-invariant reads hoisted** (vs 4 with threshold=12, vs 6 with threshold=∞)
  - ptr_coeff_1/2: 6 elements each → 2 arrays hoisted (as before)
  - inv_vert_vert_length: 12 elements → 1 array hoisted (as before)
  - inv_primal_edge_length: 12 elements → 1 array hoisted (as before)
  - **primal_normal_vert_v1: 48 unique element positions → 48 scalars hoisted**
  - **primal_normal_vert_v2: 48 unique element positions → 48 scalars hoisted**
- **0.633ms** (512-grid, K=50, single kernel, original `rbf_nabla4` stencil)

## Full progression table (rbf_nabla4, 512-grid, K=50)

| Config | Median | vs unstr | vs parallel-K | Kernels |
|--------|--------|----------|---------------|---------|
| Unstructured | 0.848ms | — | — | 1 |
| Parallel-K + FD + CSI + lb21 | 0.672ms | +26% | — | 1 |
| seqK (no LICM) | 0.685ms | +24% | — | 1 |
| seqK + LICM all (threshold=∞) | 0.718ms | +18% | −7% | 1 |
| seqK + LICM(threshold=12) | 0.645ms | +31% | +4% | 1 |
| **seqK + per-element LICM** | **0.633ms** | **+34%** | **+6%** | **1** |
| seqK + LICM(12) + prefactored v1 (numpy weights) | 0.539ms | +57% | +24% | 1 |

## Files changed (Opt 17)

| File | Change |
|------|--------|
| `../gt4py/.../transformations/licm_sequential_k.py` | Added `_hoist_per_element()` function; modified `_hoist_from_sequential_map()` to call it for n_elems > threshold instead of skipping |

---

# New-cluster results (2026-06-23): the seqK conclusion inverts; output fission

All numbers above (Opt 1–17) were measured on the **old cluster**. Re-running the `rbf_nabla4`
optimization sweep (`output/rbf_optsweep/`, 16 configs) on the **current cluster**
(516×516 grid, K=50, `dace_gpu`) gives materially different conclusions. **All timings here are
the GT4Py Timer Report median** (more exact than the `[timing] exec=` lines).

## Sweep results (GT4Py Timer median, this cluster)

| # | Config | median | vs unstruct |
|---|--------|--------|-------------|
| 05 | structured FDB lb21 (block 64×4×1) | **1.148 ms** | 0.78× (fastest) |
| 03/04/06/07/08 | structured {FD, F, none, CSI-F, CSI-FD} lb21 | 1.157–1.158 ms | 0.79× |
| 01 | **unstructured baseline** | 1.472 ms | 1.00× |
| 09/12 | seqK {F, CSI-F} lb21 no-LICM | 1.641 ms | 1.11× |
| 11/14 | seqK FR maxnreg128 LICM12 | 1.65 ms | 1.12× |
| 10/13 | seqK {F, CSI-F} lb21 LICM12 | 1.749 ms | 1.19× |
| 02 | structured FD **lb82** | 6.744 ms | 4.58× (slow) |
| 15 | rbf_nabla4_direct FD lb82 *(ignored)* | 7.40 ms | — |
| 16 | rbf_nabla4_direct FD lb21 *(ignored, no timer)* | — | — |

### Cache → run mapping (`.gt4py_cache/`, by `__launch_bounds__` + mtime)

| run | cache hash (prefix) | kernels | launch_bounds |
|-----|---------------------|---------|---------------|
| 01 unstruct | b22cb294 | 2 | (256) |
| 02 FD lb82 | 406632048a | 1 | (256, 8) |
| 03 FD lb21 | 2a6de0400b | 1 | (256, 2) |
| 04 F lb21 | f71dab4ed7 | 1 | (256, 2) |
| 05 FDB lb21 | 22830281 | 1 | (256, 2) |
| 06 none lb21 | 2cfc100f | 1 | (256, 2) |
| 07 CSI-F lb21 | 4ef46f53 | 1 | (256, 2) |
| 08 CSI-FD lb21 | 4dff099a | 1 | (256, 2) |
| 09 seqK F | b846999b | 1 | (256, 2) |
| 10 seqK F LICM12 | 9b3b87d4 | 1 | (256, 2) |
| 11 seqK FR maxnreg128 | 9a578b3c | 1 | (none → -maxrregcount) |
| 12 seqK CSI-F | f1736207 | 1 | (256, 2) |
| 13 seqK CSI-F LICM12 | c63fbad4 | 1 | (256, 2) |
| 14 seqK CSI FR maxnreg128 | 510c910b | 1 | (none) |
| 15 direct lb82 | direct_e43a9 | 1 | (256, 8) |
| 16 direct lb21 | direct_e67c2 | 1 | (256, 2) |

## Findings

1. **Structured already beats unstructured** (1.148 vs 1.472 ms, 1.28×).
2. **Launch bounds is the dominant lever**: lb82 → lb21 = **5.8× speedup**, a pure register-cap
   effect (32 vs 128 regs/thread). Register spilling is the real bottleneck.
3. **At lb21 every opt flag ties within ~1%** (FD/F/none/CSI/FDB-block all ≈1.157 ms) — the flags
   (scan-unroll, JDim-block, CSI fusion, tasklet fuse) do not touch the register ceiling.
4. **This cluster INVERTS Opt 14–17**: there seqK + per-element LICM was the winner
   (0.633 vs 0.672 ms parallel-K). Here **seqK regresses** (1.64–1.75 ms vs 1.148 ms parallel-K) and
   LICM makes seqK worse. The whole seqK + LICM line no longer applies on this GPU; the parallel-K
   path wins decisively. (Absolute times also scaled ~1.7× vs the old cluster — different GPU.)
5. **Coalescing is fine**: warp = `threadIdx.x` → IDim and `gt_change_strides` makes IDim stride-1
   on GPU, so main field reads are coalesced. ncu's 11% "excessive" is the small K-invariant
   connectivity/coefficient gathers (`primal_normal_vert`/`ptr_coeff` descriptor loads) — secondary.

## Root cause: one fused register-heavy kernel

`rbf_nabla4` is tuple-output (`out=(u_vert_out, v_vert_out)`). The final GTIR (confirmed in
`ir_out_nabla4.txt`) is a **single** `as_fieldop` whose lambda returns a tuple —
`{u_vert_out, v_vert_out} @ {dom, dom} ← as_fieldop(λ(p…) → make_tuple(u_body, v_body), dom)(args…)`
— which DaCe lowers to **one** map `map_29_fieldop` with **1323 `double` temporaries / 1256
multiplies** (both u and v fully unrolled, V2E×6 · E2C2V×4). That far exceeds the 255-register HW
limit → spilling. lb21 (128 regs) only mitigates it. (The earlier guess that the IR was
`make_tuple(as_fieldop_u, as_fieldop_v)` → two maps fused was wrong: it is one tuple-output
as_fieldop, lowered directly to one map, so there is no fusion to prevent.)

Read-count split in the generated `.cu`: **per-output (register-driving)** `primal_normal_vert_v1`
418 reads (u-only), `primal_normal_vert_v2` 418 (v-only), `ptr_coeff_1/2` 93 each;
**shared (cheap, cached)** `inv_primal_edge_length` 103, `inv_vert_vert_length` 103. So the
register-driving work is per-output and the shared work is small → fission is the right lever.

## Optimization: output fission

**Goal**: split `u_vert_out` and `v_vert_out` into **two separate kernels**, halving the
live-register footprint per kernel (the shared geometry reads are cheap, coalesced and L1-cached, so
duplicating them is far cheaper than the spilling).

- **SDFG-level `MapFission` (rejected)**: `MapFission.can_be_applied` matches (u/v *are* independent
  components), but `apply()` crashes in DaCe `propagate_memlets_state` with a dimension mismatch on
  the shared `inv_*` reads — the same structured-SDFG memlet-propagation fragility that forces
  `disable_splitting=True`. Not viable.
- **Output-aware anti-fusion alone (insufficient)**: refusing MapFusionHorizontal to merge
  disjoint-output maps does nothing here, because the tuple as_fieldop lowers to **one** map — there
  are never two maps to keep apart. (Confirmed: the first fission sweep still produced 1 kernel.)
- **IR-level split + anti-fusion (chosen)**: a new GTIR pass `TupleOutputSetAtSplitter`
  (`structured_backend_passes.py`) rewrites the single tuple-output SetAt into one single-output
  SetAt per element — `out_i @ dom_i ← as_fieldop(λ(p…) → body_i, dom)(args…)` — so each output
  lowers to its own map/kernel. It runs in `apply_common_transforms` right before the final
  `infer_program` (which repopulates domain annexes) when `DACE_FISSION_OUTPUTS=1`. Each split lambda
  keeps the full param/arg list; params unused by `body_i` are pruned by type inference / DCE in the
  DaCe lowering. The **output-aware branch of `_kolor_aware_fusion_callback`** (`translation.py`) then
  prevents DaCe from re-merging the two split maps (refuses fusing maps with disjoint non-transient
  outputs). Both gated on `DACE_FISSION_OUTPUTS=1`, off by default.

### Files changed

| File | Change |
|------|--------|
| `../gt4py/.../iterator/transforms/structured_backend_passes.py` | New `TupleOutputSetAtSplitter` pass (split tuple-output SetAt → one single-output SetAt per element) |
| `../gt4py/.../iterator/transforms/cart_unroll.py` | Re-export `TupleOutputSetAtSplitter` |
| `../gt4py/.../iterator/transforms/pass_manager.py` | Call `TupleOutputSetAtSplitter` before the final `infer_program`, gated `USE_STRUCTURED_BACKEND` + `DACE_FISSION_OUTPUTS` |
| `../gt4py/.../runners/dace/workflow/translation.py` | `_map_nontransient_output_arrays()` helper; `DACE_FISSION_OUTPUTS=1` output-aware refusal in `_kolor_aware_fusion_callback` (prevents re-fusion of split maps) |
| `commands_stencils_rbf_fission.txt` | 4-row sweep: FUSED baseline vs FISSION at lb `256,2` / `256,1` / `0` |

### Output-fission result: regression (dead end)

The split worked (the tuple SetAt became two single-output SetAts → two kernels), but **u_vert_out
and v_vert_out share the entire `z_nabla4_e2` computation** (they differ only in `ptr_coeff_1` vs
`ptr_coeff_2`), so each kernel re-computes the shared work → **~2× runtime**. Per-output param/arg
pruning helped but cannot remove the shared part. Output fission is the wrong cut for this stencil.
The code (`TupleOutputSetAtSplitter` + output-aware `_kolor_aware_fusion_callback`) is left in,
**gated off** (`DACE_FISSION_OUTPUTS`, default off) — it is valid for stencils whose outputs do not
share work.

### Materialization result: regression (dead end)

Materializing the shared `z_nabla4_e2` as its own edge kernel (opt-in `GT4PY_MATERIALIZE_SHARED=1`,
which makes `_arg_inline_predicate` in `fuse_as_fieldop.py` refuse to inline a producer read at >1
shifted positions) is **correct but slower** (1.75 ms): `z_nabla4_e2` is a `[514,514,3,50]` ≈ **317 MB**
edge field, and the global write+read round-trip costs more than the recompute it saves. Left gated
off.

---

# Optimization 18 (2026-06-23): compile-time stride + origin baking — the actual win

## Root cause (ncu roofline on the fused lb21 kernel)

The fused `rbf_nabla4` kernel (`map_29_fieldop`, 1.157 ms) is **latency/occupancy-bound, NOT memory-
or compute-bound**: SM throughput 38%, memory 54% (both <60% → ncu "latency issue"); **achieved
occupancy 22% (theoretical 25%), limited to 128 registers/thread**; eligible warps/scheduler 0.78,
61% of cycles with no eligible warp; `long_scoreboard` (global-load) stalls dominate. ncu estimated
~46% headroom from raising occupancy.

The lb sweep confirmed **lb21 (2 blocks/SM, 128 regs) is the spill-free sweet spot** (lb1=1.76 ms,
lb3=1.33 ms, lb4=1.77 ms, lb82=6.74 ms): the kernel's live-register *need* is ~128, so fewer regs
spill and more regs waste. The only way past 1.157 ms is to **reduce the register need** so a higher
block count (occupancy) fits without spilling.

The ncu Source view showed why registers are scarce: the structured backend passes **every field
stride and field origin (`range_0`) as a runtime kernel argument**, so the compiler cannot prove the
warp dim (IDim) is unit-stride. It emits **register-indirect gather loads** (`LD … [R##.64]`,
uncoalesced — 11% excessive sectors) and burns registers on per-thread address arithmetic.

## The optimization

Bake the field **strides and origins as compile-time constants** so the compiler folds the entire
neighbour-offset address arithmetic to literals → coalesced warp loads + freed address registers →
higher occupancy without spilling. Opt-in via **`GT4PY_BAKE_STRIDES=1`**.

The values must be the *exact* per-field packed strides/origins — they are **not derivable from the
grid size**, because the origin-shift / cache-line padding (Opt: `pack_vertex_field_padded`,
`pack_edge_field_compact` pad IDim to a multiple of `_STRIDE_PAD=32`) makes them per-field (e.g. a
517-vertex grid packs to stride **528**, not 517). The robust source is the **packed arrays
themselves**: `GenericStructuredWrapper.__call__` packs (line ~1336) *before* triggering compile
(line ~1343), so the exact strides/origins are available at compile time.

### Mechanism

1. **`cartesian_interceptor.py`** — `__call__`, after packing, captures per field:
   - element strides via `_packed_element_strides` (real `ndarray.strides // itemsize`, F-order
     shape fallback), and
   - range starts via `_packed_range_starts` (`packed.domain.ranges[i].start` — the *identical*
     source DaCe binds, see `get_field_domain_symbols`),
   into `sds["baked_strides"] = {field: [s0,s1,…]}` and `sds["baked_range_starts"] = {field: {dim:
   start}}` (per-call, set just before `_get_or_compile`).
2. **`translation.py`** — `_structured_field_stride_constants` (called in the structured opt block
   when `GT4PY_BAKE_STRIDES=1`) builds the `constant_symbols` dict: it matches each non-transient SDFG
   array's symbolic strides **by position** to the packed strides, and bakes `__<field>_<dim>_range_0`
   per field+dim. These feed the existing `gt_substitute_compiletime_symbols`.

### Why it is correct

The baked values are exactly what DaCe binds at call time (strides validated by
`get_array_stride_symbols`; range_0 identical to `get_field_domain_symbols`), so the kernel computes
identical addresses — only as folded constants. `TestRBFNABLA4` (compares both outputs to the
unstructured reference) passes; the stride runtime-check additionally guards every baked stride.

## Results (516 grid, K=50, dace_gpu, GT4Py Timer median)

| Config | median | vs 1.157 ms baseline |
|--------|--------|----------------------|
| baseline lb21 (no bake) | 1.157 ms | — |
| BAKE(strides) lb21 | 1.029 ms | −11% |
| **BAKE(strides) lb3** | **0.938 ms** | **−19%** |
| BAKE(strides) lb4 | 0.945 ms | −18% |
| BAKE(strides+range_0) lb21 | 1.082 ms | −6% |

With strides baked, all `stride` and `range_1` kernel args become literals; only 38 `range_0` args
remained, which the `range_0` baking also removes (80 symbols baked total → fully-constant
addressing). The win comes from freeing the address-arithmetic registers so **lb3 (37% occupancy)
beats lb21 (25%)** — exactly the occupancy unlock the roofline predicted. Coalescing of the warp
loads is a secondary benefit.

## Full optsweep with baking (all 17 configs, GT4Py median)

Re-ran the entire optsweep with `GT4PY_BAKE_STRIDES=1`, once at the (now-suboptimal) `lb21` and once
at the post-baking optimum `lb3=256,3` (`output/rbf_optsweep_baked{,_lb3}/`, jobs 4705375 / 4708325).
Baking is a **uniform win across every structured config** and shifts the occupancy optimum from
lb21 to lb3:

| Config | un-baked | baked lb21 | **baked lb3** |
|--------|----------|------------|---------------|
| unstructured baseline | 1.472 | 1.474 | 1.472 |
| FD / F / none / CSI lb | 1.157 | 1.082 | **0.932–0.933** |
| FDB block64×4×1 | 1.148 | 1.059 | 0.940 |
| `rbf_nabla4_direct` FD | — | 1.062 | 0.934 |
| FD **lb82** (spilling) | 6.744 | 4.243 | 4.243 |
| seqK {F,CSI} maxnreg128 | 1.65 | 1.37 | 1.37 |
| seqK {F,CSI} at **lb3** | 1.64 | — | 4.4–4.6 (spills) |

**Headline: structured + baking + lb3 = 0.932 ms vs unstructured 1.472 ms → 1.58× faster** (and −19%
vs the original un-baked structured best of 1.148 ms).

Findings:
- **All parallel-K configs collapse to ~0.932 ms** at lb3 — FD/F/none/CSI/FDB are within noise. Once
  baking + lb3 are in place, the other knobs (scan-unroll, JDim-blocking, CSI fusion) are
  irrelevant; **baking + occupancy dominate**. The hand-written `rbf_nabla4_direct` (0.934) ties the
  auto-generated path, confirming the CSI codegen is already optimal.
- **lb3 is the *parallel-K* optimum.** seqK needs ~128 registers (its K-loop hoisting), so the
  85-register `lb3` cap spills it catastrophically (4.4–4.6 ms); seqK wants `lb21`/`maxnreg128`. This
  re-confirms seqK is a dead end on this cluster.
- Baking helps the spilling case most in relative terms (lb82 6.74 → 4.24 ms, −37%), because folding
  the address arithmetic frees exactly the registers that were spilling.

## Files changed (Opt 18)

| File | Change |
|------|--------|
| `../gt4py/.../modules/cartesian_interceptor.py` | `_packed_element_strides`, `_packed_range_starts`; capture per-field packed strides/origins in `__call__`; inject `sds["baked_strides"]` / `sds["baked_range_starts"]` in `_get_or_compile` |
| `../gt4py/.../runners/dace/workflow/translation.py` | `_structured_field_stride_constants(ir, sdfg, symbolic_domain_sizes)` — bake strides (by SDFG-array position) and `range_0` (per field+dim) from the captured values; call gated on `USE_STRUCTURED_BACKEND` + `GT4PY_BAKE_STRIDES` |
| `commands_stencils_rbf_optsweep_baked.txt` / `..._baked_lb3.txt` | 17-config optsweep with `GT4PY_BAKE_STRIDES=1` at lb21 and at the optimal lb3 (`output/rbf_optsweep_baked{,_lb3}/`, jobs 4705375 / 4708325) |

## Follow-ups

- ✅ Full optsweep with baking re-run (jobs 4705375 / 4708325) — the lb3 table above is the
  authoritative post-baking comparison and supersedes the un-baked "New-cluster results" table.
- Consider making `GT4PY_BAKE_STRIDES` the default for the structured GPU path (strict win, always
  correct via the exact-packed-stride source).
- Remove the temporary debug prints (`[FISSION-DBG]`, `[MATERIALIZE-DBG]`, `[BAKE-STRIDES]`) and the
  leftover `sdfg.save("before_gpu_transformation.sdfg")`.

---

# Optimization 19 (2026-06-24): K-tile thread-coarsening — ILP substitutes for occupancy

## Root cause (ncu on the baked lb3 kernel, 0.932 ms)

After Opt 18 (baking) the fused `rbf_nabla4` GPU kernel is **latency-bound**, not throughput- or
occupancy-bound. ncu: compute SoL 34%, memory 67% (both < 60% → latency), **`long_scoreboard` stall =
58.7%** (global-load latency not hidden), 78% of cycles with no eligible warp. Two cheap levers were
exhausted first:

- **Occupancy push (job 4708807) is spill-limited.** 80 regs is a real floor; forcing fewer regs to
  gain blocks/SM is net-neutral (lb4 ties lb3) to worse (lb5/lb6 regress). Best was a ~1 % nudge
  (`maxnreg=72` → 0.924 ms).
- **Dedup / FMA (job 4708993) mostly inert** — nvcc's backend already CSE's the duplicate `x*x`
  (the 2nd multiply carries no register/stall attribution in SASS), and the FP64 pipe is only ~38 %
  utilised, so denser math can't move a latency wall. (`FEG @ lb4` did reach 0.898 ms by shaving
  enough registers that 50 % occupancy finally paid — a hint that register relief is the lever.)

The kernel runs **one thread per (i, j, kolor, k)** — a single K-level per thread, so a warp has only
its own neighbour loads in flight. When you can't buy occupancy, the other way to hide load latency is
**more independent work per thread (ILP / memory-level parallelism)**.

## The optimization — `DACE_KTILE=B` (env-gated, default off)

Block the vertical K dim into per-thread tiles of **B consecutive levels** via the existing
`LoopBlocking` transform, then **unroll** the inner block. Each GPU thread processes B K-levels whose
neighbour loads are mutually independent → the hardware overlaps their latencies. As a bonus, the
K-invariant geometry/coefficients are auto-hoisted *above* the inner loop (LoopBlocking relocates the
independent nodes) → computed once per thread for all B levels instead of once per level.

`LoopBlocking` produces: outer map `__gtx_coarse_K = 0:N:B` (stays in the GPU grid) + inner
**Sequential** map `K = coarse·B : Min(N, coarse·B+B)` + hoisted invariants. Wired into
`gt_auto_optimize(blocking_dim=K, blocking_size=B)`.

### Two bug fixes that made the unroll real (critical)

The whole ILP benefit depends on the inner loop **actually unrolling**, and it took two fixes to get
there (both verified in SASS — back-edge present ⇒ rolled, absent ⇒ unrolled):

1. **`Map.unroll` is left `False` by LoopBlocking** (its own TODO). Setting it `True` emits
   `#pragma unroll` — but that alone does nothing because…
2. …**the inner bound `Min(N, coarse·B+B)` is a runtime value**, which defeats `#pragma unroll`
   (nvcc keeps a rolled loop with a back-edge → the B loads stay serialized → *no ILP*). The first
   profiling of K-tiling was therefore an **invalid test** — it measured a rolled loop.

   Fix: when **N is divisible by B** (last block always full), drop the redundant `Min` so the inner
   range becomes the constant-trip `coarse·B : coarse·B+B-1`. The trip count is then the compile-time
   constant B and nvcc unrolls for real. Verified against a minimal blocked SDFG that LoopBlocking
   stores the inclusive stop as `Min(N, coarse·B+B) - 1` (Min integer operand = N exactly). When N is
   **not** divisible by B the `Min` is kept (correct, just not unrolled).

   **Deployment rule: B must divide K**, or the kernel is catastrophically slow. Measured:
   `B=2 @ K=51` (51 % 2 ≠ 0 → rolled) = **1.949 ms** vs `B=2 @ K=50` (unrolled) = **0.782 ms** — a
   2.5× penalty purely from the rolled loop.

## Results (516 grid, dace_gpu, GT4Py Timer median)

| Config | K | median | regs / blocks / occ | note |
|--------|---|--------|------|------|
| baseline (parallel-K, no tile) | 50 | 0.932 ms | 80 / 3 / 37.5 % | latency-bound, no ILP |
| KTILE=2, lb1 | 50 | 1.168 ms | 238 / 1 / 12.5 % | nvcc over-allocates → occ collapse |
| **KTILE=2, lb2** | 50 | **0.797 ms** | **128 / 2 / 25 % / 0-spill** | ILP win |
| **KTILE=2 + K_INNER (Opt 20), lb2** | 50 | **0.782 ms** | 128 / 2 / 25 % | best at K=50 |
| **KTILE=3 + K_INNER, lb2** | 51 | **0.755 ms** | **128 / 2 / 25 % / 0-spill** | best overall (~5 %/lvl faster than B=2) |
| KTILE=3, lb2 (no layout) | 51 | 0.779 ms | 128 / 2 | layout helps 0.779→0.755 |
| KTILE=3, lb1 / lb3 | 51 | 1.108 / 1.603 ms | — | wrong occupancy / spill |
| KTILE=5/10, lb2 | 50 | 1.9–2.1 ms | spills | too big for 128 regs |

### The mechanism, confirmed by SASS + ncu

- **lb2 is the sweet spot, and it's free.** At lb1 nvcc greedily uses **238 registers** (→ 1 block →
  12.5 % occ). `__launch_bounds__(256,2)` packs the *identical* kernel into **128 registers with zero
  spilling** (→ 2 blocks → 25 % occ). The SASS is byte-for-byte the same (same 112 loads, same
  unrolled body) — only the register allocation differs. So 238 was pure waste.
- **ILP substitutes for occupancy.** Baseline = 80 regs / 37.5 % occ but *no* ILP → 0.932 ms.
  KTILE=2 = 128 regs / *lower* 25 % occ but **2 independent K-levels/thread** → 0.782 ms. ncu on the
  winner: **`long_scoreboard` 58.7 % → 36.2 %** — the ILP directly cut the dominant stall.
- **The optimal B is the largest tile that still fits 128 regs / 2 blocks with no spill.** B=2 and
  **B=3 both fit 128 regs / 0 spill** (the compiler reuses registers across the unrolled iterations
  far better than a naive B× estimate); B≥4 spills.

### Optimal-tile bracket (B=2/3/4/6/8 @ K=48, all divisible — job 4712005)

| B | median @ K=48 | note |
|---|------|------|
| 2 | 0.707 ms | fits 128 |
| **3** | **0.619–0.625 ms** | **optimal** — fits 128, 3-level ILP at B=2's occupancy cost |
| 4 | 0.876 ms | register pressure (spill / 1 block) |
| 6 / 8 | 1.9 ms | heavy spill |

**B=3 is the ceiling**: the largest tile that fits 128 regs / 2 blocks cleanly. B=4 already regresses.
At equal K (apples-to-apples), B=3 beats B=2 by ~12 % (0.625 vs 0.707 @ K=48). Tuning rule: **pick the
largest B that divides K and still fits 128 regs (→ B=3 here)**.

## ncu on the 0.782 ms winner — where the *next* bottleneck moved

`scripts/profile_ncu_ktile.sh` (winning config baked in), report `ncu_out_ktile.ncu-rep`:

- **Latency stall halved** (`long_scoreboard` 58.7 → 36.2 %) ✓, but a **new #2 stall appears:
  `LG Throttle` (~30 %)** — the load/store pipe is *saturated*. We traded latency stalls for
  memory-pipe-throttle stalls.
- **`LG Throttle` is fed by the 10 % uncoalesced accesses.** Source page: every `u_vert`/`v_vert`/
  `z_nabla2_e` neighbour read is 8.5 % excessive and the `CopyND<double,1>` geometry scalar-gathers
  are 11 % excessive → extra sectors/wavefronts → LSU throttle. **Coalescing is now a first-order
  lever** (it wasn't before).
- **Occupancy gap:** achieved 19.8 % vs theoretical 25 %, flagged as *"highly different execution
  durations per warp"* → load imbalance (likely the lateral-boundary/halo path).
- Compute 38 % / memory 49 % — still latency-ish overall, but far more balanced than the baseline.

**Remaining headroom** (both harder than the levers already pulled): (a) coalescing the neighbour
gather to cut LG Throttle, (b) the occupancy/load-imbalance gap.

### Tried & rejected: warp-row alignment (`DACE_WARP_ALIGN=1`) — the LG Throttle is structural

Hypothesis: the packed `ni = 528` is a multiple of 16 (cache line) but **not 32** (warp = 256 B), so
warp bases land mid-sector → the ~8–11 % excessive sectors → LG Throttle. Fix attempt: pad the IDim
axis to a multiple of 32 (→ 544) for every field (geometry CopyND included) so all strides are
warp-multiples. Implemented in `_make_structured_field` (+ a gated unpack pad-row trim, since the
index maps use the original ni). Verified the `.cu` strides move to 544 and correctness passes.

**Result: neutral-to-negative** (B=3 @ K48 0.625 → 0.634; B=3 @ K51 0.755 → 0.837; parallel-K 0.932 →
0.930). **Why it doesn't work:** padding makes the warp base `(di + 16·j + 16·K) mod 32 → di mod 32`
— constant instead of varying, but still non-zero because `di` is the **V2E/E2C2V neighbour offset**.
Alignment fixes the stride *base*, not the *per-neighbour offset*, which is the real source of the
excess sectors. So the uncoalescing is **inherent to the unstructured-neighbour gather** (6 V2E edges
at 6 scattered offsets), and the +3 % pad memory slightly hurts. Conclusion: **LG Throttle is
structural** — cutting it would need shared-memory neighbourhood staging or vectorized loads, neither
controllable at the GT4Py/DaCe pass+packing level. `DACE_WARP_ALIGN` kept gated off as a documented
dead end.

### Tried & rejected: merging redundant geometry loads (`GT4PY_ENABLE_MISR=1`) — nvcc already CSE's them

The 118 `CopyND<double,1>` in the kernel are all **global geometry scalar-gathers**, and they look
redundant: `primal_normal_vert_v1`/`v2` are each loaded **48× from only 24 distinct addresses** (2×
each — the u/v cross-output sharing that GTIR CSE doesn't merge across separate outputs and MapFusion
then duplicates). `gt_merge_identical_scalar_reads` (`GT4PY_ENABLE_MISR=1`) merges them at the SDFG
level (reports "merged 54").

**Result: neutral** (B=3 @ K48 0.623 → 0.626; parallel-K 0.932 → 0.933; **lb3 regresses to 1.550**).
**Why:** the ref and MISR builds have the **identical 132 `LDG.E.64` global loads in SASS** — nvcc's
backend **already CSE's the duplicate-address reads**, so the source-level 2× redundancy is *not* a
runtime cost. MISR even *bloats* the source (118 → 168 CopyND) by re-materialising the merged scalars
per consumer group. It does cut the SDFG register need (128 → 80), but that buys nothing — at lb2 the
kernel is throttle-bound, not occupancy-bound (2 blocks already saturate), and at lb3 the interaction
regresses hard. `GT4PY_ENABLE_MISR` kept gated off.

### Tried & rejected: 2D horizontal block, IDim register-tiling, shared-memory staging

Three attempts to attack the LG-Throttle directly, all rejected:

- **2D horizontal block** (`DACE_UNIT_STRIDES_DIMS="IDim,JDim"`, to capture J-direction neighbour
  reuse in L1): **1.21 ms (~2× worse)**. With the K-near-IDim layout the JDim stride is huge (≈ni·K),
  so the block's 8 JDim rows land ~25 k elements apart → destroys locality instead of helping.
- **IDim register-tiling** (`DACE_ITILE=B`, block the warp dim so each thread does B adjacent IDim →
  wider coalesced loads): **broken.** Blocking the warp/unit-stride dim overflows the GPU grid-z limit
  (`grid (2,1,131841)` > 65535 → invalid launch) *and* desyncs the neighbour-offset / per-kolor domain
  bounds (98.5% wrong results). Unlike K (vertical, independent), IDim can't be tiled without
  reworking the grid assignment + shift machinery. Gated off, marked BROKEN in code.
- **Shared-memory staging** (`GT4PY_GEOM_SHARED=1`, DaCe `GPUTransformLocalStorage`): **applies to 0
  maps** — the structured SDFG is GPU_Device grid → per-thread work with no ThreadBlock scope for the
  transform to act on. And it wouldn't help anyway: the geometry loads are **already coalesced** (warp
  = 32 consecutive IDim) and **already use the read-only cache** (SASS `LDG.E.64.CONSTANT`); the only
  exploitable reuse is J-direction, which the layout fights (see 2D block). A working version needs a
  hand-built ThreadBlock scope + cooperative load + footprint mapping — multi-day from-scratch surgery
  with evidence it won't pay.

### Meta-finding: the remaining bottleneck is structural — reachable levers are exhausted

Two classes of attempt are **all neutral or broken**:
1. *Reduce redundant work* — duplicate multiplies (Opt-19 SASS read), redundant global loads (MISR),
   misaligned strides (WARP_ALIGN): **neutral**, because nvcc's backend already does the CSE /
   redundant-load elimination / coalescing.
2. *Improve locality / staging* — 2D block, IDim-tiling, shared memory: **worse or unreachable**,
   because the K-near-IDim layout (needed for the K-tile win) makes the J reuse axis expensive, and
   the structured SDFG has no block scope for shared memory.

The genuine remaining cost is the **inherent scattered V2E/E2C2V neighbour gather** — a load-*volume*
problem (≈132 coalesced, cached loads/thread saturating the LSU → LG-Throttle), not an uncoalescing
or caching one. Cutting it would require either a different mesh-data layout that makes the J reuse
cheap (conflicts with the K-tile layout) or hand-built shared-memory cooperative staging (multi-day,
evidence-against). **Conclusion: the structured DaCe backend is at its practical optimum for
`rbf_nabla4` at B=3 K-tile = 0.62 ms (~1.9× vs unstructured).** Remaining headroom is bounded and
structural, documented here as future work.

---

# Optimization 20 (2026-06-24): K-near-IDim memory layout — `DACE_K_INNER_LAYOUT=1`

## Why (memory layout finding)

Structured fields pack `(ni, nj, nkolor, K)` **Fortran-order**, so stride(IDim)=1 (warp-coalesced)
but **stride(K)=ni·nj·nkolor is the largest — K is the outermost dimension**. A per-thread K-loop
(Opt 19) therefore strides ~270 k elements (~2 MB) per level → each level's loads land in distinct
cache lines. K is forced outermost by GT4Py's canonical dim ordering
(`common.py:_dimension_kind_order`: `HORIZONTAL < LOCAL < VERTICAL`), so the easy reorder
`[IDim, K, JDim, Kolor]` is blocked (the same wall the shelved "Color rename" hit).

## The optimization

Keep the canonical dims `[IDim, JDim, Kolor, K]` but pack a **strided (non-contiguous) device array**
so K carries a small stride (= ni) next to IDim, IDim still unit-stride. This makes the B-level K-tile
loop **cache-local** instead of 2 MB apart. Implemented in `_make_structured_field`
(`cartesian_interceptor.py`): build the physical buffer `(ni, K, nj, nkolor)` F-contig on device and
return the `transpose(0,2,3,1)` strided view → strides `IDim=1, JDim=ni·K, Kolor=ni·K·nj, K=ni`.

**Feasibility gate (probed first):** `gtx.as_field` on GPU **C-contiguates a host-numpy view**
(destroys custom strides) but **preserves the strides of an on-device cupy array**. So the re-layout
must be built in cupy on-device (the helper does), and the stride-baking (Opt 18) captures whatever
`.strides` result automatically — no extra wiring. Default off → byte-for-byte unchanged.

## Result

- Standalone (parallel-K, no tile): **0.896 ms vs 0.932 ms baseline (−4 %)** — K stride literal in the
  `.cu` drops 272976 → 517, confirmed. Even parallel-K benefits (lighter address arithmetic + L2).
- Combined with the K-tile: 0.797 → **0.782 ms** (B=2), 0.779 → **0.755 ms** (B=3). The two compose:
  the tile supplies the per-thread K-loop, the layout makes that loop's B loads cache-local.

---

# Optimization 21 (2026-06-24): warp=K — correct, well-motivated, but a DEAD END for this kernel

## The idea (combine memory + algorithm)

The stencils (V2E, E2C2V, …) are **entirely horizontal**; K has **zero neighbour reuse** but the
geometry/coefficient fields are **K-invariant** (identical for all 50 levels of a vertex). In the
default layout (warp=IDim) the 50 K-levels of one vertex live in 50 different threads, each
**re-reading the full geometry** — the dominant redundancy. Making the **warp run along K** (32
K-levels of one (i,j,kolor) per warp) would: (a) **coalesce** the K-field loads (K unit-stride), and
(b) turn the K-invariance into a hardware **broadcast** — geometry read once per warp instead of once
per K (~32× less geometry traffic). This is the full-strength version of the K-tile's 2-fold
geometry hoist.

## Implementation — `DACE_K_WARP` (env-gated, default off)

- `=1`: C-order packing (K innermost, stride 1) in `_make_structured_field`.
- `=2`: K innermost **+ IDim second-innermost** (stride = K), Kolor outermost — so the block
  `(threadIdx.x=K, threadIdx.y=IDim)` is a contiguous tile (locality fix, see below).
- `translation.py`: flip the iteration order so K → threadIdx.x (warp) and set
  `unit_strides_kind=VERTICAL` so the SDFG matches the C-order arrays.

Verified correct end-to-end (passes `TestRBFNABLA4`): `.cu` shows `i_K = threadIdx.x`, K stride 1,
and the geometry access has **no `i_K` term → uniform/broadcast load** across the warp.

## Result — loses to the K-tile, for structural reasons

| variant | best lb | median |
|---------|---------|--------|
| warp=K `=1` (Kolor 2nd-innermost) | lb4 | 1.073 ms |
| warp=K `=2` (IDim 2nd-innermost) | lb4 | **1.007 ms** |
| K-tile (Opt 19+20) | lb2 | **0.755–0.782 ms** |

- **The 2nd-innermost ordering matters (validated):** `=2` is consistently 5–9 % faster than `=1` at
  every lb. With `=1` the block extends in IDim (threadIdx.y) while IDim is the memory-*outermost* dim
  (stride 25850) → the block's 8 rows are 25 k elements apart → no spatial locality. Putting **IDim
  second-innermost** (matching threadIdx.y) makes the 32K×8IDim block a contiguous ~400-element tile.
  *Lesson: the block's y-dimension and the memory second-innermost dimension must be the same dim;
  Kolor (size 1–3) belongs outermost.*
- **But warp=K plateaus at ~1.01 ms — ~29 % slower than the K-tile**, even with coalescing +
  broadcast + the locality fix. Structural reasons: (1) **K=50 vs warp-32** → the 2nd warp per column
  is 56 % full → ~22 % idle lanes baked in; (2) the horizontal reuse is inherently **2-D** (I *and*
  J) but a K-warp block is only 8-wide in IDim → most neighbour reuse spills to L2, whereas warp=IDim
  with a 2-D block captures it; (3) the broadcast saves issue slots more than DRAM traffic (geometry
  was already largely L1-resident). **Keep IDim as the warp/unit-stride dim.**

---

# Dimension-ordering — the combined memory + algorithm picture (summary)

| Dim | Size | Kind | Neighbour offsets | Cross-thread reuse | Best role |
|-----|------|------|------|------|------|
| IDim | ~528 | horizontal | small (±1,±2) | yes | **warp / unit-stride** (fills warps, 2-D coalescing) |
| JDim | ~517 | horizontal | small | yes | block-y or grid |
| Kolor | 1–3 | horizontal | 0–2 | some | **outermost** (size too small to be warp/2nd) |
| K | 50 | vertical | **0** | **none** (but K-invariant geometry) | **small per-thread tile (B=2–3)**, cache-local |

**Verdict:** IDim stays the warp/unit-stride dim. K's best role is *not* the warp (warp=K loses) and
*not* the outermost stride (default, hurts the K-loop) — it is a **small per-thread tile** (Opt 19)
with a **K-near-IDim stride** (Opt 20). When two dims share a tile/block, the block's 2nd axis must be
the memory 2nd-innermost dim (Opt 21 lesson).

## Env knob reference (all default off → suite/driver byte-for-byte unchanged)

| Env | Effect |
|-----|--------|
| `DACE_KTILE=B` | per-thread K-tile of B levels, unrolled (B must divide K); ILP. lb2 (`DACE_GPU_LAUNCH_BOUNDS="256, 2"`) is the sweet spot |
| `DACE_K_INNER_LAYOUT=1` | K stride = ni (cache-local K-tile loop), IDim unit-stride |
| `DACE_K_WARP=1\|2` | warp=K (C-order; `=2` puts IDim 2nd-innermost). Correct but slower — dead end for rbf_nabla4 |
| `DACE_WARP_ALIGN=1` | pad IDim to a multiple of 32. Correct but neutral-to-negative — dead end (uncoalescing is from the neighbour offset, not row alignment) |
| `GT4PY_ENABLE_MISR=1` | merge identical global scalar reads. Neutral — nvcc already CSE's them (132 SASS loads either way); dead end |
| `DACE_ITILE=B` | ⚠️ **BROKEN** — block the IDim warp dim. Overflows grid-z (invalid launch) + wrong results; gated off, documented dead end |
| `DACE_UNIT_STRIDES_DIMS` | **DEFAULT now `"IDim,JDim"`** (Opt 22): K→grid.z (no `ceil(K/8)` waste) + compact IDim×JDim block → **−5 to −16%** across the general Edge/Cell/V2C/V2E suite. ⚠️ rbf_nabla4 / register-heavy gather (+ `DACE_K_INNER_LAYOUT`) are ~2× worse — override those with `DACE_UNIT_STRIDES_DIMS=IDim` |
| `GT4PY_BAKE_STRIDES=1` | (Opt 18) bake packed strides/origins as compile-time constants; absorbs all of the above layouts automatically |

## Files changed (Opt 19–21)

| File | Change |
|------|--------|
| `../gt4py/.../runners/dace/workflow/translation.py` | `_unroll_inner_k_maps(sdfg, tile_size)` (set `Map.unroll`, drop the `Min` clamp when N%B==0 for a constant trip count); `DACE_KTILE` gate (feeds `blocking_dim=K` to auto-opt) + unroll post-pass; `DACE_K_WARP` iteration-order flip + `unit_strides_kind=VERTICAL` |
| `../gt4py/.../modules/cartesian_interceptor.py` | `_make_structured_field` — `DACE_K_INNER_LAYOUT=1` (K-near-IDim strided view), `DACE_K_WARP=1/2` (C-order / IDim-2nd strided view), `DACE_WARP_ALIGN=1` (pad IDim→mult-of-32, + gated unpack pad-row trim in `_unpack_to_buffer`); all on-device cupy so `as_field` preserves the strides |
| `commands_stencils_rbf_{occ,exp2,ktile,ktile2,lb2,b5lb2,kwarp,kwarp2,b3,bbracket}.txt` | the sweeps (jobs 4708807 … 4712005) |
| `scripts/profile_ncu_ktile.sh` | ncu profiling of the winning K-tile config |

## Headline

Baseline (baked, parallel-K) **0.932 ms** → **K-tile B=2 + K-near-IDim + lb2 = 0.782 ms** (−16 %) →
**K-tile B=3 = champion** (0.755 ms @ K=51; 0.619–0.625 ms @ K=48 — ~12 % faster than B=2 at equal K).
vs unstructured 1.472 ms ⇒ **~1.9×**. The win is ILP-substitutes-for-occupancy
(scoreboard 58.7 %→36.2 %), unlocked by a real unroll (constant trip count) at the free lb2 register
cap, with **B=3 = the largest tile that fits 128 regs** (B≥4 spills). The next bottleneck (LG Throttle
from the unstructured-neighbour gather) is **structural** — not addressable at the pass/packing level
(`DACE_WARP_ALIGN` confirmed this; neutral-to-negative).

---

# Optimization 22 (2026-06-26): dimension-order reorder — `IDim,JDim` default for the general suite

## Why
The old default `DACE_UNIT_STRIDES_DIMS="IDim"` put **K on threadIdx.y** (block `(32,8,1)`), so the GPU
grid quantizes K to `ceil(K/8)` (K=50→56 ⇒ ~11% wasted/masked threads) and each block's memory footprint
is a strided IDim×K slab (IDim stride 1, K stride ~10⁶ ⇒ poor L2 locality). NCU on the memory-bound Edge
stencils showed the real limiter is **Stall Long Scoreboard** (~0.5 eligible warps, waiting on global
loads) — latency-bound, which better locality directly helps.

## The change
`_usd_default` flipped `"IDim"` → `"IDim,JDim"` in `translation.py`. Keeps **x=IDim** (warp, stride-1, set
separately by `gt_change_strides` ⇒ coalescing unchanged) but moves **K off block.y into grid.z**
(sequential, no `ceil(K/8)`) and puts **JDim on block.y** ⇒ the block now tiles a compact IDim×JDim
contiguous plane (~4 KB) instead of a ~53 MB strided slab.

## Result (512 grid, GT4Py Timer median, structured, all correctness ✅)
| Stencil | conn | K | `IDim` | `IDim,JDim` | Δ |
|---|---|---|---|---|---|
| apply_2nd | Edge — | 50 | 0.319 | 0.296 | −7.1% |
| apply_2nd | Edge — | 120 | 0.708 | 0.672 | **−5.2% (pure locality, K÷8)** |
| add_analysis_to_vn | Edge — | 50/120 | 0.316/0.712 | 0.297/0.675 | −6.0% / −5.1% |
| apply_rayleigh | Cell — | 50/120 | 0.164/0.358 | 0.154/0.341 | −6.2% / −4.9% |
| horiz_kinetic | Edge — | 50 | 0.690 | 0.681 | −1.3% (multi-kernel dominated) |
| interp_cell | C2E | 50 | 0.270 | 0.227 | −15.9% |
| graddiv2 | E2C2EO | 50 | 0.455 | 0.452 | −0.7% |
| cells2verts | V2C | 50 | 0.154 | 0.145 | −5.7% |
| simple_v2e_e2c2v | V2E | 50 | 0.139 | 0.133 | −4.3% |

The −5% at **K=120 (÷8, zero quantization waste)** proves the win is genuine L2 locality, not just
quantization recovery.

## ⚠️ Exception — rbf_nabla4 must be launched with `DACE_UNIT_STRIDES_DIMS=IDim`
`IDim,JDim` is **~2.8× worse for rbf_nabla4** (V2E register-heavy gather): **2.8× with rbf's proper
`lb3` + `DACE_WARP_ALIGN=0` config (0.531→1.502 ms)** and +94% at default config — confirmed real, not an
lb0 artifact, and consistent with the Opt 21 finding ("2× worse with the K-near-IDim layout"). The gather
+ huge JDim stride under `DACE_K_INNER_LAYOUT` loses badly. So **rbf_nabla4 (and its `_direct` variants)
keep `DACE_UNIT_STRIDES_DIMS=IDim`**, alongside their existing `DACE_GPU_LAUNCH_BOUNDS="256, 3"` +
`DACE_WARP_ALIGN=0` overrides. This is a **kernel-class** carve-out, *not* an entity one: V2C
(cells2verts −5.7%) and the simple V2E stencil (−4.3%) both *gain*, so a "Vertex→IDim" rule would be
wrong. **rbf_nabla4 is the ONLY carve-out needed** — divrot (V2E), first seen at +6.3%, reconfirmed
**neutral (+0.2%, within noise)**.

## Files changed
| File | Change |
|------|--------|
| `../gt4py/.../runners/dace/workflow/translation.py` | `_usd_default` `"IDim"` → `"IDim,JDim"`; rbf carve-out noted in the comment |

---

# Optimization 23 (2026-06-27): automatic per-stencil dimension ordering — the Opt 22 blanket default was wrong for two stencil classes

## Why (full K=120 suite falsified the blanket IDim,JDim default)
The Opt 22 spot-checks were all K-independent elementwise stencils. The full K=120 suites
(`dycore_str_120` [IDim] vs `dycore_str_120_new` [IDim,JDim] and the diffusion pair) came out a
wash (dycore median ratio 0.997; diffusion 6 faster / 6 slower). Per-stencil analysis shows two
regressing classes:

1. **Vertical-dependency (Koff/scan) stencils, −10…−30%**: rho_virtual 0.707, InterpolateRhoThetaV
   0.730, ComputeResultsForThermo 0.758, ExtrapolateAtTop 0.817. Every regressor has 3–5 `Koff`
   refs in its source; every gainer has 0. Mechanism: under "IDim", K sits on threadIdx.y so 8
   consecutive K-levels share a block → k±1 reads hit L1/L2; under "IDim,JDim" K folds into grid.z
   → vertical reads lose all cache locality.
2. **E2C2V/C2V gather stencils, −18…−57%**: Nabla2AndSmag 0.425, HorizGradsForTurbulence 0.583,
   ApplyDiffusionToVn 0.739, Nabla4 0.817 — the rbf_nabla4 class (Opt 21/22); the manual rbf-only
   carve-out was too narrow. E2C2EO measured neutral/mixed → not part of the rule.

(Comparison caveat: the 4 `COPY_KERNEL_DETECTED` stencils in the new run report compute-only time
— e.g. CopyCellKdimFieldToVp 12.6 µs — and are not comparable to the old run's totals.)

## The fix — IR-based automatic selection (translation.py)
At entry of `_generate_sdfg_without_configuring_dace`, the PRE-transform GTIR is walked once
(`pre_walk_values().if_isinstance(OffsetLiteral)`; pre-transform because the structured passes
unroll E2C2V into cartesian shifts). Policy at the `_usd_default` site:

- `Koff` offset present OR `column_axis` set (scan) OR `E2C2V`/`C2V` present → `"IDim"`
- else → `"IDim,JDim"`
- `DACE_UNIT_STRIDES_DIMS` env override always wins; `DACE_SEQUENTIAL_K=1` unchanged.

Every automatic "IDim" choice prints a greppable audit line:
`DIM_ORDER: program=<name> order=IDim reason=koff|scan|gather`.

## Compute-only timer note
Reported GT4Py metric = chrono total − accumulated copy-kernel time (`SDFG_ARG_METRIC_COPY_TIME`
array filled by SDFG timing states; no report files). First active subtraction per program prints
`COMPUTE_ONLY_TIMER: program=<name> copy_ms=<x>`. If neither line nor effect appears for a
copy-kernel stencil, the binary predates the feature → clear `.gt4py_cache`.

# Optimization 24 (2026-07-07): NCU root-cause confirmation for the Opt-23 residual regressions — analysis only, no code changed

Opt 23's automatic classifier (koff/scan/E2C2V/C2V → `IDim`) fixed the originally-diagnosed
regressors, but the full K=120 suite re-run still showed ~10 residual regressions (5–42%) that
the classifier doesn't catch: `CalculateHorizontalGradientsForTurbulence` (C2E2CO, −42%),
`ComputeExnerFromRhotheta` (no connectivity, −30%), plus several more no-connectivity and C2E
stencils regressing 5–20%. NCU-profiled 4 of these (`IDim` vs `IDim,JDim`, K=50, user's local
Nsight Compute UI, screenshots in `output/ncu_screenshots/`) to understand why. **No source was
changed for this — pure diagnostic investigation.**

## Findings

3 of 4 confirm one mechanism; the 4th doesn't reproduce at this profile scale.

| Stencil | Connectivity | Duration `IDim`→`IDim,JDim` | L1 Hit `IDim`→`IDim,JDim` | Notes |
|---|---|---|---|---|
| `CalculateHorizontalGradientsForTurbulence` | C2E2CO | 320.8→515.2us (−38%) | 63.9%→29.0% | Compute-bound (54%) → DRAM-bound (91%). Scoreboard stall latency 18.1→30.3 cycles/instr. |
| `ApplyNabla2AndNabla4ToVn` | none | 439.2→507.5us (−13%) | 38.1%→13.2% | Already DRAM-bound both ways (~90%); total DRAM elapsed cycles +15.5%, matching the slowdown almost exactly. |
| `TemporaryFieldsForTurbulenceDiagnostics` | C2E | 530.3→622.1us (−15%) | 37.2%→**0.95%** | Near-total L1 collapse. DRAM 79.6%→88.0%. |
| `ComputeExnerFromRhotheta` | none | 261.5→248.7us (**+5%, no regression**) | 58.8%→61.7% (L2; L1 metric ~0 both ways) | Compute-bound (87–88%, FP64-pipeline-dominated). Contradicts the ~30% regression seen in the full K=120/101-rep suite — likely K-count-dependent (profiled at K=50) or launch-count-noise (only 3 launches here); unresolved, needs a K=120 re-profile. |

**Mechanism (confirmed, broader than originally scoped)**: it isn't specifically about
neighbor-gather connectivity. Any stencil that reuses K-independent data (geometric coefficients,
connectivity tables, or just a field that happens to be constant across a block's K-range) across
nearby K-levels loses that reuse once K moves from `threadIdx.y` (shared block → shared L1) to
`grid.z` (separate blocks → no L1 sharing). This is why even the connectivity-free
`ApplyNabla2AndNabla4ToVn` regresses. A purely connectivity-based classifier will structurally
never catch this class.

## Implication for the classifier (not yet implemented)
- `C2E2CO` should be added to the gather-detection set — confirmed same failure mode as
  `E2C2V`/`C2V`.
- The broader "K-independent reuse" mechanism suggests flipping the default: prefer `IDim` unless
  a stencil is *proven* to benefit from `IDim,JDim`, rather than opting into `IDim` only for
  known-bad classes.
- `ComputeExnerFromRhotheta`'s non-reproduction at K=50 means the classifier question for that
  stencil is still open — do not add a rule for it based on this data alone.

# Optimization 25 (2026-07-07): NCU on the two heaviest connectivity regressors independent of dim-order — analysis only, no code changed

Opt 23/24 covered regressions caused by the `IDim` vs `IDim,JDim` choice. Separately, filtering
the K=120 vs-unstructured comparison for neighbor-connectivity stencils that regress heavily
*regardless* of dim-order (excluding E2C2E/E2C2EO edge-to-edge connectivity, a known separate
issue) surfaced two standouts not yet explained by anything in Opt 23/24:

- `FusedVelocityAdvectionStencilVMomentum` (test file
  `test_compute_advection_in_vertical_momentum_equation.py`; connectivity C2E, C2E2CO, V2C) —
  ratio 0.446 (**-55%**, the single worst regression in the entire K=120 dataset), 4.612ms →
  10.334ms.
- `TemporaryFieldForGridPointColdPoolsEnhancement` (test file
  `test_temporary_field_for_grid_point_cold_pools_enhancement.py`; connectivity C2E2C, E2C) —
  ratio 0.626 (**-37%**), not previously examined.

NCU-profiled both (`IDim`, `IDim,JDim`, unstructured; user's local Nsight Compute UI,
screenshots in `output/ncu_screenshots/07_07/`). **No source was changed for this — pure
diagnostic investigation.** Note: these NCU profiles ran at the `profile_ncu.sh` default
K≈50 (not the production K=120/grid_514 the regression numbers above come from).

## `compute_advection_in_vertical_momentum_equation` — does NOT reproduce at this scale

| | `IDim` | `IDim,JDim` | unstructured |
|---|---|---|---|
| Duration | 68.64us | 66.24us | **150.78us** |
| L1 Hit Rate | 0.27% | 0.30% | 71.78% |
| Compute Throughput | 28.52% | 23.04% | 40.35% |
| Achieved Occupancy | 72.68% | 78.11% | 89.74% |

At K≈50 **structured is 2.2× faster than unstructured** — the opposite direction from the -55%
regression seen in the full K=120 production run. Both structured variants show near-zero L1 hit
rate yet still win, so this isn't explained by the Opt 24 cache-locality mechanism either. This
means the production regression is K-scaling-dependent (this stencil is the most complex one
profiled so far — three connectivities, a "Fused" mega-kernel) and this profile is not
representative of it. **Open — needs a K=120-scale NCU profile to see the real bottleneck**; a
K≈50 profile is a dead end for this specific stencil.

## `temporary_field_for_grid_point_cold_pools_enhancement` — reproduces, and the cause is register pressure, not cache locality

| | `IDim` | `IDim,JDim` | unstructured |
|---|---|---|---|
| Duration | 767.10us | 697.06us | **375.55us** (2× faster) |
| Registers/thread | **62** | **62** | **32** |
| Theoretical Occupancy | 50% (register-capped) | 50% (register-capped) | 100% |
| Top bottleneck (Est. Speedup) | Wait Stalls 35% (fixed-latency, not memory) | Wait Stalls 36% | — |

The regression direction matches production here (structured slower both ways). The structured
kernel uses **exactly double** the registers of unstructured (62 vs 32), which by itself halves
the theoretical occupancy ceiling (50% vs 100%) — identically for `IDim` and `IDim,JDim`, so this
is independent of dim-order entirely. The dominant stall reason is "Wait Stalls" (fixed-latency
execution dependency — not a memory/L1TEX stall), consistent with too few resident warps to hide
ALU/FMA latency at only 50% occupancy headroom.

This is a **different mechanism than Opt 24's cache-locality story** — it's register pressure
from how the structured C2E2C (2-hop cell gather) code is generated, not a memory access pattern
issue. It's the same *class* of problem already diagnosed and fixed once in this project: Opt 13
("CSI Let-Bindings — Share nabla4 Intermediate Between Tuple Outputs") found that an unshared
2-hop gather (there: E2C2V inside `rbf_nabla4`) gets duplicated across output branches instead of
computed once, inflating both instruction count and live-register count. Worth checking whether
`TemporaryFieldForGridPointColdPoolsEnhancement`'s C2E2C gather has the same unshared-duplication
pattern before assuming a new root cause.

## Next steps (not started)
- K=120/grid_514-scale NCU profile for `compute_advection_in_vertical_momentum_equation` to find
  its real (K-dependent) bottleneck — the K≈50 profile above is not representative.
- Check `TemporaryFieldForGridPointColdPoolsEnhancement`'s generated SDFG/IR for duplicated
  C2E2C sub-expressions (the Opt 13 pattern) before pursuing a register-pressure fix.

---

# Optimization 26 (2026-07-10): DACE_KTILE reached the unstructured backend for the first time — `rbf_nabla4`'s real "current best" is K-tile+CSI, not FD+lb3+IDim

## Two separate findings, both confirmed at full production scale (K=120, `parallelogram_514`)

### Finding 1 — the documented "current best" (`test_rbf_nabla4 [99]` in `commands_stencils_diffusion_str_512.txt`) was never actually the fastest measured config

Opt 19–21 (K-tile thread-coarsening + K-near-IDim layout) found a faster structured config than
what shipped in the production command line, but nobody folded `DACE_KTILE=3` +
`DACE_K_INNER_LAYOUT=1` into `commands_stencils_diffusion_str_512.txt` afterwards — the file kept
the pre-K-tile `FD` + `lb3` + `IDim` config as "[99] current best". This session re-ran both, back
to back, at full K=120/`parallelogram_514` scale (not the K=48–51/`grid_516` scale Opt 19–21 used)
and confirmed the gap is real and even larger at production scale:

| Label used below | Exact command (env vars only; pytest/grid/backend args identical, see full commands further down) | Median (GT4Py Timer) | vs unstructured default |
|---|---|---|---|
| `structured_FD_lb3_IDim` (old "[99] current best", still in `commands_stencils_diffusion_str_512.txt` as of this writing) | `GT4PY_BAKE_STRIDES=1 DACE_OPT_EXPERIMENT=FD DACE_GPU_LAUNCH_BOUNDS="256, 3" DACE_UNIT_STRIDES_DIMS=IDim` | **1.205 ms** | −41.1% |
| `structured_F_CSI_KTILE3_Kinner_lb2` (**the actual fastest measured `rbf_nabla4` config**) | `GT4PY_BAKE_STRIDES=1 DACE_OPT_EXPERIMENT=F GT4PY_ENABLE_CSI=1 DACE_KTILE=3 DACE_K_INNER_LAYOUT=1 DACE_GPU_LAUNCH_BOUNDS="256, 2"` | **0.867 ms** | −57.6% |

`structured_F_CSI_KTILE3_Kinner_lb2` beats `structured_FD_lb3_IDim` by **28.1%** (0.867 vs 1.205
ms) — not the ~15–20% Opt 19–21 measured at K=48–51/grid_516; the gap widens at production K/grid
scale. **Action taken**: `commands_stencils_diffusion_str_512.txt`'s `test_rbf_nabla4 [99]` entry
updated to `structured_F_CSI_KTILE3_Kinner_lb2` (renumbered `[99]`, old config kept as `[98]` for
reference — see that file). `analysis_parallelogram_gpu_nabla4.py`'s `PRODUCTION_REFERENCE_POINTS`
and both violin plots relabeled to name configs by their actual flags, not "current best", per
explicit request — "current best" drifts out of date exactly like this once, silently.

### Finding 2 — `DACE_KTILE` (and `DACE_OPT_EXPERIMENT`, `DACE_GPU_LAUNCH_BOUNDS`) never reached the unstructured backend at all; fixed in `translation.py`

While building a matched unstructured-vs-structured comparison sweep (default / K-tile / current-
best, both backends), the three "unstructured" rows produced suspiciously close numbers. Checking
`.gt4py_cache`: 3 differently-flagged unstructured `pytest` invocations produced **1** distinct
compiled binary, vs **4** distinct binaries for the 4 differently-flagged structured invocations.

**Root cause**: `_generate_sdfg_without_configuring_dace` in
`../gt4py/src/gt4py/next/program_processors/runners/dace/workflow/translation.py` had the entire
`DACE_OPT_EXPERIMENT`/`DACE_KTILE`/`DACE_GPU_LAUNCH_BOUNDS` env-var dispatch (~120 lines) nested
inside `if USE_STRUCTURED_BACKEND == 1:`. The `else:` (unstructured) branch called
`gt_auto_optimize(sdfg, gpu=on_gpu, constant_symbols=constant_symbols, **auto_optimize_args)` with
none of those env vars ever read — every unstructured config compiled identically regardless of
what was set, silently.

**Fix**: moved the `_exp`/`_extra`/`_ktile`/`_launch_bounds` computation to run unconditionally
(both backends), and the unstructured `else:` branch now forwards `_extra`/`gpu_launch_bounds`
into its own `gt_auto_optimize` call, plus applies the same `_unroll_inner_k_maps` K-tile post-
pass as structured. Two safety constraints kept the change from silently altering the *existing*
production/test suite (which never sets these env vars for unstructured runs):
- `"D"` (JDim blocking) is skipped unless `USE_STRUCTURED_BACKEND=="1"` — an unstructured SDFG has
  no `JDim` (only `Edge`/`Cell`/`Vertex`/`K`), so blocking on it would either no-op or error.
- The unstructured branch only applies `_extra`/`gpu_launch_bounds` when
  `"DACE_OPT_EXPERIMENT" in os.environ` / `"DACE_GPU_LAUNCH_BOUNDS" in os.environ` are **explicitly
  set** — not whenever they hold their defaults (`"FD"`/`"0"`). `DACE_OPT_EXPERIMENT` defaults to
  `"FD"` and `gpu_launch_bounds="0"` is not the same as gt_auto_optimize's own default `None`
  (`"0"` emits an explicit `__launch_bounds__(256)`; `None` emits no attribute at all) — forwarding
  the *defaults* unconditionally would have silently recompiled every unstructured stencil in the
  suite with different codegen. `DACE_KTILE` needed no such guard (its own default `"0"` already
  means off, explicit or not).

**Verification**: cleared `.gt4py_cache` before each of the 3 unstructured configs individually
(one `pytest` invocation per fresh cache — the GT4Py OTF cache key does not vary with these env
vars for unstructured, unlike structured which explicitly runs with
`otf_workflow__cached_translation=False`, so a warm cache across sequential commands in one job
masks the fix). Confirmed via the new `[ktile] (unstructured) B=3, unrolled 2 inner-K map(s)` log
line and 3 distinct `.gt4py_cache/rbf_nabla4_<hash>` directories.

## Results — full comparison table (GT4Py Timer median, K=120, `parallelogram_514`)

| Label | Full flags (beyond `GT4PY_BAKE_STRIDES=1 PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1`, always set) | Median | vs `unstructured_none_lb0` baseline |
|---|---|---|---|
| `structured_none_noCSI_lb0` | `USE_STRUCTURED_BACKEND=1 DACE_OPT_EXPERIMENT=none GT4PY_ENABLE_CSI=0 DACE_GPU_LAUNCH_BOUNDS="0"` | 1.716 ms | −15.7% |
| `unstructured_none_lb0` (baseline) | `USE_STRUCTURED_BACKEND=0 DACE_OPT_EXPERIMENT=none GT4PY_ENABLE_CSI=0 DACE_GPU_LAUNCH_BOUNDS="0"` | 2.043 ms | — |
| `structured_F_noCSI_KTILE3_Kinner_lb2` | `USE_STRUCTURED_BACKEND=1 DACE_OPT_EXPERIMENT=F GT4PY_ENABLE_CSI=0 DACE_KTILE=3 DACE_K_INNER_LAYOUT=1 DACE_GPU_LAUNCH_BOUNDS="256, 2"` | 1.492 ms | −27.0% |
| `unstructured_F_KTILE3_lb2` | `USE_STRUCTURED_BACKEND=0 DACE_OPT_EXPERIMENT=F DACE_KTILE=3 DACE_K_INNER_LAYOUT=1 DACE_GPU_LAUNCH_BOUNDS="256, 2"` (K_INNER_LAYOUT is a no-op for unstructured — structured-only field packing) | **1.713 ms** | **−16.2%** (K-tile genuinely helps unstructured too, once it can reach it) |
| `structured_F_CSI_KTILE3_Kinner_lb2` (**new current best**) | `USE_STRUCTURED_BACKEND=1 DACE_OPT_EXPERIMENT=F GT4PY_ENABLE_CSI=1 DACE_KTILE=3 DACE_K_INNER_LAYOUT=1 DACE_GPU_LAUNCH_BOUNDS="256, 2"` | **0.867 ms** | **−57.6%** |
| `structured_FD_lb3_IDim` (old "[99]"; CSI on by default, not explicitly set) | `USE_STRUCTURED_BACKEND=1 DACE_OPT_EXPERIMENT=FD DACE_GPU_LAUNCH_BOUNDS="256, 3" DACE_UNIT_STRIDES_DIMS=IDim` | 1.205 ms | −41.1% |
| `unstructured_FD_lb3` | `USE_STRUCTURED_BACKEND=0 DACE_OPT_EXPERIMENT=FD DACE_GPU_LAUNCH_BOUNDS="256, 3" DACE_UNIT_STRIDES_DIMS=IDim` (D silently skipped — no JDim on an unstructured SDFG; UNIT_STRIDES_DIMS is also a no-op, structured-only) | 2.015 ms | −1.4% |

Full reproducible commands (same for every row: `pytest -q
model/atmosphere/diffusion/tests/diffusion/stencil_tests/test_rbf_nabla4.py::TestRBFNABLA4::test_TestRBFNABLA4[compile_time_domain]
--backend=dace_gpu --grid ../grid_generator/parallelogram_grid_514.nc:120 --maxfail=1 -s`, mesh
`GT4PY_TRANSLATOR_MESH=.../parallelogram_grid_514.nc`) are in
`commands_stencils_rbf_nabla4_comparison_k120.txt` and
`commands_stencils_rbf_nabla4_unstructured_kfix.txt` (icon4py repo root).

## Files changed

| File | Change |
|---|---|
| `../gt4py/.../runners/dace/workflow/translation.py` | Moved `_exp`/`_extra`/`_ktile`/`_launch_bounds` computation out of the `USE_STRUCTURED_BACKEND` gate; unstructured `else:` branch now forwards them (explicit-set-only) into its own `gt_auto_optimize` call + K-tile unroll post-pass; `"D"` (JDim) letter guarded to structured-only |
| `commands_stencils_diffusion_str_512.txt` | `test_rbf_nabla4 [99]` updated to `structured_F_CSI_KTILE3_Kinner_lb2`; old `FD+lb3+IDim` config kept as `[98]` |
| `icon_structured_benchmark/plot_rbf_final_comparison.py`, `plot_rbf_gt4py_only.py` | Reference points / violin labels renamed from "current best"/"gt4py_RBFNABLA4_structured_FD_auto" to the actual flag string, e.g. `structured_F_CSI_KTILE3_Kinner_lb2` |
| `commands_stencils_rbf_nabla4_comparison_k120.txt`, `commands_stencils_rbf_nabla4_unstructured_kfix.txt` | New: the 7-row K=120/`parallelogram_514` sweep commands referenced in the table above |
