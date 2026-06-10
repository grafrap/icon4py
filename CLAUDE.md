# Project: GT4Py Structured Backend for ICON4Py on Parallelogram Mesh

## Overview

This is a master's thesis project. The goal is to run the ICON4Py atmospheric model (diffusion, dycore) on a **self-generated parallelogram mesh** using a custom **structured GT4Py backend pass** (`cart_unroll.py`). The structured backend converts unstructured-mesh field operators into Cartesian (I, J, Kolor) loop nests, which can be compiled and executed more efficiently.

### End Goal
Run the **standalone driver** (`model/atmosphere/diffusion/src/icon4py/model/atmosphere/diffusion/diffusion.py`) in structured mode, producing results that match the unstructured reference.

### Short-Term Goal
Reduce IR size by shortening the IR at the right places (kolor-split, dead-code elimination), so that compilation is faster without losing accuracy.

---

## Grid Structure

- **Mesh generator**: `../grid-generator/` — generates a parallelogram mesh as a NetCDF file
- **Grid file**: `../grid-generator/parallelogram_grid.nc` (used as `--grid ../grid-generator/parallelogram_grid.nc:N` where N = vertical levels)
- **Grid dimensions**: 76 × 66 (I × J), resulting in **15,190 total edges**
- **Parallelogram structure**: each parallelogram consists of two triangles; the dual-mesh triangulation gives 3 edge types (kolors)
- **Three edge kolors**:
  - Kolor 0: horizontal edges — shape `nx × (ny+1)` = `76 × 67` = 5,092 edges
  - Kolor 1: NE (north-east) edges — shape `(nx+1) × ny` = `77 × 66` = 5,082 edges
  - Kolor 2: diagonal edges — shape `nx × ny` = `76 × 66` = 5,016 edges
- **Cell kolors**: 2 (lower and upper triangle per parallelogram cell; 5,016 each = 10,032 total)
- **Vertex kolors**: 1

---

## Repository Layout (Key Files)

| Path | Role |
|---|---|
| `../gt4py/src/gt4py/next/iterator/transforms/structured_backend_passes.py` | **Refactored structured backend passes** — the canonical implementation. Contains `StructuredTypeRemapper`, `SetAtRemapper`, `BroadcastAxisExpander`, `ThresholdConditionRewriter`, `SymbolicSizeInliner`, `NeighborReductionUnroller`, `KolorConstantPropagation` (optional), `CanDerefRewriter`, and `StructuredBackend` orchestrator. |
| `../gt4py/src/gt4py/next/iterator/transforms/cart_unroll.py` | **Backward-compat shim** — re-exports `CartesianDomainAndTypeRemapper`, `CartesianReductionUnroller`, `CartUnroll` from `structured_backend_passes.py` so `pass_manager.py` and existing tests continue to work unchanged. Do not add new logic here. |
| `../gt4py/src/gt4py/next/iterator/transforms/pass_manager.py` | GT4Py pass pipeline — modified to call `CartesianDomainAndTypeRemapper` and `CartesianReductionUnroller` under `USE_STRUCTURED_BACKEND=1` |
| `../gt4py/src/gt4py/next/modules/cartesian_interceptor.py` | `GenericStructuredWrapper` — intercepts stencil calls, packs unstructured fields to structured layout, triggers lazy per-`horizontal_start` compilation |
| `../gt4py/src/gt4py/next/iterator/transforms/map_dict.py` | Hard-coded structured remap table: for every connectivity (E2C2EO, E2C, E2V, V2E, C2E, …) and slot, gives the structured (di, dj, dk) offset |
| `../gt4py/src/gt4py/next/modules/translator.py` | Field packing/unpacking utilities: `pack_edge_field_to_structured`, `build_index_map_from_lonlat_e2v`, `_derive_entity_start_bounds_from_mapping`, etc. |
| `model/atmosphere/dycore/tests/dycore/stencil_tests/conftest.py` | Pytest fixture that injects `GenericStructuredWrapper` for dycore stencil tests |
| `model/atmosphere/diffusion/tests/diffusion/stencil_tests/conftest.py` | Same for diffusion stencil tests |
| `scripts/compare_arrays.py` | Visualises 3-grid (kolor 0/1/2) match map: 0 = mismatch, 1 = match; boundary zeros are expected padding |
| `scripts/extract_timing_stats.py` | Reads `[timing]` lines from benchmark output files, skips first call (JIT warmup), computes min/max/mean/stddev/median of exec times per stencil. Appends stats block to each file and writes `timing_results.txt` to the folder. Usage: `python3 scripts/extract_timing_stats.py output/opt_v3fd/` |
| `scripts/compare_timing_results.py` | Compares two `timing_results.txt` files (generating them via `extract_timing_stats.py` if absent). Outputs a table of stencil name, median1, median2, abs diff, and ratio. Ratio < 1 means folder2 is faster. Writes result to `output/comparison/<folder1>_vs_<folder2>.txt`. Usage: `python3 scripts/compare_timing_results.py output/opt_v2/ output/opt_v3fd/` |
| `commands_stencils.txt` | Test commands for all implemented stencils |
| `ir_out.txt` | Captured IR output (set `GT4PY_PRINT_IR=1` env var) |
| `model/testing/src/icon4py/model/testing/stencil_tests.py` | Stencil setup for unstructured and structured test runs; This logic needs to somehow be ported to the standalone driver for end-to-end testing, but is currently only used for individual stencil tests |
| `../gt4py/src/gt4py/next/program_processors/codegens/gtfn/gtfn_module.py` | GTFN code generator module |
| `../gt4py/src/gt4py/next/program_processors/runners/dace/workflow/translation.py` | DaCe translation step — modified to read `symbolic_domain_sizes` from module-level global, augment `offset_provider_type` with IDim/JDim/Kolor, and remap arg types for bindings |
| `../gt4py/src/gt4py/next/program_processors/runners/dace/program.py` | DaCe SDFGConvertible interface — modified to pass `symbolic_domain_sizes` from thread-local |
| `../gt4py/src/gt4py/next/program_processors/runners/dace/lowering/gtir_to_sdfg_primitives.py` | DaCe GTIR→SDFG primitives — modified to handle NEVER-domain args in `translate_as_fieldop` |
| `../gt4py/src/gt4py/next/iterator/transforms/infer_domain.py` | Domain inference — modified: `_infer_as_fieldop` pre-populates `annex.domain` when `keep_existing_domains=True` to prevent overwrite by `infer_expr`; `_make_symbolic_domain_tuple` used for tuple-output as_fieldop |
| `test_narrow_kolor_fix.py` | IR-level unit tests for the Kolor-widening fix (7 tests) |
| `pass_pipeline_overview.md` | Annotated list of all passes in `apply_fieldview_transforms` in execution order |
|`test_out.txt`|Output of all tests that are not run via bash scripts with automatic output, claude should output to this file|

---

## Environment Variables

| Variable | Effect |
|---|---|
| `USE_STRUCTURED_BACKEND=1` | Activates the structured backend path in `pass_manager.py` and `conftest.py` |
| `PYTHONOPTIMIZE=1` | Required alongside structured backend (disables assertions) |
| `GT4PY_TRANSLATOR_LATERAL` | Legacy: symmetric lateral clip (no longer used in new mapping approach) |
| `GT4PY_TRANSLATOR_MESH` | Path to the grid NetCDF file (default hard-coded in `cartesian_interceptor.py`) |
| `GT4PY_PRINT_IR=1` | Dumps IR at each pipeline stage to `ir_out.txt` (or `GT4PY_PRINT_IR_FILE=<name>`) |
| `GT4PY_TRACE_INFER_DOMAIN_KOLOR=1` | Prints Kolor-specific domain info during `infer_program` (targeted, much less verbose than `GT4PY_TRACE_INFER_DOMAIN`) |
| `DACE_OPT_EXPERIMENT=FD` | Current production optimization: scan_loop_unrolling + blocking_dim=JDim. Values: `none`, `F`, `D`, `FD`. |

---

## Structured Layout

Unstructured fields are repacked into structured numpy arrays:
- **Edge field**: `[IDim, JDim, Kolor=3, K]`
- **Cell field**: `[IDim, JDim, Kolor=2, K]`
- **Vertex field**: `[IDim, JDim, Kolor=1, K]`
- **Sparse-local field** (e.g., `geofac_rot` with V2E): `[IDim, JDim, Kolor, local_slots, ...]`

The `edge_to_ijk` array (shape `[n_edges, 3]`) maps flat unstructured edge index → `(i, j, kolor)`.
The `ijk_to_edge` array (shape `[max_i, max_j, 3]`) maps `(i, j, kolor)` → flat edge index.

---

## `structured_backend_passes.py` Architecture (refactored)

The canonical implementation is now in `structured_backend_passes.py`. `cart_unroll.py` is a backward-compat shim.

### Confirmed code invariants (from hypothesis tests in `test_structured_backend_passes.py`)

| Invariant | Status |
|---|---|
| `_minus_one(ir.Literal)` branch is **live** — keep it | Confirmed live (H1) |
| `_entity_cartesian_bounds` called `_mapping_based_axis_bounds` twice for Cell | **Fixed (S1)**: result is now cached and reused |
| `SetAtRemapper.visit_FunCall` cartesian_domain+entity check fires for unusual inputs | Confirmed live — defensive guard kept |
| `_build_edge_validity_masked_expr` returns `None` only when no mapping configured | Documented in docstring — `None` return is intentional |
| `SymbolicSizeInliner` handles both `ir.Literal` and `ir.OffsetLiteral` tuple indices | **Fixed (H5b)**: `tuple_get(OffsetLiteral, make_tuple(...))` now correctly inlined |
| Three paths in `NeighborReductionUnroller.visit_FunCall` are mutually exclusive | Confirmed — comments added documenting each path's IR pattern |
| `_pick_symbolic_int` always receives integer-valued dicts at call sites | Confirmed — docstring added explaining role vs `_pick_size_param` |
| `copy.deepcopy` in `_offset_add`/`_offset_sub` zero-path | **Fixed (S5)**: deepcopy removed; zero-path returns arg directly (IR nodes are value-typed) |

### Dead code removed vs kept
- **Lateral clip path** was already removed in the refactoring.
- **`_minus_one` Literal branch**: confirmed live — NOT removed.
- **`SetAtRemapper.visit_FunCall` cartesian_domain+entity check**: confirmed needed for unusual IR forms — NOT removed.
- **`_offset_add`/`_offset_sub` zero-path deepcopy**: removed — the path is also unreachable for `OffsetLiteral` inputs (fast-path fires first), but the removal is correct for any future path through the zero-branch.

### Pass-by-pass audit simplifications (second round)

Global and per-pass simplifications applied after a full control-flow audit:

| Change | What it does |
|---|---|
| `_UNSTRUCTURED_AXES` constant | Replaced all bare `{"Edge","Vertex","Cell"}` sets — 7 sites |
| `_STRUCTURED_HORIZONTAL_AXES` constant | Replaced bare `{"IDim","JDim","Kolor"}` sets |
| `_EDGE_NUDGE_THRESHOLD_ID` constant | Replaced magic string `"start_2nd_nudge_line_idx_e"` |
| Remove `copy.deepcopy(self.generic_visit(...))` | `generic_visit` already returns new nodes; outer deepcopy was redundant (3 sites in passes 3/4/5) |
| `_extract_index_value(expr)` helper | Deduplicates `ir.Literal`/`ir.OffsetLiteral` index extraction in `SymbolicSizeInliner` (was copy-pasted twice) |
| `_collect_neighbor_tags` → `pre_walk_values()` | Replaced explicit recursive `_walk()` with Eve's built-in tree walker |
| `_mapped_connection_size` condition | Removed redundant `not idx_values or` — `list(range(0)) == []` handles the empty case |
| `_conn_name_and_is_edge_to_non_edge(key)` helper | Deduplicates conn-name extraction + `is_edge_to_non_edge` logic from Path 1, Path 3, and `_eval_list_field_at_idx` |
| `_named_range_args(nr)` helper | Deduplicates `cpm.is_call_to(nr, "named_range") and len(nr.args) == 3` guard — 5+ sites |
| `is_positive` simplification (ThresholdConditionRewriter) | Edge and Cell (non-halo) share same positivity rule; halo inverts with `not` |
| `_has_sym` docstring | Clarifies early-exit via Python `or`/`any` |
| `_get_axis_name` in domain bounds extraction | Replaced `isinstance(args[0], ir.AxisLiteral)` + `.value` direct access with the helper |
| Widened-init comment | Documents the float64 accumulator widening invariant |
| `and_` binary assumption comment | Documents GT4Py invariant in `KolorConstantPropagation` |

### Passes (in execution order)

| # | Class | Kind | Responsibility |
|---|---|---|---|
| 1 | `StructuredTypeRemapper` | **General** | Remaps `ts.FieldType` dims: Edge/Cell/Vertex → IDim/JDim/Kolor. Also remaps Program param types. |
| 2 | `SetAtRemapper` | **General dispatcher** + entity-specialized | Remaps `unstructured_domain` → `cartesian_domain`, does per-kolor split. Dispatches to `_remap_edge_setat` (3 kolors), `_remap_cell_setat` (2 kolors), `_remap_vertex_setat` (1 kolor). |
| 3 | `BroadcastAxisExpander` | **General** | Expands `broadcast(v, [Edge/Cell/Vertex, K])` → `broadcast(v, [IDim, JDim, Kolor, K])`. |
| 4 | `ThresholdConditionRewriter` | **General dispatcher** + entity-specialized | Rewrites `Edge/Cell >= threshold` comparisons to per-kolor structured conditions. `_rewrite_edge_threshold`, `_rewrite_edge_range_threshold`, `_rewrite_cell_threshold`. |
| 5 | `SymbolicSizeInliner` | **General** | Inlines symbolic sizes into `get_domain_range` / `tuple_get(get_domain_range)` expressions. |
| 6 | `NeighborReductionUnroller` | **General** + edge-specialized kolor peeling | Unrolls `reduce(op)(neighbors(conn, it), ...)` into explicit shifts. Edge-specialized: `visit_SetAt` peels outer kolor concat_where for non-split SetAts. |
| 7 | `KolorConstantPropagation` | **General** (optional, disabled) | Dead kolor branch elimination. |
| 8 | `CanDerefRewriter` | **General** | Replaces `can_deref(...)` with `True`. |
| — | `StructuredBackend` | Orchestration | Runs all passes in order; replaces `CartUnroll`. |

### Dead code removed (vs old `cart_unroll.py`)
- **Lateral clip path** (`lateral`, `lateral_bounds`, `lateral_edge`, `edge_phase_size`): superseded by mapping-based bounds — removed entirely.
- Nested closure functions in `visit_SetAt` extracted to class methods.
- Commented-out code and dead else-branches removed.

---

## `cart_unroll.py` Architecture (legacy — kept as shim only)

Two sequential passes inside `CartUnroll.apply()`:

### Pass 1: `CartesianDomainAndTypeRemapper`
- Transforms unstructured domains (`unstructured_domain(Edge, K)`) → structured Cartesian domains (`IDim, JDim, Kolor, K`)
- **Per-kolor split** (key optimisation): for each edge-domain SetAt, generates **3 separate SetAt nodes** (one per kolor) with kolor-specific I/J bounds derived from the `edge_to_ijk` mapping via `_derive_entity_start_bounds_from_mapping`. This eliminates the 3-copy `concat_where` wrapping.
- Remaps field types (Edge → IDim/JDim/Kolor)
- Transforms threshold comparisons like `Edgeₕ >= start_2nd_nudge_line_idx_e` into structured per-kolor domain conditions using precomputed bounds from `symbolic_domain_sizes`
- The `horizontal_start` value (from the stencil's runtime kwargs) is baked into `symbolic_domain_sizes` at lazy compile time

### Pass 2: `CartesianReductionUnroller`
- Expands `reduce(plus)(neighbors(E2C2EOₒ, it))` into explicit per-slot shifted field accesses
- Uses `current_kolor` (extracted from the per-kolor SetAt domain) to select the matching shift branch **directly**, avoiding inner `concat_where` kolor branching — but **only for SetAts that were actually split** (i.e. those without E2C2EO/E2C2E, see constraint below)
- **Entity-type safety**: only passes `current_kolor` to `_build_field_concat_where_from_branches` for **edge-center connectivities** (prefix "E": `E2C`, `E2V`, etc.). For cell-center (`C2E`) and vertex-center (`V2E`) connectivities, `current_kolor=None` is passed so all kolor branches are generated correctly.
- **`map_` propagation**: `current_kolor` must be forwarded through the `map_` case in `_eval_list_field_at_idx` — the recursive call for each arg of a `map_(op)(neighbors(...), weights)` expression must pass `current_kolor=current_kolor`, otherwise the `neighbors` field expansion falls back to generating a full `concat_where` (bug fixed 2026-04-24).
- **Trailing-else branch wrapping**: `_build_field_concat_where_from_branches` wraps the trailing-else branch (`cond=None`) in a `concat_where` with a narrow domain plus a `literal-0 as_fieldop` fallback. This prevents `infer_program` from over-widening the source Kolor domain via the trailing shift, which would produce out-of-bounds GPU reads. See `BUG_HISTORY.md` (Bug 04 investigation) for details.

### Key data flows

```
horizontal_start (call-time kwarg)
  → GenericStructuredWrapper._get_or_compile(horizontal_start, extra_thresholds)
    → symbolic_domain_sizes["horizontal_start"] = horizontal_start
    → symbolic_domain_sizes["start_2nd_nudge_line_idx_e_k{k}_{ilo/jlo/ihi/jhi}"] = ...
  → CartesianDomainAndTypeRemapper._per_kolor_domain(kolor, full_domain)
    → Uses _derive_entity_start_bounds_from_mapping(edge_to_ijk, horizontal_start)
    → Returns kolor-specific IDim/JDim/Kolor domain
  → CartesianReductionUnroller.visit_SetAt
    → _kolor_from_domain(new_domain) → current_kolor
    → Visits expr with current_kolor=k
  → _build_field_concat_where_from_branches(arg, branches, domain,
       current_kolor=k if is_edge_center_conn else None)
    → Selects kolor-k branch directly; returns _make_lifted_deref_shift(...)
```

---

## `GenericStructuredWrapper` (cartesian_interceptor.py)

- Intercepts every stencil call when `USE_STRUCTURED_BACKEND=1`
- Lazy compilation keyed by `(horizontal_start, extra_thresholds)` tuple
- Packs unstructured input fields → structured arrays before calling the compiled program
- Unpacks structured output fields → unstructured arrays after the call
- `_KNOWN_EDGE_THRESHOLD_PARAMS`: frozenset of parameter names (e.g., `"start_2nd_nudge_line_idx_e"`) that represent flat edge-index zone boundaries — for each found in kwargs, computes per-kolor bounds and injects into `symbolic_domain_sizes`
- `_KNOWN_CELL_THRESHOLD_PARAMS`: frozenset of cell threshold params (e.g., `"lateral_boundary_level_2"`); per-kolor cell bounds injected analogously
- `_sparse_pack_mappings`: dict cache — precomputed sparse field packing index arrays, built on first call and reused

---

## Currently Working Stencils

All tested with `USE_STRUCTURED_BACKEND=1 PYTHONOPTIMIZE=1`:

| Stencil | Connectivities | Status |
|---|---|---|
| `apply_divergence_damping_and_update_vn` | E2C, E2C2EO | ✅ passes (5-level grid) |
| `compute_avg_vn_and_graddiv_vn_and_vt` | E2C2EO, E2C2E | ✅ passes |
| `add_interpolated_horizontal_advection_of_w` | C2E | ✅ passes |
| `add_extra_diffusion_for_w_con_approaching_cfl` | C2E2CO | ✅ passes |
| `interpolate_to_cell_center` | C2E | ✅ passes |
| `mo_math_divrot_rot_vertex_ri_dsl` | V2E | ✅ passes |
| `mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl` | V2C | ✅ passes |
| `apply_diffusion_to_vn` | E2C2V, `start_2nd_nudge_line_idx_e` threshold | ✅ passes |
| `compute_advection_in_horizontal_momentum` | E2C, E2V, E2C2EO, C2E, V2E | ✅ passes (per-kolor split + entity-type-aware kolor) |
| `rbf_nabla4` | V2E + E2C2V, vertex sparse ptr_coeff_1/2[Vertex, V2EDim] | ✅ passes (origin-shift: vertex bounds, full arrays + as_field(origin=)) |

---

## Key Optimisations Implemented

### Per-Kolor SetAt Split (in `CartesianDomainAndTypeRemapper.visit_Program`)
- Edge SetAts are split into 3 kolor-specific SetAts at the type-remapping stage
- Each kolor's expression is visited with `current_kolor=k`, so `CartesianReductionUnroller` can select shift branches directly without generating inner `concat_where`
- **IR size reduction**: eliminates the inner kolor `concat_where` nesting (was `concat_where` per neighbor slot × 3 kolor branches = O(n_neighbors × 3) → now O(n_neighbors)). For `compute_avg_vn_and_graddiv_vn_and_vt`: concat_where count 75 → 3.
- **Critical constraint — edge-to-edge connectivity**: SetAts that contain **E2C2EO or E2C2E** are **NOT split** into per-kolor pieces. Reason: E2C2EO/E2C2E always accesses a **different** kolor than the source (e.g. kolor-0 source always reads kolor-1 or kolor-2 neighbors). When the accessed field is a **local intermediate** computed inside the same SetAt lambda, splitting to kolor-0-only means the intermediate is only materialized at kolor 0, so the kolor-1/2 neighbor accesses read garbage → wrong results. Guard in code: `_expr_uses_edge_to_edge_connectivity(stmt.expr)` (uses `_EDGE_TO_EDGE_CONNECTIVITIES = frozenset({"E2C2EO", "E2C2E"})`).

### Kolor-Branch Peeling in `CartesianReductionUnroller.visit_SetAt`
- For **non-split** SetAts (domain `Kolor:[0,3)`), the expression is wrapped in a 3-branch `concat_where` by `_build_edge_validity_masked_expr`. `_visit_expr_with_kolor_branches` peels the outer 3-kolor domain concat_where and visits each kolor-k branch with `current_kolor=k`. This allows the inner shift concat_where for **E2C, E2V** etc. to be resolved to the single correct per-kolor shift.
- **Excluded**: SetAts where `_e2c2e_on_local_intermediate(expr)` returns True — i.e. any E2C2EO/E2C2E reduce that operates on a **lambda-bound** (local intermediate) edge field.
- E2C2EO/E2C2E connectivity always passes `current_kolor=None` to `_build_field_concat_where_from_branches` even when `current_kolor` is set (guard: `normalized_conn not in _EDGE_TO_EDGE_CONNECTIVITIES`).

### Mapping-Based Domain Bounds (replaces `GT4PY_TRANSLATOR_LATERAL`)
- Per-kolor I/J bounds derived from `edge_to_ijk[horizontal_start:]` via `_derive_entity_start_bounds_from_mapping`
- Exact bounds (not symmetric heuristic) for each kolor's valid domain
- `start_2nd_nudge_line_idx_e` (and other threshold params) similarly handled with per-kolor bounds

### Origin-Shift Optimization (cache-aligned GPU writes)
- **Goal**: first interior write lands at GPU buffer offset 0 (cache-aligned), avoiding partial cache-line writes at the write-domain boundary.
- **Approach**: `as_field([IDim, JDim, Kolor, ...], full_packed_array, origin={IDim: shift_i, JDim: shift_j})` — keeps the FULL array, shifts the coordinate origin. DaCe accesses `ptr[i_logical + shift_i]` which equals `ptr[i_original]`. No slicing.
- **Entity-aware shift**: `_detect_output_entity()` inspects PAST body call `kwargs["out"]` → first output field dim → "Vertex"/"Edge"/"Cell". Then `_compute_horizontal_shift(entity, horizontal_start)` calls `_derive_entity_start_bounds_from_mapping` with the correct mapping (vertex_to_ij for vertex stencils, edge_to_ijk for edge stencils). Cached in `_shift_cache` per horizontal_start value.
- **Uniform shift**: same `(shift_i, shift_j)` applied to ALL fields (edge, vertex, cell, sparse). Neighbor reads stay in bounds because `ptr[(i_v_logical + di_V2E) + shift_v] = ptr[i_v_original + di_V2E]` — the shift cancels for any topology-valid offset.
- **Full array unpack**: `unpack_edge_field(struct_np, m, n_edge)` and `unpack_vertex_field_to_unstructured(struct_np, m)` read from original positions — boundary values are preserved since `pack_edge/vertex_field` fills ALL positions (including boundary).
- **IR domain**: `_mapping_based_axis_bounds` subtracts shift from vertex bounds → IR vertex domain `[0, 21)` with origin shift_i=3 → physical writes to `[3, 24)`. Edge per-kolor bounds subtracted in `_per_kolor_domain` via `sds["horizontal_start_shift_i"]`.
- **Files modified**: `../gt4py/src/gt4py/next/modules/cartesian_interceptor.py` (new methods `_detect_output_entity`, `_compute_horizontal_shift`, cache `_shift_cache`; updated `_pack_argument` and `_unpack_to_buffer`).
- **Why compact slicing was wrong for vertex stencils**: `edge_compact = edge_full[shift_i:, ...]` puts edge IDim=shift_i at ptr[0]. But V2E of a vertex at i_v_compact=0 reads edge at i_v_compact + di = -1 → OOB. The origin approach avoids slicing entirely.

---

## Known Issues / In-Progress

### Future: per-kolor split for E2C2EO stencils
- SetAts using E2C2EO/E2C2E are currently excluded from the per-kolor split (fall back to full concat_where), because local intermediate edge fields can't provide cross-kolor neighbor data.
- The right long-term fix is to compute the intermediate field over the union of required kolors (not just the output kolor), then write only the output kolor's slice. This requires restructuring how local intermediates are materialized — e.g. computing the intermediate over the full 3-kolor domain but restricting the final write.
- Alternatively, restructure the stencil so that cross-kolor intermediate fields are program parameters (pre-computed globally) rather than local lambdas.

### `_edge_shape_domain` (resolved)
- Previously `_edge_shape_domain` was over-clipping the per-kolor domain. Fix: return `None` from `_edge_shape_domain` when `current_kolor is not None` — the per-kolor SetAt domain already restricts to valid interior source positions. **Landed in commit `c75f8290`.**

---

## How to Run Tests

```bash
# Standard pattern for stencil tests
export USE_STRUCTURED_BACKEND=1
export PYTHONOPTIMIZE=1
# Always clear cache before tests (symbolic_domain_sizes keyed in OTF cache)
rm -rf .gt4py_cache ~/.gt4py_cache 2>/dev/null

# IMPORTANT: Always redirect full output to test_out.txt so both Claude and the user
# can inspect the complete log (IR prints, compilation messages, assertion details).
pytest -q <test_file> --backend=gtfn_cpu --grid ../grid-generator/parallelogram_grid.nc:<levels> --maxfail=2 -s > test_out.txt 2>&1; tail -40 test_out.txt

# Check result quality (boundary zeros are correct; interior zeros indicate bugs)
python scripts/compare_arrays.py stencil_output.txt
```

See `commands_stencils.txt` for all stencil-specific commands.
There is also a `run_all_stencils.sh` script that runs all stencil tests sequentially, but it's recommended to run stencils individually during development for faster feedback.

### Running the comparison script

```bash
cd /home/raphael/Documents/Studium/Msc_thesis/icon4py
./scripts/run_comparison.sh              # full structured vs unstructured comparison
./scripts/run_comparison.sh --structured-only  # only run structured backend
```

**Important**: The comparison compiles and runs the entire standalone driver, which takes several minutes per run. **Do not spin or poll** while it is running — start the script in the background (or as a long-running command), then wait for it to complete. Output is in `output/structured_run.log` and `output/unstructured_run.log`, and the final comparison summary is printed to stdout / `output/compare_out.txt`. A successful run looks like:

```
vn     : 100.00% match  max_abs=1.591e-12
w      : 100.00% match  max_abs=0.000e+00
...
✓ All fields match within tolerance (99% or higher)
```

---

## Writing Tests

When modifying `cart_unroll.py` or related passes, write unit tests in the relevant unit test file (e.g., `../gt4py/tests/next_tests/unit_tests/iterator_tests/transforms_tests/test_cart_unroll.py`). 
The `StencilTest` base class handles reference vs. structured comparison. Use `STATIC_PARAMS` to test with `NONE`, `COMPILE_TIME_DOMAIN`, and `COMPILE_TIME_VERTICAL` variants.

---

## Standalone Driver

- Target: `model/standalone_driver/src/icon4py/model/standalone_driver/standalone_driver.py`
- Run command: `export USE_STRUCTURED_BACKEND=1 && export PYTHONOPTIMIZE=0 && python3 model/standalone_driver/src/icon4py/model/standalone_driver/main.py --grid-file-path /home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc --icon4py-backend gtfn_cpu > main_out.txt`
- Note: `PYTHONOPTIMIZE=0` for the driver (unlike stencil tests which use `=1`)
- Status: **in progress** — individual stencil tests all pass, driver being brought up

### Driver Wrapper Architecture

When `USE_STRUCTURED_BACKEND=1`:
- `_resolve_symbolic_domain_sizes_from_mesh_metadata` in `gtfn_module.py` now also injects `use_horizontal_start_mapping=True` + `edge_to_ijk` + `vertex_to_ij` (loaded via `get_global_grid_mapping`). This means ALL stencils compiled through the standard GTFN backend get the mapping path, not the lateral path.
- `_validate_structured_remap_requirements` in `cart_unroll.py` returns `False` (skip transform gracefully) when `has_symbolic_structured_sizes=True` but neither mapping nor lateral is configured — handles stencils compiled before index_map is available.
- `CartesianReductionUnroller.visit_SetAt` skips unrolling for SetAts with unstructured domains (i.e. where the type remapper was skipped). Guard: `_is_structured_domain(new_domain)`.
- `type_synthesizer.py deref()`: The `USE_STRUCTURED_BACKEND` assertion that blocked non-derefable iterator types was removed — it was too aggressive.

### Wrapper Injection Points

Two injection points for `GenericStructuredWrapper`:
1. **`model_options.setup_program`** (used by diffusion/dycore): wraps each GT4Py program with `GenericStructuredWrapper` when `USE_STRUCTURED_BACKEND=1`. Returns a `functools.partial(wrapper, ...)` instead of `functools.partial(static_args_program, ...)`.
2. **`factory.ProgramFieldProvider._compute`** (used by geometry/metrics stencils): wraps `self._func` with `GenericStructuredWrapper` before calling, when `USE_STRUCTURED_BACKEND=1`.

### Known Issues (driver in progress)
- Driver reaches diffusion init and initial condition setup successfully. Further stencil issues may appear as diffusion timestep stencils compile.
- No more lateral clip logic: only `horizontal_start` mapping-based domain bounds should be active for global bounds; other threshold params like `start_2nd_nudge_line_idx_e` do further per-kolor clipping via `_KNOWN_EDGE_THRESHOLD_PARAMS`.
- Output: Driver output goes to `main_out.txt`.
- See `BUG_HISTORY.md` — "Driver Fixes Landed (2026-04-27)" for the 8 driver-level fixes that were required to make driver stencils compile correctly.

---

## DaCe Backend Integration

The structured backend passes (`CartesianDomainAndTypeRemapper` + `CartesianReductionUnroller`) now also run in the **DaCe backend** pipeline, in addition to the GTfn backend.

### Architecture

The GTfn backend uses `apply_common_transforms` (the full pass pipeline). The DaCe backend uses `apply_fieldview_transforms` (a lighter pipeline). The structured passes were added to `apply_fieldview_transforms` gated on `USE_STRUCTURED_BACKEND=1`.

**DaCe compatibility findings**:
- DaCe is **fully axis-agnostic**: `IDim`, `JDim`, `Kolor` are treated identically to `Edge`, `Cell`, `Vertex`.
- `concat_where` → handled by `gtir_to_sdfg_concat_where.py` along any named dimension ✅
- Cartesian shifts `shift(IDim, di)` → `_make_cartesian_shift()` fires when `offset_provider_type[key]` is a `Dimension` ✅
- `build_sdfg_from_gtir` is dimension-agnostic ✅

### Key Files Modified

| File | Change |
|---|---|
| `../gt4py/src/gt4py/next/iterator/transforms/pass_manager.py` | `apply_fieldview_transforms`: added `symbolic_domain_sizes` param; added `NormalizeShifts` + `InlineLifts` + structured passes block under `USE_STRUCTURED_BACKEND=1`; added `expand_tuple_args` + `dead_code_elimination` after the structured block; added fusion loop (InlineLambdas, FuseAsFieldOp, etc.) after `infer_program` to prevent IR explosion on large grids |
| `../gt4py/src/gt4py/next/program_processors/runners/dace/workflow/translation.py` | `_generate_sdfg_without_configuring_dace`: reads `symbolic_domain_sizes` from module-level global; augments `offset_provider_type` with IDim/JDim/Kolor; unified optimization via `gt_auto_optimize(disable_splitting=True, unit_strides_kind=VERTICAL)` with JSON-snapshot fallback. `DaCeTranslator.__call__`: uses `StructuredTypeRemapper._cartesian_remapped_type()` on each arg type so bindings match the structured SDFG. |
| `../gt4py/src/gt4py/next/program_processors/runners/dace/program.py` | Same `symbolic_domain_sizes` plumbing from thread-local for the SDFGConvertible interface path. |
| `../gt4py/src/gt4py/next/modules/cartesian_interceptor.py` | Module-level global `_CURRENT_COMPILE_SDS` + `get_compile_sds()` (replaces threading.local — DaCe pool threads inherit module globals); `_get_or_compile` sets global before `_setup()`, cleared in `finally`; backend factory uses `make_dace_backend(gpu=False, cached=True, auto_optimize=True, otf_workflow__cached_translation=False, use_metrics=False)`. |
| `../gt4py/src/gt4py/next/program_processors/runners/dace/lowering/gtir_to_sdfg_primitives.py` | `_parse_fieldop_arg`: returns `None` when `visit_SymRef` returns `None` (NEVER domain) instead of raising. `translate_as_fieldop`: filters dead (None) args and their corresponding lambda params before calling `translate_lambda_to_dataflow`. |
| `../gt4py/src/gt4py/next/iterator/transforms/infer_domain.py` | `_infer_as_fieldop`: pre-populates `annex.domain` on new node when `keep_existing_domains=True`; uses `_make_symbolic_domain_tuple` to handle tuple-output `as_fieldop`. |
| `model/testing/src/icon4py/model/testing/stencil_tests.py` | `_configured_program`: detects DaCe backend via `backend_like.get("backend_factory")` name and uses `DaCeBackendFactory` instead of `GTFNBackendFactory` for `GenericStructuredWrapper`. |
| `model/standalone_driver/src/icon4py/model/standalone_driver/standalone_driver.py` | `_wrap_granule_programs_for_structured_backend`: added `backend_like` parameter; selects `DaCeBackendFactory` or `GTFNBackendFactory` based on backend. Guard extended from `"gtfn" not in backend_name` to also allow `"dace"`. |

### Missing Prerequisites Added to `apply_fieldview_transforms`

GTfn ran these before the structured passes; DaCe did not:

| Pass | Why needed |
|---|---|
| `NormalizeShifts` | Structured passes pattern-match on canonical shift forms |
| `InlineLifts` | Structured passes must see bare `neighbors(conn, it)` — not `lift(lambda it: ...)` wrappers |

### `symbolic_domain_sizes` Flow for DaCe

The `symbolic_domain_sizes` dict (containing per-kolor bounds, `horizontal_start`, `edge_to_ijk`, etc.) needs to reach `apply_fieldview_transforms` inside the DaCe pipeline.

**Mechanism**: Module-level global in `cartesian_interceptor.py`:
1. `GenericStructuredWrapper._get_or_compile(horizontal_start, extra_thresholds)` builds `sds`
2. Sets `_CURRENT_COMPILE_SDS = sds` before calling `_setup()`, cleared in `finally`
3. `translation._generate_sdfg_without_configuring_dace` reads via `get_compile_sds()`
4. Falls back to `GTFNTranslationStep._resolve_symbolic_domain_sizes_from_mesh_metadata()` when not called from wrapper

Note: A module-level global (not `threading.local`) is required because DaCe submits compilation to a `ThreadPoolExecutor`, and thread-locals are NOT inherited by thread-pool threads in Python < 3.12. The main thread is blocked while DaCe compiles, so no race condition. See `BUG_HISTORY.md` (Bug 5) for the full investigation.

### DaCe Backend Factory Details

`GenericStructuredWrapper._get_or_compile` detects DaCe via `isinstance(self._backend_factory, type) and issubclass(self._backend_factory, DaCeBackendFactory)`. When DaCe is detected:
- Uses `make_dace_backend(gpu=False, cached=True, auto_optimize=False)` — this correctly sets all required `DaCeTranslator` init params (`auto_optimize_args`, `async_sdfg_call`, etc.)
- Does NOT pass `otf_workflow__bare_translation__symbolic_domain_sizes` (not supported by DaCe factory) — relies on module-level global instead

For GTfn: continues to use `self._backend_factory(cached=True, otf_workflow__cached_translation=True, otf_workflow__bare_translation__symbolic_domain_sizes=sds)` as before.

### How to Run Stencil Tests with DaCe Backend

```bash
export USE_STRUCTURED_BACKEND=1
export PYTHONOPTIMIZE=1
rm -rf .gt4py_cache

# C2E stencil (simple, good starting point)
pytest -q model/atmosphere/dycore/tests/dycore/stencil_tests/test_interpolate_to_cell_center.py \
  --backend=dace_cpu --grid ../grid-generator/parallelogram_grid.nc:5 --maxfail=1 -s

# V2E stencil
pytest -q model/atmosphere/dycore/tests/dycore/stencil_tests/test_mo_math_divrot_rot_vertex_ri_dsl.py \
  --backend=dace_cpu --grid ../grid-generator/parallelogram_grid.nc:5 --maxfail=1 -s
```

See `BUG_HISTORY.md` for the history of bugs fixed during initial DaCe integration.

---

## DaCe Optimization Strategy for Structured Code

### Key Finding: Only Phase 1 Splitting Block is Unsafe

Analysis of `auto_optimize.py` reveals that **`propagate_memlets_sdfg` is called at exactly two places** (lines 557 and 572), both inside the `if not disable_splitting:` block of `_gt_auto_process_top_level_maps`. These calls collapse single-element `Kolor:[k,k+1)` map ranges → `InvalidSDFGEdgeError`.

- **Phase 1 (non-splitting transforms)**: MapFusionVertical, MapFusionHorizontal, MapPromoter, GT4PyMapBufferElimination — all safe, no propagate_memlets.
- **Phase 3** (`_gt_auto_process_dataflow_inside_maps`): `FuseHorizontalConditionBlocks`, `MoveDataflowIntoIfBody`, `RemoveScalarCopies` etc. — safe (no propagate_memlets).
- **Phase 4**: `gt_set_iteration_order(unit_strides_kind=VERTICAL)` + `gt_change_strides` — makes K the innermost loop for CPU cache-coherent access.

### Unified Optimization Strategy (opt_v2 — current)

**File**: `../gt4py/src/gt4py/next/program_processors/runners/dace/workflow/translation.py`

A single `gt_auto_optimize(disable_splitting=True, unit_strides_kind=VERTICAL, validate=False)` call is used for **all** structured stencils. A JSON snapshot fallback handles the rare case where Phase 1 map fusion creates an invalid SDFG:

```python
structured_opt_args = {
    "unit_strides_kind": common.DimensionKind.VERTICAL,
    "disable_splitting": True,
    "validate": False,
    **(auto_optimize_args or {}),
}
sdfg_snapshot = sdfg.to_json()
try:
    gtx_transformations.gt_auto_optimize(sdfg, gpu=on_gpu,
        constant_symbols=None, **structured_opt_args)
except Exception:
    # Restore and apply safe fallback: buffer elimination + iteration order only.
    sdfg = dace.SDFG.from_json(sdfg_snapshot)
    sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyMapBufferElimination(assume_pointwise=True),
        validate=False, validate_all=False)
    gtx_transformations.gt_set_iteration_order(
        sdfg, unit_strides_kind=common.DimensionKind.VERTICAL, validate=False)
```

**Kolor-aware fusion callback** (`_kolor_aware_fusion_callback` in `translation.py`): passed to both `MapFusionVertical` and `MapFusionHorizontal` via `optimization_hooks`. The callback refuses to fuse two maps whose concrete Kolor start indices differ (e.g., `Kolor:[0,0)` vs `Kolor:[1,1)`). `_KOLOR_MAP_PARAM = 'i_Kolor_gtx_horizontal'` (computed via `gtx_dace_lowering.get_map_variable(common.Dimension("Kolor"))` to guarantee it matches SDFG lowering).

### Benchmark Results — 76×66 grid, K=5

Exec-only median from `[timing]` lines (excludes pack/unpack). Unstructured = total (no wrapper). See `BUG_HISTORY.md` for the full optimization experiment history (A-G).

| # | Stencil | Unstruct | opt_v2 | Exp D (blocking) | Exp E (fuse_t) | Exp F (unroll) |
|---|---------|----------|--------|------------------|----------------|----------------|
| 01 | nabla2_smag (E2C2V) | 0.52 ms | 1.40 ms | **1.20 ms** | ❌ 19.27 ms | 1.15 ms |
| 02 | horiz_adv (C2E) | 0.07 ms | 0.30 ms | 0.30 ms | 0.32 ms | **0.24 ms** |
| 03 | extra_diff (C2E2CO) | 5.40 ms | 0.60 ms | 0.60 ms | **0.45 ms** | 0.54 ms |
| 04 | div_damp (E2C+E2C2EO) | 0.83 ms | 5.95 ms | 5.90 ms | 5.89 ms | **5.75 ms** |
| 05 | avg_vn (E2C2EO+E2C2E) | 1.04 ms | 3.05 ms | 3.40 ms | 3.87 ms | **2.89 ms** |
| 06 | adv_hmom (complex) | 0.27 ms | 2.40 ms | 2.30 ms | 2.62 ms | **2.17 ms** |
| 07 | interp_cell (C2E) | 0.06 ms | 0.30 ms | **0.20 ms** | 0.23 ms | 0.22 ms |
| 08 | cells2verts (V2C) | 0.08 ms | 0.20 ms | 0.20 ms | 0.21 ms | 0.20 ms |
| 09 | rot_vertex (V2E) | 0.08 ms | 0.20 ms | 0.20 ms | 0.20 ms | 0.20 ms |
| 10 | diffusion_vn (E2C2V) | 0.42 ms | 3.05 ms | 2.90 ms | 3.59 ms | **2.71 ms** |

**Key finding**: Test 03 (C2E2CO) structured exec (0.6ms) is already **9× faster than unstructured** (5.4ms). The structured layout wins on 2-hop stencils even at small grid sizes.

**Current production default**: `DACE_OPT_EXPERIMENT=FD` (scan_loop_unrolling + blocking_dim=JDim, blocking_size=8). F+D improves 8/10 stencils over opt_v2.

**How to re-run** (env var selector in `translation.py`, default = FD):
```bash
DACE_OPT_EXPERIMENT=none  ./scripts/run_stencil_commands.sh -o output/opt_v2_rerun/  # pure opt_v2 baseline
DACE_OPT_EXPERIMENT=F     ./scripts/run_stencil_commands.sh -o output/opt_v3f/
DACE_OPT_EXPERIMENT=D     ./scripts/run_stencil_commands.sh -o output/opt_v3d/
DACE_OPT_EXPERIMENT=FD    ./scripts/run_stencil_commands.sh -o output/opt_v3fd/      # current default
```
Compare: `python3 scripts/compare_timing_results.py output/opt_v2/ output/opt_v3fd/`

---

## Performance Optimizations for Structured DaCe Backend

Pack/unpack dominated total structured time at small grids. Two fixes landed:

**Fix 1: Vectorized pack/unpack** (`../gt4py/src/gt4py/next/modules/translator.py`)

Replaced all Python triple-nested loops with numpy fancy indexing:
- `pack_edge_field` / `unpack_edge_field`: `valid = ijk_to_edge >= 0; out[valid] = field[ijk_to_edge[valid]]`
- `pack_cell_field` / `unpack_cell_field` / `unpack_cell_field_from_structured`
- `pack_vertex_field_to_structured` / `unpack_vertex_field_to_unstructured` / `pack_vertex_field`

**Fix 2: Precomputed sparse pack mapping** (`../gt4py/src/gt4py/next/modules/translator.py` + `cartesian_interceptor.py`)

`pack_sparse_local_field_to_structured` (used for weight fields like `e_bln_c_s`) previously used Python loops. Solution:
- Added `precompute_sparse_pack_mapping(conn, index_map, local_dim_name, ...)` — runs the Python loop ONCE and returns 6 numpy index arrays
- Added `apply_sparse_pack_mapping(coeff, mapping, out_shape)` — fast O(1) numpy indexing using precomputed arrays
- `self._sparse_pack_mappings: dict` cache in `GenericStructuredWrapper` — builds mapping on first call, reuses for all subsequent calls
- E2C special case uses vectorized `np.repeat`/`np.tile`; `_neighbor_ijk` uses `rfind("2")+1` (not `[-1]`) for the neighbor element type (fixes `C2E2CO` where `[-1]="O"`)

**Result** (76×66 grid, K=5, total test duration including compilation):

| Stencil | Before (dace1) | After (fix1_v2) | Speedup |
|---|---|---|---|
| nabla2_smag (E2C2V) | 238s | 135s | 1.8x |
| horizontal_advection (C2E) | 23s | 15s | 1.5x |
| avg_vn_graddiv (E2C2EO) | 153s | 118s | 1.3x |
| advection_momentum | 414s | 235s | 1.8x |
| interpolate_cell (C2E) | 21s | 14s | 1.5x |

**Fix 3 (investigated, not beneficial)**: Changing `[IDim, JDim, Kolor, K]` to `[Kolor, IDim, JDim, K]` required renaming `Kolor` to `Color` (GT4Py enforces alphabetical dim ordering). The resulting change to the SDFG loop nest order produced 2–6x **regression** on all stencils. Layout change is shelved.

---

## What Still Needs Work for Full DaCe Integration

- `cells2verts` (V2C, test 08) is 1.4× slower than the no-optimization baseline at 76×66. V2C has only 1 Kolor and very few cells; map fusion overhead outweighs the benefit at small grid sizes. Larger grids are expected to flip this.
- The JSON-snapshot fallback in `translation.py` remains as a safety net for any future unknown MapFusion patterns.

---

## Potential Future Improvement: Kolor-Aware auto_optimize

Currently `disable_splitting=True` is needed because `propagate_memlets_sdfg` in Phase 1 collapses single-element `Kolor:[k,k+1)` map ranges. A targeted fix is to make `MapFusionVertical`/`MapFusionHorizontal` Kolor-aware via a fusion callback that refuses to fuse maps with disjoint Kolor ranges. This would allow removing `disable_splitting=True`, unlocking the full Phase 1 pipeline.

```python
def _kolor_aware_fusion_callback(state, sdfg, first_exit, second_entry, missing_params):
    kolor_param = "__Kolor"
    first_range = dict(zip(first_exit.map.params, first_exit.map.range))
    second_range = dict(zip(second_entry.map.params, second_entry.map.range))
    if kolor_param in first_range and kolor_param in second_range:
        k1_start, k1_stop, _ = first_range[kolor_param][0]
        k2_start, k2_stop, _ = second_range[kolor_param][0]
        try:
            if int(k1_stop) < int(k2_start) or int(k2_stop) < int(k1_start):
                return False  # disjoint Kolor ranges — refuse fusion
        except (TypeError, ValueError):
            pass
    return True

structured_opt_args["optimization_hooks"] = {
    GT4PyAutoOptHook.TopLevelDataFlowMapFusionVerticalCallBack: _kolor_aware_fusion_callback,
    GT4PyAutoOptHook.TopLevelDataFlowMapFusionHorizontalCallBack: _kolor_aware_fusion_callback,
}
```

---

## DaCe GPU Backend — Current State

### Cache Management (important!)

The gt4py OTF compilation cache lives at `icon4py/.gt4py_cache` (controlled by `GT4PY_BUILD_CACHE_DIR`). The SLURM script sets `GT4PY_BUILD_CACHE_DIR=${WORKDIR}` so the cache is at `${WORKDIR}/.gt4py_cache = icon4py/.gt4py_cache`. The non-SLURM runner script (`run_stencil_commands.sh`) clears `$repo_root/.gt4py_cache` before each run. **Always clear the cache when changing grids or switching branches** — a stale compiled binary for the wrong grid size causes wrong results or crashes.

```bash
rm -rf /scratch/mch/rgraf/icon4py/.gt4py_cache
```

### Small-Grid Workflow (26×26, K=5)

Use `commands_stencils_small.txt` for fast iteration during development. Run on the SLURM **debug** partition:

```bash
sbatch --partition=debug --time=02:00:00 \
  scripts/run_stencil_commands_slurm.sh \
  --command-file commands_stencils_small.txt \
  --output-dir output/small_26_grafrap3/
```

Grid files:
- Small (26×26): `grid_generator/parallelogram_grid_26.nc` — K=5 levels, fast debug runs
- Standard (512×512): `grid_generator/parallelogram_grid.nc` — K=50 levels, full benchmark

### Current Stencil Status (dace_gpu, grafrap3 branch)

**10/10 stencils pass** on `dace_gpu` at 512×512 (confirmed in `output/big_update_dace_gpu/` after Bug 04 fix). Post-fix sweeps showed stencils 06 and 10 failing on small grid — the `annex.domain` overwrite fix (2026-05-21) addresses these. Pending cluster confirmation as of 2026-05-26.

| # | Stencil | dace_gpu 512×512 | dace_gpu 26×26 | Notes |
|---|---------|-----------------|----------------|-------|
| 01 | nabla2_smag (E2C2V) | ✅ | ✅ | FuseAsFieldOp ratio-guard + tuple-domain fix |
| 02 | horiz_advection (C2E) | ✅ | ✅ | — |
| 03 | extra_diffusion (C2E2CO) | ✅ | ⚠️ recheck | Asymmetric cell bounds at 26×26 only |
| 04 | div_damping (E2C+E2C2EO) | ✅ | ✅ | `domain_union` empty/inverted range fix (Bug 06, 2026-05-28) |
| 05 | avg_vn_graddiv (E2C2EO) | ✅ | ✅ | allow_uninferred=True for literal-0 fallback |
| 06 | adv_hmom (complex) | ✅ | ⚠️ fix pending | annex.domain overwrite fix (2026-05-21) |
| 07 | interp_cell (C2E) | ✅ | ✅ | — |
| 08 | cells2verts (V2C) | ✅ | ✅ | — |
| 09 | rot_vertex (V2E) | ✅ | ✅ | horizontal_start test fix |
| 10 | diffusion_vn (E2C2V) | ✅ | ⚠️ fix pending | annex.domain overwrite fix (2026-05-21) |

See `BUG_HISTORY.md` for the detailed investigation and fix history for each stencil.

---

## Visualization Scripts

| Script | Field type | Key args |
|---|---|---|
| `scripts/compare_arrays.py` | Edge (3 kolors) | `--nx <nx> --ny <ny>` |
| `scripts/compare_cell_arrays.py` | Cell (2 kolors) | `--nx <nx> --ny <ny>` |
| `scripts/compare_vertex_arrays.py` | Vertex (1 kolor) | `--nx <nx+1> --ny <ny+1>` |

For the 26×26 grid: cell `--nx 26 --ny 26`, vertex `--nx 27 --ny 27`.
For the 512×512 grid: cell `--nx 512 --ny 512`, vertex `--nx 513 --ny 513`.

---

## WORKFLOW CONSTRAINTS (mandatory)

- **Stencil tests** (pytest with `--backend=dace_*` or `--backend=gtfn_*`): **always via `sbatch scripts/run_stencil_commands_slurm.sh`**. Never on login node, never via bare `srun`.
- **Non-stencil Python tests** (e.g. `test_narrow_kolor_fix.py`, unit tests): **always via `srun`** on a debug/compute node. Never on login node.
- **Small grid first**: run 26×26 before 512×512. Small-grid failures almost always predict large-grid failures (exception: GPU OOB that CPU silently absorbs — stencil 03 pattern).
- **DaCe GPU — always run sanitize first**: when working on a DaCe stencil fix on this machine, **always** submit `commands_stencils_sanitize.txt` (uses `compute-sanitizer --tool memcheck`) before any regular small-grid sweep. CUDA OOB errors are silently absorbed by the CPU and GPU-small grid path but caught by the sanitizer. Use `commands_stencils_small.txt` only after sanitize passes cleanly.

srun example for unit tests:
```bash
srun --partition=debug --time=00:05:00 --uenv=icon/25.2:v3 --view=default \
  bash -c 'cd /scratch/mch/rgraf/icon4py && source .venv/bin/activate && python test_narrow_kolor_fix.py'
```

sbatch example for small-grid stencil sweep:
```bash
rm -rf /scratch/mch/rgraf/icon4py/.gt4py_cache
sbatch scripts/run_stencil_commands_slurm.sh \
  --command-file commands_stencils_small.txt \
  --output-dir output/small_after_infer_fix/
```

---

## See Also

- [BUG_HISTORY.md](BUG_HISTORY.md) — complete history of bugs investigated and fixed:
  - Driver Fixes Landed (2026-04-27): 8 bugs fixed to bring up the standalone driver
  - Initial DaCe Integration Bugs (1-5): foundational DaCe lowering fixes
  - Bug 6: CUDA ILLEGAL_ADDRESS on large grids — IR fusion missing from `apply_fieldview_transforms`
  - DaCe GPU stencil-by-stencil debug history (stencils 01, 03, 04, 05, 06, 09, 10)
  - DaCe optimization experiment history (Experiments A-G)
