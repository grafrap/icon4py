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
| `ir_out.txt` | Captured IR output (set `print_ir=True` or env var) |
| `model/testing/src/icon4py/model/testing/stencil_tests.py` | Stencil setup for unstructured and structured test runs; This logic needs to somehow be ported to the standalone driver for end-to-end testing, but is currently only used for individual stencil tests |
| `../gt4py/src/gt4py/next/program_processors/codegens/gtfn/gtfn_module.py` | GTFN code generator module |
| `../gt4py/src/gt4py/next/program_processors/runners/dace/workflow/translation.py` | DaCe translation step — modified to read `symbolic_domain_sizes` from thread-local, augment `offset_provider_type` with IDim/JDim/Kolor, and remap arg types for bindings |
| `../gt4py/src/gt4py/next/program_processors/runners/dace/program.py` | DaCe SDFGConvertible interface — modified to pass `symbolic_domain_sizes` from thread-local |
| `../gt4py/src/gt4py/next/program_processors/runners/dace/lowering/gtir_to_sdfg_primitives.py` | DaCe GTIR→SDFG primitives — modified to handle NEVER-domain args in `translate_as_fieldop` |
|`test_out.txt`|Output of all tests that are not run via bash scripts with automatic output, claude should output to this file|
---

## Environment Variables

| Variable | Effect |
|---|---|
| `USE_STRUCTURED_BACKEND=1` | Activates the structured backend path in `pass_manager.py` and `conftest.py` |
| `PYTHONOPTIMIZE=1` | Required alongside structured backend (disables assertions) |
| `GT4PY_TRANSLATOR_LATERAL` | Legacy: symmetric lateral clip (no longer used in new mapping approach) |
| `GT4PY_TRANSLATOR_MESH` | Path to the grid NetCDF file (default hard-coded in `cartesian_interceptor.py`) |

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
- **Entity-type safety**: only passes `current_kolor` to `_build_field_concat_where_from_branches` for **edge-center connectivities** (prefix "E": `E2C`, `E2V`, etc.). For cell-center (`C2E`) and vertex-center (`V2E`) connectivities, `current_kolor=None` is passed so all kolor branches are generated correctly. This prevents cross-entity kolor confusion (the bug with `compute_advection_in_horizontal_momentum` was exactly that `C2E` cell-kolor branches were being folded using edge-kolor context).
- **`map_` propagation**: `current_kolor` must be forwarded through the `map_` case in `_eval_list_field_at_idx` — the recursive call for each arg of a `map_(op)(neighbors(...), weights)` expression must pass `current_kolor=current_kolor`, otherwise the `neighbors` field expansion falls back to generating a full `concat_where` (bug fixed 2026-04-24).

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

---

## Key Optimisations Implemented

### Per-Kolor SetAt Split (in `CartesianDomainAndTypeRemapper.visit_Program`)
- Edge SetAts are split into 3 kolor-specific SetAts at the type-remapping stage
- Each kolor's expression is visited with `current_kolor=k`, so `CartesianReductionUnroller` can select shift branches directly without generating inner `concat_where`
- **IR size reduction**: eliminates the inner kolor `concat_where` nesting (was `concat_where` per neighbor slot × 3 kolor branches = O(n_neighbors × 3) → now O(n_neighbors)). For `compute_avg_vn_and_graddiv_vn_and_vt`: concat_where count 75 → 3.
- **Critical constraint — edge-to-edge connectivity**: SetAts that contain **E2C2EO or E2C2E** are **NOT split** into per-kolor pieces. Reason: E2C2EO/E2C2E always accesses a **different** kolor than the source (e.g. kolor-0 source always reads kolor-1 or kolor-2 neighbors). When the accessed field is a **local intermediate** computed inside the same SetAt lambda (e.g. `horizontal_gradient_of_total_divergence` in `apply_divergence_damping_and_update_vn`), splitting to kolor-0-only means the intermediate is only materialized at kolor 0, so the kolor-1/2 neighbor accesses read garbage → wrong results. Guard in code: `_expr_uses_edge_to_edge_connectivity(stmt.expr)` (uses `_EDGE_TO_EDGE_CONNECTIVITIES = frozenset({"E2C2EO", "E2C2E"})`). These SetAts fall back to the old single-SetAt path with full concat_where, which is correct because the intermediate field spans all 3 kolors there.

### Kolor-Branch Peeling in `CartesianReductionUnroller.visit_SetAt`
- For **non-split** SetAts (domain `Kolor:[0,3)`), the expression is wrapped in a 3-branch `concat_where` by `_build_edge_validity_masked_expr`. Previously, all 3 branches were visited with `current_kolor=None`, so each branch's inner shift `concat_where` (from `_build_field_concat_where_from_branches`) was left fully expanded with all 3 kolor sub-branches.
- **New**: `_visit_expr_with_kolor_branches` peels the outer 3-kolor domain concat_where and visits each kolor-k branch with `current_kolor=k`. This allows the inner shift concat_where for **E2C, E2V** etc. to be resolved to the single correct per-kolor shift, eliminating the redundant 3-branch structure inside each already-kolor-constrained branch.
- **Excluded**: SetAts where `_e2c2e_on_local_intermediate(expr)` returns True — i.e. any E2C2EO/E2C2E reduce that operates on a **lambda-bound** (local intermediate) edge field. For those, the full 3-branch E2C2EO inner concat_where is needed since the backend uses it to select the right shift at runtime. `_e2c2e_on_local_intermediate` detects this by checking whether the `neighbors(E2C2EO/E2C2E, it)(field)` argument, after unwrapping cast/identity wrappers, is a `SymRef` that is a parameter of an enclosing lambda.
- E2C2EO/E2C2E connectivity always passes `current_kolor=None` to `_build_field_concat_where_from_branches` even when `current_kolor` is set (guard: `normalized_conn not in _EDGE_TO_EDGE_CONNECTIVITIES`), because E2C2EO always accesses a different kolor than the source edge.

### Mapping-Based Domain Bounds (replaces `GT4PY_TRANSLATOR_LATERAL`)
- Per-kolor I/J bounds derived from `edge_to_ijk[horizontal_start:]` via `_derive_entity_start_bounds_from_mapping`
- Exact bounds (not symmetric heuristic) for each kolor's valid domain
- `start_2nd_nudge_line_idx_e` (and other threshold params) similarly handled with per-kolor bounds

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

### Driver Fixes Landed (2026-04-27)

Three classes of issues were fixed to advance driver progress:

#### Fix 1: Cell threshold params (`compute_exner_exfac`)
`_compute_exner_exfac` uses `concat_where(CellDim >= lateral_boundary_level_2, ...)`. After structured remapping this generated `gtfn::index(Cell)` in C++ (Cell not declared in structured domain).

**Fix**:
- Added `_KNOWN_CELL_THRESHOLD_PARAMS = frozenset({"lateral_boundary_level_2"})` in `cartesian_interceptor.py`.
- `_get_or_compile` now also injects per-kolor cell bounds from `cell_to_ijk` for cell threshold params (analogous to edge threshold params).
- `_derive_entity_start_bounds_from_mapping` updated: Cell now returns per-kolor `{0:..., 1:...}` bounds (using `cell_to_ijk` `(i,j,kolor)` format), not just a single global bound.
- `_mapping_based_threshold_condition` in `visit_FunCall` now handles `n_kolors=2` for Cell, `n_kolors=3` for Edge.
- `has_threshold_mapping` check extended from `entity_axis == "Edge"` to `entity_axis in {"Edge", "Cell"}`.
- `__call__` collects both `_KNOWN_EDGE_THRESHOLD_PARAMS` and `_KNOWN_CELL_THRESHOLD_PARAMS` in `extra_thresholds`.

#### Fix 2: AND of two edge threshold bounds (`compute_pressure_gradient_downward_extrapolation_mask_distance`)
Stencil uses `concat_where((start <= EdgeDim) & (EdgeDim < end), ...)`. Previously the upper bound was remapped as `not_(interior_cond)` which produced invalid C++ (not a valid domain expression).

**Fix**:
- Added `_derive_entity_range_bounds_from_mapping(entity, mapping_rows, horizontal_start, horizontal_end, ...)` in `cart_unroll.py` — computes per-kolor bounding boxes for `mapping_rows[start:end]`.
- Added `_inject_edge_range_bounds(...)` module-level function in `cartesian_interceptor.py` — detects paired threshold params by name (`horizontal_start_X` + `horizontal_end_X` with matching suffix), computes range bounds for `[start, end)`, injects under key `{start_name}|{end_name}_k{k}_{ilo/jlo/ihi/jhi}`.
- Added early `and_()` interception at top of `visit_FunCall` in `CartesianDomainAndTypeRemapper`: before children are visited, checks if the node is `and_(EdgeCmp_a, EdgeCmp_b)` and both threshold IDs form a known range pair. If so, returns the range domain condition directly — no `not_()` needed.

**Note on end = total**: When `horizontal_end_distance == total_edges` (= 15,190 for this grid), the range `[start, total)` equals the start-only condition. The bounding box is exact in this case. When `end < total` (true lateral boundary slice), the bounding box may be slightly loose (the rectangular hull covers more than the exact set of edges), but the SetAt domain already restricts computation to `[start, end)` so results remain correct.

#### Fix 3: K-broadcast type inference compatibility (`inference.py`)
After structured remapping, an `EdgeField` (no K) used in an `EdgeKField` context (e.g. `z_me - extrapolation_distance` where `z_me` is EdgeKField) caused type inference conflicts:
- `existing=Field[[IDim,JDim,Kolor]]`, `inferred=Field[[IDim,JDim,Kolor,K]]` → `TypeError`
- `existing=IteratorType(pos=[IDim,JDim,Kolor,K])`, `inferred=IteratorType(pos=[IDim,JDim,Kolor])` → `AssertionError`

**Fix** in `_is_structured_remap_compatibility_case` (`inference.py`):
- **FieldType case**: added Case 2 — if `inferred.dims[-1].value in {"K","KHalf"}` and `inferred.dims[:-1] == existing.dims`, return True (K appended by broadcast).
- **IteratorType case**: added Case A — if `existing.position_dims[-1].value in {"K","KHalf"}` and `existing.position_dims[:-1] == inferred.position_dims` and `defined_dims` match, return True (iterator in K-context has extra K in position).

#### Fix 4: `GenericStructuredWrapper.__call__` positional argument support
Diffusion `init_run` calls stencils like `init_diffusion_local_fields_for_regular_timestep(K4, substep, *smagorinski_factor, ...)` with positional args. `GenericStructuredWrapper.__call__(**kwargs)` rejected these.

**Fix**: Changed `__call__` to accept `*args`. When positional args are given, maps them to kwarg names by looking at `self._operator.past_stage.past_node.params` (the program's declared param order). Silently skips if params not accessible.

#### Fix 5: IteratorType K-broadcast compatibility (second pattern)
`__acc_init` for `max_over` reductions on EdgeKField produced a conflict where both `position_dims` AND `defined_dims` lost K:
- `existing=IteratorType(pos=[IDim,JDim,Kolor,K], defined=[IDim,JDim,Kolor,K])`
- `inferred=IteratorType(pos=[IDim,JDim,Kolor], defined=[IDim,JDim,Kolor])`

**Fix**: Extended Case A in `_is_structured_remap_compatibility_case` (`inference.py`) to also accept when `existing.defined_dims[-1]` is K and `existing.defined_dims[:-1] == inferred.defined_dims` (i.e. K removed from both position and defined dims simultaneously).

#### Fix 6: Direct stencil calls bypassing `GenericStructuredWrapper` (`initial_condition.py`)
Stencils called with `.with_backend(backend)` directly (e.g. `edge_2_cell_vector_rbf_interpolation`, `compute_difference_on_cell_k` in `initial_condition.py`) compiled with the structured pass (because `USE_STRUCTURED_BACKEND=1`) but received unstructured fields → C++ shape mismatch.

**Fix**: Wrapped both with `GenericStructuredWrapper` (same pattern as `cell_2_edge_interpolation` which was already wrapped). Changed `offset_provider={}` → `offset_provider=grid.connectivities` for `compute_difference_on_cell_k` so the wrapper can build cell packing tables.

#### Fix 7: Double-wrapping in `_wrap_granule_programs_for_structured_backend` (`standalone_driver.py`)
`model_options.setup_program` already wraps each diffusion/dycore stencil as `functools.partial(GenericStructuredWrapper_instance, **static_args)`. Then `_wrap_granule_programs_for_structured_backend` extracted `.func` (= the wrapper) and wrapped it **again** → `AttributeError: 'GenericStructuredWrapper' has no attribute '__name__'` (from `customize_backend` in `setup_program`).

**Fix**: In `_wrap_granule_programs_for_structured_backend`, if `operator = attr_value.func` is already a `GenericStructuredWrapper`, reuse it directly instead of double-wrapping. Get param names from `wrapper._operator` (the inner original program).

#### Fix 8: `calculate_nabla2_and_smag_coefficients_for_vn` — E2C2V vertex field kolor stride mismatch
The stencil used `u_vert_wp(E2C2V) * coeff` (full neighbor collection via `map_` + `neighbors`) then indexed `v_n[E2C2VDim(i)]`. In the structured backend, this generates a `map_(mul)(neighbors(E2C2V, u_vert_wp), coeff)` IR node that, when expanded per-slot, emits C++ kolor shifts (`at_key<Kolor_implicit=2>(vertex_kolor_stride)`). Vertex fields have Kolor=1, so the kolor stride is stored as `integral_constant<int, 0>` (compile-time zero scalar), and `at_key<2>` on a scalar fails at compile time.

**Fix**: Rewrote the stencil body to use explicit per-slot access `u_vert_wp(E2C2V[i]) * coeff[E2C2VDim(i)]` for i=0,1,2,3 (matching the working `calculate_nabla4.py` pattern). This generates `deref(shift(E2C2V, i)(it))` nodes (not `map_` + `neighbors`), which the GTFN backend handles correctly.

**Pattern rule**: When accessing vertex fields via E2C2V, always use explicit `E2C2V[i]` slot access, NOT the full `field(E2C2V)` + `[E2C2VDim(i)]` indexing. The latter creates a `map_/neighbors` IR pattern that triggers invalid kolor stride lookups in the structured C++ code for 1-kolor (vertex) fields.

### Known Issues (driver in progress)
- Driver reaches diffusion init and initial condition setup successfully. Further stencil issues may appear as diffusion timestep stencils compile.
- No more lateral clip logic: only `horizontal_start` mapping-based domain bounds should be active for global bounds; other threshold params like `start_2nd_nudge_line_idx_e` do further per-kolor clipping via `_KNOWN_EDGE_THRESHOLD_PARAMS`.
- Output: Driver output goes to `main_out.txt`.

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
| `../gt4py/src/gt4py/next/iterator/transforms/pass_manager.py` | `apply_fieldview_transforms`: added `symbolic_domain_sizes` param; added `NormalizeShifts` + `InlineLifts` + structured passes block under `USE_STRUCTURED_BACKEND=1`; added `expand_tuple_args` + `dead_code_elimination` after the structured block |
| `../gt4py/src/gt4py/next/program_processors/runners/dace/workflow/translation.py` | `_generate_sdfg_without_configuring_dace`: reads `symbolic_domain_sizes` from module-level global; augments `offset_provider_type` with IDim/JDim/Kolor; unified optimization via `gt_auto_optimize(disable_splitting=True, unit_strides_kind=VERTICAL)` with JSON-snapshot fallback. `DaCeTranslator.__call__`: uses `StructuredTypeRemapper._cartesian_remapped_type()` on each arg type so bindings match the structured SDFG. |
| `../gt4py/src/gt4py/next/program_processors/runners/dace/program.py` | Same `symbolic_domain_sizes` plumbing from thread-local for the SDFGConvertible interface path. |
| `../gt4py/src/gt4py/next/modules/cartesian_interceptor.py` | Module-level global `_CURRENT_COMPILE_SDS` + `get_compile_sds()` (replaces threading.local — DaCe pool threads inherit module globals); `_get_or_compile` sets global before `_setup()`, cleared in `finally`; backend factory uses `make_dace_backend(gpu=False, cached=True, auto_optimize=True, otf_workflow__cached_translation=False, use_metrics=False)`. |
| `../gt4py/src/gt4py/next/program_processors/runners/dace/lowering/gtir_to_sdfg_primitives.py` | `_parse_fieldop_arg`: returns `None` when `visit_SymRef` returns `None` (NEVER domain) instead of raising. `translate_as_fieldop`: filters dead (None) args and their corresponding lambda params before calling `translate_lambda_to_dataflow`. |
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

**Mechanism**: Thread-local in `cartesian_interceptor.py`:
1. `GenericStructuredWrapper._get_or_compile(horizontal_start, extra_thresholds)` builds `sds`
2. Sets `_compile_sds_local.current = sds` before calling `_setup()`
3. `translation._generate_sdfg_without_configuring_dace` reads via `get_compile_sds()`
4. Falls back to `GTFNTranslationStep._resolve_symbolic_domain_sizes_from_mesh_metadata()` when not called from wrapper (gives base mesh metadata without `horizontal_start`-specific bounds)

### DaCe Backend Factory Details

`GenericStructuredWrapper._get_or_compile` detects DaCe via `isinstance(self._backend_factory, type) and issubclass(self._backend_factory, DaCeBackendFactory)`. When DaCe is detected:
- Uses `make_dace_backend(gpu=False, cached=True, auto_optimize=False)` — this correctly sets all required `DaCeTranslator` init params (`auto_optimize_args`, `async_sdfg_call`, etc.)
- Does NOT pass `otf_workflow__bare_translation__symbolic_domain_sizes` (not supported by DaCe factory) — relies on thread-local instead

For GTfn: continues to use `self._backend_factory(cached=True, otf_workflow__cached_translation=True, otf_workflow__bare_translation__symbolic_domain_sizes=sds)` as before.

### Bugs Fixed in DaCe Lowering

#### Bug 1: `DaCeBackendFactory` missing `auto_optimize`
`DaCeWorkflowFactory` uses `factory.SelfAttribute("..auto_optimize")` which only resolves when the parent explicitly passes `auto_optimize`. Without it, factory-boy raises `AttributeError: The parameter 'auto_optimize' is unknown`. Calling `DaCeBackendFactory(cached=True, otf_workflow__cached_translation=True)` alone is not enough.

**Fix**: Use `make_dace_backend(gpu=False, cached=True, auto_optimize=False)` which sets all required params.

#### Bug 2: NEVER-domain args crash `translate_as_fieldop`
The structured unroller generates `as_fieldop(λ __x → 0)(e_bln_c_s)` for accumulator zero-init. The lambda ignores `__x`, so domain inference marks `e_bln_c_s` as `DomainAccessDescriptor.NEVER`. `translate_symbol_ref` returns `None` for NEVER symbols. `_parse_fieldop_arg` got `None` instead of `FieldopData` → `ValueError: Expected a field, found a tuple of fields.`

**Fix** in `gtir_to_sdfg_primitives.py`:
- `_parse_fieldop_arg`: when `sdfg_builder.visit(node)` returns `None`, return `None` (don't raise)
- `translate_as_fieldop`: before calling `translate_lambda_to_dataflow`, zip params with args and filter out pairs where `arg is None`. Create a new `Lambda` with only live params. This correctly handles `as_fieldop(λ x → 0)(f)` by stripping `x`/`f` since the body doesn't use them.

#### Bug 3: SDFG array shape vs binding param type mismatch
After the structured passes, the SDFG arrays have 4 dims (`IDim × JDim × Kolor × K`). But `DaCeTranslator.__call__` used `inp.args.args` (PAST-level, unstructured types: `Edge × K` = 2 dims) for `program_parameters` → the Python bindings generated by `_create_sdfg_bindings` used the wrong (2-dim) types → `zip(param_type.dims, sdfg_arg_desc.shape, strict=True)` failed.

Root cause: `apply_fieldview_transforms` returns a NEW `itir.Program` with remapped params, but this remapped program is discarded after SDFG building. The `__call__` method still held the original (unremapped) `program` from `inp.data`. Additionally, `StructuredTypeRemapper.visit_Program` creates new `ir.Sym` objects with remapped types, but these live only in the remapped program returned by the pass.

**Fix** in `DaCeTranslator.__call__`: apply `StructuredTypeRemapper._cartesian_remapped_type()` to each type in `inp.args.args` when `USE_STRUCTURED_BACKEND=1`. This remaps `Edge → [IDim, JDim, Kolor]` etc. so the binding code uses types that match the structured SDFG arrays.

#### Bug 4: `offset_provider_type` missing IDim/JDim/Kolor
After the structured passes, the IR uses `shift(IDim, di)` etc. DaCe's `_make_cartesian_shift()` fires only when `offset_provider_type[key]` is a `Dimension` object. The original `offset_provider_type` had only unstructured entries (E2C → `NeighborConnectivityType`, etc.). IDim/JDim/Kolor were absent → DaCe couldn't lower the structured shifts.

**Fix** in `translation._generate_sdfg_without_configuring_dace`: augment `offset_provider_type` with `{"IDim": Dimension("IDim"), "JDim": Dimension("JDim"), "Kolor": Dimension("Kolor")}` before `build_sdfg_from_gtir`.

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

Both confirmed passing (2026-05-06).

### Bug 5 — `horizontal_start` not reaching `apply_fieldview_transforms` (DaCe thread pool)

**Symptom**: Stencils with `horizontal_start > 0` (interior/nudging zones) produced wrong bounds — always `[0, max_i)` × `[0, max_j)` — and failed numerical comparison. Stencils with `horizontal_start=0` passed.

**Root cause**: DaCe submits compilation to a `ThreadPoolExecutor`. `threading.local` and Python 3.10 `contextvars.ContextVar` are **NOT** inherited by thread-pool threads (ContextVar propagation to ThreadPoolExecutor was only added in Python 3.12). So `_CURRENT_COMPILE_SDS` set in the main thread was invisible in the pool thread, causing `get_compile_sds()` to return `None` → fallback to mesh metadata without `horizontal_start`.

**Fix**: Replace `threading.local` / `contextvars.ContextVar` with a plain module-level global `_CURRENT_COMPILE_SDS: dict | None`. This works because the main thread is **blocked** inside `_compiled_programs()` (which waits for DaCe to finish compiling and executing), so the global is stable while the pool thread reads it — no race condition.

Set the global:
1. In `_get_or_compile`: `_CURRENT_COMPILE_SDS = sds` before `_setup()`, cleared in `finally`
2. In `__call__`: `_CURRENT_COMPILE_SDS = _call_sds` before `_compiled_programs()`, cleared in `finally`

`_sds_cache: dict[tuple, dict]` stores the sds per `(horizontal_start, extra_thresholds)` so `__call__` can look it up for subsequent calls (DaCe compilation is JIT — fired from `_compiled_programs()` on first call, not from `_setup()`).

### DaCe Optimization Strategy for Structured Code

#### Key Finding: Only Phase 1 Splitting Block is Unsafe

Analysis of `auto_optimize.py` reveals that **`propagate_memlets_sdfg` is called at exactly two places** (lines 557 and 572), both inside the `if not disable_splitting:` block of `_gt_auto_process_top_level_maps`. These calls collapse single-element `Kolor:[k,k+1)` map ranges → `InvalidSDFGEdgeError`.

- **Phase 1 (non-splitting transforms)**: MapFusionVertical, MapFusionHorizontal, MapPromoter, GT4PyMapBufferElimination — all safe, no propagate_memlets.
- **Phase 3** (`_gt_auto_process_dataflow_inside_maps`): `FuseHorizontalConditionBlocks`, `MoveDataflowIntoIfBody`, `RemoveScalarCopies` etc. — safe (no propagate_memlets). However, `FuseHorizontalConditionBlocks` hardcodes `validate=True`, so if Phase 1 map fusion creates a corrupt SDFG (dimensionality mismatch in a fusion temporary), Phase 3 will catch and raise it.
- **Phase 4**: `gt_set_iteration_order(unit_strides_kind=VERTICAL)` + `gt_change_strides` — makes K the innermost loop for CPU cache-coherent access. Previously never ran for CPU (unit_strides_kind defaulted to None).

#### Unified Optimization Strategy (opt_v2)

**File**: `../gt4py/src/gt4py/next/program_processors/runners/dace/workflow/translation.py`

A single `gt_auto_optimize(disable_splitting=True, unit_strides_kind=VERTICAL, validate=False)` call is used for **all** structured stencils — no per-kolor split vs non-split branching. A JSON snapshot fallback handles the rare case where Phase 1 map fusion creates an invalid SDFG:

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

**Kolor-aware fusion callback** (`_kolor_aware_fusion_callback` in `translation.py`): passed to both `MapFusionVertical` and `MapFusionHorizontal` via `optimization_hooks`. The callback refuses to fuse two maps whose concrete Kolor start indices differ (e.g., `Kolor:[0,0)` vs `Kolor:[1,1)`). This prevents the `InvalidSDFGEdgeError` that previously caused `apply_diffusion_to_vn` to fall back to the safe subset. `_KOLOR_MAP_PARAM = 'i_Kolor_gtx_horizontal'` (computed via `gtx_dace_lowering.get_map_variable(common.Dimension("Kolor"))` to guarantee it matches SDFG lowering).

#### Benchmark Results — 76×66 grid, K=5

Pytest-benchmark auto-scales units per stencil. Structured times (fix3/opt_v1/opt_v2) are **total round-trip including pack+exec+unpack** — all in ms. Unstructured (`USE_STRUCTURED_BACKEND=0`, `dace_cpu`) has no pack/unpack overhead — units vary (µs for fast stencils, ms for slow ones, as reported by pytest-benchmark). Unstructured is the **goal to beat**; comparison is approximate at this small grid size because pack/unpack (~5–20ms) dominates over the actual SDFG exec.

`fix3` = no SDFG optimization; `opt_v1` = old conditional (GT4PyMapBufferElimination+gt_simplify for per-kolor split, full gt_auto_optimize for non-split); `opt_v2` = current unified strategy.
Results: `output/` (unstruct), `output/fix3_stash/`, `output/opt_v1/`, `output/opt_v2/`.

| # | Stencil | Unstruct (µs) | fix3 (ms) | opt_v1 (ms) | opt_v2 (ms) | v2 vs opt_v1 |
|---|---------|---------------|-----------|-------------|-------------|--------------|
| 01 | nabla2_smag (E2C2V) | 523 µs | 29.33 | 29.50 | **26.60** | 1.1× ✅ |
| 02 | horiz_advection (C2E) | 68 µs | 9.12 | 103.23 | **7.28** | **14.2×** ✅ |
| 03 | extra_diffusion (C2E2CO) | 5,400 µs (5.4 ms) | 37.95 | 23.44 | **12.02** | 1.95× ✅ |
| 04 | div_damping (E2C+E2C2EO) | 829 µs | 43.62 | 52.17 | **37.16** | 1.40× ✅ |
| 05 | avg_vn_graddiv (E2C2EO+E2C2E) | 1,039 µs | 52.83 | 89.24 | **20.65** | **4.32×** ✅ |
| 06 | advection_hmom (complex) | 273 µs | 33.25 | 31.10 | **28.27** | 1.10× ✅ |
| 07 | interpolate_cell (C2E) | 62 µs | 9.02 | 6.72 | **5.85** | 1.15× ✅ |
| 08 | cells2verts (V2C) | 80 µs | 4.63 | 8.08 | 6.44 | 1.25× ✅ |
| 09 | rot_vertex (V2E) | 81 µs | 4.82 | 5.34 | **4.80** | 1.11× ✅ |
| 10 | diffusion_vn (E2C2V, fallback) | 417 µs | 18.96 | 16.72 | 20.17 | 0.83× (fallback ❌) |

_opt_v2 tests 01–09: first run (machine under load); test 10: second run (verified passing after fallback fix)._

**Note on unstructured comparison**: At 76×66 grid, structured pack+unpack (~5–20ms) dominates the total time, making direct structured vs unstructured comparison misleading. For example, test 01 structured exec-only is ~4ms vs 523µs unstructured — the overhead comes from pack/unpack, not the SDFG kernel. At production ICON grid sizes (millions of edges), pack/unpack is amortised and the structured kernel is expected to win via better cache behaviour and vectorisability. The table is most useful for **structured vs structured** comparisons (v2 vs v1 vs fix3).

**opt_v2 vs opt_v1**: Better for all 10 stencils. The big wins are tests 02 (14.2×) and 05 (4.3×) which were regressed in opt_v1. Test 10 is slightly slower (fallback path) — would recover once the MapFusion bug is fixed upstream.

#### Optimization Experiment History

The optimization strategy was developed through three experiments on the `dace_cpu` backend:

**Experiment A — `gt_set_iteration_order` only (not implemented as standalone)**
Hypothesis: K is stride-1 in `[IDim,JDim,Kolor,K]`; making K the innermost loop via `unit_strides_kind=VERTICAL` would improve cache behavior. Superseded by Experiment C which includes this and more.

**Experiment B — drop `gt_simplify` from per-kolor split path (diagnostic, not run)**
The opt_v1 per-kolor split path applied `GT4PyMapBufferElimination + gt_simplify`. Test 02 (C2E horiz_advection) was 11× SLOWER than the no-opt baseline (103ms vs 9ms), while test 07 (also C2E) was unchanged and test 03 (C2E2CO) got 1.6× faster. Exp B was proposed to isolate whether `gt_simplify` caused the regression by dropping it and keeping only `GT4PyMapBufferElimination`. Not needed: Exp C superseded it.

**Experiment C — `gt_auto_optimize(disable_splitting=True)` for all structured stencils (→ opt_v2, IMPLEMENTED)**
Root cause analysis showed only the two `propagate_memlets_sdfg` calls inside Phase 1's `if not disable_splitting:` block are unsafe. Phase 3 (`FuseHorizontalConditionBlocks`, `MoveDataflowIntoIfBody`, etc.) does NOT call `propagate_memlets_sdfg` — it is safe. `disable_splitting=True` skips only the unsafe Phase 1 block, preserving all other phases including map fusion (Phase 1), dataflow optimization (Phase 3), and iteration order setting (Phase 4 with `unit_strides_kind=VERTICAL`).

Results (opt_v2): test 02 improved 14.2× over opt_v1; test 05 improved 4.3×; all 10 stencils improved vs opt_v1. One stencil (`apply_diffusion_to_vn`) triggers an `InvalidSDFGEdgeError` from a MapFusion dimensionality mismatch — handled via JSON-snapshot fallback.

**Experiments D–G — further `gt_auto_optimize` parameter search (IMPLEMENTED as selector, TESTED on subset)**

A `DACE_OPT_EXPERIMENT` env-var selector was added to `translation.py` to test each parameter independently without code changes. Tested on tests 01 (E2C2V) and 05 (E2C2EO) using **median exec time from `[timing]` lines** (excludes pack/unpack). Comparison target is the unstructured DaCe Median (no pack/unpack).

Full benchmark results (all 10 stencils, all experiments), **exec-only median (ms)** from `[timing]` lines. Unstructured column = total benchmark median (no pack/unpack wrapper). Results in `output/opt_v3d/`, `output/opt_v3e/`, `output/opt_v3f/`. Comparison files (using `scripts/extract_timing_stats.py` + `scripts/compare_timing_results.py`) in `output/comparison/`.

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

**Key findings:**
- **Test 03 (C2E2CO)**: structured exec (0.6ms) is already **9× faster than unstructured** (5.4ms). The 2-hop C2E2CO stencil is where structured layout wins at this grid size.
- **Exp D (blocking_dim=JDim, blocking_size=8)**: Improves E2C2V stencils (nabla2_smag, interp_cell). Big regression on adv_hmom (1.72×). Neutral on simple stencils.
- **Exp E (fuse_tasklets=True)**: Catastrophic for nabla2_smag (6.47×) and diffusion_vn (1.94×). E2C2V sparse slot expressions become unfused tasklets that the C++ compiler cannot vectorise. **Never use alone.**
- **Exp F (scan_loop_unrolling=True)**: Broadly helpful — improves 7/10 stencils. Regressions on nabla2_smag (1.51×) and rot_vertex (1.2×). Earlier apparent 10× regression on test 02 was measurement noise (re-run confirmed).
- **Exp G (reuse_transients=True)**: Not run on full benchmark — smoke test showed exec 134s → catastrophic SDFG corruption.

**Experiment B — Further search on top of F (F+D, F+E):**

Using `scripts/compare_timing_results.py` on full benchmark runs. Ratio = opt_v3X / opt_v2; < 1 is faster.

| Stencil | F ratio | D ratio | E ratio | F+D ratio | F+E ratio |
|---------|---------|---------|---------|-----------|-----------|
| nabla2_smag (E2C2V) | ❌ 1.51 | ✅ 0.86 | ❌❌ 6.47 | ❌ 1.11 | ✅ 0.85 |
| horiz_adv (C2E) | ✅ 0.80 | 1.00 | 1.00 | ✅ 0.83 | ✅ 0.83 |
| extra_diff (C2E2CO) | ✅ 0.90 | 1.00 | ✅ 0.75 | ✅ 0.88 | ✅ 0.87 |
| div_damp (E2C+E2C2EO) | ❌ 1.02 | 1.00 | 1.00 | ✅ 0.96 | ✅ 0.97 |
| avg_vn (E2C2EO) | ✅ 0.95 | ❌ 1.10 | ❌ 1.23 | ✅ 1.01 | ❌ 1.28 |
| adv_hmom (complex) | ✅ 0.78 | ❌❌ 1.72 | ✅ 0.80 | ✅ 0.77 | ❌❌❌ 28.1 |
| diffusion_vn (E2C2V) | ✅ 0.97 | ✅ 0.94 | ❌❌ 1.94 | ❌ 1.07 | ✅ 0.98 |
| interp_cell (C2E) | ✅ 0.77 | ✅ 0.67 | ✅ 0.77 | ✅ 0.73 | ❌❌ 9.27 |
| cells2verts (V2C) | ✅ 0.95 | 1.00 | 1.05 | ✅ 0.90 | ❌❌ 8.70 |
| rot_vertex (V2E) | ❌ 1.20 | 1.00 | 1.00 | ✅ 0.95 | ❌ 1.35 |
| **Better/Worse** | **7/3** | **4/2** | **5/4** | **8/2** | **5/5** |

**F+D is the best overall**: 8/10 stencils improved, regressions are small (nabla2_smag 11%, diffusion_vn 7% — both E2C2V stencils where blocking disrupts sparse-slot access). F+E is unusable: adv_hmom 28×, interp_cell 9×, cells2verts 9× slower.

**Current production default**: `DACE_OPT_EXPERIMENT=FD` (scan_loop_unrolling + blocking_dim=JDim, blocking_size=8).

**How to re-run** (env var selector in `translation.py`, default = FD):
```bash
DACE_OPT_EXPERIMENT=none  ./scripts/run_stencil_commands.sh -o output/opt_v2_rerun/  # pure opt_v2 baseline
DACE_OPT_EXPERIMENT=F     ./scripts/run_stencil_commands.sh -o output/opt_v3f/
DACE_OPT_EXPERIMENT=D     ./scripts/run_stencil_commands.sh -o output/opt_v3d/
DACE_OPT_EXPERIMENT=FD    ./scripts/run_stencil_commands.sh -o output/opt_v3fd/      # current default
```
Compare: `python3 scripts/compare_timing_results.py output/opt_v2/ output/opt_v3fd/`

### Performance Optimizations for Structured DaCe Backend

**Problem (observed)**: Benchmark measured pack+compute+unpack together. For a C2E stencil on 76×66 grid: unstructured=69μs vs structured=150ms (2000x). Root causes:
1. Python triple-nested loops in `pack_edge_field`, `unpack_edge_field`, `pack_cell_field` etc.
2. `pack_sparse_local_field_to_structured` uses Python loop + dict lookups (84ms/call for `e_bln_c_s`)
3. `np.zeros` + `gtx.as_field` allocation on every call

**Fix 1: Vectorized pack/unpack** (`../gt4py/src/gt4py/next/modules/translator.py`)

Replaced all Python triple-nested loops with numpy fancy indexing:
- `pack_edge_field` / `unpack_edge_field`: `valid = ijk_to_edge >= 0; out[valid] = field[ijk_to_edge[valid]]`
- `pack_cell_field` / `unpack_cell_field` / `unpack_cell_field_from_structured`
- `pack_vertex_field_to_structured` / `unpack_vertex_field_to_unstructured` / `pack_vertex_field`

The commented-out vectorized version for 1D edge fields (line 468) already existed — extended to handle K dimension.

**Fix 2: Precomputed sparse pack mapping** (`../gt4py/src/gt4py/next/modules/translator.py` + `cartesian_interceptor.py`)

`pack_sparse_local_field_to_structured` (used for weight fields like `e_bln_c_s`) still required Python loops because of remap table lookups. Solution:
- Added `precompute_sparse_pack_mapping(conn, index_map, local_dim_name, ...)` — runs the Python loop ONCE and returns 6 numpy index arrays
- Added `apply_sparse_pack_mapping(coeff, mapping, out_shape)` — fast O(1) numpy indexing using precomputed arrays
- Added `self._sparse_pack_mappings: dict` cache in `GenericStructuredWrapper` — builds mapping on first call, reuses for all subsequent calls

**Critical bug fixed in precomputed sparse mapping**: `pack_sparse_local_field_to_structured` has a special case for `E2C` connectivity that bypasses the remap table — it directly assigns `coeff[edge, local]` → `out[edge_ijk, local]` (slot=local). The precomputed mapping in `precompute_sparse_pack_mapping` missed this special case and tried to look up cell neighbors via `cell_to_ijk`, which produced an empty mapping → all zeros. Added the E2C special case using vectorized `np.repeat`/`np.tile`. Also fixed `_neighbor_ijk` in all three locations to use `rfind("2")+1` instead of `[-1]` for the neighbor element type (fixes e.g. `C2E2CO` where `[-1]="O"` defaults to "Edge" instead of "Cell").

**Result of Fix 1+2** (76×66 grid, K=5, total test duration including compilation):

| Stencil | Before (dace1) | After (fix1_v2) | Speedup |
|---|---|---|---|
| nabla2_smag (E2C2V) | 238s | 135s | 1.8x |
| horizontal_advection (C2E) | 23s | 15s | 1.5x |
| avg_vn_graddiv (E2C2EO) | 153s | 118s | 1.3x |
| advection_momentum | 414s | 235s | 1.8x |
| interpolate_cell (C2E) | 21s | 14s | 1.5x |

**Fix 3 (investigated, not beneficial): Memory layout Kolor-first** — Changing `[IDim, JDim, Kolor, K]` to `[Kolor, IDim, JDim, K]` required renaming `Kolor` to `Color` (GT4Py enforces alphabetical dim ordering; "Color" < "IDim" < "JDim"). The rename itself was straightforward but changing the canonical domain order from `["IDim","JDim","Kolor"]` to `["Color","IDim","JDim"]` changed the DaCe SDFG loop nest order, producing 2–6x **regression** on all stencils. The layout change is shelved; cache efficiency on larger grids must be achieved via SDFG-level loop interchange instead.

### What Still Needs Work for Full DaCe Integration

`GenericStructuredWrapper` injection points in `standalone_driver.py` and `stencil_tests.py` now detect the backend and switch to `DaCeBackendFactory` when DaCe is requested. The `_get_or_compile` DaCe path uses `make_dace_backend(auto_optimize=True, ...)` — the conditional optimization strategy is now active.

Remaining gaps:
- Field packing/unpacking is backend-agnostic (works for both GTfn and DaCe) — wrapper packs unstructured → structured before calling the SDFG.
- `apply_diffusion_to_vn` previously used the JSON-snapshot fallback due to a MapFusion dimensionality mismatch. **Fixed** by the Kolor-aware fusion callback (see above) — the stencil now uses the full `gt_auto_optimize` path. The JSON-snapshot fallback remains in the code as a safety net for any future unknown patterns.
- `cells2verts` (V2C, test 08) is 1.4× slower than the no-optimization baseline. V2C has only 1 Kolor and very few cells; map fusion overhead outweighs the benefit at 76×66 grid size. Larger grids are expected to flip this.

### Potential Future Improvement: Kolor-Aware auto_optimize

Currently `disable_splitting=True` is needed because `propagate_memlets_sdfg` in Phase 1 collapses single-element `Kolor:[k,k+1)` map ranges. This blocks the full Phase 1 pipeline (MapSplitter, SplitConsumerMemlet, gt_vertical/horizontal_map_split_fusion) which could provide additional speedup for some stencils.

A targeted fix is to make `MapFusionVertical`/`MapFusionHorizontal` Kolor-aware via a fusion callback that refuses to fuse maps with disjoint Kolor ranges (e.g., prevents merging `Kolor:[0,1)` map with `Kolor:[1,2)` map). This would:
1. Allow removing `disable_splitting=True` — unlocking the full splitting pipeline
2. Prevent the `InvalidSDFGEdgeError` from test 10's MapFusion dimensionality mismatch
3. Make the blocking_dim (Exp D) more effective since the split maps would be better separated

Implementation: pass a `check_fusion_callback` via `optimization_hooks` in `gt_auto_optimize` that inspects the Kolor range of both maps being fused. If they are both concrete (post `gt_substitute_compiletime_symbols`) and disjoint, return False to prevent fusion.

```python
def _kolor_aware_fusion_callback(state, sdfg, first_exit, second_entry, missing_params):
    # Extract Kolor range from both maps and refuse to fuse if disjoint
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
    return True  # allow fusion (default)

structured_opt_args["optimization_hooks"] = {
    GT4PyAutoOptHook.TopLevelDataFlowMapFusionVerticalCallBack: _kolor_aware_fusion_callback,
    GT4PyAutoOptHook.TopLevelDataFlowMapFusionHorizontalCallBack: _kolor_aware_fusion_callback,
}
```

This is the most promising path to close the remaining gap with the unstructured baseline while keeping full SDFG optimization.

---

### Bug 6 — CUDA `ILLEGAL_ADDRESS` on big grids: `apply_fieldview_transforms` is missing IR fusion

**Symptom**: On `grafrap3` + `dace_gpu` + 512×512 grid, every kernel of `compute_advection_in_corrector_vertical_momentum` fails at first launch with `CUDA_ERROR_ILLEGAL_ADDRESS (700)`. The same code on the same branch with a 26×26 grid passes. The previous `grafrap_dace` branch passes on the 512×512 grid.

**Root cause**: When the DaCe path was changed from `apply_common_transforms` (with `apply_common_transform=True`) to `apply_fieldview_transforms`, every IR fusion pass after the structured passes was dropped. Concretely, `apply_fieldview_transforms` is missing the 10-iteration inlining loop with `InlineLambdas`, `CollapseTuple`, `InlineScalar`, `SimplifyCartesianShifts`, `FuseAsFieldOp`, `ConstantFolding`; plus `CommonSubexpressionElimination`, `FuseMaps`, `CollapseListGet`, the `_apply_unroll_reduce_pipeline` (note: `unroll_reduce` parameter is accepted but never used in `apply_fieldview_transforms`), and the final `InlineLambdas(force_inline_lambda_args=True)`.

Without `FuseAsFieldOp`, every `+`, `×`, `cast_`, `if`, `list_get` lands in its own `as_fieldop` over the full structured domain. For `compute_advection_in_corrector_vertical_momentum` the post-pipeline IR is 1632 lines / **289 `as_fieldop`** on `grafrap3` vs 630 lines / **53 `as_fieldop`** on `grafrap_dace`. Each `as_fieldop` becomes a separate DaCe nested-SDFG/map writing to a separate device-side transient.

**Why the small grid works:**
- Per-transient size: 26² × 3 × 50 × 8 B ≈ 800 KB vs 513² × 3 × 50 × 8 B ≈ 316 MB at the big grid.
- Total transient memory: 26² → ~160 MB (fits trivially); 513² → ~60 GB. DaCe must aggressively reuse transients on the big grid — same device pointer rebound to the next kernel. Any latent off-by-one in a declared shape/stride/memlet subset becomes a use-after-free or write-past-end.
- CUDA's allocator rounds up: a small array sits inside a still-mapped page, so a tiny OOB silently reads zeros or stale data. The same OOB on a 316 MB array lands outside the backing memory and CUDA traps it as `700`.
- CUDA grid Y/Z dim cap (65535): at 26² no kernel comes near it; at 512² some auto-generated kernels skirt the limit, and an extra factor in any un-fused intermediate would exceed it.

**Diagnostic** (offline, no GPU job needed): set `print_ir=True` and compare the post-`apply_fieldview_transforms` IR with the post-`apply_common_transforms` IR. Counts that matter for `compute_advection_in_corrector_vertical_momentum`:

| | grafrap_dace (works at 512²) | grafrap3 (fails at 512²) |
|---|---|---|
| Lines | 630 | **1632** |
| `as_fieldop` | 53 | **289** |
| `concat_where` | 34 | 44 |

If you see a ratio anywhere near 5× more `as_fieldop`, this bug is in play.

**Fix (landed, verified on 512×512 grid)**: in `../gt4py/src/gt4py/next/iterator/transforms/pass_manager.py`, at the **very end** of `apply_fieldview_transforms` (after `ir = remove_broadcast.RemoveBroadcast.apply(ir)`), append a structured-backend-gated inlining loop **plus a second `infer_domain.infer_program` call**. Three subtle constraints had to be satisfied in order:

1. **Order: fuse AFTER `infer_program`**. `transform_to_as_fieldop` (and the rest of the DaCe lowering) reads `node.annex.domain` set by `infer_program`. Earlier I placed the fusion block right after the structured `cart_unroll` block (before `infer_program`) → `AttributeError: 'Namespace' object has no attribute 'domain'` inside `transform_to_as_fieldop.visit_FunCall`.

2. **Do NOT call `concat_where.transform_to_as_fieldop`** inside the structured-backend block. `grafrap3`'s `gtir_to_sdfg_primitives.translate_as_fieldop` explicitly rejects tuple-output `as_fieldop` (`raise NotImplementedError("Unexpected 'as_fieldop' with tuple output in SDFG lowering.")`, line 255). `transform_to_as_fieldop` collapses a tuple-returning `concat_where` (e.g. the stencil's 3-tuple `(advective_tendency, corrected_w, cfl)`) into one tuple-output `as_fieldop`, which DaCe then can't lower. Skip the transform → `concat_where` survives and is lowered by `gtir_to_sdfg_concat_where.translate_concat_where`. This matches `grafrap_dace`'s `transform_concat_where_to_as_fieldop=False`.

3. **Re-run `infer_program` AFTER the fusion loop**. `InlineLambdas` / `FuseAsFieldOp` / `CollapseTuple` rebuild the IR tree → newly created `concat_where` nodes have no `annex.domain`. `grafrap3`'s simplified `gtir_to_sdfg_concat_where.py` (shrunk 225 lines vs `grafrap_dace`) hard-relies on the annex at line 241/277: `output_domain = gtir_domain.get_field_domain(node.annex.domain)`. Without the second `infer_program` call, the same `AttributeError` reappears at DaCe lowering time. The bumped recursion limit in `infer_domain.py` is required for this second call to complete on the 512² IR.

```python
# At the end of apply_fieldview_transforms, after `ir = remove_broadcast.RemoveBroadcast.apply(ir)`:
if os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1":
    for _ in range(10):
        inlined = ir
        inlined = InlineLambdas.apply(inlined, opcount_preserving=True)
        inlined = ConstantFolding.apply(inlined)
        inlined = CollapseTuple.apply(
            inlined,
            enabled_transformations=~CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES,
            uids=uids,
            offset_provider_type=offset_provider_type,
        )
        inlined = InlineScalar.apply(inlined, offset_provider_type=offset_provider_type)
        inlined = simplify_cart_shifts.SimplifyCartesianShifts.apply(inlined)
        try:
            inlined = fuse_as_fieldop.FuseAsFieldOp.apply(
                inlined, uids=uids, offset_provider_type=offset_provider_type
            )
        except Exception:
            pass
        inlined = ConstantFolding.apply(inlined)
        if inlined == ir:
            break
        ir = inlined
    ir = NormalizeShifts().visit(ir)
    ir = InlineLambdas.apply(ir, opcount_preserving=True, force_inline_lambda_args=True)
    # Repopulate node.annex.domain on concat_where nodes rebuilt by the fusion loop;
    # gtir_to_sdfg_concat_where.translate_concat_where reads it at lowering time.
    ir = infer_domain.infer_program(
        ir,
        symbolic_domain_sizes=symbolic_domain_sizes,
        offset_provider=offset_provider,
    )
```

**Verification (slurm job 4311561, 512×512 grid, K=50, dace_gpu, `test_compute_advection_in_vertical_momentum_equation`)**:

| | grafrap3 (broken) | grafrap3 (patched) |
|---|---|---|
| Result | `CUDA_ERROR_ILLEGAL_ADDRESS (700)` at first kernel | **5/5 variants passed** |
| Steady-state GPU exec / call | n/a | **38 ms** |
| Steady-state pack / unpack | n/a | 1.29 s / 1.65 s |
| First-call JIT + first exec | n/a | 238 s |
| Total wall-time (5 variants × 3 rounds × 10 iter) | n/a | 24:22 |

**Failure-mode trail** (each successive run revealed the next blocker):
1. `CUDA_ERROR_ILLEGAL_ADDRESS (700)` — original IR explosion (289 vs 53 `as_fieldop`); transient pool exhausted at 512².
2. `AttributeError: 'Namespace' object has no attribute 'domain'` inside `transform_to_as_fieldop.visit_FunCall` — fusion block placed before `infer_program`.
3. `NotImplementedError: Unexpected 'as_fieldop' with tuple output in SDFG lowering.` — `transform_to_as_fieldop` collapsed tuple-returning `concat_where` into a single tuple-output `as_fieldop`.
4. `AttributeError: 'Namespace' object has no attribute 'domain'` inside `gtir_to_sdfg_concat_where.translate_concat_where` — fusion loop rebuilt `concat_where` nodes, stripping the annex set by the first `infer_program`.
5. Pass.

**Alternative fix (not landed)**: swap `apply_fieldview_transforms` for `apply_common_transforms` (with `force_inline_lambda_args=True`, retry with `unroll_reduce=True` if `neighbors` survive) in `translation.py:_generate_sdfg_without_configuring_dace`. Restores `CommonSubexpressionElimination`, `FuseMaps`, `CollapseListGet`, and the real `_apply_unroll_reduce_pipeline` too — potentially fewer SDFG nodes, but a bigger blast radius (all DaCe stencils go through the heavier pipeline).

**Recursion-limit bumps** in `constant_folding.py` and `infer_domain.py` are still needed (the second `infer_program` on the 512² IR recurses past the default 1000 frames).

**Related lateral env-var note**: `unset GT4PY_TRANSLATOR_LATERAL` and `export GT4PY_TRANSLATOR_LATERAL=0` are not equivalent. Default in code is `"1"`, so the small-grid command (`unset …`) runs with `lateral=1` and the big-grid command (`=0`) runs with `lateral=0`. This changes `start_i/start_j/end_i/end_j` in `StructuredRemapSizes`. Unrelated to the fusion bug, but worth aligning between runs if comparing IRs.

**Trade-off note**: the steady-state pack (1.29 s) and unpack (1.65 s) dominate over the actual GPU exec (38 ms). At production grid sizes the GPU exec scales linearly but pack/unpack scale similarly — the structured layout's win shows up only if pack/unpack can be made async with compute, or amortised across many kernels in the same call.

---

## DaCe GPU Backend — Known Bugs and Fixes

### Cache management (important!)

The gt4py OTF compilation cache lives at `icon4py/.gt4py_cache` (controlled by `GT4PY_BUILD_CACHE_DIR`). The SLURM script sets `GT4PY_BUILD_CACHE_DIR=${WORKDIR}` so the cache is at `${WORKDIR}/.gt4py_cache = icon4py/.gt4py_cache`. The non-SLURM runner script (`run_stencil_commands.sh`) clears `$repo_root/.gt4py_cache` before each run. **Always clear the cache when changing grids or switching branches** — a stale compiled binary for the wrong grid size causes wrong results or crashes.

```bash
rm -rf /scratch/mch/rgraf/icon4py/.gt4py_cache
```

### Small-grid workflow (26×26, K=5)

Use `commands_stencils_small.txt` for fast iteration during development. Run on the SLURM **debug** partition:

```bash
# From icon4py/ directory, after sbatch or srun:
bash scripts/run_stencil_commands.sh \
  --command-file commands_stencils_small.txt \
  --output-dir output/small_26_grafrap3/ \
  --jobs 1
```

For SBATCH:
```bash
sbatch --partition=debug --time=02:00:00 \
  scripts/run_stencil_commands_slurm.sh \
  --command-file commands_stencils_small.txt \
  --output-dir output/small_26_grafrap3/
```

Grid files:
- Small (26×26): `grid_generator/parallelogram_grid_26.nc` — K=5 levels, fast debug runs
- Standard (512×512): `grid_generator/parallelogram_grid.nc` — K=50 levels, full benchmark

### Small-grid status (grafrap3, 26×26, K=5) — updated 2026-05-15

**9/10 stencils pass** on `dace_gpu`. Only stencil 03 (C2E2CO) remains failing.

| # | Stencil | Status | Root cause / fix |
|---|---------|--------|-----------------|
| 01 | nabla2_smag (E2C2V) | ✅ PASSES | FuseAsFieldOp ratio-guard fix |
| 02 | horiz_advection (C2E) | ✅ PASSES | - |
| 03 | extra_diffusion (C2E2CO) | ❌ FAILS | Wrong `cell_to_ijk` mapping → asymmetric bounds |
| 04 | div_damping (E2C+E2C2EO) | ✅ PASSES | - |
| 05 | avg_vn_graddiv (E2C2EO) | ✅ PASSES | - |
| 06 | adv_hmom (complex) | ✅ PASSES | - |
| 07 | interp_cell (C2E) | ✅ PASSES | - |
| 08 | cells2verts (V2C) | ✅ PASSES | - |
| 09 | rot_vertex (V2E) | ✅ PASSES | Test `horizontal_start` fix |
| 10 | diffusion_vn (E2C2V) | ✅ PASSES | - |

---

### Fix 1 — Stencil 01 (E2C2V): FuseAsFieldOp ratio-guard (FIXED)

**Symptom**: `InvalidSDFGNodeError: Dangling in-connector __tlet_arg0 (at tlet_42_minus)` for `calculate_nabla2_and_smag_coefficients_for_vn`.

**Root cause**: `FuseAsFieldOp.apply` reduces 249→3 as_fieldop (83×) in one step for this stencil. The per-kolor split creates 9 SetAts (3 outputs × 3 kolors) with structurally identical lambda bodies → they share the same `as_fieldop` node. FuseAsFieldOp inlines 249 inner nodes into this single shared as_fieldop, creating a fused lambda with broken data-flow (one `as_fieldop` referenced from 3 kolor domains → dangling `tlet_42_minus` connector).

Confirmed: disabling FuseAsFieldOp (`GT4PY_DISABLE_FUSE_AS_FIELDOP=1`) makes stencil 01 pass in 189s.

**Fix** (`gt4py/src/gt4py/next/iterator/transforms/pass_manager.py`): reject fusion when the reduction ratio exceeds 20× in one step:
```python
_ratio = int(os.environ.get("GT4PY_FUSE_RATIO_THRESHOLD", "20"))
if _n_before > 0 and _n_after > 0 and (_n_before // _n_after) > _ratio:
    inlined = _pre_fuse  # restore pre-fusion IR
```
Also added `CommonSubexpressionElimination + MergeLet` before `FuseAsFieldOp` each iteration.

**Note**: With the ratio-guard, stencil 01 retains ~249 as_fieldop nodes. At 512×512 this causes IR explosion (slow execution / GPU transient OOM). Long-term fix: de-share SetAt expressions before fusion (deepcopy each SetAt's `as_fieldop` after kolor-split).

---

### Fix 2 — Stencil 09 (V2E): Test `horizontal_start` fix (FIXED)

**Symptom**: Wrong `rot_vec` values — FuseAsFieldOp over-fused (earlier), then wrong reference mismatch when `horizontal_start` was added.

**Fix**: Changed test `horizontal_start` from 0 to `grid.start_index(vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))` AND updated the reference function to also respect `horizontal_start`, so vertices in the lateral boundary halo are excluded from both computation and reference comparison.

File: `model/atmosphere/dycore/tests/dycore/stencil_tests/test_mo_math_divrot_rot_vertex_ri_dsl.py`

---

### Remaining Bug — Stencil 03 (C2E2CO): Wrong `cell_to_ijk` mapping (open)

**Symptom**: `ddt_w_adv` has ~1% wrong elements on `dace_gpu`; passes on `dace_cpu` and `gtfn`. FuseAsFieldOp is **not** the cause (confirmed: disabling fusion gives same ~1% failure).

**Root cause**: The IR domain for the C2E2CO stencil is `IDimₕ:[1,26), JDimₕ:[0,26)` — **asymmetric**. IDim is correctly clamped to [1,26) by `LATERAL_BOUNDARY_LEVEL_2`, but JDim starts at 0. Cells at JDim=0 (kolor=0) have C2E2CO neighbors at JDim-1=-1 → OOB GPU access → wrong values. Similarly, cells at IDim=25/JDim=25 (kolor=1) access IDim+1=26 / JDim+1=26 → OOB. On CPU, the OOB reads accidentally land on valid adjacent memory → passes.

The domain should be **symmetric**: either [0,26) for both (if all boundary cells are in the halo) or [1,25) for both (if interior domain contracts). The asymmetry means `LATERAL_BOUNDARY_LEVEL_2` in the unstructured cell ordering excludes IDim=0 cells but NOT JDim=0 / IDim=25 / JDim=25 cells from the structured layout.

**The hypothesis**: The `build_cell_ijk_maps` function in `gt4py/src/gt4py/next/modules/translator.py` is incorrectly assigning some cells to boundary structured positions (JDim=0) when they should be at interior positions. This makes `LATERAL_BOUNDARY_LEVEL_2` unable to exclude all OOB cells. According to ICON convention, `LATERAL_BOUNDARY_LEVEL_2` should already exclude all cells with invalid C2E2CO neighbors.

**File to investigate**: `gt4py/src/gt4py/next/modules/translator.py`, function `build_cell_ijk_maps` (lines 737–764) — specifically whether the `i_min = min(i_coords)` / `j_min = min(j_coords)` formula correctly assigns cells to structured positions for the parallelogram mesh.

**Next step**: Print the structured (IDim, JDim, kolor) positions for cells near the LATERAL_BOUNDARY zone boundary and verify they match expected interior positions.

### Visualization scripts

| Script | Field type | Key args |
|---|---|---|
| `scripts/compare_arrays.py` | Edge (3 kolors) | `--nx <nx> --ny <ny>` |
| `scripts/compare_cell_arrays.py` | Cell (2 kolors) | `--nx <nx> --ny <ny>` |
| `scripts/compare_vertex_arrays.py` | Vertex (1 kolor) | `--nx <nx+1> --ny <ny+1>` |

For the 26×26 grid: cell `--nx 26 --ny 26`, vertex `--nx 27 --ny 27`.
For the 512×512 grid: cell `--nx 512 --ny 512`, vertex `--nx 513 --ny 513`.