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
- **Grid dimensions**: 16 × 13 (I × J), resulting in **653 total edges**
- **Parallelogram structure**: each parallelogram consists of two triangles; the dual-mesh triangulation gives 3 edge types (kolors)
- **Three edge kolors**:
  - Kolor 0: horizontal edges — shape `nx × (ny+1)` = `16 × 14`
  - Kolor 1: NE (north-east) edges — shape `(nx+1) × ny` = `17 × 13`
  - Kolor 2: diagonal edges — shape `nx × ny` = `16 × 13`
- **Cell kolors**: 2 (lower and upper triangle per parallelogram cell)
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
| `commands_stencils.txt` | Test commands for all implemented stencils |
| `ir_out.txt` | Captured IR output (set `print_ir=True` or env var) |
| `model/testing/src/icon4py/model/testing/stencil_tests.py` | Stencil setup for unstructured and structured test runs; This logic needs to somehow be ported to the standalone driver for end-to-end testing, but is currently only used for individual stencil tests |
| `../gt4py/src/gt4py/next/program_processors/codegens/gtfn/gtfn_module.py` | GTFN code generator module |
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
rm -rf .gt4py_cache

pytest -q <test_file> --backend=gtfn_cpu --grid ../grid-generator/parallelogram_grid.nc:<levels> --maxfail=2 -s > test_out.txt

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

**Note on end = total**: When `horizontal_end_distance == total_edges` (= 653 for this grid), the range `[start, total)` equals the start-only condition. The bounding box is exact in this case. When `end < total` (true lateral boundary slice), the bounding box may be slightly loose (the rectangular hull covers more than the exact set of edges), but the SetAt domain already restricts computation to `[start, end)` so results remain correct.

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