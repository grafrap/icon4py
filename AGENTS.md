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
| `../gt4py/src/gt4py/next/iterator/transforms/cart_unroll.py` | **Custom GT4Py IR pass** — maps unstructured domains to structured I/J/Kolor domains. Written from scratch. Contains `CartesianDomainAndTypeRemapper`, `CartesianReductionUnroller`, `KolorConstantPropagation` (disabled), and helpers. |
| `../gt4py/src/gt4py/next/iterator/transforms/pass_manager.py` | GT4Py pass pipeline — modified to call `CartesianDomainAndTypeRemapper` and `CartesianReductionUnroller` under `USE_STRUCTURED_BACKEND=1` |
| `../gt4py/src/gt4py/next/modules/cartesian_interceptor.py` | `GenericStructuredWrapper` — intercepts stencil calls, packs unstructured fields to structured layout, triggers lazy per-`horizontal_start` compilation |
| `../gt4py/src/gt4py/next/iterator/transforms/map_dict.py` | Hard-coded structured remap table: for every connectivity (E2C2EO, E2C, E2V, V2E, C2E, …) and slot, gives the structured (di, dj, dk) offset |
| `../gt4py/src/gt4py/next/modules/translator.py` | Field packing/unpacking utilities: `pack_edge_field_to_structured`, `build_index_map_from_lonlat_e2v`, `_derive_entity_start_bounds_from_mapping`, etc. |
| `model/atmosphere/dycore/tests/dycore/stencil_tests/conftest.py` | Pytest fixture that injects `GenericStructuredWrapper` for dycore stencil tests |
| `model/atmosphere/diffusion/tests/diffusion/stencil_tests/conftest.py` | Same for diffusion stencil tests |
| `scripts/compare_arrays.py` | Visualises 3-grid (kolor 0/1/2) match map: 0 = mismatch, 1 = match; boundary zeros are expected padding |
| `commands_stencils.txt` | Test commands for all implemented stencils |
| `ir_out.txt` | Captured IR output (set `print_ir=True` or env var) |

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

## `cart_unroll.py` Architecture

Two sequential passes inside `CartUnroll.apply()`:

### Pass 1: `CartesianDomainAndTypeRemapper`
- Transforms unstructured domains (`unstructured_domain(Edge, K)`) → structured Cartesian domains (`IDim, JDim, Kolor, K`)
- **Per-kolor split** (key optimisation): for each edge-domain SetAt, generates **3 separate SetAt nodes** (one per kolor) with kolor-specific I/J bounds derived from the `edge_to_ijk` mapping via `_derive_entity_start_bounds_from_mapping`. This eliminates the 3-copy `concat_where` wrapping.
- Remaps field types (Edge → IDim/JDim/Kolor)
- Transforms threshold comparisons like `Edgeₕ >= start_2nd_nudge_line_idx_e` into structured per-kolor domain conditions using precomputed bounds from `symbolic_domain_sizes`
- The `horizontal_start` value (from the stencil's runtime kwargs) is baked into `symbolic_domain_sizes` at lazy compile time

### Pass 2: `CartesianReductionUnroller`
- Expands `reduce(plus)(neighbors(E2C2EOₒ, it))` into explicit per-slot shifted field accesses
- Uses `current_kolor` (extracted from the per-kolor SetAt domain) to select the matching shift branch **directly**, avoiding inner `concat_where` kolor branching
- **Entity-type safety**: only passes `current_kolor` to `_build_field_concat_where_from_branches` for **edge-center connectivities** (prefix "E": `E2C`, `E2V`, `E2C2EO`, etc.). For cell-center (`C2E`) and vertex-center (`V2E`) connectivities, `current_kolor=None` is passed so all kolor branches are generated correctly. This prevents cross-entity kolor confusion (the bug with `compute_advection_in_horizontal_momentum` was exactly that `C2E` cell-kolor branches were being folded using edge-kolor context).

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

### Mapping-Based Domain Bounds (replaces `GT4PY_TRANSLATOR_LATERAL`)
- Per-kolor I/J bounds derived from `edge_to_ijk[horizontal_start:]` via `_derive_entity_start_bounds_from_mapping`
- Exact bounds (not symmetric heuristic) for each kolor's valid domain
- `start_2nd_nudge_line_idx_e` (and other threshold params) similarly handled with per-kolor bounds

---

## Known Issues / In-Progress

### `_edge_shape_domain` over-clipping with per-kolor split
- When `current_kolor is not None`, the `_edge_shape_domain` function clips the per-kolor-specific domain (which is already restricted), causing over-clipping that excludes valid source positions.
- **Symptom**: `apply_divergence_damping_and_update_vn` fails when run with the per-kolor split at `:2` levels.
- **Fix**: Skip `_edge_shape_domain` when `current_kolor is not None` — the per-kolor SetAt domain already correctly restricts to valid interior source positions, so neighbors are within valid range.

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
---

## Writing Tests

When modifying `cart_unroll.py` or related passes, write unit tests in the relevant unit test file (e.g., `../gt4py/tests/next_tests/unit_tests/iterator_tests/transforms_tests/test_cart_unroll.py`). 
The `StencilTest` base class handles reference vs. structured comparison. Use `STATIC_PARAMS` to test with `NONE`, `COMPILE_TIME_DOMAIN`, and `COMPILE_TIME_VERTICAL` variants.

---

## Standalone Driver (Future)

- Target: `model/standalone_driver/src/icon4py/model/standalone_driver/standalone_driver.py`
- Requires wrapping each granule's programs with `GenericStructuredWrapper` at model init time
- The wrapper's lazy compilation + threshold param support handles different `horizontal_start` values per stencil
- Still deferred; focus is on getting all individual stencil tests correct first
