# Bug Fix and Investigation History

This file documents the history of bugs investigated and fixed during development of the structured GT4Py backend. For current project state and architecture, see [CLAUDE.md](CLAUDE.md).

---

## Driver Fixes Landed (2026-04-27)

Three classes of issues were fixed to advance driver progress:

### Fix 1: Cell threshold params (`compute_exner_exfac`)
`_compute_exner_exfac` uses `concat_where(CellDim >= lateral_boundary_level_2, ...)`. After structured remapping this generated `gtfn::index(Cell)` in C++ (Cell not declared in structured domain).

**Fix**:
- Added `_KNOWN_CELL_THRESHOLD_PARAMS = frozenset({"lateral_boundary_level_2"})` in `cartesian_interceptor.py`.
- `_get_or_compile` now also injects per-kolor cell bounds from `cell_to_ijk` for cell threshold params (analogous to edge threshold params).
- `_derive_entity_start_bounds_from_mapping` updated: Cell now returns per-kolor `{0:..., 1:...}` bounds (using `cell_to_ijk` `(i,j,kolor)` format), not just a single global bound.
- `_mapping_based_threshold_condition` in `visit_FunCall` now handles `n_kolors=2` for Cell, `n_kolors=3` for Edge.
- `has_threshold_mapping` check extended from `entity_axis == "Edge"` to `entity_axis in {"Edge", "Cell"}`.
- `__call__` collects both `_KNOWN_EDGE_THRESHOLD_PARAMS` and `_KNOWN_CELL_THRESHOLD_PARAMS` in `extra_thresholds`.

### Fix 2: AND of two edge threshold bounds (`compute_pressure_gradient_downward_extrapolation_mask_distance`)
Stencil uses `concat_where((start <= EdgeDim) & (EdgeDim < end), ...)`. Previously the upper bound was remapped as `not_(interior_cond)` which produced invalid C++ (not a valid domain expression).

**Fix**:
- Added `_derive_entity_range_bounds_from_mapping(entity, mapping_rows, horizontal_start, horizontal_end, ...)` in `cart_unroll.py` — computes per-kolor bounding boxes for `mapping_rows[start:end]`.
- Added `_inject_edge_range_bounds(...)` module-level function in `cartesian_interceptor.py` — detects paired threshold params by name (`horizontal_start_X` + `horizontal_end_X` with matching suffix), computes range bounds for `[start, end)`, injects under key `{start_name}|{end_name}_k{k}_{ilo/jlo/ihi/jhi}`.
- Added early `and_()` interception at top of `visit_FunCall` in `CartesianDomainAndTypeRemapper`: before children are visited, checks if the node is `and_(EdgeCmp_a, EdgeCmp_b)` and both threshold IDs form a known range pair. If so, returns the range domain condition directly — no `not_()` needed.

**Note on end = total**: When `horizontal_end_distance == total_edges` (= 15,190 for this grid), the range `[start, total)` equals the start-only condition. The bounding box is exact in this case. When `end < total` (true lateral boundary slice), the bounding box may be slightly loose (the rectangular hull covers more than the exact set of edges), but the SetAt domain already restricts computation to `[start, end)` so results remain correct.

### Fix 3: K-broadcast type inference compatibility (`inference.py`)
After structured remapping, an `EdgeField` (no K) used in an `EdgeKField` context (e.g. `z_me - extrapolation_distance` where `z_me` is EdgeKField) caused type inference conflicts:
- `existing=Field[[IDim,JDim,Kolor]]`, `inferred=Field[[IDim,JDim,Kolor,K]]` → `TypeError`
- `existing=IteratorType(pos=[IDim,JDim,Kolor,K])`, `inferred=IteratorType(pos=[IDim,JDim,Kolor])` → `AssertionError`

**Fix** in `_is_structured_remap_compatibility_case` (`inference.py`):
- **FieldType case**: added Case 2 — if `inferred.dims[-1].value in {"K","KHalf"}` and `inferred.dims[:-1] == existing.dims`, return True (K appended by broadcast).
- **IteratorType case**: added Case A — if `existing.position_dims[-1].value in {"K","KHalf"}` and `existing.position_dims[:-1] == inferred.position_dims` and `defined_dims` match, return True (iterator in K-context has extra K in position).

### Fix 4: `GenericStructuredWrapper.__call__` positional argument support
Diffusion `init_run` calls stencils like `init_diffusion_local_fields_for_regular_timestep(K4, substep, *smagorinski_factor, ...)` with positional args. `GenericStructuredWrapper.__call__(**kwargs)` rejected these.

**Fix**: Changed `__call__` to accept `*args`. When positional args are given, maps them to kwarg names by looking at `self._operator.past_stage.past_node.params` (the program's declared param order). Silently skips if params not accessible.

### Fix 5: IteratorType K-broadcast compatibility (second pattern)
`__acc_init` for `max_over` reductions on EdgeKField produced a conflict where both `position_dims` AND `defined_dims` lost K:
- `existing=IteratorType(pos=[IDim,JDim,Kolor,K], defined=[IDim,JDim,Kolor,K])`
- `inferred=IteratorType(pos=[IDim,JDim,Kolor], defined=[IDim,JDim,Kolor])`

**Fix**: Extended Case A in `_is_structured_remap_compatibility_case` (`inference.py`) to also accept when `existing.defined_dims[-1]` is K and `existing.defined_dims[:-1] == inferred.defined_dims` (i.e. K removed from both position and defined dims simultaneously).

### Fix 6: Direct stencil calls bypassing `GenericStructuredWrapper` (`initial_condition.py`)
Stencils called with `.with_backend(backend)` directly (e.g. `edge_2_cell_vector_rbf_interpolation`, `compute_difference_on_cell_k` in `initial_condition.py`) compiled with the structured pass (because `USE_STRUCTURED_BACKEND=1`) but received unstructured fields → C++ shape mismatch.

**Fix**: Wrapped both with `GenericStructuredWrapper` (same pattern as `cell_2_edge_interpolation` which was already wrapped). Changed `offset_provider={}` → `offset_provider=grid.connectivities` for `compute_difference_on_cell_k` so the wrapper can build cell packing tables.

### Fix 7: Double-wrapping in `_wrap_granule_programs_for_structured_backend` (`standalone_driver.py`)
`model_options.setup_program` already wraps each diffusion/dycore stencil as `functools.partial(GenericStructuredWrapper_instance, **static_args)`. Then `_wrap_granule_programs_for_structured_backend` extracted `.func` (= the wrapper) and wrapped it **again** → `AttributeError: 'GenericStructuredWrapper' has no attribute '__name__'` (from `customize_backend` in `setup_program`).

**Fix**: In `_wrap_granule_programs_for_structured_backend`, if `operator = attr_value.func` is already a `GenericStructuredWrapper`, reuse it directly instead of double-wrapping. Get param names from `wrapper._operator` (the inner original program).

### Fix 8: `calculate_nabla2_and_smag_coefficients_for_vn` — E2C2V vertex field kolor stride mismatch
The stencil used `u_vert_wp(E2C2V) * coeff` (full neighbor collection via `map_` + `neighbors`) then indexed `v_n[E2C2VDim(i)]`. In the structured backend, this generates a `map_(mul)(neighbors(E2C2V, u_vert_wp), coeff)` IR node that, when expanded per-slot, emits C++ kolor shifts (`at_key<Kolor_implicit=2>(vertex_kolor_stride)`). Vertex fields have Kolor=1, so the kolor stride is stored as `integral_constant<int, 0>` (compile-time zero scalar), and `at_key<2>` on a scalar fails at compile time.

**Fix**: Rewrote the stencil body to use explicit per-slot access `u_vert_wp(E2C2V[i]) * coeff[E2C2VDim(i)]` for i=0,1,2,3 (matching the working `calculate_nabla4.py` pattern). This generates `deref(shift(E2C2V, i)(it))` nodes (not `map_` + `neighbors`), which the GTFN backend handles correctly.

**Pattern rule**: When accessing vertex fields via E2C2V, always use explicit `E2C2V[i]` slot access, NOT the full `field(E2C2V)` + `[E2C2VDim(i)]` indexing. The latter creates a `map_/neighbors` IR pattern that triggers invalid kolor stride lookups in the structured C++ code for 1-kolor (vertex) fields.

---

## Bugs Fixed in DaCe Lowering (initial DaCe integration, 2026-05-06)

### Bug 1: `DaCeBackendFactory` missing `auto_optimize`
`DaCeWorkflowFactory` uses `factory.SelfAttribute("..auto_optimize")` which only resolves when the parent explicitly passes `auto_optimize`. Without it, factory-boy raises `AttributeError: The parameter 'auto_optimize' is unknown`. Calling `DaCeBackendFactory(cached=True, otf_workflow__cached_translation=True)` alone is not enough.

**Fix**: Use `make_dace_backend(gpu=False, cached=True, auto_optimize=False)` which sets all required params.

### Bug 2: NEVER-domain args crash `translate_as_fieldop`
The structured unroller generates `as_fieldop(λ __x → 0)(e_bln_c_s)` for accumulator zero-init. The lambda ignores `__x`, so domain inference marks `e_bln_c_s` as `DomainAccessDescriptor.NEVER`. `translate_symbol_ref` returns `None` for NEVER symbols. `_parse_fieldop_arg` got `None` instead of `FieldopData` → `ValueError: Expected a field, found a tuple of fields.`

**Fix** in `gtir_to_sdfg_primitives.py`:
- `_parse_fieldop_arg`: when `sdfg_builder.visit(node)` returns `None`, return `None` (don't raise)
- `translate_as_fieldop`: before calling `translate_lambda_to_dataflow`, zip params with args and filter out pairs where `arg is None`. Create a new `Lambda` with only live params. This correctly handles `as_fieldop(λ x → 0)(f)` by stripping `x`/`f` since the body doesn't use them.

### Bug 3: SDFG array shape vs binding param type mismatch
After the structured passes, the SDFG arrays have 4 dims (`IDim × JDim × Kolor × K`). But `DaCeTranslator.__call__` used `inp.args.args` (PAST-level, unstructured types: `Edge × K` = 2 dims) for `program_parameters` → the Python bindings generated by `_create_sdfg_bindings` used the wrong (2-dim) types → `zip(param_type.dims, sdfg_arg_desc.shape, strict=True)` failed.

Root cause: `apply_fieldview_transforms` returns a NEW `itir.Program` with remapped params, but this remapped program is discarded after SDFG building. The `__call__` method still held the original (unremapped) `program` from `inp.data`. Additionally, `StructuredTypeRemapper.visit_Program` creates new `ir.Sym` objects with remapped types, but these live only in the remapped program returned by the pass.

**Fix** in `DaCeTranslator.__call__`: apply `StructuredTypeRemapper._cartesian_remapped_type()` to each type in `inp.args.args` when `USE_STRUCTURED_BACKEND=1`. This remaps `Edge → [IDim, JDim, Kolor]` etc. so the binding code uses types that match the structured SDFG arrays.

### Bug 4: `offset_provider_type` missing IDim/JDim/Kolor
After the structured passes, the IR uses `shift(IDim, di)` etc. DaCe's `_make_cartesian_shift()` fires only when `offset_provider_type[key]` is a `Dimension` object. The original `offset_provider_type` had only unstructured entries (E2C → `NeighborConnectivityType`, etc.). IDim/JDim/Kolor were absent → DaCe couldn't lower the structured shifts.

**Fix** in `translation._generate_sdfg_without_configuring_dace`: augment `offset_provider_type` with `{"IDim": Dimension("IDim"), "JDim": Dimension("JDim"), "Kolor": Dimension("Kolor")}` before `build_sdfg_from_gtir`.

---

## Bug 5 — `horizontal_start` not reaching `apply_fieldview_transforms` (DaCe thread pool)

**Symptom**: Stencils with `horizontal_start > 0` (interior/nudging zones) produced wrong bounds — always `[0, max_i)` × `[0, max_j)` — and failed numerical comparison. Stencils with `horizontal_start=0` passed.

**Root cause**: DaCe submits compilation to a `ThreadPoolExecutor`. `threading.local` and Python 3.10 `contextvars.ContextVar` are **NOT** inherited by thread-pool threads (ContextVar propagation to ThreadPoolExecutor was only added in Python 3.12). So `_CURRENT_COMPILE_SDS` set in the main thread was invisible in the pool thread, causing `get_compile_sds()` to return `None` → fallback to mesh metadata without `horizontal_start`.

**Fix**: Replace `threading.local` / `contextvars.ContextVar` with a plain module-level global `_CURRENT_COMPILE_SDS: dict | None`. This works because the main thread is **blocked** inside `_compiled_programs()` (which waits for DaCe to finish compiling and executing), so the global is stable while the pool thread reads it — no race condition.

Set the global:
1. In `_get_or_compile`: `_CURRENT_COMPILE_SDS = sds` before `_setup()`, cleared in `finally`
2. In `__call__`: `_CURRENT_COMPILE_SDS = _call_sds` before `_compiled_programs()`, cleared in `finally`

`_sds_cache: dict[tuple, dict]` stores the sds per `(horizontal_start, extra_thresholds)` so `__call__` can look it up for subsequent calls (DaCe compilation is JIT — fired from `_compiled_programs()` on first call, not from `_setup()`).

---

## Bug 6 — CUDA `ILLEGAL_ADDRESS` on big grids: `apply_fieldview_transforms` missing IR fusion

**Symptom**: On `grafrap3` + `dace_gpu` + 512×512 grid, every kernel of `compute_advection_in_corrector_vertical_momentum` fails at first launch with `CUDA_ERROR_ILLEGAL_ADDRESS (700)`. The same code on the same branch with a 26×26 grid passes. The previous `grafrap_dace` branch passes on the 512×512 grid.

**Root cause**: When the DaCe path was changed from `apply_common_transforms` to `apply_fieldview_transforms`, every IR fusion pass after the structured passes was dropped. Without `FuseAsFieldOp`, every `+`, `×`, `cast_`, `if`, `list_get` lands in its own `as_fieldop` over the full structured domain. For `compute_advection_in_corrector_vertical_momentum` the post-pipeline IR is 1632 lines / **289 `as_fieldop`** on `grafrap3` vs 630 lines / **53 `as_fieldop`** on `grafrap_dace`. Each `as_fieldop` becomes a separate DaCe nested-SDFG/map writing to a separate device-side transient.

**Why the small grid works:**
- Per-transient size: 26² × 3 × 50 × 8 B ≈ 800 KB vs 513² × 3 × 50 × 8 B ≈ 316 MB at the big grid.
- Total transient memory: 26² → ~160 MB (fits trivially); 513² → ~60 GB. DaCe must aggressively reuse transients on the big grid.
- CUDA's allocator rounds up: a small array sits inside a still-mapped page, so a tiny OOB silently reads zeros or stale data. The same OOB on a 316 MB array lands outside the backing memory and CUDA traps it as `700`.

**Fix (landed, verified on 512×512 grid)**: in `../gt4py/src/gt4py/next/iterator/transforms/pass_manager.py`, at the **very end** of `apply_fieldview_transforms` (after `ir = remove_broadcast.RemoveBroadcast.apply(ir)`), append a structured-backend-gated inlining loop **plus a second `infer_domain.infer_program` call**. Three subtle constraints:

1. **Order: fuse AFTER `infer_program`**. Earlier placing the fusion block before `infer_program` → `AttributeError: 'Namespace' object has no attribute 'domain'` inside `transform_to_as_fieldop.visit_FunCall`.
2. **Do NOT call `concat_where.transform_to_as_fieldop`** inside the structured-backend block. `translate_as_fieldop` explicitly rejects tuple-output `as_fieldop` (`raise NotImplementedError("Unexpected 'as_fieldop' with tuple output in SDFG lowering.")`).
3. **Re-run `infer_program` AFTER the fusion loop**. `InlineLambdas` / `FuseAsFieldOp` / `CollapseTuple` rebuild the IR tree → newly created `concat_where` nodes have no `annex.domain`. Without the second `infer_program` call, `AttributeError` reappears at DaCe lowering time.

```python
# At the end of apply_fieldview_transforms, after `ir = remove_broadcast.RemoveBroadcast.apply(ir)`:
if os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1":
    for _ in range(10):
        inlined = ir
        inlined = InlineLambdas.apply(inlined, opcount_preserving=True)
        inlined = ConstantFolding.apply(inlined)
        inlined = CollapseTuple.apply(...)
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
    ir = infer_domain.infer_program(
        ir,
        symbolic_domain_sizes=symbolic_domain_sizes,
        offset_provider=offset_provider,
    )
```

**Verification (slurm job 4311561, 512×512 grid, K=50, dace_gpu)**:

| | grafrap3 (broken) | grafrap3 (patched) |
|---|---|---|
| Result | `CUDA_ERROR_ILLEGAL_ADDRESS (700)` at first kernel | **5/5 variants passed** |
| Steady-state GPU exec / call | n/a | **38 ms** |
| Steady-state pack / unpack | n/a | 1.29 s / 1.65 s |
| First-call JIT + first exec | n/a | 238 s |

**Failure-mode trail** (each successive run revealed the next blocker):
1. `CUDA_ERROR_ILLEGAL_ADDRESS (700)` — original IR explosion
2. `AttributeError: 'Namespace' object has no attribute 'domain'` inside `transform_to_as_fieldop.visit_FunCall` — fusion block placed before `infer_program`.
3. `NotImplementedError: Unexpected 'as_fieldop' with tuple output in SDFG lowering.` — `transform_to_as_fieldop` collapsed tuple-returning `concat_where`.
4. `AttributeError: 'Namespace' object has no attribute 'domain'` inside `gtir_to_sdfg_concat_where.translate_concat_where` — fusion loop rebuilt `concat_where` nodes, stripping the annex set by the first `infer_program`.
5. Pass.

**Recursion-limit bumps** in `constant_folding.py` and `infer_domain.py` are still needed (the second `infer_program` on the 512² IR recurses past the default 1000 frames).

---

## DaCe Optimization Experiment History

The optimization strategy was developed through experiments on the `dace_cpu` backend. Current production default: `DACE_OPT_EXPERIMENT=FD`.

### Experiments A-C (2026-05-06 to 2026-05-07)

**Experiment A — `gt_set_iteration_order` only**: K is stride-1 in `[IDim,JDim,Kolor,K]`; making K the innermost loop via `unit_strides_kind=VERTICAL` would improve cache behavior. Superseded by Experiment C.

**Experiment B — drop `gt_simplify` from per-kolor split path**: Test 02 (C2E horiz_advection) was 11× SLOWER than the no-opt baseline (103ms vs 9ms), while test 07 (also C2E) was unchanged. Not needed: Exp C superseded it.

**Experiment C — `gt_auto_optimize(disable_splitting=True)` for all structured stencils (→ opt_v2, IMPLEMENTED)**: Only the two `propagate_memlets_sdfg` calls inside Phase 1's `if not disable_splitting:` block are unsafe. `disable_splitting=True` skips only the unsafe Phase 1 block, preserving all other phases. Test 02 improved 14.2× over opt_v1; test 05 improved 4.3×; all 10 stencils improved vs opt_v1.

### Experiments D-G (2026-05-08 to 2026-05-10)

A `DACE_OPT_EXPERIMENT` env-var selector was added to `translation.py` to test each parameter independently.

Full benchmark results (all 10 stencils), **exec-only median (ms)** from `[timing]` lines:

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
- **Exp D (blocking_dim=JDim, blocking_size=8)**: Improves E2C2V stencils (nabla2_smag, interp_cell). Big regression on adv_hmom (1.72×).
- **Exp E (fuse_tasklets=True)**: Catastrophic for nabla2_smag (6.47×) and diffusion_vn (1.94×). **Never use alone.**
- **Exp F (scan_loop_unrolling=True)**: Broadly helpful — improves 7/10 stencils. Regressions on nabla2_smag (1.51×) and rot_vertex (1.2×).
- **Exp G (reuse_transients=True)**: Not run on full benchmark — smoke test showed catastrophic SDFG corruption.

### Experiment B2 — F+D and F+E search

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

**F+D is the best overall**: 8/10 stencils improved, regressions are small. F+E is unusable: adv_hmom 28×, interp_cell 9×, cells2verts 9× slower.

---

## DaCe GPU Stencil-by-Stencil Bug History

### Fix 1 — Stencil 01 (E2C2V): FuseAsFieldOp ratio-guard (FIXED, 2026-05-14)

**Symptom**: `InvalidSDFGNodeError: Dangling in-connector __tlet_arg0 (at tlet_42_minus)` for `calculate_nabla2_and_smag_coefficients_for_vn`.

**Root cause**: `FuseAsFieldOp.apply` reduces 249→3 as_fieldop (83×) in one step for this stencil. The per-kolor split creates 9 SetAts (3 outputs × 3 kolors) with structurally identical lambda bodies → they share the same `as_fieldop` node. FuseAsFieldOp inlines 249 inner nodes into this single shared as_fieldop, creating a fused lambda with broken data-flow (one `as_fieldop` referenced from 3 kolor domains → dangling `tlet_42_minus` connector).

**Fix** (`gt4py/src/gt4py/next/iterator/transforms/pass_manager.py`): reject fusion when the reduction ratio exceeds 20× in one step:
```python
_ratio = int(os.environ.get("GT4PY_FUSE_RATIO_THRESHOLD", "20"))
if _n_before > 0 and _n_after > 0 and (_n_before // _n_after) > _ratio:
    inlined = _pre_fuse  # restore pre-fusion IR
```
Also added `CommonSubexpressionElimination + MergeLet` before `FuseAsFieldOp` each iteration.

**Note**: With the ratio-guard, stencil 01 retains ~249 as_fieldop nodes. At 512×512 this causes IR explosion. Long-term fix: de-share SetAt expressions before fusion (deepcopy each SetAt's `as_fieldop` after kolor-split).

---

### Fix 2 — Stencil 09 (V2E): Test `horizontal_start` fix (FIXED, 2026-05-14)

**Symptom**: Wrong `rot_vec` values — FuseAsFieldOp over-fused (earlier), then wrong reference mismatch when `horizontal_start` was added.

**Fix**: Changed test `horizontal_start` from 0 to `grid.start_index(vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))` AND updated the reference function to also respect `horizontal_start`, so vertices in the lateral boundary halo are excluded from both computation and reference comparison.

File: `model/atmosphere/dycore/tests/dycore/stencil_tests/test_mo_math_divrot_rot_vertex_ri_dsl.py`

---

### Stencil 03 (C2E2CO): 26×26 vs 512×512 behavior

**Status**: ✅ PASSES on `dace_gpu` at 512×512 (confirmed `output/big_update_dace_gpu/`, 2026-05-18).

**Earlier 26×26 failure (2026-05-15)**: `ddt_w_adv` had ~1% wrong elements on `dace_gpu` 26×26 only. The IR domain was `IDimₕ:[1,26), JDimₕ:[0,26)` — asymmetric. Cells at JDim=0 (kolor=0) had C2E2CO neighbors at JDim-1=-1 → GPU OOB. On CPU the OOB reads landed in valid adjacent pages → silently passed.

**Why 512×512 passes**: The 512×512 parallelogram grid generates different boundary cell orderings — the structured positions assigned by `build_cell_ijk_maps` happen to produce a symmetric domain at this size.

**If 26×26 still fails**: investigate `build_cell_ijk_maps` in `gt4py/src/gt4py/next/modules/translator.py` (lines ~737–764) — the `i_min = min(i_coords)` / `j_min = min(j_coords)` formula may incorrectly assign some small-grid cells to JDim=0 positions that should be interior.

---

### Bug 06 — Stencil 04: `CUDA_ERROR_ILLEGAL_ADDRESS` (big grid) from empty/inverted ranges in `domain_union` (FIXED, 2026-05-28)

#### Symptom

Stencil 04 (`apply_divergence_damping_and_update_vn`) passed the small grid (26×26, sanitize) but failed with `CUDA_ERROR_ILLEGAL_ADDRESS` on the big grid (512×512, K=50) even after Bug 05's fix. Sanitize on the small grid passed (no OOB errors), ruling out obvious wrong-shift bugs.

#### Root cause

`domain_union` in `domain_utils.py` did not handle "empty" ranges (start ≥ stop) when computing the per-dimension union. These arise naturally when `_infer_concat_where` intersects a domain with a condition complement that doesn't overlap:

1. `canonicalize_domain_argument` (called at `pass_manager.py:511`, **before** `infer_program`) converts every finite concat_where condition like `Kolor:[0,1)` to canonical semi-infinite form:
   ```
   concat_where(Kolor:[0,1), expr, rest)
   →  concat_where(Kolor:[-inf,0), rest, concat_where(Kolor:[1,+inf), rest, expr))
   ```
   The `rest` expression appears as the TRUE branch of **both** `[-inf,0)` and `[1,+inf)`.

2. `_infer_concat_where` processes `concat_where(Kolor:[-inf,0), rest, ...)` with outer domain `Kolor:[0,3)`:
   - TRUE branch: `Kolor:[-inf,0) ∩ Kolor:[0,3)` = `Kolor:[max(-inf,0), min(0,3))` = **`Kolor:[0,0)`** — zero-width empty range.
   - Similarly, `Kolor:[2,3) ∩ Kolor:[-inf,1)` = `Kolor:[2,1)` — **inverted range** (start > stop), also semantically empty.

3. **Old `_range_union` / `domain_union`** included these degenerate ranges in the union:
   ```
   _range_union([0,0), [1,3)) = [min(0,1), max(0,3)) = [0,3)  ← WRONG, should be [1,3)
   _range_union([2,1), [1,3)) = [min(2,1), max(1,3)) = [1,3)  ← start=1 ok, but [2,1) shouldn't contribute
   ```

4. This caused the `__cwcda_field_b` / `_cs_0` let-bound stencil to receive a union domain of `Kolor:[0,3)` instead of the correct `Kolor:[1,3)` (for the kolor-1+2 context), and ultimately to get `Kolor:[-2,3)` through cascading widening from E2C shift back-propagation. DaCe allocated GPU temporaries spanning `Kolor:[-2,3)` (5 kolor slices) while the actual `dwdz` field had only 2 cell kolors → **GPU OOB** on big grid (small grid had enough memory slack to absorb it silently).

#### Fix

**File**: `../gt4py/src/gt4py/next/iterator/ir_utils/domain_utils.py`

- Added `_range_is_empty(range_) -> bool`: returns True when `int(start.value) >= int(stop.value)` after constant folding. Handles both `OffsetLiteral` (int `.value`) and `Literal` (str `.value`) node types, which can arise from the same arithmetic constant-folded differently. `InfinityLiteral` nodes lack a numeric `.value` and are safely treated as non-empty.

- Rewrote `domain_union` (was `functools.partial(_reduce_domains, ...)`) as a proper function that filters out empty/inverted per-dimension ranges before computing the union:
  ```python
  def domain_union(*domains: SymbolicDomain) -> SymbolicDomain:
      ...
      for dim in dims:
          all_ranges = [domain.ranges[dim] for domain in promoted_domains]
          non_empty = [r for r in all_ranges if not _range_is_empty(r)]
          if non_empty:
              new_domain_ranges[dim] = _range_union(*non_empty)
          else:
              new_domain_ranges[dim] = all_ranges[0]  # all empty, keep first
  ```

Two iteration of the fix were needed:
1. **First**: checked only `start == stop` (zero-width empty ranges) — removed `Kolor:[0,0)` contributions. IR still showed inverted ranges `Kolor:[1,0)` and `Kolor:[2,0)` from disjoint intersections.
2. **Second** (extended to `start >= stop`): also filters inverted ranges like `Kolor:[2,1)` from `Kolor:[2,3) ∩ Kolor:[-inf,1)`. This cleaned up all remaining degenerate domains in the IR.

#### Why small grid passed silently

The small grid (26×26, K=5) has enough GPU memory slack after each allocated buffer that out-of-bounds reads at negative kolor indices landed in valid but uninitialized memory, producing wrong values that happened to be tolerated (close to zero or overwritten). The big grid (512×512, K=50) has tightly-packed large allocations, so the OOB reads crossed allocation boundaries → kernel abort.

#### Diagnostic unit test

`/scratch/mch/rgraf/icon4py/test_kolor_domain_inference.py` — three tests:
1. `test_domain_union_skips_empty_ranges`: directly tests `domain_union([0,0), [1,3)) = [1,3)` and `domain_union([2,1), [1,3)) = [1,3)`.
2. `test_canonical_three_branch_gives_correct_domains`: end-to-end `infer_expr` on canonical semi-infinite concat_where; `field_k12` domain = `[1,3)` not `[0,3)`.
3. `test_cs0_canonical_cond_empty_range_union`: CSE let-binding structure; `_cs_0` let arg kolor range = `[1,3)` not `[0,3)`.

#### Note on dead code in `_infer_concat_where`

Fix D (the `and_()` case in `_infer_concat_where`, added for Bug 05) is **dead code** in the production pipeline: `canonicalize_domain_argument` converts all `and_()` conditions to semi-infinite before `infer_program` runs. Fix E (pre-fusion `infer_program`) also does not help because it uses the same broken `domain_union`. Both remain in the code and are harmless.

#### Validation

| Grid | Test | Result |
|---|---|---|
| 26×26 | sanitize (`compute-sanitizer --tool memcheck`) | ✅ 0 errors |
| 26×26 | small grid | ✅ 1 passed |
| 512×512 | big grid | ✅ 1 passed |

---

### Bug 05 — Stencil 04: `_infer_concat_where` crashes on `and_()` + numerical bug from `_EDGE_TO_EDGE_SPLIT_ALLOWLIST` (FIXED, 2026-05-27)

#### Problem 1: `AssertionError` in `_infer_concat_where` when `cond = and_(...)`

**Symptom**: After adding the second `infer_program` pass (with `keep_existing_domains=True`) to `apply_fieldview_transforms`, stencil 04 crashed with:
```
AssertionError: assert cpm.is_call_to(node, ("unstructured_domain", "cartesian_domain"))
```
inside `_infer_concat_where` at the line `symbolic_cond = domain_utils.SymbolicDomain.from_expr(cond)`.

**Root cause**: `_build_edge_validity_masked_expr` (called from the non-split `NeighborReductionUnroller` path) generates the `concat_where` condition as a conjunction of three sub-domains:
```
and_(
    cartesian_domain(named_range(Kolor, k, k+1)),
    cartesian_domain(named_range(IDim, ilo, ihi)),
    cartesian_domain(named_range(JDim, jlo, jhi)),
)
```
`SymbolicDomain.from_expr` only accepts a single `cartesian_domain` or `unstructured_domain`, so it asserts on `and_(...)`.

**Why `and_()` was used**: The condition masks to a single Kolor slice AND restricts to valid I/J positions simultaneously — all three constraints are genuinely present.

**Naive fix that was tried and failed**: Merging all three sub-domains into `symbolic_cond`. This produced a CUDA OOB error because the IDim and JDim upper bounds were included in the complement computation:
```
cond_complement = {Kolor:[k+1,+∞), IDim:[ihi,+∞), JDim:[jhi,+∞)}
```
Intersecting with the outer domain `IDim:[ilo,ihi)` gives `IDim:[ihi,+∞) ∩ IDim:[ilo,ihi) = ∅` → the FALSE branch becomes empty after intersection → `prune_empty_concat_where` removes all valid entries → GPU reads from uninitialized memory.

**Correct fix** (`infer_domain.py`, `_infer_concat_where`): Extract ONLY the Kolor sub-domain from `and_()` to use as `symbolic_cond`. IDim/JDim sub-conditions are spatial restrictions that are handled elsewhere and must NOT be included in the complement computation.
- `symbolic_cond = Kolor:[k,k+1)` only
- TRUE branch: `intersection(outer, Kolor:[k,k+1))` — correct per-kolor restriction ✓
- FALSE complement: `Kolor:[k+1,+∞)` → intersected with outer `Kolor:[0,3)` → `Kolor:[k+1,3)` ✓
- IDim/JDim ranges are unchanged in both branches ✓

```python
if cpm.is_call_to(cond, "and_"):
    kolor_sub = None
    for sub_cond in cond.args:
        if not cpm.is_call_to(sub_cond, "cartesian_domain"):
            continue
        for rng in sub_cond.args:
            if (cpm.is_call_to(rng, "named_range") and len(rng.args) == 3
                    and _get_axis_name(rng.args[0]) == "Kolor"):
                kolor_sub = sub_cond
                break
        if kolor_sub is not None:
            break
    symbolic_cond = (
        domain_utils.SymbolicDomain.from_expr(kolor_sub)
        if kolor_sub is not None
        else domain_utils.SymbolicDomain.from_expr(cond.args[0])
    )
else:
    symbolic_cond = domain_utils.SymbolicDomain.from_expr(cond)
```

#### Problem 2: Numerical mismatches (2808/10400 wrong) from `_EDGE_TO_EDGE_SPLIT_ALLOWLIST`

**Symptom**: After the initial `_infer_concat_where` crash was worked around by adding stencil 04 to a `_EDGE_TO_EDGE_SPLIT_ALLOWLIST` (forcing per-kolor split), all elements in all 3 kolors were numerically wrong. The mismatch count (2808) was not a multiple of 5 (the K-level count), indicating the error pattern was horizontal, not vertical.

**Root cause**: Per-kolor split forces 3 separate SetAts (Kolor:[0,1), [1,2), [2,3)). In the Kolor-k SetAt, the intermediate field `horizontal_gradient_of_total_divergenceᐞ0` (an E2C reduction) was computed using kolor-k-specific E2C shifts (e.g. kolor-0: `IDim-1, Kolor+1`). However, `infer_domain.infer_program` back-propagates from the outer domain `Kolor:[0,3)` (needed because E2C2EO reads all 3 kolors) and widens the E2C `as_fieldop` domain to `Kolor:[0,3)`. The kolor-0 E2C shift (`IDim-1, Kolor+1`) is then applied at Kolor=1 and Kolor=2 positions — computing the wrong cell neighbors — producing wrong intermediate values used by the E2C2EO reduction. All 3 kolor results were corrupted.

**Fix** (`structured_backend_passes.py`): Remove `_EDGE_TO_EDGE_SPLIT_ALLOWLIST` entirely and simplify `_allow_edge_to_edge_split` back to the original one-liner:
```python
@staticmethod
def _allow_edge_to_edge_split(expr: ir.Expr) -> bool:
    return not _expr_uses_edge_to_edge_connectivity(expr)
```

**Why the non-split path works correctly**: `_build_field_concat_where_from_branches` with `current_kolor=None` and `_narrow_kolor` generates 3 per-kolor `concat_where` branches, each visited with `current_kolor=k`. Each branch uses the correct kolor-k E2C shifts and has an explicit narrow domain `Kolor:[k,k+1)`. With Problem 1's fix in place, `_infer_concat_where` correctly restricts back-propagation to each narrow Kolor slice, so the E2C `as_fieldop` inside each branch stays at `Kolor:[k,k+1)`. `prune_empty_concat_where` then eliminates the dead branches. The full-domain `Kolor:[0,3)` that appears on the E2C2EO input field is correct and expected (it needs all 3 kolors for its 6 neighbors).

#### Files changed
- `../gt4py/src/gt4py/next/iterator/transforms/infer_domain.py` — `_infer_concat_where`: Kolor-only extraction from `and_()` (replaces `from_expr(cond)` direct call)
- `../gt4py/src/gt4py/next/iterator/transforms/structured_backend_passes.py` — Remove `_EDGE_TO_EDGE_SPLIT_ALLOWLIST`; simplify `_allow_edge_to_edge_split`; update call site

#### Note on `Kolor:[0,3)` in the IR after the fix

After the fix, `Kolor:[0,3)` still appears on the E2C2EO input `as_fieldop` nodes — this is correct and not a bug. Each E2C2EO access from a kolor-k edge has 6 neighbors drawn from all 3 kolors, so the input field must be available at all kolors. Only the E2C intermediate (the local per-edge computation) must be narrow — and with the non-split path it now is, via the `concat_where` branching.

---

### Bug 04 — Stencil 04 (E2C + E2C2EO): `apply_divergence_damping_and_update_vn` CUDA OOB

#### Initial investigation (2026-05-14 to 2026-05-15)

**Symptom**: `CUDA_ERROR_ILLEGAL_ADDRESS (700)` on `dace_gpu` with the 512×512 grid. Passes on `dace_cpu` and `gtfn_cpu`.

**Root cause (confirmed)**: `FuseAsFieldOp` in the fusion loop inside `apply_fieldview_transforms` inlines a **kolor-restricted** inner `as_fieldop` (e.g. `as_fieldop(shift(Kolor,-2), Kolor:[2,3))(dwdz)`) into an outer `as_fieldop` with domain `Kolor:[0,3)`. The fused node keeps the OUTER domain. DaCe then evaluates `shift(Kolor,-2)` for all kolors 0–2, reading `dwdz[i,j,-2,k]` for kolor=0 → GPU OOB.

Multiple false leads ruled out (2026-05-15):
1. `FuseAsFieldOp` cross-kolor fusion guard (added, correct, but didn't address root cause)
2. Negative kolor IR ranges (not the root cause)
3. Padding cell fields to 3 Kolors (irrelevant)

**Concrete root cause (CUDA kernel level)**: The failing kernel reads `dwdz_at_cells_on_model_levels` (Kolor=2) at offsets `__j2 + {-2, -1, 0, +1}`. The kernel iterates `__j2 ∈ {0}` only, so absolute Kolor indices `{-2, -1, 0, 1}` → reads at Kolor `-2` and `-1` are OOB.

#### IR-level root cause (2026-05-16)

The widening enters between `=== AFTER INFERRING DOMAIN OPS ===` and `=== AFTER PRUNING EMPTY CONCAT WHERE ===`. The second `infer_program` call at `pass_manager.py:544` re-infers source domain from output `Kolor:[0,3)` plus shift, giving `[2,5)` for `dkolor=+2`, `[1,4)` for `dkolor=+1`, then unions to `[0,5)` on the dwdz access (a cell field with only Kolor 0,1).

The let-lifting comes from `concat_where.canonicalize_domain_argument` which transforms `concat_where(and_(d1, d2), a, b)` into a let-bound nested form. After let-lifting, the second `infer_program` re-infers source domain from the outer `Kolor:[0,3)` and the shift offset — the per-branch Kolor restriction is gone.

#### Fix — `keep_existing_domains=True` + trailing-else concat_where wrap

Fixing required two independent changes:

**1. `keep_existing_domains=True` on the second `infer_program`** (`pass_manager.py:544`): When an `as_fieldop` has an explicit `(stencil, domain)` form, use the existing domain as the target instead of propagating the outer target_domain in. The docstring at `infer_domain.py:536-541` describes exactly our scenario: "This is useful in cases where after a transformation some nodes are missing domain information that needs to be repopulated, but we can't reinfer everything because some domain access information has been lost."

**2. Trailing-else branch wrap in `_build_field_concat_where_from_branches`** (`structured_backend_passes.py`): When handling the trailing-else branch (`cond=None`), wrap the bare `as_fieldop` in an inner `concat_where`:

```python
if len(branches) == 1:
    if source_kolor is not None and source_kolor > 0 and isinstance(branch_domain, ir.FunCall):
        narrow_trailing_domain = _narrow_domain_kolor(
            branch_domain, (source_kolor, source_kolor + 1)
        )
        narrow_trailing_expr = _make_lifted_deref_shift(arg, shift_spec, narrow_trailing_domain)
        identity_domain = _narrow_domain_kolor(branch_domain, (0, source_kolor))
        identity_expr = _make_lifted_deref_shift(arg, (), identity_domain)
        narrow_cond = im.call("cartesian_domain")(
            im.named_range(ir.AxisLiteral(value="Kolor", ...), ir.OffsetLiteral(source_kolor), ir.OffsetLiteral(source_kolor + 1))
        )
        return im.concat_where(narrow_cond, narrow_trailing_expr, identity_expr)
```

This prevents the trailing branch from using a wide `Kolor:[0,3)` domain that would cause OOB reads after `canonicalize_domain_argument` let-lifting.

**Note on intervening CPU-only narrowing attempt (2026-05-17 to 2026-05-18)**: An intermediate fix that narrowed the trailing-else to `Kolor:[2,3)` without the concat_where wrap passed CPU 512×512 numerically but SIGSEGVed on a later run. The trailing-else narrowing without the wrap caused `+inf location mismatch` because the let-lifted 1-slot temp got consumed at other Kolor positions. The concat_where wrap is the correct fix.

#### Validation matrix (post-fix, 2026-05-18)

| Grid | Backend | Result | Steady-state exec |
|---|---|---|---|
| 26×26 | dace_cpu | ✅ PASSED | median 4.94 ms |
| 26×26 | dace_gpu | ✅ PASSED | median 10.87 ms |
| 512×512 | dace_cpu | ✅ PASSED | median 11.28 s |
| 512×512 | dace_gpu | ✅ PASSED | 68 ms exec, 3.7 s round-trip |

---

## Bug: E2C2V Kolor OOB in Vertex SetAt with Nested V2E+E2C2V (`rbf_nabla4`) — 2026-06-05

### Stencil
`rbf_nabla4` — new diffusion stencil combining `_calculate_nabla4` (uses E2C2V to interpolate vertex fields to edge normals) and `_mo_intp_rbf_rbf_vec_interpol_vertex` (uses V2E to scatter edge field back to vertex). Output is `tuple[VertexKField, VertexKField]`.

### Symptoms
- Unstructured backend (gtfn_cpu): 3/3 test variants pass ✅
- Structured backend (gtfn_cpu / dace_cpu before fix): output is all zeros or wrong values, ~37% mismatch

### Root cause
`NeighborReductionUnroller.visit_SetAt` calls `_kolor_from_domain(new_domain)` to extract the SetAt's Kolor range. For a **vertex SetAt** with domain `Kolor:[0,1)` this returns `0`. For a **split edge kolor-0 SetAt** (generated by `SetAtRemapper`) the domain also has `Kolor:[0,1)` and also returns `0`. The two cases are indistinguishable from the domain alone.

With `current_kolor=0` set from the vertex SetAt, the bottom-up visitor processes E2C2V first (inside the z_nabla4_e2 sub-expression) and calls `_build_field_concat_where_from_branches` with `current_kolor=0`. Because `is_ene=True` for E2C2V (Edge→Vertex), the condition `current_kolor if (is_ene or ...)` passes `0` through, selecting the E2C2V kolor-0 branch (`Kolor: +0` shift). The u_vert access is baked in at `Kolor=0`.

When the outer V2E reduce is then unrolled, each slot applies an additional `Kolor: +dk_s` shift (e.g. slot 1: `Kolor:+1`, slot 2: `Kolor:+2`, slot 4: `Kolor:+1`, slot 5: `Kolor:+2`). After this shift, u_vert lands at `Kolor = dk_s`. For `dk_s ∈ {1,2}` this is out-of-bounds — vertex fields only have `Kolor:[0,1)`.

On gtfn_cpu: numpy OOB returns 0 → all-zero output.  
On dace_cpu: SDFG reads uninitialised memory → garbage values.

### Unrelated suspicion: padding commit
The padding commit `b5ec5e12` added `pack_edge_field_compact` and `pack_vertex_field_padded` (applied only when `shift_i > 0`). For this test, `horizontal_start=150` (vertex index) produces `shift_i=0` from the edge bounds, so neither compact nor padded packing is activated. The padding commit has **no effect** on this test.

### Fix
**`NeighborReductionUnroller` in `structured_backend_passes.py`**:

1. Added `_expr_has_connectivity(expr, conn_name)` static helper — walks the expression tree and returns True if `conn_name` appears as an `OffsetLiteral` value.

2. In `visit_SetAt`: when `current_kolor==0`, check if the SetAt expression contains V2E connectivity. `V2E` (vertex→edge) only appears in vertex-output stencils, uniquely identifying a vertex SetAt. Set `setat_is_vertex=True` and propagate via kwargs.

3. In `visit_FunCall`, `_eval_list_field_at_idx`, and `_build_generic_unrolled_reduce_expr`: when `is_ene=True` (E2C2V, E2C, E2V, etc.) **and** `setat_is_vertex=True`, pass `current_kolor=None` to `_build_field_concat_where_from_branches` instead of the vertex kolor 0.

Passing `current_kolor=None` causes a full 3-way concat_where to be generated for E2C2V. After V2E slot s applies `Kolor:+dk_s`, the concat_where condition `Kolor:[dk_s, dk_s+1)` fires and selects the E2C2V branch for edge kolor `dk_s`, which has a `Kolor: -dk_s` offset. The final u_vert access lands at `Kolor = dk_s - dk_s = 0` ✓ (vertex kolor).

For the edge-kolor-0 SetAt case (also `current_kolor=0` but `setat_is_vertex=False`): unchanged — E2C2V still uses `current_kolor=0` to select the kolor-0 branch directly, which is correct.

### Validation
| Backend | Variants | Result |
|---|---|---|
| gtfn_cpu | none, compile_time_domain, compile_time_vertical | ✅ all pass |
| dace_cpu | none, compile_time_domain, compile_time_vertical | ✅ all pass |

Existing vertex stencils `mo_math_divrot_rot_vertex_ri_dsl` (V2E) and `mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl` (V2C) continue to pass on both backends — no regression.

---

## Full diffusion + dycore stencil-sweep audit (2026-06-25) — ⚠️ candidate fixes, validation PENDING

Triggered by the question "do *all* diffusion and dycore stencil tests pass?" — they don't. Audited
the complete sweeps (`output/dycore_baked_str/` 82 tests, `output/diffusion_baked_str/` 23 tests).

**Important — these are pre-existing structured-backend bugs, NOT caused by the lb0 / warp-align
perf-default changes.** Proven by comparing the lb0 run (`output/dycore_lb0_str/`) against the baked
baseline (old `lb 256,8` + warp-align off): the identical set fails in both. The perf defaults are
orthogonal to correctness.

**Real failures: 16** (dycore 12 + diffusion 4). The 5 dycore "no-result" files are non-failures
(33/35/36/37 are `skipped`; 77 is exit-code-5 no-collection). Grouped into shared root causes below.
Each candidate fix is **unverified** until the numerical `assert_dallclose` passes — a compile that
gets past the crash can still be *masking* malformed IR that yields wrong numbers (see the dyc04
lesson, and Fix 8 above where a malformed-`deref` had to be fixed in the IR, not tolerated).
Validation in flight: job **959071** (`output/fixval_str/`, baked-baseline config to isolate the IR
fix from perf knobs).

### Group A — `'Sentinel' object has no attribute 'deref'` (trace_shifts) → dyc 40, 50
**Symptom**: during type inference, `trace_stencil` → `_deref(Sentinel.VALUE)` → AttributeError.
**Root cause**: the structured IR contains `deref(<value>)` (e.g. a literal-0 fallback in a
degenerate per-kolor branch); trace_shifts' `_deref` assumed a Tracer (iterator).
**Candidate fix** (`trace_shifts.py`): `_deref` returns `Sentinel.VALUE` for a non-Tracer arg —
deref of a non-iterator records no shift access pattern; mirrors `_can_deref`, which already returns
`Sentinel.VALUE` unconditionally. **Risk**: if the `deref(value)` IR is genuinely wrong (not benign),
this masks it → would show as a numerical mismatch. **Status: PENDING (dyc 40, 50).**

### Group B — iterator-type compatibility too narrow (inference) → dyc 81, 82, dif 03
**Symptom**: `TypeError: Incompatible inferred type for node z_q` (dyc 81/82, tridiag scan temp) /
`wᐞ0` (dif 03). `position_dims` are IDENTICAL; only `defined_dims` differ by one dim — z_q: existing
has trailing `K`, inferred doesn't; wᐞ0: inferred has a stale leading `Cell`, existing doesn't.
**Root cause**: `_is_structured_remap_compatibility_case` (extended before in Fix 3 / Fix 5) only
accepted `position_dims` differing by one K; it `return False`s when positions are equal, so these
benign defined-dims-only artifacts raised.
**Candidate fix** (`inference.py`): new **Case C** — when `position_dims` are identical (so the
iteration structure is provably the same), accept a `defined_dims` mismatch of exactly one trailing
`K` (C1, either direction) or one leading `Edge/Cell/Vertex` (C2, either direction); keep `existing`.
**Risk**: forces `existing` to win; if `existing` is the wrong type this propagates → wrong numerics.
**Status: PENDING (dyc 81, 82, dif 03).**

### Group F — `Undefined symbol vertical_start` (trace_shifts) → dif 22
**Symptom**: trace_shifts `visit_SymRef` raises on `vertical_start`, a free scalar left in a vertical
`concat_where` condition (`less(K, vertical_start)`); `trace_stencil` ctx only has builtins + iterator
args.
**Candidate fix** (`trace_shifts.py`): recognize the four canonical GT4Py domain-bound scalars
`{horizontal,vertical}_{start,end}` (never iterators) as `Sentinel.VALUE`; keep the general
undefined-symbol guard for real typos. **Probe**: if dif 22 then fails *numerically*, the vertical
condition is misplaced and needs a structured-pass fix, not tolerance. **Status: PENDING (dif 22).**

### Group C — `InvalidSDFGEdgeError: Memlet other_subset out-of-bounds`, Kolor index 3 → dyc 20, 21, 25
**Symptom**: `sdfg.validate()` rejects a memlet whose Kolor subset is `[3 - Kolor_range_0 : 3 -
Kolor_range_0]` — index **3**, one past the last valid edge kolor (size 3 → 0,1,2). Empty-volume, but
DaCe flags the OOB offset. All three are E2C2EO/E2C2E edge-to-edge stencils.
**Root cause (suspected)**: the **trailing-else `concat_where` fallback** in
`_build_field_concat_where_from_branches` emits a degenerate Kolor-3 subset. This is the danger zone
documented above (Bug 04 trailing-else: a narrowing there once "passed CPU numerically but SIGSEGVed").
**Status: NOT fixed — deferred to careful investigation with the saved `_dacegraphs/invalid.sdfgz`.**

### Group D — numerical mismatch `rtol=3e-6` → dyc 12, 15, 45
**Symptom**: compiles + runs, fails `assert_dallclose` (1 of N parametrizations). Needs per-kolor
output comparison (`scripts/compare_arrays.py`) to localize. **Status: NOT yet investigated.**

### Test-file issues (not backend)
- **dyc 04** `add_extra_diffusion_for_normal_wind_tendency_approaching_cfl`: numpy reference had a
  stray `exit(1)` + header "DOES NOT WORK YET; NEITHER STRUCTURED NOR UNSTRUCTURED." Removing `exit(1)`
  exposed a real **6.65% mismatch (max rel ~2e9)** → broken reference, confirmed in both backends.
  **Resolved → `@pytest.mark.skip`** with the evidence (consistent with the 4 already-skipped tests).
  Definitive confirmation would be an unstructured run (expected to also mismatch).
- **dyc 39** `compute_horizontal_velocity_quantities_and_fluxes`: the `reference()` calls
  `compute_avg_vn_and_graddiv_vn_and_vt_numpy(...)` (a sub-step) **without** `horizontal_start`, which
  became a required positional arg. **Fix**: pass `horizontal_start` (already in `reference()` scope).
  **Status: PENDING (in validation sweep).**

### Non-bug — flaky infrastructure
- **dif 13** `calculate_nabla2_for_z`: `CompilationError: ... can't create ...cuda.cu.o: Stale file
  handle` — a transient Lustre error from the parallel-compile sweep, **not a code bug**. A clean
  re-run is expected to pass; to be confirmed.

## Re-check 2026-07-31 — thesis benchmark table stencils (structured, DaCe GPU, 512-grid/K=120, FD)

Re-ran the structured backend for every dycore/diffusion stencil still missing from the thesis's
main speedup tables at the time of measurement, to recover as many as possible for the tables and
document the rest. Two were recovered; the remaining five reproduce (or supersede) existing
Group A/C/D/F entries above.

### Recovered — isolating the `compile_time_domain` parametrized variant (now in the main tables)
- **dyc 12** `apply_divergence_damping_and_update_vn`: confirms **Group D**'s numerical mismatch is
  specifically the `is_iau_active[False]` sub-case (unstructured 2.057 ms — the "faster one" — vs.
  structured producing wrong results). `is_iau_active[True]` passes cleanly (unstructured 3.876 ms,
  structured 9.764 ms, 30 runs). **New finding not previously in Group D**: running the full test
  file (all `STATIC_PARAMS`, no `-k` filter) additionally hangs indefinitely on the
  `COMPILE_TIME_VERTICAL` variant (reached "Reference outputs computed", then no further output for
  2500 s) — unrelated to the Group D mismatch, avoided entirely by filtering to `compile_time_domain`.
- **dyc 15** `compute_advection_in_horizontal_momentum_equation` (internal class name
  `FusedVelocityAdvectionStencilsHMomentum`): same pattern as dyc 12 — `apply_extra_diffusion_on_vn
  [False]` is the faster sub-case that currently produces wrong results (suspected cause: reuses a
  cached compiled CUDA binary keyed insufficiently to distinguish it from the `[True]` sub-case);
  `apply_extra_diffusion_on_vn[True]` passes cleanly (unstructured 4.698 ms, structured 7.519 ms, 30
  runs). This gives a likely root-cause hypothesis for Group D generally: a compiled-artifact cache
  key that doesn't fully capture which boolean-parametrized branch was compiled.

### Confirmed unchanged — Group C (`InvalidSDFGEdgeError: Memlet other_subset out-of-bounds`)
- **dyc 20** `compute_averaged_vn_and_fluxes_and_prepare_tracer_advection` and **dyc 21**
  `compute_avg_vn_and_graddiv_vn_and_vt` both reproduce the exact same error text as documented
  (`InvalidSDFGEdgeError: Memlet other_subset out-of-bounds`, invalid SDFG dumped to
  `_dacegraphs/invalid.sdfgz`), on `compile_time_domain`. dyc 21 has no parametrized variants, so
  there is no alternative sub-case to recover it with. Still **NOT fixed**.
- **dyc 25** `compute_diagnostics_from_normal_wind`: not confirmed to reproduce the same error —
  it never got far enough. It timed out with **zero output** after 1000 s (not even reaching the
  first `STATIC_PARAMS` variant), then after retrying with `-k compile_time_domain` and 3000 s it
  still timed out with zero output. Given dyc 20/21 (same Group C) compile in ~12 min each, this is
  either an unusually slow compile or a genuine hang; not distinguished within the time available.

### Symptom changed since originally documented (Group A / Group F) — worth re-triaging
Both of these no longer show their originally-recorded symptom; a different failure now occurs
first, suggesting the original bug was fixed upstream but a new blocker sits behind it.
- **dyc 50** `compute_theta_rho_face_values_and_pressure_gradient_and_update_vn`: Group A recorded
  `'Sentinel' object has no attribute 'deref'`. Today's re-run instead fails with a **Group
  B-style** type-inference `TypeError` on the very first collected variant
  (`is_iau_active[True]-none`): `Incompatible inferred type for node
  horizontal_pressure_gradient␞0_: existing=Field[[IDim, JDim, Kolor, K], float64],
  inferred=DeferredType(constraint=None)`. Same shape of error as Group B's `z_q`/`wᐞ0` cases
  (SSA-renamed node, `existing` vs. `inferred` type mismatch) — likely the same underlying
  compatibility gap in `_is_structured_remap_compatibility_case`, just untriggered for this node
  before. `compile_time_domain` specifically was not isolated (masked by `--maxfail=1` on the first
  variant); worth retrying in isolation, but the error looks structural rather than
  parametrization-dependent.
- **dif 22** `truly_horizontal_diffusion_nabla_of_theta_over_steep_points`: Group F recorded
  `Undefined symbol vertical_start`. Today's re-run instead fails with a `cupy.core.core.ValueError`
  on buffer allocation, preceded by `RuntimeWarning: overflow encountered in scalar multiply` while
  computing the padded allocation size. Consistent with this test's `@pytest.mark.uses_as_offset`
  marker: it reads a data-dependent vertical offset field (`zd_vertoffset`), which likely isn't
  representable in the structured backend's compile-time-constant-shift model and produces a
  bogus/oversized domain size instead of a compile-time error.

### New failure mode, not previously documented
- **dyc 45** `compute_perturbed_quantities_and_interpolation`: Group D listed this as a numerical
  mismatch, but today's re-run (all `STATIC_PARAMS`, no `-k` filter) fails on the `[none]` variant
  before `compile_time_domain` is reached, with two chained `KeyError`s instead of a numerical
  assertion: `KeyError: ((), -2979764658528722721, None)` in
  `otf/compiled_program.py:419` (a compiled-program cache lookup), followed by `KeyError: SDFGState
  (stmt_2_false_branch)` in `dace/sdfg/graph.py:676`. Neither the originally-documented mismatch nor
  the cache-key hypothesis from dyc 12/15 above were confirmed for this stencil specifically;
  `compile_time_domain` was not isolated. Worth a targeted `-k compile_time_domain` re-run.
