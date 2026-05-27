
#### Side-fixes landed at the same time (storage-migration fallout, 2026-05-18)

1. `.venv/bin/python` symlink dangling after MCH admins moved `$HOME` from mchstor1 to mchstor2. Fix: copy `cpython-3.10.17-linux-x86_64-gnu/` tree to `/scratch/mch/rgraf/cpython-3.10.17/`, relink `.venv/bin/python`, update `home=` in `.venv/pyvenv.cfg`. Venv is now fully `/scratch/`-resident.
2. CuPy kernel cache defaulted to `$HOME/.cupy/kernel_cache`, unwritable on compute nodes post-migration. Fix in `scripts/run_stencil_commands_slurm.sh`: export `CUPY_CACHE_DIR=${WORKDIR}/.cupy_cache`.

---

### Update 2026-05-19 — Stencils 01 and 10: two new fixes after 512×512 sweep

After the Bug 04 concat_where-wrap fix, a full 10-stencil `dace_gpu` sweep (`output/big_update_dace_gpu/`) showed stencils **01, 05, 06, 10** failing.

#### Fix A — Stencil 01: `infer_domain.py` crashes on tuple-output `as_fieldop`

**File**: `../gt4py/src/gt4py/next/iterator/transforms/infer_domain.py`, line 239.

**Symptom**: `AssertionError` inside `_infer_as_fieldop` when `keep_existing_domains=True` and the `as_fieldop` has a `make_tuple(domain_a, domain_b)` as `fun.args[1]` (tuple-output fieldop from fused E2C2EO sums in `calculate_nabla2_and_smag_coefficients_for_vn`).

**Root cause**: `SymbolicDomain.from_expr(applied_fieldop.fun.args[1])` only handles single-domain expressions. `make_tuple(...)` hit the assertion.

**Fix**: Replace with `_make_symbolic_domain_tuple(applied_fieldop.fun.args[1])` (already defined at line 603; handles both single-domain and `make_tuple`):
```python
if len(applied_fieldop.fun.args) == 2 and keep_existing_domains:
    target_domain = _make_symbolic_domain_tuple(applied_fieldop.fun.args[1])
```

#### Fix B — Stencil 10: literal-0 fallback for vertex-safe trailing-else wrap

**File**: `../gt4py/src/gt4py/next/iterator/transforms/structured_backend_passes.py`, `_build_field_concat_where_from_branches`, ~line 450–480.

**Symptom**: `apply_diffusion_to_vn` (E2C2V) crashed with `InlineSDFG` error on `gtir_tmp_108[0:509, 0:508, 0:0, ...]` — zero-sized Kolor dimension.

**Root cause**: Previous identity-based fallback read `arg` (a vertex field, Kolor=1) in the fallback region `Kolor:[0, source_kolor)`. For vertex fields `source_kolor=1`, this 1-slot temp was let-lifted by `canonicalize_domain_argument` and consumed at multiple Kolor positions → DaCe produced a zero-sized Kolor dim in the concatenated result.

**Fix**: Three changes in `_build_field_concat_where_from_branches`:
1. Added `cond is None and` guard — the wrap only fires for the actual trailing-else branch.
2. Fallback uses **full outer domain** (`domain`) — safe because the lambda body is constant (NEVER-arg → `arg` never read).
3. Fallback is a **literal-0 constant** `as_fieldop` (not identity deref):
```python
fallback_expr = im.as_fieldop(
    im.lambda_("__cart_trailing_unused")(im.literal("0.0", "float64")),
    domain,  # full outer domain, not narrowed
)(copy.deepcopy(arg))
```

**Confirmed incompatible**: Adding `keep_existing_domains=True` to the final `infer_program` call alongside these fixes causes stencil 4 numerical errors (2.19% mismatch) and stencils 5/6 `InvalidSDFGEdgeError`. Do not add `keep_existing_domains=True` to the final infer_program call.

#### Stencils 05 and 06 status after 2026-05-19

Both stencils now pass on `dace_cpu` locally after two additional fixes:
1. `allow_uninferred=True` added to the final `infer_program` call in `apply_fieldview_transforms` — required because the literal-0 fallback `as_fieldop(λ __cart_trailing_unused → 0.0)(arg)` has a NEVER-arg.
2. `keep_existing_domains=True` was **removed** from the final `infer_program` call (it caused issues with stencils 4/5/6). Kolor-widening is now prevented structurally by the inner concat_where wrap.

---

### Update 2026-05-20 — Stencil 06 `_edge_shape_domain` fix incomplete; stencil 10 regression

**Resume keyword: EDGE-SHAPE-FIX**

#### What was done

- Added `_idim_shift`, `_jdim_shift`, `_apply_offset` helpers to `_build_field_concat_where_from_branches`
- Changed `_edge_shape_domain` clip from `i_hi - 1` to `_apply_offset(i_hi, -1 - di)` (shift-aware)
- Removed dead code from `fuse_as_fieldop.py` (`_kolor_interval`, debug prints)
- Gated `_print_ir_block` calls on `GT4PY_PRINT_IR` env var

#### Current test result (small grid dace_gpu, `output/small_edge_shape_fix/`)

8/10 pass. Failing: stencils 06 and 10.

**Stencil 06** (E2C/E2V/E2C2EO): SAME error as before — `InvalidSDFGEdgeError: Memlet subset out-of-bounds`. The shift-aware clip fix did NOT take effect. **Real root cause confirmed as the `annex.domain` overwrite bug (see Update 2026-05-21)**. The `_edge_shape_domain` clips were already correctly removed.

**Stencil 10** (E2C2V): `InlineSDFG` crash (`RuntimeError: generator raised StopIteration`). **Root cause confirmed as the same `annex.domain` bug**.

---

### Update 2026-05-21 — Stencils 06 and 10: `annex.domain` overwrite fix (FIXED)

#### Root cause (confirmed via `GT4PY_TRACE_INFER_DOMAIN_KOLOR=1` trace)

Both stencils 06 (`compute_advection_in_horizontal_momentum`) and 10 (`apply_diffusion_to_vn`) failed with zero-sized Kolor transients (`Kolor:[0:0]` or `Kolor:[1:1]`). The trace showed:

```
[INFER_LET_KOLOR] depth=N param=__cwcda_field_b kolor=Kolor:[1,1) EMPTY
[INFER_AS_FIELDOP_KOLOR] target_kolor_before=Kolor:[1,1) has_explicit_domain=True keep_existing=True
```

**Call chain**:
1. `_build_edge_validity_masked_expr` wraps the stencil in `concat_where(AND(Kolor:[k,k+1), IDim, JDim), expr, rest)`.
2. `canonicalize_domain_argument` let-lifts the AND condition → `let __cwcda_field_b = as_fieldop(stencil, Kolor:[0,3))(field) in concat_where(OR(Kolor:(-inf,k), Kolor:[k+1,inf)), __cwcda_field_b, ...)`
3. `infer_program` processes the outer let at context `Kolor:[0,k+1)`. The concat_where body assigns `accessed_domain[__cwcda_field_b] = Kolor:[1,1)` (empty intersection).
4. `_infer_as_fieldop` is called with `target_domain = Kolor:[1,1)` on the `as_fieldop` with `has_explicit_domain=True`. `keep_existing_domains=True` correctly **replaces `target_domain` with `Kolor:[0,3)`** (the kept explicit domain).
5. **BUG**: `_infer_as_fieldop` creates a fresh `transformed_call` node with no `annex.domain`. The outer wrapper `infer_expr` (line 712) checks `not hasattr(expr.annex, "domain")` → True → **sets `annex.domain = Kolor:[1,1)`** (the caller's empty domain, overwriting the intended `Kolor:[0,3)`).
6. DaCe reads `annex.domain` to size the copy-destination transient → Kolor `[1:1]` → size 0 → `InvalidSDFGEdgeError: Dimensionality mismatch` or `InlineSDFG: generator raised StopIteration`.

#### Fix

**File**: `../gt4py/src/gt4py/next/iterator/transforms/infer_domain.py`, inside `_infer_as_fieldop`, after `transformed_call = im.as_fieldop(stencil, target_domain_expr)(*transformed_inputs)`:

```python
# When keep_existing_domains kept an explicit domain, pre-populate annex.domain
# on the new node with that kept domain. Without this, infer_expr (line ~712)
# overwrites annex.domain with the caller's target_domain (Kolor:[1,1) empty).
# DaCe reads annex.domain to size the copy-destination transient → zero-sized.
if keep_existing_domains and len(applied_fieldop.fun.args) == 2:
    if isinstance(target_domain, domain_utils.SymbolicDomain):
        transformed_call.annex.domain = target_domain
```

This pre-sets `annex.domain` to the KEPT explicit domain on the fresh node, so `infer_expr` line 712 sees it already set and skips overwriting.

#### Key implementation detail: `_build_field_concat_where_from_branches` does NOT narrow domains

The current codebase does NOT narrow branch `as_fieldop` domains to `Kolor:[k,k+1)`. All branches use the full outer domain (e.g. `Kolor:[0,3)`). The fix operates entirely in `infer_domain.py`. This is a simpler and more robust approach than domain narrowing, which previously caused additional issues (zero-sized vertex-field transients, CPU `+inf` mismatches, etc.).

#### Diagnostic tools added (2026-05-21)

- `GT4PY_TRACE_INFER_DOMAIN_KOLOR=1` — prints Kolor-specific domain info for every let param and as_fieldop in `infer_domain.py`
- `GT4PY_PRINT_IR_FILE=ir_out_06.txt` — controls the IR dump filename
- `GT4PY_PRINT_IR=1` — gates the IR dump (now env-var controlled)
- `=== FIELDVIEW IR AFTER CANONICALIZE DOMAIN ARG ===` and `=== FIELDVIEW IR AFTER INFER DOMAIN ===` sections added to IR dumps
- `commands_stencils_kolor_trace.txt` — runs stencils 06 and 10 with `GT4PY_TRACE_INFER_DOMAIN_KOLOR=1`
- `pass_pipeline_overview.md` — complete annotated list of all passes in `apply_fieldview_transforms`

#### Unit tests (test_narrow_kolor_fix.py — 7 tests as of 2026-05-21)

| Test | Status | What it checks |
|---|---|---|
| `test_build_field_concat_where_uses_full_outer_domain` | ✅ | All branches use `Kolor:[0,3)` (no narrowing in current code) |
| `test_build_field_concat_where_trailing_else_uses_full_domain` | ✅ | Trailing else also uses full outer domain |
| `test_post_unroll_pipeline_no_widening_with_fix` | ✅ | Full pipeline: no widened Kolor ranges after inference |
| `test_post_unroll_pipeline_widens_without_keep_existing` | ✅ | Without `keep_existing_domains`, widening occurs |
| `test_infer_program_handles_tuple_output_as_fieldop` | ✅ | Tuple-output `as_fieldop` (stencil 01 fix) doesn't raise |
| `test_edge_shape_domain_no_clip_on_idim` | ✅ | No IDim clip in `_edge_shape_domain` |
| `test_infer_as_fieldop_keeps_annex_domain_with_keep_existing` | ✅ | `annex.domain` on returned node is the kept explicit domain, NOT the empty caller domain |

Run:
```bash
srun --partition=debug --time=00:05:00 --uenv=icon/25.2:v3 --view=default \
  bash -c '.venv/bin/python test_narrow_kolor_fix.py'
```

#### Verification pending (2026-05-26)

Stencil sweep `output/annex_domain_fix/` (job 4349646, 26×26 small grid, dace_gpu). Expected: 5/5 pass (stencils 01, 04, 05, 06, 10). Full 10-stencil small-grid sweep then 512×512 needed to confirm no regressions.
