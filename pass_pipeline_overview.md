# `apply_fieldview_transforms` Pass Pipeline Overview

File: `../gt4py/src/gt4py/next/iterator/transforms/pass_manager.py`

This document maps every pass in the DaCe path (`apply_fieldview_transforms`) in
execution order.  The structured-backend block is gated on `USE_STRUCTURED_BACKEND=1`.

---

## Pass sequence

| # | Pass / function | Source | Purpose |
|---|---|---|---|
| 1 | `_process_symbolic_domains_option` | `pass_manager.py` | Preprocesses symbolic domain sizes; inserts max-range expressions for unstructured shifts |
| 2 | `InlineFundefs.visit` + `prune_unreferenced_fundefs` | `inline_fundefs.py` | Inlines top-level function definitions into the IR |
| 3 | `expand_tuple_args` | `concat_where/__init__.py` | Expands tuple arguments in `concat_where` expressions |
| 4 | `dead_code_elimination` | `dead_code_elimination.py` | Eliminates unreachable dead code |
| 5 | `InlineDynamicShifts.apply` | `inline_dynamic_shifts.py` | Inlines dynamic (non-static) shifts; required before domain inference |
| 6★ | `NormalizeShifts` | `normalize_shifts.py` | Canonicalises shift order (required by structured-backend pattern matching) |
| 7★ | `InlineLifts` | `inline_lifts.py` | Inlines `lift(λ it→…)` wrappers so structured passes see bare `neighbors(conn, it)` |
| 8★ | `CartesianDomainAndTypeRemapper.apply` | `cart_unroll.py` → `structured_backend_passes.py` | ① Remaps `Edge/Cell/Vertex` types → `IDim/JDim/Kolor`. ② Converts `unstructured_domain` → `cartesian_domain`. ③ Per-kolor SetAt split for Edge (if mapping enabled). ④ Wraps non-split SetAts with `_build_edge_validity_masked_expr` (AND conditions). ⑤ Rewrites threshold comparisons (`Edge ≥ N` → per-kolor bounds). |
| 9★ | `CartesianReductionUnroller.apply` | `cart_unroll.py` → `structured_backend_passes.py` | Unrolls `reduce(op)(neighbors(conn,it), …)` → explicit per-slot shifts. For non-split SetAts: calls `_visit_expr_with_kolor_branches` to peel the AND-conditioned concat_where and visit each kolor branch with `current_kolor=k`. |
| 10★ | `NormalizeShifts` | `normalize_shifts.py` | Re-canonicalises after unrolling |
| 11★ | `expand_tuple_args` | `concat_where/__init__.py` | Handles any new tuple args from unrolling |
| 12★ | `dead_code_elimination` | `dead_code_elimination.py` | DCE after unrolling |
| 13 | `InferDomainOps.apply` | `infer_domain_ops.py` | **First** domain annotation: sets `node.annex.domain` on `as_fieldop` nodes based on the IR structure. Does NOT use offset_provider. |
| 14 | `canonicalize_domain_argument` | `concat_where/canonicalize_domain_argument.py` | Converts `concat_where(AND(d1,d2), a, b)` → nested let-bindings `__cwcda_field_a/b` + canonical complement form (`OR(-inf,lo) | [hi,inf)`). Also converts finite single-dim conditions → complement. **This is where the let-lifting that triggers the Kolor:0:0 bug happens.** |
| 15 | `ConstantFolding.apply` | `constant_folding.py` | Folds constant expressions |
| 16 | `infer_domain.infer_program` | `infer_domain.py` | **Second** domain annotation: propagates domains top-down through the IR, including into let-bindings. `keep_existing_domains=True` preserves explicit `as_fieldop` domain args. `allow_uninferred=True` allows NEVER-domain args (from the literal-0 fallback). **This is where the Kolor:0:0 empty domain gets assigned to `__cwcda_field_b`.** |
| 17 | `prune_empty_concat_where` | `prune_empty_concat_where.py` | Prunes `concat_where` branches whose `annex.domain` is empty (start==stop). Uses `annex.domain` set by pass 16. **Only prunes direct concat_where nodes; let-bound dead values are NOT pruned here.** |
| 18 | `RemoveBroadcast.apply` | `remove_broadcast.py` | Removes `broadcast` nodes |
| 19 | `ConstantFolding.apply` | `constant_folding.py` | Final constant folding |

★ = only runs when `USE_STRUCTURED_BACKEND=1`

---

## IR dump section headers (from `_print_ir_block` calls)

Set `GT4PY_PRINT_IR=1` to write these to `ir_out.txt`:

| Header | After pass # |
|---|---|
| `=== FIELDVIEW IR BEFORE TRANSFORMS ===` | (start) |
| `=== FIELDVIEW IR AFTER PROCESSING DOMAIN OPTIONS ===` | 1 |
| `=== FIELDVIEW IR AFTER INLINING FUNDEFS ===` | 2 |
| `=== FIELDVIEW IR AFTER DEAD CODE ELIMINATION ===` | 4 |
| `=== FIELDVIEW IR AFTER CARTESIAN UNROLLING ===` | 12★ |
| `=== FIELDVIEW IR AFTER INFERRING DOMAIN OPS ===` | 13 |
| `=== FIELDVIEW IR AFTER PRUNING EMPTY CONCAT WHERE ===` | 17 |
| `=== FINAL FIELDVIEW IR ===` | 19 |

---

## The Kolor:0:0 bug — root cause map

```
Pass 8 (SetAtRemapper):
  _build_edge_validity_masked_expr creates:
    concat_where(AND(Kolor:[k,k+1), IDim:[lo,hi), JDim:[lo,hi)), expr_k, rest)

Pass 9 (ReductionUnroller):
  _visit_expr_with_kolor_branches peels this → visits each kolor-k branch
  with current_kolor=k → _build_field_concat_where_from_branches creates:
    as_fieldop(λ→deref(shift(…)), Kolor:[k,k+1))(field)  ← explicit narrow domain

Pass 13 (InferDomainOps):
  Sets annex.domain on as_fieldop nodes.
  For the narrow domains: annex.domain = Kolor:[k,k+1)

Pass 14 (canonicalize_domain_argument):
  Expands AND(Kolor:[k,k+1), IDim, JDim) → let __cwcda_field_a = kolor-k branch
                                                let __cwcda_field_b = rest
  Then Kolor:[k,k+1) → complement OR(Kolor:(-inf,k), Kolor:[k+1,inf))
  → concat_where(OR(…), __cwcda_field_b, <IDim/JDim inner structure>)

Pass 16 (infer_program with keep_existing_domains=True):
  For the top-level SetAt context Kolor:[0,3):
    concat_where(OR(Kolor:(-inf,0), Kolor:[1,inf)), __b, inner) at domain [0,3):
      __b accessed at Kolor:[1,3)  ← CORRECT for the case k=0
  But for the INNER IDim/JDim let structure, __b2 (= __cwcda_field_b inside the IDim/JDim AND expansion)
  may get a narrowed context. If there is a concat_where with condition Kolor:[1,inf)
  inside an outer domain Kolor:[0,1) (from keep_existing_domains on a narrow as_fieldop):
    __b accessed at Kolor:[1,inf) ∩ Kolor:[0,1) = Kolor:[1,1)  ← EMPTY!

Pass 17 (prune_empty_concat_where):
  Prunes concat_where that references __b at empty Kolor:[1,1).
  But the let-binding `let __b = <expr>` STILL EXISTS in the IR.
  DaCe materialises __b as a transient with domain Kolor:[1,1) → zero-sized.
```

---

## Debug environment variables

| Variable | Effect |
|---|---|
| `GT4PY_PRINT_IR=1` | Write per-pass IR to `ir_out.txt` |
| `GT4PY_TRACE_INFER_DOMAIN=1` | Print every let / as_fieldop domain in `infer_domain.py` |
| `GT4PY_TRACE_INFER_DOMAIN_SHRINK=1` | Print IDim/JDim changes in `_infer_as_fieldop` |
| `GT4PY_TRACE_INFER_DOMAIN_KOLOR=1` | Print Kolor-range details for let params and as_fieldop targets (NEW) |
| `GT4PY_KEEP_EXISTING_DOMAINS` | `1` (default) = keep_existing_domains=True; `0` = False |

---

## Key functions for the Kolor:0:0 bug

| Function | File | Role |
|---|---|---|
| `_build_edge_validity_masked_expr` | `structured_backend_passes.py` | Creates AND-conditioned concat_where (source of let-lifting) |
| `_build_field_concat_where_from_branches` | `structured_backend_passes.py` | Creates narrow `as_fieldop(shift, Kolor:[k,k+1))` for each branch |
| `_CanonicalizeDomainArgument.transform` | `concat_where/canonicalize_domain_argument.py` | AND → let-lift + complement form |
| `_infer_let` | `infer_domain.py` | Propagates domain to let-bound args via `accessed_domains_let_args` |
| `_infer_as_fieldop` | `infer_domain.py` | Optionally preserves explicit domain via `keep_existing_domains` |
| `prune_empty_concat_where` | `prune_empty_concat_where.py` | Removes branches but NOT enclosing let-bindings |
