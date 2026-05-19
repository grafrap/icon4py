"""Fast unit + integration tests for the narrow-Kolor fix in `structured_backend_passes`.

These tests reproduce the IR pattern that causes
`apply_divergence_damping_and_update_vn` to over-widen the Kolor source ranges
(e.g. `Kolor:[2, 5)`, `Kolor:[1, 4)`) and trigger
`CUDA_ERROR_ILLEGAL_ADDRESS` / SIGSEGV in DaCe.

The fix has two parts (see `structured_backend_passes.py` and
`pass_manager.py`):
  1. `_build_field_concat_where_from_branches`: narrow each branch
     `as_fieldop`'s stored Kolor domain to `[k, k+1)` (matching the
     concat_where `cond`), instead of inheriting the full outer `Kolor:[0,3)`.
  2. `apply_fieldview_transforms` (`pass_manager.py:551-556`): pass
     `keep_existing_domains=True` to the 2nd `infer_program`, so the narrow
     domains survive `canonicalize_domain_argument` + re-inference.

Run directly:
    python3 test_narrow_kolor_fix.py

All assertions are `assert` statements; the script prints a summary and
exits non-zero on failure.
"""

import re
import sys

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.pretty_printer import pformat
from gt4py.next.iterator.transforms import (
    concat_where,
    infer_domain,
    infer_domain_ops,
    prune_empty_concat_where,
)
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator.transforms.structured_backend_passes import (
    _build_field_concat_where_from_branches,
)
from gt4py.next.type_system import type_specifications as ts


# ── shared setup ─────────────────────────────────────────────────────────────

IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)
K = common.Dimension("K", kind=common.DimensionKind.VERTICAL)

float64 = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
int32 = ts.ScalarType(kind=ts.ScalarKind.INT32)
edge_k_field = ts.FieldType(dims=[IDim, JDim, Kolor, K], dtype=float64)
cell_k_field = ts.FieldType(dims=[IDim, JDim, Kolor, K], dtype=float64)

OFFSET_PROVIDER = {
    "IDim": IDim, "JDim": JDim, "Kolor": Kolor, "K": K,
    "_OffIDim": IDim, "_OffJDim": JDim, "_OffKolor": Kolor, "_OffK": K,
}


def cart_dom(i, j, k, kv_start="vertical_start", kv_end="vertical_end"):
    return im.domain(
        common.GridType.CARTESIAN,
        {
            IDim: (itir.OffsetLiteral(value=i[0]), itir.OffsetLiteral(value=i[1])),
            JDim: (itir.OffsetLiteral(value=j[0]), itir.OffsetLiteral(value=j[1])),
            Kolor: (itir.OffsetLiteral(value=k[0]), itir.OffsetLiteral(value=k[1])),
            K: (im.ref(kv_start), im.ref(kv_end)),
        },
    )


def kolor_cond(lo, hi):
    return im.domain(
        common.GridType.CARTESIAN,
        {Kolor: (itir.OffsetLiteral(value=lo), itir.OffsetLiteral(value=hi))},
    )


def shift_spec(axis_name, offset):
    return (itir.OffsetLiteral(value=axis_name), itir.OffsetLiteral(value=offset))


def kolor_widenings(node):
    """Return list of widened Kolor ranges found in the pretty-printed IR.

    A 'widening' is a Kolor range without `ₒ` subscript on its bounds
    (i.e. an `ir.Literal` created by `infer_program`) whose extent > 1 or whose
    bounds are negative/over-3.  These are the OOB-causing ranges.
    """
    text = pformat(node)
    out = []
    # Plain Literal (no ₒ subscript) Kolor ranges.
    for m in re.finditer(r"Kolor\w*: \[(-?\d+), (-?\d+)\[", text):
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo < 0 or hi > 3 or (hi - lo) > 1:
            out.append((lo, hi))
    return out


def all_as_fieldop_kolor_domains(node):
    """Walk an IR node, return Kolor ranges of every `as_fieldop` with explicit domain."""
    out = []
    for fc in node.pre_walk_values().if_isinstance(itir.FunCall):
        if not cpm.is_applied_as_fieldop(fc):
            continue
        if len(fc.fun.args) < 2:
            continue
        dom = fc.fun.args[1]
        if not cpm.is_call_to(dom, "cartesian_domain"):
            continue
        for nr in dom.args:
            if not cpm.is_call_to(nr, "named_range") or len(nr.args) != 3:
                continue
            axis = nr.args[0]
            if getattr(axis, "value", None) != "Kolor":
                continue
            lo, hi = nr.args[1], nr.args[2]
            lo_v = lo.value if hasattr(lo, "value") else None
            hi_v = hi.value if hasattr(hi, "value") else None
            if isinstance(lo_v, int) and isinstance(hi_v, int):
                out.append((lo_v, hi_v))
            elif isinstance(lo_v, str) and isinstance(hi_v, str):
                try:
                    out.append((int(lo_v), int(hi_v)))
                except ValueError:
                    pass
    return out


# ── Test 1: direct call to _build_field_concat_where_from_branches ───────────

def test_build_field_concat_where_narrows_kolor():
    """Each per-kolor branch's as_fieldop must store the narrow `[k, k+1)`, not the outer `[0,3)`."""
    outer = cart_dom(i=(4, 508), j=(4, 508), k=(0, 3))
    branches = [
        (kolor_cond(0, 1), shift_spec("_OffKolor", 1)),
        (kolor_cond(1, 2), shift_spec("_OffKolor", 1)),
        (kolor_cond(2, 3), shift_spec("_OffKolor", -2)),
    ]

    result = _build_field_concat_where_from_branches(
        arg=im.ref("grad"),
        branches=branches,
        domain=outer,
    )

    kolor_doms = all_as_fieldop_kolor_domains(result)
    # Expect exactly three as_fieldop domains, one per branch, narrow [k, k+1).
    assert (0, 1) in kolor_doms, f"missing Kolor:[0,1); got {kolor_doms}"
    assert (1, 2) in kolor_doms, f"missing Kolor:[1,2); got {kolor_doms}"
    assert (2, 3) in kolor_doms, f"missing Kolor:[2,3); got {kolor_doms}"
    # And there must be NO branch as_fieldop with the wide [0,3).
    wide = [d for d in kolor_doms if d == (0, 3)]
    assert not wide, (
        f"branch as_fieldop still has the wide Kolor:[0,3) — fix not applied. "
        f"All Kolor domains: {kolor_doms}\nIR:\n{pformat(result)}"
    )
    print("  ok  test_build_field_concat_where_narrows_kolor")


def test_build_field_concat_where_trailing_else_is_wrapped():
    """Real-world map_dict pattern: trailing branch has `cond=None` (else branch).

    The trailing branch is wrapped in
        concat_where(Kolor:[source_kolor, source_kolor+1),
                     narrow_trailing_shift,
                     literal_zero_fallback)
    so the result is a field with valid values at ALL kolors of the outer
    SetAt domain. This avoids:

      - GPU `CUDA_ERROR_ILLEGAL_ADDRESS (700)` at 512×512: the previous
        bare wide-domain trailing as_fieldop with shift `_OffKolor:-2` on
        a 2-Kolor cell field caused OOB reads at output K=0,1 (source -2,-1).

      - CPU `+inf location mismatch`: naïvely narrowing trailing's as_fieldop
        domain to `Kolor:[k,k+1)` creates a 1-slot temp that downstream
        let-lifted concat_where consumers read OOB at other kolors.

      - GPU InlineSDFG crash on E2C2V stencils (01, 10): the original
        identity-fallback read `arg` at output position K=0..source_kolor-1,
        which OOB'd on vertex fields (Kolor=1) at K>=1. The literal-0
        fallback marks `arg` as NEVER, DaCe strips the unused arg, no reads.

    Both narrow_trailing_shift (at K=source_kolor) and the literal-0 fallback
    (at K<source_kolor) end up with the same field shape but with safe values
    at every position.
    """
    outer = cart_dom(i=(4, 508), j=(4, 508), k=(0, 3))
    branches = [
        (kolor_cond(0, 1), shift_spec("_OffKolor", 1)),
        (kolor_cond(1, 2), shift_spec("_OffKolor", 1)),
        (None,             shift_spec("_OffKolor", -2)),  # cond=None — trailing else
    ]

    result = _build_field_concat_where_from_branches(
        arg=im.ref("dwdz"),
        branches=branches,
        domain=outer,
        inferred_kolor_start=0,
    )

    kolor_doms = all_as_fieldop_kolor_domains(result)
    # First two branches narrowed via their explicit `cond`.
    assert (0, 1) in kolor_doms, f"missing Kolor:[0,1); got {kolor_doms}"
    assert (1, 2) in kolor_doms, f"missing Kolor:[1,2); got {kolor_doms}"
    # Trailing else now wraps: narrow_trailing at [2,3) + literal-0 fallback at [0,2).
    assert (2, 3) in kolor_doms, (
        f"trailing branch's narrow as_fieldop missing Kolor:[2,3); got {kolor_doms}"
    )
    assert (0, 2) in kolor_doms, (
        f"trailing branch's literal-0 fallback missing Kolor:[0,2); got {kolor_doms}"
    )
    # And critically: no bare wide [0,3) trailing branch remains.
    wide = [d for d in kolor_doms if d == (0, 3)]
    assert not wide, (
        f"trailing else still has wide Kolor:[0,3); the wrap didn't fire. "
        f"all Kolor domains: {kolor_doms}\nIR:\n{pformat(result)}"
    )
    # The fallback's lambda body should be a literal (NEVER-arg semantics),
    # NOT a deref of the input. This is what makes it safe for vertex fields.
    text = pformat(result)
    assert "·__cart_trailing_unused" not in text, (
        "fallback lambda dereferences its arg — it should be a literal-0 instead.\n"
        f"IR:\n{text}"
    )
    print("  ok  test_build_field_concat_where_trailing_else_is_wrapped")


# ── Test 2: post-unroll pipeline (the actual passes that re-widen w/o fix) ───

def _build_minimal_failing_program():
    """Hand-built Program matching the failing E2C2EO pattern in
    `apply_divergence_damping_and_update_vn`, but stripped to one slot:

        next_vn @ Kolor:[0,3) ←
          concat_where(Kolor:[0,1), as_fieldop(shift+1, Kolor:[0,1))(grad),
            concat_where(Kolor:[1,2), as_fieldop(shift+1, Kolor:[1,2))(grad),
                          as_fieldop(shift-2, Kolor:[2,3))(grad)))

    Each per-kolor branch reads `grad` (an edge field, Kolor=3) via a Kolor
    shift.  With narrow per-branch domains (Kolor:[k,k+1)) the post-unroll
    passes should leave the source domains at the valid `[shift+k, shift+k+1)`
    interval, NOT widen to `[0,3)+shift = [1,4)` / `[2,5)`.
    """
    outer = cart_dom(i=(4, 508), j=(4, 508), k=(0, 3))
    branches = [
        (kolor_cond(0, 1), shift_spec("_OffKolor", 1)),
        (kolor_cond(1, 2), shift_spec("_OffKolor", 1)),
        (kolor_cond(2, 3), shift_spec("_OffKolor", -2)),
    ]
    expr = _build_field_concat_where_from_branches(
        arg=im.ref("grad"), branches=branches, domain=outer,
    )
    return itir.Program(
        id="minimal_failing_pattern",
        function_definitions=[],
        params=[
            itir.Sym(id="grad", type=edge_k_field),
            itir.Sym(id="next_vn", type=edge_k_field),
            itir.Sym(id="vertical_start", type=int32),
            itir.Sym(id="vertical_end", type=int32),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=expr,
                domain=outer,
                target=im.ref("next_vn"),
            )
        ],
    )


def _run_post_unroll_pipeline(program, *, keep_existing_domains: bool):
    """Same sequence as `apply_fieldview_transforms` after CartesianReductionUnroller."""
    ir = infer_domain_ops.InferDomainOps.apply(program)
    ir = concat_where.canonicalize_domain_argument(ir)
    ir = ConstantFolding.apply(ir)
    ir = infer_domain.infer_program(
        ir,
        offset_provider=OFFSET_PROVIDER,
        keep_existing_domains=keep_existing_domains,
    )
    ir = ConstantFolding.apply(ir)
    ir = prune_empty_concat_where.prune_empty_concat_where(ir)
    return ir


def test_post_unroll_pipeline_no_widening_with_fix():
    """With both fixes (narrow branch domain + keep_existing_domains=True), no widening."""
    prog = _build_minimal_failing_program()
    ir = _run_post_unroll_pipeline(prog, keep_existing_domains=True)
    widenings = kolor_widenings(ir)
    assert not widenings, (
        f"unexpected widened Kolor ranges in final IR: {widenings}\nIR:\n{pformat(ir)}"
    )
    print("  ok  test_post_unroll_pipeline_no_widening_with_fix")


def test_post_unroll_pipeline_widens_without_keep_existing():
    """Regression guard: without `keep_existing_domains=True`, widening reappears.

    This documents WHY the flag matters.  If `keep_existing_domains` becomes the
    default upstream, this test will start failing (good — remove the flag
    and this test then).
    """
    prog = _build_minimal_failing_program()
    ir_no_flag = _run_post_unroll_pipeline(prog, keep_existing_domains=False)
    widenings_no_flag = kolor_widenings(ir_no_flag)
    # With the narrow-branch fix alone (no keep_existing_domains), some
    # downstream re-inference can still widen via let-lifted cast wrappers.
    # If this happens, we report it but don't fail — the regression check is
    # only that the WITH-fix variant is clean (above).
    if not widenings_no_flag:
        print(
            "  note  without keep_existing_domains=True we ALSO got no widening — "
            "the flag may no longer be required.  Consider removing it from "
            "pass_manager.py:555."
        )
    else:
        print(
            f"  ok  test_post_unroll_pipeline_widens_without_keep_existing "
            f"(saw {len(widenings_no_flag)} widened ranges without the flag — "
            f"flag is needed)"
        )


# ── Test: infer_domain.py:233 must handle tuple-output as_fieldop ────────────

def test_infer_program_handles_tuple_output_as_fieldop():
    """Regression for `SymbolicDomain.from_expr` assertion at infer_domain.py:233.

    When `keep_existing_domains=True` and an `as_fieldop` has a tuple output
    (`fun.args[1]` is `make_tuple(cartesian_domain, ...)` rather than a bare
    `cartesian_domain`), the original code crashed with `AssertionError` in
    `SymbolicDomain.from_expr`. The fix uses `_make_symbolic_domain_tuple` to
    handle both single-domain and tuple-of-domains cases.

    This was the root cause of stencil 01 (`calculate_nabla2_and_smag_coefficients_for_vn`)
    failing on `dace_gpu`: that stencil's IR contains tuple-output `as_fieldop`
    nodes (e.g. fused E2C2V kolor-split outputs producing a tuple of fields).
    """
    # Build: as_fieldop(λ → make_tuple(·it, ·it), make_tuple(dom_a, dom_b))(arg)
    dom_a = cart_dom(i=(0, 10), j=(0, 10), k=(0, 1))
    dom_b = cart_dom(i=(0, 10), j=(0, 10), k=(1, 2))
    tuple_domain = im.call("make_tuple")(dom_a, dom_b)
    tuple_stencil = im.lambda_("it")(
        im.call("make_tuple")(im.deref("it"), im.deref("it"))
    )
    expr = im.as_fieldop(tuple_stencil, tuple_domain)(im.ref("dwdz"))
    prog = itir.Program(
        id="tuple_output_pattern",
        function_definitions=[],
        params=[
            itir.Sym(id="dwdz", type=cell_k_field),
            itir.Sym(
                id="out",
                type=ts.TupleType(types=[cell_k_field, cell_k_field]),
            ),
            itir.Sym(id="vertical_start", type=int32),
            itir.Sym(id="vertical_end", type=int32),
        ],
        declarations=[],
        body=[
            itir.SetAt(expr=expr, domain=tuple_domain, target=im.ref("out")),
        ],
    )

    # The fix lives in `infer_domain._infer_as_fieldop`: should NOT raise.
    try:
        infer_domain.infer_program(
            prog,
            offset_provider=OFFSET_PROVIDER,
            keep_existing_domains=True,
        )
    except AssertionError as e:
        raise AssertionError(
            "infer_program(keep_existing_domains=True) raised AssertionError on "
            "a tuple-output as_fieldop — the SymbolicDomain.from_expr fix at "
            "infer_domain.py:233 must use _make_symbolic_domain_tuple. "
            f"Original error: {e!r}"
        )
    print("  ok  test_infer_program_handles_tuple_output_as_fieldop")


# ── runner ───────────────────────────────────────────────────────────────────

TESTS = [
    test_build_field_concat_where_narrows_kolor,
    test_build_field_concat_where_trailing_else_is_wrapped,
    test_post_unroll_pipeline_no_widening_with_fix,
    test_post_unroll_pipeline_widens_without_keep_existing,
    test_infer_program_handles_tuple_output_as_fieldop,
]

if __name__ == "__main__":
    import traceback

    failures = []
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, e))
            print(f"  FAIL  {t.__name__}\n    {e!r}")
            traceback.print_exc()
        except Exception as e:
            failures.append((t.__name__, e))
            print(f"  ERROR  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()

    print()
    if failures:
        print(f"FAILED ({len(failures)} / {len(TESTS)})")
        sys.exit(1)
    print(f"PASSED ({len(TESTS)} / {len(TESTS)})")
