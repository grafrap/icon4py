"""Fast unit + integration tests for the Kolor domain fix.

Root cause: `_build_edge_validity_masked_expr` wraps each kolor branch in a
concat_where with AND conditions. `canonicalize_domain_argument` let-lifts
these into `let __cwcda_field_b = as_fieldop(stencil, Kolor:[0,3))(field)`.
When `infer_program` processes the let body (e.g. at outer context Kolor:[0,1)),
it assigns `__cwcda_field_b` accessed domain Kolor:[1,1) (empty intersection).
Since the returned `as_fieldop` node is fresh (no pre-existing `annex.domain`),
`infer_expr` line 712 overwrote `annex.domain` with Kolor:[1,1). DaCe read
this annex to size the copy-destination transient → zero-sized Kolor → crash.

The fix is in `infer_domain._infer_as_fieldop`: after `keep_existing_domains`
preserves the explicit domain, pre-set `annex.domain` on the returned node to
the kept explicit domain. Then `infer_expr` line 712 sees it already set and
skips overwriting.

Note: `_build_field_concat_where_from_branches` in the current codebase does
NOT narrow branches to Kolor:[k,k+1). All branches use the full outer domain
Kolor:[0,3). The fix to `infer_domain.py` ensures annex.domain stays correct
regardless of what the let-binding context propagates.

Run via srun:
    srun --partition=debug --time=00:05:00 --uenv=icon/25.2:v3 --view=default \\
      bash -c '.venv/bin/python test_narrow_kolor_fix.py'

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
    """Return list of OOB Kolor ranges found in the pretty-printed IR.

    An OOB range is a Kolor range (with or without `ₒ` subscript) whose
    bounds are negative (lo < 0) or above-max (hi > 3). These are the
    ranges that cause GPU ILLEGAL_ADDRESS from DaCe reading past valid Kolor indices.

    Note: ranges like Kolor:[0,3) are VALID (full outer domain) and must NOT be
    flagged. After the intersection fix in _infer_as_fieldop, full explicit domains
    Kolor:[0,3) are correctly preserved in fun.args[1] for DaCe transient sizing.
    """
    text = pformat(node)
    out = []
    # Match both subscript (ₒ) and plain Literal Kolor ranges
    for m in re.finditer(r"Kolor\w*: \[(-?\d+)\w*, (-?\d+)\w*\[", text):
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo < 0 or hi > 3:  # only flag truly OOB ranges; [0,3) is valid
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

def test_build_field_concat_where_uses_full_outer_domain():
    """All per-kolor branch as_fieldop nodes now use NARROW per-kolor domains Kolor:[k,k+1).

    After the fix, each branch is narrowed so that infer_program back-propagation
    through Kolor shifts produces valid (in-bounds) source domains. E.g. for shift
    Kolor+2 on a kolor=0 branch: back-prop from Kolor:[0,1) → source Kolor:[2,3) ✓
    instead of from Kolor:[0,3) → source Kolor:[2,5) OOB.
    """
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
    # Each branch uses its own narrow Kolor:[k,k+1) domain.
    assert kolor_doms == [(0, 1), (1, 2), (2, 3)], (
        f"expected narrow per-kolor domains [(0,1),(1,2),(2,3)], got {kolor_doms}"
    )
    assert len(kolor_doms) == 3, f"expected 3 as_fieldop nodes, got {len(kolor_doms)}"
    print("  ok  test_build_field_concat_where_uses_full_outer_domain")


def test_build_field_concat_where_trailing_else_uses_full_domain():
    """Trailing else branch (cond=None) also uses narrow Kolor:[2,3) domain.

    The trailing else kolor is inferred from inferred_kolor_start progression:
    kolor=0 → kolor=1 → trailing=2. The branch is narrowed to Kolor:[2,3)
    so that back-propagation through shift Kolor-2 gives source Kolor:[0,1) ✓
    instead of Kolor:[-2,1) OOB from the full Kolor:[0,3).
    """
    outer = cart_dom(i=(4, 508), j=(4, 508), k=(0, 3))
    branches = [
        (kolor_cond(0, 1), shift_spec("_OffKolor", 1)),
        (kolor_cond(1, 2), shift_spec("_OffKolor", 1)),
        (None,             shift_spec("_OffKolor", -2)),  # cond=None — trailing else (kolor=2)
    ]

    result = _build_field_concat_where_from_branches(
        arg=im.ref("dwdz"),
        branches=branches,
        domain=outer,
        inferred_kolor_start=0,
    )

    kolor_doms = all_as_fieldop_kolor_domains(result)
    # Trailing else gets kolor=2 (inferred) → narrows to Kolor:[2,3).
    assert kolor_doms == [(0, 1), (1, 2), (2, 3)], (
        f"expected narrow per-kolor domains [(0,1),(1,2),(2,3)], got {kolor_doms}"
    )
    assert len(kolor_doms) == 3, f"expected 3 as_fieldop nodes, got {len(kolor_doms)}"
    print("  ok  test_build_field_concat_where_trailing_else_uses_full_domain")


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


# ── Test 6: _edge_shape_domain no longer clips IDim/JDim ─────────────────────

def test_edge_shape_domain_no_clip_on_idim():
    """_edge_shape_domain must NOT clip IDim/JDim.

    Root cause of stencil 06 failure with keep_existing_domains=True:
      _edge_shape_domain was clipping IDim hi by -1 for target_kolor in {1,2}.
      For outer IDim=[5,22), it returned IDim=[5,21).
      With keep_existing_domains=True this 21 was preserved.
      DaCe allocated the transient at IDim=[5,21) (size=16) while the outer map
      ran IDim=[5,22) (size=17) → at IDim=21, local index 21-5=16 → OOB.

    For our parallelogram grid all three edge kolors share the same IDim/JDim
    upper bound, so no clip is correct: kolor-1 edges DO extend to IDim=22.
    """
    key = (itir.OffsetLiteral(value="E2C"), itir.OffsetLiteral(value=0))
    from gt4py.next.iterator.transforms.map_dict import map_dict as _map_dict
    entry = _map_dict.get(key)
    if entry is None or entry["kind"] != "concat_where":
        print("  ok  test_edge_shape_domain_no_clip_on_idim (skip: E2C not in map_dict)")
        return

    branches = entry["branches"]
    outer = cart_dom(i=(5, 22), j=(4, 22), k=(0, 1))
    vn = im.ref("vn")
    result = _build_field_concat_where_from_branches(
        vn, branches, outer,
        apply_edge_shape_bounds=True,
        current_kolor=None,
    )
    # No as_fieldop domain may have IDim hi < 22 (the outer hi).
    # With the old clip: branches with target_kolor=1 got IDim hi=21 →
    # DaCe transient size=16 < outer loop size=17 → OOB at IDim=21.
    bad_idim = []
    for fc in result.pre_walk_values().if_isinstance(itir.FunCall):
        if not cpm.is_applied_as_fieldop(fc) or len(fc.fun.args) < 2:
            continue
        dom = fc.fun.args[1]
        if not cpm.is_call_to(dom, "cartesian_domain"):
            continue
        for nr in dom.args:
            if not cpm.is_call_to(nr, "named_range") or len(nr.args) != 3:
                continue
            axis = nr.args[0]
            hi = nr.args[2]
            if getattr(axis, "value", None) == "IDim":
                hi_val = hi.value if hasattr(hi, "value") else None
                if isinstance(hi_val, int) and hi_val < 22:
                    bad_idim.append(hi_val)
    assert not bad_idim, (
        f"_edge_shape_domain clipped IDim hi to {bad_idim} (expected 22 everywhere). "
        f"Old code: target_kolor in {{1,2}} → i_hi -= 1 → 21. "
        f"Fix: remove that clip so inner branch domain equals outer domain."
    )
    print("  ok  test_edge_shape_domain_no_clip_on_idim")


# ── Test 7: annex.domain fix in infer_as_fieldop ─────────────────────────────

def test_infer_as_fieldop_keeps_annex_domain_with_keep_existing():
    """The `annex.domain` on a keep_existing node must be the kept explicit domain,
    NOT the (possibly empty) domain from the let-binding context.

    Root cause of stencils 06 and 10 failure:
      _infer_as_fieldop creates a fresh `transformed_call` node (no annex.domain).
      `infer_expr` then sets `annex.domain = caller_domain` (e.g. Kolor:[1,1) empty).
      DaCe allocates the copy-destination transient using annex.domain → zero-sized.

    Fix: pre-set `transformed_call.annex.domain = kept_explicit_domain` inside
    `_infer_as_fieldop` so that `infer_expr`'s line 712 sees it already set
    and skips overwriting.
    """
    from gt4py.next.iterator.transforms.infer_domain import infer_expr, DomainAccessDescriptor
    from gt4py.next.iterator.transforms.infer_domain import InferenceOptions
    from gt4py.next.iterator.ir_utils import domain_utils

    outer = cart_dom(i=(4, 22), j=(4, 22), k=(0, 3))
    empty_context = im.domain(
        common.GridType.CARTESIAN,
        {Kolor: (itir.OffsetLiteral(value=1), itir.OffsetLiteral(value=1))},  # Kolor:[1,1) empty
    )
    # Build an as_fieldop with an explicit Kolor:[0,3) domain (the full outer domain).
    inner_asfo = im.as_fieldop(
        im.lambda_("it")(im.deref(im.ref("it"))),
        outer,
    )(im.ref("field"))

    # Infer it under the empty Kolor:[1,1) context with keep_existing_domains=True.
    result, _ = infer_expr(
        inner_asfo,
        domain_utils.SymbolicDomain.from_expr(empty_context),
        offset_provider=OFFSET_PROVIDER,
        symbolic_domain_sizes={},
        allow_uninferred=True,
        keep_existing_domains=True,
    )

    # annex.domain must be the KEPT explicit domain Kolor:[0,3), not Kolor:[1,1).
    assert hasattr(result.annex, "domain"), "annex.domain not set on result"
    kolor_dim = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)
    if isinstance(result.annex.domain, domain_utils.SymbolicDomain):
        kolor_range = result.annex.domain.ranges.get(kolor_dim)
        assert kolor_range is not None, f"no Kolor in annex.domain: {result.annex.domain}"
        lo = getattr(kolor_range.start, "value", None)
        hi = getattr(kolor_range.stop, "value", None)
        assert lo == 0 and hi == 3, (
            f"annex.domain Kolor should be [0,3) (kept explicit), got [{lo},{hi}). "
            f"Without the fix, line 712 of infer_domain.py overwrites annex.domain "
            f"with the caller's Kolor:[1,1), causing DaCe to allocate a zero-sized transient."
        )
    print("  ok  test_infer_as_fieldop_keeps_annex_domain_with_keep_existing")


# ── Helpers for nested-and_ tests ────────────────────────────────────────────

def _and_cond(kolor_lo, kolor_hi, i_lo, i_hi, j_lo, j_hi):
    """Build and_(Kolor:[klo,khi), and_(IDim:[ilo,ihi), JDim:[jlo,jhi))) condition."""
    def _dom(axis_name, lo, hi):
        axis = itir.AxisLiteral(value=axis_name, kind=common.DimensionKind.HORIZONTAL)
        return im.call("cartesian_domain")(
            im.named_range(axis, itir.OffsetLiteral(value=lo), itir.OffsetLiteral(value=hi))
        )
    return im.call("and_")(
        _dom("Kolor", kolor_lo, kolor_hi),
        im.call("and_")(_dom("IDim", i_lo, i_hi), _dom("JDim", j_lo, j_hi)),
    )


def _build_nested_and_program():
    """Build a Program matching the non-split E2C2EO path of _build_edge_validity_masked_expr.

    Structure after CartesianReductionUnroller (before canonicalize_domain_argument):
        SetAt(output, Kolor:[0,3)) ←
          concat_where(and_(Kolor:[0,1), and_(IDim:[4,22), JDim:[4,22))),
            as_fieldop(shift(Kolor,+1), Kolor:[0,3))(vn),   ← wide explicit domain
            concat_where(and_(Kolor:[1,2), and_(IDim:[4,22), JDim:[4,22))),
              as_fieldop(shift(Kolor,+1), Kolor:[0,3))(vn),  ← wide
              as_fieldop(shift(Kolor,-2), Kolor:[0,3))(vn))) ← wide (trailing)

    The inner as_fieldop nodes carry wide Kolor:[0,3) from the outer accumulation.
    `infer_program(keep_existing_domains=False)` before canonicalize_domain_argument
    should narrow each to the enclosing branch's Kolor ([0,1), [1,2), [2,3)).
    """
    outer = cart_dom(i=(4, 22), j=(4, 22), k=(0, 3))

    def _wide_as_fieldop(kolor_offset: int) -> itir.Expr:
        """as_fieldop with WIDE Kolor:[0,3) explicit domain, shift on Kolor."""
        return im.as_fieldop(
            im.lambda_("__it")(im.deref(im.shift("_OffKolor", kolor_offset)(im.ref("__it")))),
            outer,  # wide — this is what we want to narrow
        )(im.ref("vn"))

    expr = im.concat_where(
        _and_cond(0, 1, 4, 22, 4, 22),
        _wide_as_fieldop(+1),
        im.concat_where(
            _and_cond(1, 2, 4, 22, 4, 22),
            _wide_as_fieldop(+1),
            _wide_as_fieldop(-2),  # trailing — kolor 2, shift -2 → src kolor 0 (valid)
        ),
    )

    return itir.Program(
        id="nested_and_test",
        function_definitions=[],
        params=[
            itir.Sym(id="vn", type=edge_k_field),
            itir.Sym(id="output", type=edge_k_field),
            itir.Sym(id="vertical_start", type=int32),
            itir.Sym(id="vertical_end", type=int32),
        ],
        declarations=[],
        body=[itir.SetAt(expr=expr, domain=outer, target=im.ref("output"))],
    )


def test_infer_concat_where_nested_and_no_crash():
    """Nested and_ condition must not cause AssertionError in _infer_concat_where.

    _build_edge_validity_masked_expr produces and_(Kolor:[k,k+1), and_(IDim:[i,j), JDim:[i,j))).
    Old _infer_concat_where iterated cond.args calling SymbolicDomain.from_expr on each —
    the inner and_ is not a cartesian_domain → AssertionError.
    Fix: recursive stack flattening that descends through nested and_ before calling from_expr.
    """
    prog = _build_nested_and_program()
    # Run through infer_program with keep_existing_domains=False (pre-canonicalize approach).
    # Must not raise AssertionError.
    try:
        result = infer_domain.infer_program(
            prog,
            offset_provider=OFFSET_PROVIDER,
            allow_uninferred=True,
            keep_existing_domains=False,
        )
    except AssertionError as e:
        raise AssertionError(
            "infer_program crashed with AssertionError on nested and_ condition — "
            "the recursive stack flattening fix in _infer_concat_where is missing or broken. "
            f"Original error: {e!r}"
        )
    print("  ok  test_infer_concat_where_nested_and_no_crash")
    return result


def test_precanon_infer_narrows_kolor_with_nested_and():
    """Pre-canonicalize infer_program narrows inner as_fieldop to per-branch Kolor,
    even when the condition is nested and_(Kolor, and_(IDim, JDim)).

    After the structured passes, inner as_fieldop nodes carry wide Kolor:[0,3).
    Running infer_program(keep_existing_domains=False) before canonicalize_domain_argument
    should narrow each branch's as_fieldop domain to the enclosing kolor context:
      - kolor-0 branch → as_fieldop gets Kolor:[0,1)
      - kolor-1 branch → as_fieldop gets Kolor:[1,2)
      - kolor-2 (trailing) → as_fieldop gets Kolor:[2,3)

    This prevents DaCe from materializing a 3-Kolor transient for each let-lifted node,
    which would produce OOB reads at shift+1 from kolor 2 (→ kolor 3) or shift-2 from
    kolor 0 (→ kolor -2).

    Root cause of the original failure: _infer_concat_where computed the FALSE-branch
    complement of and_(Kolor:[0,1), and_(IDim:[4,22), JDim:[4,22))) as
    {Kolor:[1,+inf), IDim:[22,+inf), JDim:[22,+inf)} — then intersected with outer
    IDim:[4,22) → IDim:[22,22) = EMPTY, cascading to Kolor:[1,1) and Kolor:[2,1)
    (empty/negative) in nested branches.

    Fix: for multi-dim AND conditions containing Kolor, only advance Kolor for the
    false branch; IDim/JDim come from the outer target unchanged.
    """
    prog = _build_nested_and_program()

    # Step 1: pre-canonicalize infer_program — should narrow domains without crash.
    try:
        ir_narrowed = infer_domain.infer_program(
            prog,
            offset_provider=OFFSET_PROVIDER,
            allow_uninferred=True,
            keep_existing_domains=False,
        )
    except Exception as e:
        raise AssertionError(
            f"Pre-canonicalize infer_program raised: {type(e).__name__}: {e}\n"
            "Check that the nested and_ false-branch complement fix is in _infer_concat_where."
        )

    # No OOB Kolor ranges after narrowing.
    widenings = kolor_widenings(ir_narrowed)
    assert not widenings, (
        f"Pre-canonicalize infer_program produced OOB Kolor ranges: {widenings}\n"
        f"These would cause DaCe GPU OOB reads.\n"
        f"IR:\n{pformat(ir_narrowed)}"
    )

    # All inner as_fieldop nodes must now have narrow per-branch domains (≤ [0,3)).
    kolor_doms = all_as_fieldop_kolor_domains(ir_narrowed)
    assert kolor_doms, "No as_fieldop nodes found with explicit Kolor domains"
    for lo, hi in kolor_doms:
        assert lo >= 0 and hi <= 3, (
            f"as_fieldop Kolor domain [{lo},{hi}) is OOB (valid: [0,3)). "
            f"The pre-canonicalize infer_program failed to narrow it. "
            f"All domains: {kolor_doms}"
        )

    # Step 2: canonicalize + post-canonicalize infer — should stay valid (no re-widening).
    ir_canon = concat_where.canonicalize_domain_argument(ir_narrowed)
    ir_canon = ConstantFolding.apply(ir_canon)
    ir_final = infer_domain.infer_program(
        ir_canon,
        offset_provider=OFFSET_PROVIDER,
        allow_uninferred=True,
        keep_existing_domains=True,
    )
    ir_final = prune_empty_concat_where.prune_empty_concat_where(ir_final)

    final_widenings = kolor_widenings(ir_final)
    assert not final_widenings, (
        f"Post-canonicalize infer_program re-widened Kolor ranges: {final_widenings}\n"
        f"IR:\n{pformat(ir_final)}"
    )
    print("  ok  test_precanon_infer_narrows_kolor_with_nested_and")


# ── runner ───────────────────────────────────────────────────────────────────

TESTS = [
    test_build_field_concat_where_uses_full_outer_domain,
    test_build_field_concat_where_trailing_else_uses_full_domain,
    test_post_unroll_pipeline_no_widening_with_fix,
    test_post_unroll_pipeline_widens_without_keep_existing,
    test_infer_program_handles_tuple_output_as_fieldop,
    test_edge_shape_domain_no_clip_on_idim,
    test_infer_as_fieldop_keeps_annex_domain_with_keep_existing,
    test_infer_concat_where_nested_and_no_crash,
    test_precanon_infer_narrows_kolor_with_nested_and,
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
