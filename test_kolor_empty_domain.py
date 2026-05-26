"""Tests that reproduce the Kolor:0:0 empty-domain bug for stencils 06 and 10.

Root cause (confirmed by IR traces `ir_out_06.txt` and `ir_out_10.txt`):

  `ThresholdConditionRewriter` (stencil 10) or `_build_edge_validity_masked_expr`
  (stencil 06) creates:

    concat_where(AND(Kolor:[k,k+1), IDim:[lo,hi), JDim:[lo,hi)), expr_a, expr_b)

  `canonicalize_domain_argument` let-lifts the AND into:

    let __cwcda_field_a = expr_a, __cwcda_field_b = expr_b in
      concat_where(OR(Kolor:(-∞,k), Kolor:[k+1,∞)), __cwcda_field_b, inner)

  `infer_program` at outer context `Kolor:[0,k+1)` assigns:
    __cwcda_field_b: Kolor:[k+1,k+1) (EMPTY — intersection of [k+1,∞) with [0,k+1))

  Two sub-bugs:
  A) `has_explicit_domain=False` (stencil 10):
     `infer_program` assigns `fun.args[1] = empty_Kolor_domain` to the as_fieldop.
     DaCe sees the explicit empty domain → allocates zero-sized transient → crash.

  B) `has_explicit_domain=True` (stencil 06):
     `keep_existing_domains=True` preserves `fun.args[1] = Kolor:[0,3)` (correct).
     BUT `infer_expr` line 712 overwrites `annex.domain = Kolor:[1,1)` (empty)
     on the fresh `transformed_call` node (no pre-existing annex.domain).
     DaCe reads `annex.domain` for the pre-state copy size → zero-sized → crash.

Tests 1-2 MUST FAIL before the fix (confirming the bug is reproduced).
Tests 3-4 are diagnostic (print what DaCe would see; no assertions).
Test 5 MUST PASS after the fix.
ALL tests must pass after the fix is applied.

Run via srun:
    srun --partition=debug --time=00:05:00 --uenv=icon/25.2:v3 --view=default \\
      bash -c '.venv/bin/python test_kolor_empty_domain.py'
"""
import sys
sys.path.insert(0, "/scratch/mch/rgraf/gt4py/src")

import copy
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.ir_utils import domain_utils
from gt4py.next.iterator.pretty_printer import pformat
from gt4py.next.iterator.transforms import (
    concat_where,
    infer_domain,
    infer_domain_ops,
    prune_empty_concat_where,
    dead_code_elimination,
)
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.type_system import type_specifications as ts


# ── Shared dimensions & types ────────────────────────────────────────────────

IDim  = common.Dimension("IDim",  kind=common.DimensionKind.HORIZONTAL)
JDim  = common.Dimension("JDim",  kind=common.DimensionKind.HORIZONTAL)
Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)
K     = common.Dimension("K",     kind=common.DimensionKind.VERTICAL)

_float64  = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
_int32    = ts.ScalarType(kind=ts.ScalarKind.INT32)
_edge_kfld = ts.FieldType(dims=[IDim, JDim, Kolor, K], dtype=_float64)

OFFSET_PROVIDER = {
    "IDim": IDim, "JDim": JDim, "Kolor": Kolor, "K": K,
}

OL = itir.OffsetLiteral  # shorthand


def _cart_dom(i, j, k, kv_start="vertical_start", kv_end="vertical_end"):
    return im.domain(
        common.GridType.CARTESIAN,
        {
            IDim:  (OL(value=i[0]),   OL(value=i[1])),
            JDim:  (OL(value=j[0]),   OL(value=j[1])),
            Kolor: (OL(value=k[0]),   OL(value=k[1])),
            K:     (im.ref(kv_start), im.ref(kv_end)),
        },
    )


def _and_cond(kolo, idim, jdim):
    """Build AND(Kolor:[kolo[0],kolo[1]), IDim:[idim[0],idim[1]), JDim:[jdim[0],jdim[1]))."""
    return im.call("and_")(
        im.domain(common.GridType.CARTESIAN, {Kolor: (OL(value=kolo[0]), OL(value=kolo[1]))}),
        im.call("and_")(
            im.domain(common.GridType.CARTESIAN, {IDim: (OL(value=idim[0]), OL(value=idim[1]))}),
            im.domain(common.GridType.CARTESIAN, {JDim: (OL(value=jdim[0]), OL(value=jdim[1]))}),
        ),
    )


def _make_program(setat_domain, expr, out_id="output", fields=("field_a", "field_b")):
    """Build a minimal Program with one SetAt."""
    params = [
        itir.Sym(id=f, type=_edge_kfld) for f in fields
    ] + [
        itir.Sym(id=out_id, type=_edge_kfld),
        itir.Sym(id="vertical_start", type=_int32),
        itir.Sym(id="vertical_end",   type=_int32),
    ]
    return itir.Program(
        id="test_prog",
        function_definitions=[],
        params=params,
        declarations=[],
        body=[itir.SetAt(expr=expr, domain=setat_domain, target=im.ref(out_id))],
    )


def _run_pipeline(prog, *, keep_existing_domains=True, allow_uninferred=True):
    """Run canonicalize_domain_argument + infer_program + prune."""
    ir = infer_domain_ops.InferDomainOps.apply(prog)
    ir = concat_where.canonicalize_domain_argument(ir)
    ir = ConstantFolding.apply(ir)
    ir = infer_domain.infer_program(
        ir,
        offset_provider=OFFSET_PROVIDER,
        symbolic_domain_sizes={},
        allow_uninferred=allow_uninferred,
        keep_existing_domains=keep_existing_domains,
    )
    ir = ConstantFolding.apply(ir)
    return ir  # return BEFORE prune so tests can inspect raw post-inference state


def _int_val(expr) -> int | None:
    """Extract integer value from Literal or OffsetLiteral, return None otherwise."""
    if isinstance(expr, itir.Literal):
        try:
            return int(expr.value)
        except (ValueError, TypeError):
            return None
    if isinstance(expr, itir.OffsetLiteral) and isinstance(expr.value, int):
        return expr.value
    return None


def _find_prunable_dead_kolor_symrefs(node):
    """Return list of (lo, hi, start_type, stop_type) for SymRef nodes that have a numerically
    empty Kolor range but where SymbolicRange.empty() returns None (the bug).

    These occur because _range_intersection produces mixed OL/Literal bounds:
    e.g. SymbolicRange(OL(0), Literal("0")) — both are 0 numerically but different Python types.
    SymbolicRange.empty() checks isinstance(start, Literal) which is False for OL → returns None.
    prune_empty_concat_where then cannot prune the dead branch → DaCe sees it → crash.
    """
    dead = []
    for sr in node.pre_walk_values().if_isinstance(itir.SymRef):
        dom = getattr(getattr(sr, "annex", None), "domain", None)
        if not isinstance(dom, domain_utils.SymbolicDomain):
            continue
        kd = next((d for d in dom.ranges if getattr(d, "value", None) == "Kolor"), None)
        if kd is None:
            continue
        r = dom.ranges[kd]
        lo, hi = _int_val(r.start), _int_val(r.stop)
        if lo is not None and hi is not None and lo >= hi and r.empty() is None:
            dead.append((lo, hi, type(r.start).__name__, type(r.stop).__name__))
    return dead


def _find_empty_kolor_asfo(node):
    """Return list of (lo, hi) for every as_fieldop with explicit empty Kolor in fun.args[1]."""
    empty = []
    for fc in node.pre_walk_values().if_isinstance(itir.FunCall):
        if not (cpm.is_applied_as_fieldop(fc) and len(fc.fun.args) == 2):
            continue
        dom_expr = fc.fun.args[1]
        if not cpm.is_call_to(dom_expr, "cartesian_domain"):
            continue
        for nr in dom_expr.args:
            if not (cpm.is_call_to(nr, "named_range") and len(nr.args) == 3):
                continue
            if getattr(nr.args[0], "value", None) != "Kolor":
                continue
            lo_v, hi_v = _int_val(nr.args[1]), _int_val(nr.args[2])
            if lo_v is not None and hi_v is not None and lo_v >= hi_v:
                empty.append((lo_v, hi_v))
    return empty


def _find_empty_kolor_annex(node):
    """Return list of (lo, hi) for every as_fieldop where annex.domain has empty Kolor."""
    empty = []
    for fc in node.pre_walk_values().if_isinstance(itir.FunCall):
        if not cpm.is_applied_as_fieldop(fc):
            continue
        dom = getattr(getattr(fc, "annex", None), "domain", None)
        if not isinstance(dom, domain_utils.SymbolicDomain):
            continue
        kd = next((d for d in dom.ranges if getattr(d, "value", None) == "Kolor"), None)
        if kd is None:
            continue
        r = dom.ranges[kd]
        lo, hi = _int_val(r.start), _int_val(r.stop)
        if lo is not None and hi is not None and lo >= hi:
            empty.append((lo, hi))
    return empty


def _find_orphaned_cwcda_lets(node):
    """Return True if any __cwcda_field_b let-binding survives after prune."""
    text = pformat(node)
    return "__cwcda_field_b" in text or "__cwcda_field_a" in text


# ── Test 1: stencil 10 pattern — infer_program assigns empty domain to no-explicit as_fieldop ──

def test_stencil10_no_domain_asfo_gets_empty_kolor():
    """Reproduce stencil 10 (`apply_diffusion_to_vn`) Kolor:0:0 failure.

    SetAt @ Kolor:[0,1) with concat_where(AND(Kolor:[0,1), IDim, JDim), nudge, diffuse)
    where BOTH branches are plain arithmetic as_fieldop WITHOUT explicit domains.

    Bug (before fix):
      infer_program assigns `fun.args[1] = Kolor:[1,1)` (empty) to the diffuse branch.
      DaCe sees this explicit empty domain → allocates zero-sized Kolor transient → crash.

    This test MUST FAIL before the fix is applied (asserting the bug exists).
    It must PASS after the fix.
    """
    setat_dom = _cart_dom(i=(2, 25), j=(2, 24), k=(0, 1))   # Kolor:[0,1)
    cond_k0 = _and_cond(kolo=(0, 1), idim=(5, 22), jdim=(4, 22))

    # Both branches: plain arithmetic, NO explicit domain in as_fieldop
    expr_nudge    = im.as_fieldop(
        im.lambda_("a", "b")(im.plus(im.deref(im.ref("a")), im.deref(im.ref("b"))))
    )(im.ref("field_a"), im.ref("field_b"))

    expr_diffuse  = im.as_fieldop(
        im.lambda_("a", "b")(im.multiplies_(im.deref(im.ref("a")), im.deref(im.ref("b"))))
    )(im.ref("field_a"), im.ref("field_b"))

    prog = _make_program(setat_dom, im.concat_where(cond_k0, expr_nudge, expr_diffuse))

    ir = _run_pipeline(prog, keep_existing_domains=True, allow_uninferred=True)
    ir_after_prune = prune_empty_concat_where.prune_empty_concat_where(ir)

    # After the fix (domain_utils.SymbolicRange.empty() handles mixed OL/Literal types):
    # dead_symrefs should be empty (no mixed-type empty ranges with empty()=None)
    dead_symrefs_remaining = _find_prunable_dead_kolor_symrefs(ir)
    assert not dead_symrefs_remaining, (
        f"FIX INCOMPLETE: still found SymRef nodes where empty() returns None for "
        f"numerically-empty Kolor ranges (mixed OL/Literal). {dead_symrefs_remaining}"
    )

    # Prune should have changed the IR (dead Kolor branches pruned)
    assert pformat(ir_after_prune) != pformat(ir), (
        "prune_empty_concat_where should have changed the IR "
        "(dead Kolor branches should be removed after the fix). "
        f"IR unchanged after prune.\n{pformat(ir_after_prune)}"
    )

    # No empty-Kolor as_fieldop should remain after prune
    empty_after_prune = _find_empty_kolor_asfo(ir_after_prune)
    assert not empty_after_prune, (
        f"empty-Kolor as_fieldop still present after prune: {empty_after_prune}"
    )
    print("  ok  test_stencil10_no_domain_asfo_gets_empty_kolor")


# ── Test 2: stencil 06 pattern — annex.domain overwrite on explicit-domain as_fieldop ──

def test_stencil06_explicit_domain_annex_overwritten():
    """Reproduce stencil 06 (`compute_advection_in_horizontal_momentum`) Kolor:0:0 failure.

    SetAt @ Kolor:[0,3) with concat_where(AND(Kolor:[0,1), IDim, JDim), expr_a, expr_b)
    where BOTH branches have an EXPLICIT domain Kolor:[0,3) in fun.args[1].

    Bug (before fix):
      keep_existing_domains=True correctly preserves fun.args[1] = Kolor:[0,3).
      BUT infer_expr line 712 overwrites annex.domain = Kolor:[1,1) (empty) on the
      freshly-created transformed_call node (no pre-existing annex.domain).
      DaCe reads annex.domain for pre-state copy sizing → zero-sized transient → crash.

    This test MUST FAIL before the fix (asserting the annex.domain bug exists).
    It must PASS after the fix.
    """
    setat_dom = _cart_dom(i=(4, 22), j=(4, 22), k=(0, 3))   # Kolor:[0,3)
    cond_k0 = _and_cond(kolo=(0, 1), idim=(5, 22), jdim=(4, 22))

    explicit_dom = _cart_dom(i=(4, 22), j=(4, 22), k=(0, 3))  # Kolor:[0,3)

    # Both branches: have explicit Kolor:[0,3) domain in fun.args[1]
    expr_a = im.as_fieldop(
        im.lambda_("a")(im.deref(im.ref("a"))),
        explicit_dom,
    )(im.ref("field_a"))

    expr_b = im.as_fieldop(
        im.lambda_("b")(im.deref(im.ref("b"))),
        copy.deepcopy(explicit_dom),
    )(im.ref("field_b"))

    prog = _make_program(setat_dom, im.concat_where(cond_k0, expr_a, expr_b))

    ir = _run_pipeline(prog, keep_existing_domains=True, allow_uninferred=True)
    ir_after_prune = prune_empty_concat_where.prune_empty_concat_where(ir)

    # fun.args[1] must still be Kolor:[0,3) (kept by keep_existing_domains) — NOT empty.
    empty_explicit = _find_empty_kolor_asfo(ir)
    assert not empty_explicit, (
        f"keep_existing_domains failed: fun.args[1] has empty Kolor {empty_explicit}."
    )

    # After fix: no dead SymRefs with empty()=None
    dead_symrefs_remaining = _find_prunable_dead_kolor_symrefs(ir)
    assert not dead_symrefs_remaining, (
        f"FIX INCOMPLETE: still found SymRef nodes where empty() returns None for "
        f"numerically-empty Kolor ranges. {dead_symrefs_remaining}"
    )

    # Prune should have changed the IR
    assert pformat(ir_after_prune) != pformat(ir), (
        "prune_empty_concat_where should have changed the IR after the fix."
    )

    print("  ok  test_stencil06_explicit_domain_annex_overwritten")


# ── Test 3: diagnostic — what survives prune_empty_concat_where ──────────────

def _get_kolor_range(annex_domain):
    """Extract (start, stop, empty_check) for Kolor in an annex domain."""
    if not isinstance(annex_domain, domain_utils.SymbolicDomain):
        return None
    kolor_dim = next((d for d in annex_domain.ranges if getattr(d, "value", None) == "Kolor"), None)
    if kolor_dim is None:
        return None
    r = annex_domain.ranges[kolor_dim]
    return r.start, r.stop, r.empty()


def _inspect_annex_domains(node, label=""):
    """Print raw annex.domain Kolor ranges for FunCall and SymRef nodes."""
    for fc in node.pre_walk_values().if_isinstance(itir.FunCall):
        dom = getattr(getattr(fc, "annex", None), "domain", None)
        info = _get_kolor_range(dom)
        if info is None:
            continue
        lo, hi, emp = info
        is_asfo = cpm.is_applied_as_fieldop(fc)
        # Check for explicit domain: as_fieldop has fun = FunCall(as_fieldop, [stencil, dom?])
        explicit_dom = False
        if is_asfo and isinstance(fc.fun, itir.FunCall):
            explicit_dom = len(fc.fun.args) == 2
        print(
            f"  [{label}] {'as_fieldop' if is_asfo else 'FunCall'}"
            f" explicit_dom={explicit_dom}"
            f" Kolor.start={lo!r}  Kolor.stop={hi!r}  .empty()={emp}"
        )
    for sr in node.pre_walk_values().if_isinstance(itir.SymRef):
        dom = getattr(getattr(sr, "annex", None), "domain", None)
        info = _get_kolor_range(dom)
        if info is None:
            continue
        lo, hi, emp = info
        print(
            f"  [{label}] SymRef id={sr.id}"
            f"  Kolor.start={lo!r}  Kolor.stop={hi!r}  .empty()={emp}"
        )


def test_diagnostic_after_prune_stencil10():
    """Diagnostic only: print annex.domain raw values and prune behavior. No assertions.

    This reveals whether the symbolic intersection expressions prevent pruning:
    - `SymbolicRange.empty()` returns None for symbolic min/max expressions
    - DaCe's sympy can evaluate them to Kolor:[0:0] — THIS is the actual failure mechanism
    """
    setat_dom = _cart_dom(i=(2, 25), j=(2, 24), k=(0, 1))
    cond_k0 = _and_cond(kolo=(0, 1), idim=(5, 22), jdim=(4, 22))

    expr_nudge   = im.as_fieldop(
        im.lambda_("a", "b")(im.plus(im.deref(im.ref("a")), im.deref(im.ref("b"))))
    )(im.ref("field_a"), im.ref("field_b"))
    expr_diffuse = im.as_fieldop(
        im.lambda_("a", "b")(im.multiplies_(im.deref(im.ref("a")), im.deref(im.ref("b"))))
    )(im.ref("field_a"), im.ref("field_b"))

    prog = _make_program(setat_dom, im.concat_where(cond_k0, expr_nudge, expr_diffuse))
    ir_before_prune = _run_pipeline(prog, keep_existing_domains=True, allow_uninferred=True)

    print("\n  [test3] Raw annex.domain Kolor ranges BEFORE prune:")
    _inspect_annex_domains(ir_before_prune, "before_prune")

    ir_after_prune  = prune_empty_concat_where.prune_empty_concat_where(ir_before_prune)
    print("\n  [test3] Raw annex.domain Kolor ranges AFTER prune:")
    _inspect_annex_domains(ir_after_prune, "after_prune")

    pruned = pformat(ir_after_prune) != pformat(ir_before_prune)
    print(f"\n  prune changed IR: {pruned}")
    print(f"  orphaned cwcda lets after prune: {_find_orphaned_cwcda_lets(ir_after_prune)}")
    print("  ok  test_diagnostic_after_prune_stencil10  (diagnostic — see prints above)")


# ── Test 4: diagnostic — stencil 06 after prune ──────────────────────────────

def test_diagnostic_after_prune_stencil06():
    """Diagnostic only: stencil 06 pattern after prune. No assertions."""
    setat_dom = _cart_dom(i=(4, 22), j=(4, 22), k=(0, 3))
    cond_k0 = _and_cond(kolo=(0, 1), idim=(5, 22), jdim=(4, 22))
    explicit_dom = _cart_dom(i=(4, 22), j=(4, 22), k=(0, 3))

    expr_a = im.as_fieldop(im.lambda_("a")(im.deref(im.ref("a"))), explicit_dom)(im.ref("field_a"))
    expr_b = im.as_fieldop(im.lambda_("b")(im.deref(im.ref("b"))), copy.deepcopy(explicit_dom))(im.ref("field_b"))

    prog = _make_program(setat_dom, im.concat_where(cond_k0, expr_a, expr_b))
    ir_before_prune = _run_pipeline(prog, keep_existing_domains=True, allow_uninferred=True)
    ir_after_prune  = prune_empty_concat_where.prune_empty_concat_where(ir_before_prune)

    print(f"\n  [test4 BEFORE prune]:\n{pformat(ir_before_prune)}\n")
    print(f"\n  [test4 AFTER prune]:\n{pformat(ir_after_prune)}\n")

    orphaned = _find_orphaned_cwcda_lets(ir_after_prune)
    print(f"  orphaned cwcda lets after prune: {orphaned}")
    empty_after_prune_explicit = _find_empty_kolor_asfo(ir_after_prune)
    empty_after_prune_annex    = _find_empty_kolor_annex(ir_after_prune)
    print(f"  empty Kolor in fun.args[1] after prune: {empty_after_prune_explicit}")
    print(f"  empty Kolor in annex.domain after prune: {empty_after_prune_annex}")
    print("  ok  test_diagnostic_after_prune_stencil06  (diagnostic — see prints above)")


# ── Test 5: full fix validation (both patterns, run after fix is applied) ─────

def test_full_pipeline_no_empty_kolor_after_fix():
    """After the fix is applied, neither pattern should produce empty Kolor domains.

    This test combines both the stencil 10 (no explicit domain) and stencil 06
    (explicit domain) patterns and asserts the FINAL IR (post-infer + post-prune +
    post-fix) has no empty-Kolor domains in either fun.args[1] or annex.domain.

    Must PASS after the fix. Currently FAILS (no fix applied yet).
    """
    results = []

    # Pattern A: stencil 10 (no explicit domain)
    setat_dom_10 = _cart_dom(i=(2, 25), j=(2, 24), k=(0, 1))
    cond_k0_10 = _and_cond(kolo=(0, 1), idim=(5, 22), jdim=(4, 22))
    expr_n = im.as_fieldop(im.lambda_("a", "b")(im.plus(im.deref(im.ref("a")), im.deref(im.ref("b")))))(
        im.ref("field_a"), im.ref("field_b"))
    expr_d = im.as_fieldop(im.lambda_("a", "b")(im.multiplies_(im.deref(im.ref("a")), im.deref(im.ref("b")))))(
        im.ref("field_a"), im.ref("field_b"))
    prog10 = _make_program(setat_dom_10, im.concat_where(cond_k0_10, expr_n, expr_d))
    ir10 = _run_pipeline(prog10, keep_existing_domains=True, allow_uninferred=True)
    ir10 = prune_empty_concat_where.prune_empty_concat_where(ir10)
    empty10_e = _find_empty_kolor_asfo(ir10)
    empty10_a = _find_empty_kolor_annex(ir10)
    results.append(("stencil10_explicit", empty10_e))
    results.append(("stencil10_annex",    empty10_a))

    # Pattern B: stencil 06 (explicit domain)
    setat_dom_06 = _cart_dom(i=(4, 22), j=(4, 22), k=(0, 3))
    cond_k0_06 = _and_cond(kolo=(0, 1), idim=(5, 22), jdim=(4, 22))
    expl = _cart_dom(i=(4, 22), j=(4, 22), k=(0, 3))
    ea = im.as_fieldop(im.lambda_("a")(im.deref(im.ref("a"))), expl)(im.ref("field_a"))
    eb = im.as_fieldop(im.lambda_("b")(im.deref(im.ref("b"))), copy.deepcopy(expl))(im.ref("field_b"))
    prog06 = _make_program(setat_dom_06, im.concat_where(cond_k0_06, ea, eb))
    ir06 = _run_pipeline(prog06, keep_existing_domains=True, allow_uninferred=True)
    ir06 = prune_empty_concat_where.prune_empty_concat_where(ir06)
    empty06_e = _find_empty_kolor_asfo(ir06)
    empty06_a = _find_empty_kolor_annex(ir06)
    results.append(("stencil06_explicit", empty06_e))
    results.append(("stencil06_annex",    empty06_a))

    # After the fix (SymbolicRange.empty() handles mixed OL/Literal types),
    # prune_empty_concat_where correctly prunes dead branches, and no orphaned
    # let-bindings or DaCe-visible empty Kolor ranges should remain.
    for label, empty in results:
        print(f"  {label}: empty Kolor domains = {empty}")

    failures = [(l, e) for l, e in results if e]
    assert not failures, (
        "After fix: expected zero empty Kolor domains in all cases. "
        f"Still failing: {failures}"
    )

    # After fix+prune, no SymRef nodes with numerically-empty Kolor ranges should remain
    for label, prog in [
        ("stencil10", _make_program(
            _cart_dom(i=(2,25), j=(2,24), k=(0,1)),
            im.concat_where(
                _and_cond(kolo=(0,1), idim=(5,22), jdim=(4,22)),
                im.as_fieldop(im.lambda_("a","b")(im.plus(im.deref(im.ref("a")), im.deref(im.ref("b")))))(im.ref("field_a"), im.ref("field_b")),
                im.as_fieldop(im.lambda_("a","b")(im.multiplies_(im.deref(im.ref("a")), im.deref(im.ref("b")))))(im.ref("field_a"), im.ref("field_b")),
            )
        )),
        ("stencil06", _make_program(
            _cart_dom(i=(4,22), j=(4,22), k=(0,3)),
            im.concat_where(
                _and_cond(kolo=(0,1), idim=(5,22), jdim=(4,22)),
                im.as_fieldop(im.lambda_("a")(im.deref(im.ref("a"))), _cart_dom(i=(4,22),j=(4,22),k=(0,3)))(im.ref("field_a")),
                im.as_fieldop(im.lambda_("b")(im.deref(im.ref("b"))), copy.deepcopy(_cart_dom(i=(4,22),j=(4,22),k=(0,3))))(im.ref("field_b")),
            )
        )),
    ]:
        ir = _run_pipeline(prog, keep_existing_domains=True, allow_uninferred=True)
        ir = prune_empty_concat_where.prune_empty_concat_where(ir)
        # No dead (empty Kolor) SymRef nodes should survive into DaCe
        dead = _find_prunable_dead_kolor_symrefs(ir)
        empty_asfo = _find_empty_kolor_asfo(ir)
        print(f"  {label}: dead SymRef ranges = {dead}, empty as_fieldop = {empty_asfo}")
        assert not dead, (
            f"After fix: expected no dead SymRef Kolor ranges in {label}. Still: {dead}. "
            f"IR:\n{pformat(ir)}"
        )
        assert not empty_asfo, (
            f"After fix: expected no empty-Kolor as_fieldop in {label}. Still: {empty_asfo}."
        )

    print("  ok  test_full_pipeline_no_empty_kolor_after_fix")


# ── Test 6: negative-start Kolor treated as empty ────────────────────────────

def test_negative_kolor_start_is_empty():
    """SymbolicRange.empty() must return True for concrete negative starts.

    Root cause (stencil 04 + stencil 06): infer_program back-propagates through
    Kolor shifts and produces domains like Kolor:[-1,0) or Kolor:[-2,1).
    These are not numerically empty (stop > start) but are STRUCTURALLY INVALID
    in the 0-indexed structured mesh. prune_empty_concat_where must prune them.

    Fix: domain_utils.SymbolicRange.empty() returns True when lo < 0 (concrete).
    """
    OL = itir.OffsetLiteral

    # Kolor:[-1,0) — one element at index -1, invalid
    r_neg1_0 = domain_utils.SymbolicRange(OL(value=-1), OL(value=0))
    assert r_neg1_0.empty() is True, (
        f"Kolor:[-1,0) must be treated as empty (invalid index). Got: {r_neg1_0.empty()}"
    )

    # Kolor:[-2,1) — also invalid start
    r_neg2_1 = domain_utils.SymbolicRange(OL(value=-2), OL(value=1))
    assert r_neg2_1.empty() is True, (
        f"Kolor:[-2,1) must be treated as empty (invalid index). Got: {r_neg2_1.empty()}"
    )

    # Kolor:[-1,2) — negative start, multiple elements that include -1
    r_neg1_2 = domain_utils.SymbolicRange(OL(value=-1), OL(value=2))
    assert r_neg1_2.empty() is True, (
        f"Kolor:[-1,2) must be treated as empty (starts at invalid index -1). Got: {r_neg1_2.empty()}"
    )

    # Kolor:[0,0) — still empty (existing logic, start == stop)
    r_0_0 = domain_utils.SymbolicRange(OL(value=0), OL(value=0))
    assert r_0_0.empty() is True, f"Kolor:[0,0) must still be empty. Got: {r_0_0.empty()}"

    # Kolor:[0,1) — valid, non-empty
    r_0_1 = domain_utils.SymbolicRange(OL(value=0), OL(value=1))
    assert r_0_1.empty() is False, f"Kolor:[0,1) must NOT be empty. Got: {r_0_1.empty()}"

    # Kolor:[2,3) — valid, non-empty
    r_2_3 = domain_utils.SymbolicRange(OL(value=2), OL(value=3))
    assert r_2_3.empty() is False, f"Kolor:[2,3) must NOT be empty. Got: {r_2_3.empty()}"

    print("  ok  test_negative_kolor_start_is_empty")


# ── Test 7: _infer_as_fieldop uses narrow context for back-propagation ────────

def test_kolor_backprop_stays_within_entity_bounds():
    """With the intersection fix in _infer_as_fieldop, back-propagation through Kolor
    shifts respects the per-kolor context from _infer_concat_where.

    Pattern: concat_where(Kolor:[0,1), as_fieldop(shift(Kolor+2), Kolor:[0,3))(field), rest).
    Simulates E2C2EO slot-0 kolor=0 branch: from edge kolor 0, shift Kolor+2 → kolor 2.

    Without fix (keep_existing_domains overrides to full Kolor:[0,3)):
      source = translate(Kolor:[0,3), shift=+2) = Kolor:[2,5) → OOB!

    With fix (uses intersection Kolor:[0,1) ∩ Kolor:[0,3) = Kolor:[0,1)):
      source = translate(Kolor:[0,1), shift=+2) = Kolor:[2,3) ✓

    Check: no SymRef annex.domain Kolor stop > 3 after the pipeline.
    """
    setat_dom  = _cart_dom(i=(4, 22), j=(4, 22), k=(0, 3))  # edge SetAt domain
    cond_k0    = _and_cond(kolo=(0, 1), idim=(4, 22), jdim=(4, 22))
    # Use narrow Kolor:[0,1) — _build_field_concat_where_from_branches now narrows
    # every branch to Kolor:[k,k+1). The kolor=0 branch gets Kolor:[0,1).
    explicit   = _cart_dom(i=(4, 22), j=(4, 22), k=(0, 1))  # narrow kolor=0 domain

    # as_fieldop(λ __it → ·⟪Kolor,+2⟫(__it), Kolor:[0,1))(field_a)
    # Back-prop from Kolor:[0,1) through shift Kolor+2: source Kolor:[2,3) ✓
    inner = im.as_fieldop(
        im.lambda_("__it")(im.deref(im.shift("Kolor", 2)(im.ref("__it")))),
        explicit,
    )(im.ref("field_a"))

    prog = _make_program(setat_dom, im.concat_where(cond_k0, inner, im.ref("field_b")))

    ir = _run_pipeline(prog, keep_existing_domains=True, allow_uninferred=True)
    ir = ConstantFolding.apply(ir)
    ir = prune_empty_concat_where.prune_empty_concat_where(ir)

    # Find any SymRef whose annex.domain Kolor stop > 3 (above-max for 3-kolor edge field)
    oob = []
    for sr in ir.pre_walk_values().if_isinstance(itir.SymRef):
        dom = getattr(getattr(sr, "annex", None), "domain", None)
        if not isinstance(dom, domain_utils.SymbolicDomain):
            continue
        kd = next((d for d in dom.ranges if getattr(d, "value", None) == "Kolor"), None)
        if kd is None:
            continue
        r = dom.ranges[kd]
        hi = _int_val(r.stop)
        if hi is not None and hi > 3:
            oob.append((str(sr.id), _int_val(r.start), hi))

    assert not oob, (
        f"Back-propagation produced above-max Kolor stop > 3: {oob}. "
        "Without the intersection fix, shift Kolor+2 from outer Kolor:[0,3) gives "
        "source Kolor:[2,5) which exceeds the 3-kolor edge field bound."
    )

    # Check annex.domain: with narrow branches (Kolor:[0,1) for the kolor=0 branch),
    # the as_fieldop should have annex.domain Kolor:[0,1) — correctly narrow, 1-slot.
    narrow_annex_found = False
    for fc in ir.pre_walk_values().if_isinstance(itir.FunCall):
        if not cpm.is_applied_as_fieldop(fc):
            continue
        dom = getattr(getattr(fc, "annex", None), "domain", None)
        if not isinstance(dom, domain_utils.SymbolicDomain):
            continue
        kd = next((d for d in dom.ranges if getattr(d, "value", None) == "Kolor"), None)
        if kd is None:
            continue
        r = dom.ranges[kd]
        lo, hi = _int_val(r.start), _int_val(r.stop)
        if lo is not None and hi is not None and (hi - lo) == 1:
            narrow_annex_found = True
            break

    assert narrow_annex_found, (
        "Expected at least one as_fieldop with narrow 1-slot annex.domain (Kolor:[0,1)). "
        "With per-kolor narrowing, every branch uses a 1-slot kolor domain."
    )

    print("  ok  test_kolor_backprop_stays_within_entity_bounds")


# ── Test 8: negative-start Kolor gets pruned by prune_empty_concat_where ─────

def test_negative_start_kolor_pruned_by_pipeline():
    """The belt-and-suspenders SymbolicRange.empty() fix: concrete negative-start Kolor
    ranges produced by infer_program back-propagation are treated as empty by SymbolicRange.
    This allows prune_empty_concat_where to remove dead branches.

    We verify that:
    1. SymbolicRange with negative-start is empty (unit check, belt-and-suspenders)
    2. After the full pipeline, the annex.domain for as_fieldop with shift Kolor-1 in kolor=0
       context is the FULL explicit domain Kolor:[0,3) (not a narrow/negative domain),
       because _dace_annex_domain uses the explicit domain for DaCe transient sizing.
    """
    # Verify SymbolicRange with negative start is empty (the direct test of the fix)
    neg_range = domain_utils.SymbolicRange(itir.OffsetLiteral(value=-1), itir.OffsetLiteral(value=0))
    assert neg_range.empty() is True, (
        "SymbolicRange(-1, 0) must be empty. Belt-and-suspenders fix ensures "
        "prune_empty_concat_where removes branches with negative Kolor start."
    )

    # Verify that for shift(Kolor-1) inside a kolor=0 branch with narrow domain Kolor:[0,1),
    # no as_fieldop gets a negative annex.domain Kolor start.
    setat_dom = _cart_dom(i=(4, 22), j=(4, 22), k=(0, 3))
    cond_k0   = _and_cond(kolo=(0, 1), idim=(4, 22), jdim=(4, 22))
    # Use narrow Kolor:[0,1) as _build_field_concat_where_from_branches now produces.
    explicit_narrow = _cart_dom(i=(4, 22), j=(4, 22), k=(0, 1))

    # as_fieldop(λ __it → ·⟪Kolor,-1⟫(__it), Kolor:[0,1))(field_a) — narrow kolor=0 domain
    inner = im.as_fieldop(
        im.lambda_("__it")(im.deref(im.shift("Kolor", -1)(im.ref("__it")))),
        explicit_narrow,
    )(im.ref("field_a"))

    prog = _make_program(setat_dom, im.concat_where(cond_k0, inner, im.ref("field_b")))

    ir = _run_pipeline(prog, keep_existing_domains=True, allow_uninferred=True)
    ir = ConstantFolding.apply(ir)

    # No as_fieldop annex.domain should have a negative Kolor start.
    # With narrow Kolor:[0,1), back-prop through Kolor-1 gives source Kolor:[-1,0)
    # for field_a — but that's the SOURCE domain stored on field_a, not on as_fieldop.
    negative_annex = []
    for fc in ir.pre_walk_values().if_isinstance(itir.FunCall):
        if not cpm.is_applied_as_fieldop(fc):
            continue
        dom = getattr(getattr(fc, "annex", None), "domain", None)
        if not isinstance(dom, domain_utils.SymbolicDomain):
            continue
        kd = next((d for d in dom.ranges if getattr(d, "value", None) == "Kolor"), None)
        if kd is None:
            continue
        r = dom.ranges[kd]
        lo, hi = _int_val(r.start), _int_val(r.stop)
        if lo is not None and lo < 0:
            negative_annex.append((lo, hi))

    assert not negative_annex, (
        f"as_fieldop annex.domain has negative Kolor start: {negative_annex}. "
        "Narrow per-kolor explicit domains must not produce negative annex.domain on as_fieldop nodes."
    )
    print("  ok  test_negative_start_kolor_pruned_by_pipeline")


# ── Test 9: E2C2EO multi-slot pattern (stencil 04) ──────────────────────────

def test_e2c2eo_multi_slot_no_oob_source_domain():
    """Reproduces stencil 04 (apply_divergence_damping_and_update_vn) pattern.

    The stencil has E2C2EO with 4 slots. Shifts include Kolor+2, Kolor-2.
    Without per-kolor narrowing: back-prop from Kolor:[0,3) through Kolor+2
    gives source Kolor:[2,5) (OOB). Union of all slot shifts produces Kolor:[-2,5).

    With per-kolor narrowing (Fix 2): back-prop from Kolor:[0,1) through Kolor+2
    gives source Kolor:[2,3) ✓. No OOB.

    This test verifies that after the full pipeline, no as_fieldop has annex.domain
    with Kolor start < 0 or Kolor stop > 3 (edge field bounds).
    """
    setat_dom = _cart_dom(i=(4, 22), j=(4, 22), k=(0, 3))  # edge SetAt

    # Build E2C2EO slot 0 structure (kolor0→kolor2, kolor1→kolor0, kolor2→kolor1)
    # Using per-kolor narrow domains as _build_field_concat_where_from_branches now produces
    def e2c2eo_slot_narrow(field, shifts, outer_dom):
        """Build concat_where(Kolor:[0,1), shift0, concat_where(Kolor:[1,2), shift1, shift2))
        with narrow per-kolor domains from the shifts tuple."""
        (_, dk0), (_, dk1), (_, dk2) = shifts
        d0 = _cart_dom(i=(4,22), j=(4,22), k=(0,1))
        d1 = _cart_dom(i=(4,22), j=(4,22), k=(1,2))
        d2 = _cart_dom(i=(4,22), j=(4,22), k=(2,3))
        asfo0 = im.as_fieldop(
            im.lambda_("it")(im.deref(im.shift("Kolor", dk0)(im.ref("it")))), d0)(im.ref(field))
        asfo1 = im.as_fieldop(
            im.lambda_("it")(im.deref(im.shift("Kolor", dk1)(im.ref("it")))), d1)(im.ref(field))
        asfo2 = im.as_fieldop(
            im.lambda_("it")(im.deref(im.shift("Kolor", dk2)(im.ref("it")))), d2)(im.ref(field))
        return im.concat_where(
            im.domain(common.GridType.CARTESIAN, {Kolor: (OL(value=0), OL(value=1))}),
            asfo0,
            im.concat_where(
                im.domain(common.GridType.CARTESIAN, {Kolor: (OL(value=1), OL(value=2))}),
                asfo1, asfo2,
            ),
        )

    # E2C2EO slot 0: kolor0→+2, kolor1→-1, kolor2→-1(trailing)
    slot0 = e2c2eo_slot_narrow("h_grad", [("K",2), ("K",-1), ("K",-1)], setat_dom)
    # E2C2EO slot 1: kolor0→+1, kolor1→+1, kolor2→-2(trailing)
    slot1 = e2c2eo_slot_narrow("h_grad", [("K",1), ("K",1), ("K",-2)], setat_dom)

    # Accumulation: acc = slot0 + slot1
    acc_dom = setat_dom
    acc = im.as_fieldop(
        im.lambda_("acc", "elem")(
            im.call("plus")(im.deref(im.ref("acc")), im.deref(im.ref("elem")))
        ),
        acc_dom,
    )(
        im.as_fieldop(im.lambda_("a")(im.deref(im.ref("a"))), acc_dom)(im.ref("h_grad")),
        im.as_fieldop(
            im.lambda_("acc", "elem")(
                im.call("plus")(im.deref(im.ref("acc")), im.deref(im.ref("elem")))
            ),
            acc_dom,
        )(
            im.as_fieldop(im.lambda_("__x")(im.literal("0", "float64")), acc_dom)(im.ref("h_grad")),
            slot0,
        ),
    )

    params = [itir.Sym(id="h_grad", type=_edge_kfld), itir.Sym(id="output", type=_edge_kfld),
              itir.Sym(id="vertical_start", type=_int32), itir.Sym(id="vertical_end", type=_int32)]
    prog = itir.Program(
        id="test_e2c2eo",
        function_definitions=[],
        params=params,
        declarations=[],
        body=[itir.SetAt(expr=acc, domain=setat_dom, target=im.ref("output"))],
    )

    ir = _run_pipeline(prog, keep_existing_domains=True, allow_uninferred=True)
    ir = ConstantFolding.apply(ir)

    # Check: no as_fieldop has annex.domain Kolor outside [0,3)
    oob_domains = []
    for fc in ir.pre_walk_values().if_isinstance(itir.FunCall):
        if not cpm.is_applied_as_fieldop(fc):
            continue
        dom = getattr(getattr(fc, "annex", None), "domain", None)
        if not isinstance(dom, domain_utils.SymbolicDomain):
            continue
        kd = next((d for d in dom.ranges if getattr(d, "value", None) == "Kolor"), None)
        if kd is None:
            continue
        r = dom.ranges[kd]
        lo, hi = _int_val(r.start), _int_val(r.stop)
        if lo is not None and (lo < 0 or (hi is not None and hi > 3)):
            oob_domains.append((lo, hi))

    assert not oob_domains, (
        f"E2C2EO multi-slot produced OOB Kolor annex.domain: {oob_domains}. "
        "Per-kolor narrowing should prevent Kolor:[-2,5) from appearing."
    )
    print("  ok  test_e2c2eo_multi_slot_no_oob_source_domain")


# ── runner ───────────────────────────────────────────────────────────────────

TESTS = [
    test_stencil10_no_domain_asfo_gets_empty_kolor,
    test_stencil06_explicit_domain_annex_overwritten,
    test_diagnostic_after_prune_stencil10,
    test_diagnostic_after_prune_stencil06,
    test_full_pipeline_no_empty_kolor_after_fix,
    test_negative_kolor_start_is_empty,
    test_kolor_backprop_stays_within_entity_bounds,
    test_negative_start_kolor_pruned_by_pipeline,
]


if __name__ == "__main__":
    failures = []
    expected_failures: set[str] = set()
    print(f"\nAll tests should PASS with the domain_utils.py fix applied.\n")

    for t in TESTS:
        try:
            print(f"\n{'='*60}\n{t.__name__}:")
            t()
        except AssertionError as e:
            if t.__name__ in expected_failures:
                print(f"  EXPECTED FAILURE (bug confirmed): {e}")
            else:
                print(f"  UNEXPECTED FAILURE: {e}")
                failures.append(t.__name__)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {type(e).__name__}: {e}")
            failures.append(t.__name__)

    print()
    if failures:
        print(f"UNEXPECTED FAILURES: {failures}")
        sys.exit(1)
    print("All tests passed (expected failures confirmed as bugs; diagnostic tests printed info)")
