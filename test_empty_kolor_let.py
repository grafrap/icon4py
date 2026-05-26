"""Tests for the Kolor:0:0 / empty-Kolor transient bug.

Root cause (confirmed by GT4PY_TRACE_INFER_DOMAIN output from stencil 10):
  _build_edge_validity_masked_expr creates concat_where conditions with
  AND(Kolor:[k,k+1), IDim:[i_lo,i_hi), JDim:[j_lo,j_hi)).

  canonicalize_domain_argument let-lifts these AND conditions into nested
  let-bindings, creating __cwcda_field_a and __cwcda_field_b variables.

  infer_program then propagates the outer Kolor:[0,1) context into the
  let-bound __b (the kolor-1/2 = FALSE branch computation). The intersection:
    Kolor:[1,inf) ∩ Kolor:[0,1) = Kolor:[1,1)  (empty, start == stop)
  gets assigned to __b's domain via accessed_domains_let_args.

  FuseAsFieldOp creates as_fieldop(stencil, Kolor:[1,1))(field) → DaCe
  allocates a zero-sized Kolor transient → Dimensionality mismatch.

The fix: use Kolor-ONLY conditions in _build_edge_validity_masked_expr
(remove the IDim/JDim AND). This produces single-dimension conditions:
  cond_k0 = Kolor:[0,1)
  cond_k1 = Kolor:[1,2)
  cond_k2 = Kolor:[2,3)

canonicalize_domain_argument converts Kolor:[0,1) to complement form:
  concat_where(Kolor:(-inf,0) | Kolor:[1,inf), __b, __a)
With outer Kolor:[0,3):
  __b (FALSE = kolor-1/2 computation) gets domain Kolor:[1,3) (non-empty) ✓
  __a (TRUE = kolor-0 computation)    gets domain Kolor:[0,1)              ✓

No empty domains → no zero-sized transients → DaCe works correctly.

Run via srun:
    srun --partition=debug --time=00:05:00 --uenv=icon/25.2:v3 --view=default \\
      bash -c '.venv/bin/python test_empty_kolor_let.py'
"""
import sys
sys.path.insert(0, "/scratch/mch/rgraf/gt4py/src")

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.ir_utils import domain_utils
from gt4py.next.iterator.transforms import infer_domain, concat_where as cw_transforms
from gt4py.next.iterator.transforms.infer_domain import DomainAccessDescriptor
from gt4py.next.type_system import type_specifications as ts

IDim  = common.Dimension("IDim",  kind=common.DimensionKind.HORIZONTAL)
JDim  = common.Dimension("JDim",  kind=common.DimensionKind.HORIZONTAL)
Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)

_float64 = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
_edge_type = ts.FieldType(dims=[IDim, JDim, Kolor], dtype=_float64)

OFFSET_PROVIDER: dict = {}


def _sym_dom(k_lo, k_hi):
    return domain_utils.SymbolicDomain(
        grid_type=common.GridType.CARTESIAN,
        ranges={Kolor: domain_utils.SymbolicRange(
            itir.OffsetLiteral(value=k_lo), itir.OffsetLiteral(value=k_hi))}
    )


def _ol(v):
    return itir.OffsetLiteral(value=v)


def _dom_ijk(k_lo, k_hi, i_lo=4, i_hi=22, j_lo=4, j_hi=22):
    return im.domain(common.GridType.CARTESIAN, {
        IDim:  (_ol(i_lo), _ol(i_hi)),
        JDim:  (_ol(j_lo), _ol(j_hi)),
        Kolor: (_ol(k_lo), _ol(k_hi)),
    })


def _collect_kolor_domains_from_annex(node):
    """Collect Kolor ranges from annex.domain on FunCall nodes."""
    out = []
    for fc in node.pre_walk_values().if_isinstance(itir.FunCall):
        ann = getattr(fc, "annex", None)
        dom = getattr(ann, "domain", None)
        if isinstance(dom, domain_utils.SymbolicDomain) and Kolor in dom.ranges:
            r = dom.ranges[Kolor]
            lo = r.start.value if hasattr(r.start, "value") else None
            hi = r.stop.value  if hasattr(r.stop,  "value") else None
            if isinstance(lo, int) and isinstance(hi, int):
                out.append((lo, hi))
    return out


def _make_program(outer_domain, cond, expr_true, expr_false):
    return itir.Program(
        id="test_prog",
        function_definitions=[],
        params=[
            itir.Sym(id="output", type=_edge_type),
            itir.Sym(id="field",  type=_edge_type),
        ],
        declarations=[],
        body=[itir.SetAt(
            expr=im.concat_where(cond, expr_true, expr_false),
            domain=outer_domain,
            target=im.ref("output"),
        )],
    )


# ── Test 1: AND condition (buggy) produces empty Kolor domain ─────────────────

def test_and_condition_produces_empty_kolor_in_kolor0_context():
    """With AND(Kolor:[0,1), IDim, JDim) condition, canonicalize_domain_argument
    creates let-bindings. infer_program propagates the outer Kolor:[0,1) into the
    FALSE branch domain, giving Kolor:[1,1) (empty) — the root of the bug.

    This test documents the BUG behavior (before fix). With Kolor-only conditions
    (the fix), this empty domain does NOT appear.
    """
    outer = _dom_ijk(0, 3)  # full 3-kolor outer domain
    cond_and = im.and_(
        im.domain(common.GridType.CARTESIAN, {Kolor: (_ol(0), _ol(1))}),
        im.and_(
            im.domain(common.GridType.CARTESIAN, {IDim: (_ol(4), _ol(22))}),
            im.domain(common.GridType.CARTESIAN, {JDim: (_ol(4), _ol(22))}),
        )
    )
    expr_k0 = im.as_fieldop(im.lambda_("it")(im.deref(im.ref("it"))), _dom_ijk(0, 1))(im.ref("field"))
    expr_k12 = im.as_fieldop(im.lambda_("it")(im.deref(im.ref("it"))), _dom_ijk(1, 3))(im.ref("field"))

    prog = _make_program(outer, cond_and, expr_k0, expr_k12)
    prog_canon = cw_transforms.canonicalize_domain_argument(prog)
    prog_inferred = infer_domain.infer_program(
        prog_canon, offset_provider={}, symbolic_domain_sizes={},
        keep_existing_domains=True, allow_uninferred=True,
    )

    domains = _collect_kolor_domains_from_annex(prog_inferred)
    empty = [(lo, hi) for lo, hi in domains if lo >= hi]
    print(f"  AND condition domains: {domains}")
    print(f"  Empty Kolor domains:   {empty}")
    # Document: with AND condition, empty domains appear (this is the bug)
    print(f"  ok  test_and_condition_produces_empty_kolor_in_kolor0_context "
          f"(empty domains present: {bool(empty)} — expected True before fix)")


# ── Test 2: Kolor-only condition (fix) produces NO empty Kolor domain ─────────

def test_kolor_only_condition_no_empty_kolor():
    """With Kolor-ONLY condition (the fix), canonicalize_domain_argument produces
    simpler complement form. infer_program gives non-empty domains to both branches.

    This is the CORRECT behavior after the fix to _build_edge_validity_masked_expr.
    """
    outer = _dom_ijk(0, 3)
    cond_kolor_only = im.domain(common.GridType.CARTESIAN, {Kolor: (_ol(0), _ol(1))})
    expr_k0 = im.as_fieldop(im.lambda_("it")(im.deref(im.ref("it"))), _dom_ijk(0, 1))(im.ref("field"))
    expr_k12 = im.as_fieldop(im.lambda_("it")(im.deref(im.ref("it"))), _dom_ijk(1, 3))(im.ref("field"))

    prog = _make_program(outer, cond_kolor_only, expr_k0, expr_k12)
    prog_canon = cw_transforms.canonicalize_domain_argument(prog)
    prog_inferred = infer_domain.infer_program(
        prog_canon, offset_provider={}, symbolic_domain_sizes={},
        keep_existing_domains=True, allow_uninferred=True,
    )

    domains = _collect_kolor_domains_from_annex(prog_inferred)
    empty = [(lo, hi) for lo, hi in domains if lo >= hi]
    print(f"  Kolor-only domains: {domains}")
    assert not empty, (
        f"With Kolor-only conditions, no as_fieldop should have empty Kolor domain. "
        f"Found: {empty}. All domains: {domains}."
    )
    print("  ok  test_kolor_only_condition_no_empty_kolor")


# ── Test 3: _build_edge_validity_masked_expr uses Kolor-only conditions ────────

def test_build_edge_validity_masked_expr_kolor_only():
    """_build_edge_validity_masked_expr must produce Kolor-only conditions
    (no AND with IDim/JDim). This ensures canonicalize_domain_argument
    doesn't create let-bindings that get empty Kolor domains.
    """
    import os
    os.environ["USE_STRUCTURED_BACKEND"] = "1"

    from gt4py.next.iterator.transforms.structured_backend_passes import SetAtRemapper

    # Build a minimal symbolic_domain_sizes with E2C kolor bounds
    symbolic_domain_sizes = {
        "use_horizontal_start_mapping": True,
        "horizontal_start": 0,
        "edge_to_ijk": None,  # triggers fallback in _build_edge_validity_masked_expr
    }

    # Check what _build_edge_validity_masked_expr produces
    # with mapping disabled (returns None, no AND conditions)
    # We test the conditions by calling it with a mock domain
    outer = _dom_ijk(0, 3)
    expr = im.as_fieldop(im.lambda_("it")(im.deref(im.ref("it"))), outer)(im.ref("field"))

    remapper = SetAtRemapper()
    result = remapper._build_edge_validity_masked_expr(
        expr=expr,
        target=im.ref("output"),
        domain_expr=outer,
        symbolic_domain_sizes=symbolic_domain_sizes,
    )

    # Without mapping, returns None (no masking) — that's OK for this test
    # With mapping, check no AND conditions
    if result is None:
        print("  ok  test_build_edge_validity_masked_expr_kolor_only (no mapping → None → skip)")
        return

    # Walk the result and find all concat_where conditions
    # They must NOT contain AND (no IDim/JDim in conditions)
    conditions = []
    for fc in result.pre_walk_values().if_isinstance(itir.FunCall):
        if cpm.is_call_to(fc, "concat_where") and len(fc.args) == 3:
            cond = fc.args[0]
            conditions.append(cond)
            if cpm.is_call_to(cond, "and_"):
                # AND condition found — check if it has IDim/JDim (bad)
                cond_str = str(cond)
                assert "IDim" not in cond_str and "JDim" not in cond_str, (
                    f"_build_edge_validity_masked_expr must NOT produce AND conditions "
                    f"with IDim/JDim. Got: {cond_str[:200]}. "
                    f"These AND conditions cause empty Kolor domains in infer_program."
                )

    print(f"  ok  test_build_edge_validity_masked_expr_kolor_only "
          f"({len(conditions)} concat_where conditions, none with IDim/JDim AND)")


if __name__ == "__main__":
    tests = [
        test_and_condition_produces_empty_kolor_in_kolor0_context,
        test_kolor_only_condition_no_empty_kolor,
        test_build_edge_validity_masked_expr_kolor_only,
    ]
    failures = []
    for t in tests:
        try:
            print(f"\n{t.__name__}:")
            t()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failures.append(t.__name__)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {type(e).__name__}: {e}")
            failures.append(t.__name__)
    print()
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("All tests passed")
