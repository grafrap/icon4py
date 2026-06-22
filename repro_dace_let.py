"""Minimal reproducers: does DaCe correctly lower shared scalar let-bindings
inside an as_fieldop with TUPLE output? Tests both simple arithmetic and
list_get-on-sparse-field patterns (what CSE actually extracts).


Stencil: out0 = x*x + 1, out1 = x*x + 2, where x*x is a shared let-binding _cs.
If DaCe's is_let handling is buggy for tuple outputs, the result won't match.
"""

import numpy as np

from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import infer_domain, pass_manager
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.program_processors.runners.dace import lowering as dace_lowering

IDim = gtx_common.Dimension("IDim", kind=gtx_common.DimensionKind.HORIZONTAL)
KDim = gtx_common.Dimension("KDim", kind=gtx_common.DimensionKind.VERTICAL)
FLOAT = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
FTYPE = ts.FieldType(dims=[IDim], dtype=FLOAT)
N = 10


def build(ir, offset_provider):
    ir = infer_domain.infer_program(
        ir, offset_provider=offset_provider,
        symbolic_domain_sizes=pass_manager._max_domain_range_sizes(offset_provider),
    )
    opt = gtx_common.offset_provider_to_type(offset_provider)
    return dace_lowering.build_sdfg_from_gtir(ir, opt, column_axis=KDim)


def main():
    domain = im.get_field_domain(gtx_common.GridType.CARTESIAN, "x", [IDim])

    # Stencil body mimicking nabla4: a shared compound subexpression _cs used in BOTH
    # tuple outputs with different coefficients (like ptr_coeff_1 vs ptr_coeff_2):
    #   _cs = 4.0 * (deref(a)*deref(a) - 2.0*deref(a))   [stands in for nabla4]
    #   out0 = _cs * 1.5 ; out1 = _cs * 2.5
    nabla_like = im.multiplies_(
        4.0,
        im.minus(
            im.multiplies_(im.deref("a"), im.deref("a")),
            im.multiplies_(2.0, im.deref("a")),
        ),
    )
    inner = im.let("_cs", nabla_like)(
        im.make_tuple(
            im.multiplies_(im.ref("_cs"), 1.5),
            im.multiplies_(im.ref("_cs"), 2.5),
        )
    )
    stencil = im.lambda_("a")(inner)

    testee = gtir.Program(
        id="repro_let",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=FTYPE),
            gtir.Sym(id="out0", type=FTYPE),
            gtir.Sym(id="out1", type=FTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.as_fieldop(stencil, domain)("x"),
                domain=im.make_tuple(domain, domain),
                target=im.make_tuple("out0", "out1"),
            )
        ],
    )

    offset_provider = {}
    sdfg = build(testee, offset_provider)

    x = np.random.rand(N)
    out0 = np.zeros(N)
    out1 = np.zeros(N)

    FSYMBOLS = dict(
        __x_IDim_range_0=0, __x_IDim_range_1=N, __x_IDim_stride=1,
        __out0_IDim_range_0=0, __out0_IDim_range_1=N, __out0_IDim_stride=1,
        __out1_IDim_range_0=0, __out1_IDim_range_1=N, __out1_IDim_stride=1,
    )
    sdfg(x=x, out0=out0, out1=out1, **FSYMBOLS)

    cs = 4.0 * (x * x - 2.0 * x)
    exp0 = cs * 1.5
    exp1 = cs * 2.5
    ok0 = np.allclose(out0, exp0)
    ok1 = np.allclose(out1, exp1)
    print(f"out0 match: {ok0}  (max diff {np.max(np.abs(out0-exp0)):.2e})")
    print(f"out1 match: {ok1}  (max diff {np.max(np.abs(out1-exp1)):.2e})")
    if ok0 and ok1:
        print("RESULT: PASS — DaCe handles shared scalar let in tuple output correctly")
    else:
        print("RESULT: FAIL — DaCe MISHANDLES shared scalar let in tuple output")


def main_listget():
    """Reproducer 2: shared list_get result — what CSE actually extracts.

    CSE extracts `list_get(0, deref(sparse_param))` (used in multiple V2E edges)
    as a shared let-binding. This tests whether DaCe handles list_get-derived
    ValueExpr in a multi-use let context.
    """
    # Use a sparse-local field: Field[IDim, V2EDim] (6 neighbors)
    V2EDim = gtx_common.Dimension("V2EDim", kind=gtx_common.DimensionKind.LOCAL)
    SFTYPE = ts.FieldType(dims=[IDim, V2EDim], dtype=FLOAT)
    FTYPE2 = ts.FieldType(dims=[IDim], dtype=FLOAT)
    domain = im.get_field_domain(gtx_common.GridType.CARTESIAN, "s", [IDim])

    # _cs_slot0 = list_get(0, deref(s))  shared between out0 and out1
    inner = im.let("_cs_slot0",
        im.call("list_get")(0, im.deref("s"))
    )(
        im.make_tuple(
            im.multiplies_(im.ref("_cs_slot0"), 1.0),  # out0 = slot0 * 1
            im.multiplies_(im.ref("_cs_slot0"), 2.0),  # out1 = slot0 * 2
        )
    )
    stencil = im.lambda_("s")(inner)

    testee = gtir.Program(
        id="repro_listget_let",
        function_definitions=[],
        params=[
            gtir.Sym(id="sparse", type=SFTYPE),
            gtir.Sym(id="out0", type=FTYPE2),
            gtir.Sym(id="out1", type=FTYPE2),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.as_fieldop(stencil, domain)("sparse"),
                domain=im.make_tuple(domain, domain),
                target=im.make_tuple("out0", "out1"),
            )
        ],
    )

    from gt4py.next import common as gtx_common2
    offset_provider = {"V2E": gtx_common2.Dimension("V2EDim", kind=gtx_common2.DimensionKind.LOCAL)}
    try:
        sdfg = build(testee, offset_provider)

        sparse = np.random.rand(N, 6)
        out0 = np.zeros(N)
        out1 = np.zeros(N)
        FSYMBOLS = dict(
            __sparse_IDim_range_0=0, __sparse_IDim_range_1=N, __sparse_IDim_stride=6,
            __sparse_V2EDim_range_0=0, __sparse_V2EDim_range_1=6, __sparse_V2EDim_stride=1,
            __out0_IDim_range_0=0, __out0_IDim_range_1=N, __out0_IDim_stride=1,
            __out1_IDim_range_0=0, __out1_IDim_range_1=N, __out1_IDim_stride=1,
        )
        sdfg(sparse=sparse, out0=out0, out1=out1, **FSYMBOLS)
        exp0 = sparse[:, 0] * 1.0
        exp1 = sparse[:, 0] * 2.0
        ok0 = np.allclose(out0, exp0)
        ok1 = np.allclose(out1, exp1)
        print(f"[listget] out0 match: {ok0}  out1 match: {ok1}")
        if ok0 and ok1:
            print("[listget] PASS — list_get shared let works")
        else:
            print("[listget] FAIL — list_get shared let broken")
    except Exception as e:
        print(f"[listget] ERROR: {e}")


if __name__ == "__main__":
    print("=== Test 1: arithmetic shared let ===")
    main()
    print()
    print("=== Test 2: list_get shared let ===")
    main_listget()
