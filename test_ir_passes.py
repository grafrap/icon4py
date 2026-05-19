"""Unit-test style reproducer for the Kolor widening bug.

Builds a minimal IR matching the failing pattern from the stripped v3 stencil
(z_graddiv_vn + (dwdz(E2C[1]) - dwdz(E2C[0]))) and runs each suspect pass on it
individually, printing the IR before/after each pass. This lets us pinpoint
exactly which pass introduces the Kolor:[0,5) / Kolor:[2,5) widening without
having to recompile DaCe on the GPU.

Run with:
    cd /scratch/mch/rgraf/icon4py
    python3 test_ir_passes.py
"""

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import (
    infer_domain,
    infer_domain_ops,
    concat_where,
)
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator.pretty_printer import pformat as pretty_format
from gt4py.next.type_system import type_specifications as ts


# ---------- dimensions, types, offset provider -------------------------------

IDim = common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL)
JDim = common.Dimension(value="JDim", kind=common.DimensionKind.HORIZONTAL)
Kolor = common.Dimension(value="Kolor", kind=common.DimensionKind.HORIZONTAL)
K = common.Dimension(value="K", kind=common.DimensionKind.VERTICAL)

float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
edge_k_field = ts.FieldType(dims=[IDim, JDim, Kolor, K], dtype=float_type)
cell_k_field = ts.FieldType(dims=[IDim, JDim, Kolor, K], dtype=float_type)

offset_provider = {
    "IDim": IDim,
    "JDim": JDim,
    "Kolor": Kolor,
    "K": K,
    "_OffIDim": IDim,
    "_OffJDim": JDim,
    "_OffKolor": Kolor,
    "_OffK": K,
}


# ---------- helpers ----------------------------------------------------------

def cart_dom(*ranges):
    """Build a cartesian_domain with given (axis_name, lo, hi) tuples."""
    return im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value=name, kind=common.DimensionKind.HORIZONTAL
                          if name != "K" else common.DimensionKind.VERTICAL): (lo, hi)
         for name, lo, hi in ranges},
    )


def kolor_widenings(ir_node: itir.Node) -> int:
    """Count Kolor:[lo, hi) ranges where hi > 3 (i.e. widened past edge extent)."""
    text = pretty_format(ir_node)
    import re
    n = 0
    for m in re.finditer(r"Kolor\w*: \[(\-?\d+), (\-?\d+)", text):
        lo, hi = int(m.group(1)), int(m.group(2))
        if hi > 3 or lo < 0:
            n += 1
    return n


def show(label: str, prog: itir.Program) -> None:
    text = pretty_format(prog)
    n = kolor_widenings(prog)
    print(f"\n========== {label}  [widenings={n}] ==========")
    print(text)


# ---------- minimal IR matching the failing pattern --------------------------
# One slot of the failing 5-slot E2C2EO sum, from ir_out.txt:551–694 stripped to
# its essence:
#
#   next_vn @ Kolor:[0,3) ←
#     as_fieldop(_OffKolor:+2, Kolor:[0,1))(
#       ⇑(+)(z_graddiv_vn,
#            ⇑(-)(as_fieldop(id, Kolor:[0,1))(dwdz),
#                 as_fieldop(_OffIDim:-1, _OffKolor:+1, Kolor:[0,1))(dwdz))))
#
# Output writeback is wider (Kolor:[0,3)) than the body's outermost domain
# (Kolor:[0,1)) — this is what the structured unroller produces for one of the
# per-Kolor branches. The bug we are hunting is whether `canonicalize_domain_argument`
# or the second `infer_program` then widens the inner Kolor:[0,1) up to [0,3) and
# pushes the +1/+2 shifts to source [1,4) / [2,5) on dwdz (a 2-Kolor cell field).

OUTER_DOM = cart_dom(("IDim", 4, 508), ("JDim", 4, 508), ("Kolor", 0, 3),
                     ("K", im.ref("vertical_start"), im.ref("vertical_end")))
INNER_DOM = cart_dom(("IDim", 5, 508), ("JDim", 4, 508), ("Kolor", 0, 1),
                     ("K", im.ref("vertical_start"), im.ref("vertical_end")))

# Build the lifted (`⇑`) deref-shift pattern by hand:
# as_fieldop(λ(it) → ·shift(off, n)(it), domain)(arg)
def make_multi_shift(*pairs):
    """Build shift(s1, n1, s2, n2, ...)(it) — multi-offset shift call."""
    args = []
    for p in pairs:
        if isinstance(p, str):
            args.append(itir.OffsetLiteral(value=p))
        else:
            args.append(itir.OffsetLiteral(value=int(p)))
    return im.call(im.call("shift")(*args))


def shift_fieldop(arg, shifts, domain):
    """as_fieldop(λ(it) → ·shift(s1,n1, s2,n2,...)(it), domain)(arg)."""
    inner = im.deref(make_multi_shift(*shifts)("__it"))
    return im.as_fieldop(im.lambda_("__it")(inner), domain)(arg)


def id_fieldop(arg, domain):
    """as_fieldop(λ(it) → ·it, domain)(arg)."""
    return im.as_fieldop(im.lambda_("__it")(im.deref("__it")), domain)(arg)


# For type-system compatibility, use plain as_fieldop(+/-) instead of lifted (↑).
# Real IR has `(↑(λ → +))(field_ref, ...)` which is a typing edge case the type-inference
# downstream of canonicalize_domain_argument handles only in context. Plain as_fieldop
# captures the same dataflow shape that triggers `infer_program` widening.

def plus_fieldop(a, b, domain):
    return im.as_fieldop(
        im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))), domain
    )(a, b)


def minus_fieldop(a, b, domain):
    return im.as_fieldop(
        im.lambda_("a", "b")(im.minus(im.deref("a"), im.deref("b"))), domain
    )(a, b)


dwdz_id = id_fieldop("dwdz_at_cells_on_model_levels", INNER_DOM)
dwdz_shift = shift_fieldop(
    "dwdz_at_cells_on_model_levels", ("_OffIDim", -1, "_OffKolor", 1), INNER_DOM
)
diff_expr = minus_fieldop(dwdz_id, dwdz_shift, INNER_DOM)
graddiv_expr = plus_fieldop(
    im.ref("horizontal_gradient_of_normal_wind_divergence"), diff_expr, INNER_DOM
)

# One outer slot: as_fieldop(_OffKolor:+2, Kolor:[0,1))(graddiv_expr)
slot = shift_fieldop(graddiv_expr, ("_OffKolor", 2), INNER_DOM)

prog = itir.Program(
    id="failing_pattern_minimal",
    function_definitions=[],
    params=[
        itir.Sym(id="horizontal_gradient_of_normal_wind_divergence", type=edge_k_field),
        itir.Sym(id="dwdz_at_cells_on_model_levels", type=cell_k_field),
        itir.Sym(id="next_vn", type=edge_k_field),
        itir.Sym(id="vertical_start", type=ts.ScalarType(kind=ts.ScalarKind.INT32)),
        itir.Sym(id="vertical_end", type=ts.ScalarType(kind=ts.ScalarKind.INT32)),
    ],
    declarations=[],
    body=[
        itir.SetAt(
            expr=slot,
            domain=OUTER_DOM,
            target=im.ref("next_vn"),
        )
    ],
)


# ---------- run each suspect pass in isolation -------------------------------

print(">>> Starting from the post-`AFTER CARTESIAN UNROLLING` shape (one slot only).")
show("input (= structured unroller output)", prog)

# Pass 1: InferDomainOps (line 536 in pass_manager.py)
ir1 = infer_domain_ops.InferDomainOps.apply(prog)
show("after InferDomainOps.apply (= pass_manager.py:536)", ir1)

# Pass 2: canonicalize_domain_argument (line 538)
ir2 = concat_where.canonicalize_domain_argument(ir1)
show("after concat_where.canonicalize_domain_argument (= line 538)", ir2)

# Pass 3: ConstantFolding (line 540)
ir3 = ConstantFolding.apply(ir2)
show("after ConstantFolding.apply (= line 540)", ir3)

# Pass 4: second infer_program (line 546) — prime suspect for widening
ir4 = infer_domain.infer_program(ir3, offset_provider=offset_provider)
show("after infer_domain.infer_program (= line 546, SECOND inference)", ir4)

# Pass 5: ConstantFolding again (line 551)
ir5 = ConstantFolding.apply(ir4)
show("after second ConstantFolding (= line 551)", ir5)


# ---------- summary ---------------------------------------------------------
print("\n========== SUMMARY ==========")
for label, ir in [("input", prog), ("InferDomainOps", ir1),
                   ("canonicalize_domain_argument", ir2), ("ConstantFolding(1)", ir3),
                   ("infer_program(2nd)", ir4), ("ConstantFolding(2)", ir5)]:
    print(f"  {label:35s}  widenings = {kolor_widenings(ir)}")
