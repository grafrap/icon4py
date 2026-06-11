# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C2VDim, V2E, V2E2C2V, V2EDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _rbf_nabla4_direct(
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    ptr_coeff_1: gtx.Field[gtx.Dims[dims.VertexDim, V2EDim], wpfloat],
    ptr_coeff_2: gtx.Field[gtx.Dims[dims.VertexDim, V2EDim], wpfloat],
    primal_normal_vert_v1: gtx.Field[gtx.Dims[dims.EdgeDim, E2C2VDim], wpfloat],
    primal_normal_vert_v2: gtx.Field[gtx.Dims[dims.EdgeDim, E2C2VDim], wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
) -> tuple[fa.VertexKField[wpfloat], fa.VertexKField[wpfloat]]:
    u_wp, v_wp = astype((u_vert, v_vert), wpfloat)

    # Read each of the 7 unique vertex positions exactly once.
    # V2E2C2V slot mapping: 0=center(0,0), 1=N(0,+1), 2=NW(-1,+1),
    #   3=E(+1,0), 4=SE(+1,-1), 5=S(0,-1), 6=W(-1,0). All Kolor=0.
    u0, v0 = u_wp(V2E2C2V[0]), v_wp(V2E2C2V[0])
    u1, v1 = u_wp(V2E2C2V[1]), v_wp(V2E2C2V[1])
    u2, v2 = u_wp(V2E2C2V[2]), v_wp(V2E2C2V[2])
    u3, v3 = u_wp(V2E2C2V[3]), v_wp(V2E2C2V[3])
    u4, v4 = u_wp(V2E2C2V[4]), v_wp(V2E2C2V[4])
    u5, v5 = u_wp(V2E2C2V[5]), v_wp(V2E2C2V[5])
    u6, v6 = u_wp(V2E2C2V[6]), v_wp(V2E2C2V[6])

    # nabla4 formula per V2E slot s:
    #   tang_s = sum over E2C2V[0,1]: u/v * primal_normal  (tangential component pair)
    #   norm_s = sum over E2C2V[2,3]: u/v * primal_normal  (normal component pair)
    #   n4_s = 4 * ((norm_s - 2*nabla2_s)*ivvl_s^2 + (tang_s - 2*nabla2_s)*ipel_s^2)
    #
    # Syntax: field(V2E[s])[E2C2VDim(j)] — shift to the V2E[s] edge FIRST,
    # then index the local E2C2VDim slot j. This produces list_get(j, deref(shift(V2E,s)(it)))
    # in GTIR, where deref yields a ListType that list_get can index correctly.
    #
    # (V2E slot, E2C2V slot) -> V2E2C2V slot lookup:
    #   V2E[0]: E2C2V[0]->0, [1]->1, [2]->2, [3]->3  (tang: center,N;  norm: NW,E)
    #   V2E[1]: E2C2V[0]->0, [1]->3, [2]->1, [3]->4  (tang: center,E;  norm: N,SE)
    #   V2E[2]: E2C2V[0]->0, [1]->4, [2]->3, [3]->5  (tang: center,SE; norm: E,S)
    #   V2E[3]: E2C2V[0]->5, [1]->0, [2]->6, [3]->4  (tang: S,center;  norm: W,SE)
    #   V2E[4]: E2C2V[0]->6, [1]->0, [2]->2, [3]->5  (tang: W,center;  norm: NW,S)
    #   V2E[5]: E2C2V[0]->2, [1]->0, [2]->1, [3]->6  (tang: NW,center; norm: N,W)

    n2_0 = z_nabla2_e(V2E[0])
    ivvl0 = inv_vert_vert_length(V2E[0])
    ipel0 = inv_primal_edge_length(V2E[0])
    tang0 = (u0 * primal_normal_vert_v1(V2E[0])[E2C2VDim(0)]
             + v0 * primal_normal_vert_v2(V2E[0])[E2C2VDim(0)]
             + u1 * primal_normal_vert_v1(V2E[0])[E2C2VDim(1)]
             + v1 * primal_normal_vert_v2(V2E[0])[E2C2VDim(1)])
    norm0 = (u2 * primal_normal_vert_v1(V2E[0])[E2C2VDim(2)]
             + v2 * primal_normal_vert_v2(V2E[0])[E2C2VDim(2)]
             + u3 * primal_normal_vert_v1(V2E[0])[E2C2VDim(3)]
             + v3 * primal_normal_vert_v2(V2E[0])[E2C2VDim(3)])
    n4_0 = wpfloat("4.0") * (
        (norm0 - wpfloat("2.0") * n2_0) * (ivvl0 * ivvl0)
        + (tang0 - wpfloat("2.0") * n2_0) * (ipel0 * ipel0)
    )

    n2_1 = z_nabla2_e(V2E[1])
    ivvl1 = inv_vert_vert_length(V2E[1])
    ipel1 = inv_primal_edge_length(V2E[1])
    tang1 = (u0 * primal_normal_vert_v1(V2E[1])[E2C2VDim(0)]
             + v0 * primal_normal_vert_v2(V2E[1])[E2C2VDim(0)]
             + u3 * primal_normal_vert_v1(V2E[1])[E2C2VDim(1)]
             + v3 * primal_normal_vert_v2(V2E[1])[E2C2VDim(1)])
    norm1 = (u1 * primal_normal_vert_v1(V2E[1])[E2C2VDim(2)]
             + v1 * primal_normal_vert_v2(V2E[1])[E2C2VDim(2)]
             + u4 * primal_normal_vert_v1(V2E[1])[E2C2VDim(3)]
             + v4 * primal_normal_vert_v2(V2E[1])[E2C2VDim(3)])
    n4_1 = wpfloat("4.0") * (
        (norm1 - wpfloat("2.0") * n2_1) * (ivvl1 * ivvl1)
        + (tang1 - wpfloat("2.0") * n2_1) * (ipel1 * ipel1)
    )

    n2_2 = z_nabla2_e(V2E[2])
    ivvl2 = inv_vert_vert_length(V2E[2])
    ipel2 = inv_primal_edge_length(V2E[2])
    tang2 = (u0 * primal_normal_vert_v1(V2E[2])[E2C2VDim(0)]
             + v0 * primal_normal_vert_v2(V2E[2])[E2C2VDim(0)]
             + u4 * primal_normal_vert_v1(V2E[2])[E2C2VDim(1)]
             + v4 * primal_normal_vert_v2(V2E[2])[E2C2VDim(1)])
    norm2 = (u3 * primal_normal_vert_v1(V2E[2])[E2C2VDim(2)]
             + v3 * primal_normal_vert_v2(V2E[2])[E2C2VDim(2)]
             + u5 * primal_normal_vert_v1(V2E[2])[E2C2VDim(3)]
             + v5 * primal_normal_vert_v2(V2E[2])[E2C2VDim(3)])
    n4_2 = wpfloat("4.0") * (
        (norm2 - wpfloat("2.0") * n2_2) * (ivvl2 * ivvl2)
        + (tang2 - wpfloat("2.0") * n2_2) * (ipel2 * ipel2)
    )

    n2_3 = z_nabla2_e(V2E[3])
    ivvl3 = inv_vert_vert_length(V2E[3])
    ipel3 = inv_primal_edge_length(V2E[3])
    tang3 = (u5 * primal_normal_vert_v1(V2E[3])[E2C2VDim(0)]
             + v5 * primal_normal_vert_v2(V2E[3])[E2C2VDim(0)]
             + u0 * primal_normal_vert_v1(V2E[3])[E2C2VDim(1)]
             + v0 * primal_normal_vert_v2(V2E[3])[E2C2VDim(1)])
    norm3 = (u6 * primal_normal_vert_v1(V2E[3])[E2C2VDim(2)]
             + v6 * primal_normal_vert_v2(V2E[3])[E2C2VDim(2)]
             + u4 * primal_normal_vert_v1(V2E[3])[E2C2VDim(3)]
             + v4 * primal_normal_vert_v2(V2E[3])[E2C2VDim(3)])
    n4_3 = wpfloat("4.0") * (
        (norm3 - wpfloat("2.0") * n2_3) * (ivvl3 * ivvl3)
        + (tang3 - wpfloat("2.0") * n2_3) * (ipel3 * ipel3)
    )

    n2_4 = z_nabla2_e(V2E[4])
    ivvl4 = inv_vert_vert_length(V2E[4])
    ipel4 = inv_primal_edge_length(V2E[4])
    tang4 = (u6 * primal_normal_vert_v1(V2E[4])[E2C2VDim(0)]
             + v6 * primal_normal_vert_v2(V2E[4])[E2C2VDim(0)]
             + u0 * primal_normal_vert_v1(V2E[4])[E2C2VDim(1)]
             + v0 * primal_normal_vert_v2(V2E[4])[E2C2VDim(1)])
    norm4 = (u2 * primal_normal_vert_v1(V2E[4])[E2C2VDim(2)]
             + v2 * primal_normal_vert_v2(V2E[4])[E2C2VDim(2)]
             + u5 * primal_normal_vert_v1(V2E[4])[E2C2VDim(3)]
             + v5 * primal_normal_vert_v2(V2E[4])[E2C2VDim(3)])
    n4_4 = wpfloat("4.0") * (
        (norm4 - wpfloat("2.0") * n2_4) * (ivvl4 * ivvl4)
        + (tang4 - wpfloat("2.0") * n2_4) * (ipel4 * ipel4)
    )

    n2_5 = z_nabla2_e(V2E[5])
    ivvl5 = inv_vert_vert_length(V2E[5])
    ipel5 = inv_primal_edge_length(V2E[5])
    tang5 = (u2 * primal_normal_vert_v1(V2E[5])[E2C2VDim(0)]
             + v2 * primal_normal_vert_v2(V2E[5])[E2C2VDim(0)]
             + u0 * primal_normal_vert_v1(V2E[5])[E2C2VDim(1)]
             + v0 * primal_normal_vert_v2(V2E[5])[E2C2VDim(1)])
    norm5 = (u1 * primal_normal_vert_v1(V2E[5])[E2C2VDim(2)]
             + v1 * primal_normal_vert_v2(V2E[5])[E2C2VDim(2)]
             + u6 * primal_normal_vert_v1(V2E[5])[E2C2VDim(3)]
             + v6 * primal_normal_vert_v2(V2E[5])[E2C2VDim(3)])
    n4_5 = wpfloat("4.0") * (
        (norm5 - wpfloat("2.0") * n2_5) * (ivvl5 * ivvl5)
        + (tang5 - wpfloat("2.0") * n2_5) * (ipel5 * ipel5)
    )

    u_out = (ptr_coeff_1[V2EDim(0)] * n4_0
             + ptr_coeff_1[V2EDim(1)] * n4_1
             + ptr_coeff_1[V2EDim(2)] * n4_2
             + ptr_coeff_1[V2EDim(3)] * n4_3
             + ptr_coeff_1[V2EDim(4)] * n4_4
             + ptr_coeff_1[V2EDim(5)] * n4_5)
    v_out = (ptr_coeff_2[V2EDim(0)] * n4_0
             + ptr_coeff_2[V2EDim(1)] * n4_1
             + ptr_coeff_2[V2EDim(2)] * n4_2
             + ptr_coeff_2[V2EDim(3)] * n4_3
             + ptr_coeff_2[V2EDim(4)] * n4_4
             + ptr_coeff_2[V2EDim(5)] * n4_5)
    return u_out, v_out


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def rbf_nabla4_direct(
    u_vert_out: fa.VertexKField[wpfloat],
    v_vert_out: fa.VertexKField[wpfloat],
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    ptr_coeff_1: gtx.Field[gtx.Dims[dims.VertexDim, V2EDim], wpfloat],
    ptr_coeff_2: gtx.Field[gtx.Dims[dims.VertexDim, V2EDim], wpfloat],
    primal_normal_vert_v1: gtx.Field[gtx.Dims[dims.EdgeDim, E2C2VDim], wpfloat],
    primal_normal_vert_v2: gtx.Field[gtx.Dims[dims.EdgeDim, E2C2VDim], wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _rbf_nabla4_direct(
        u_vert,
        v_vert,
        z_nabla2_e,
        ptr_coeff_1,
        ptr_coeff_2,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        inv_vert_vert_length,
        inv_primal_edge_length,
        out=(u_vert_out, v_vert_out),
        domain={
            dims.VertexDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
