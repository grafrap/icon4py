# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C2V, E2C2VDim
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.dimension import IDim, JDim, Kolor


@gtx.field_operator
def _calculate_nabla4(
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_v1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_v2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
) -> fa.EdgeKField[vpfloat]:
    u_vert_wp, v_vert_wp = astype((u_vert, v_vert), wpfloat)

    nabv_tang_vp = astype(
        (
            u_vert_wp(E2C2V[0]) * primal_normal_vert_v1[E2C2VDim(0)]
            + v_vert_wp(E2C2V[0]) * primal_normal_vert_v2[E2C2VDim(0)]
            + u_vert_wp(E2C2V[1]) * primal_normal_vert_v1[E2C2VDim(1)]
            + v_vert_wp(E2C2V[1]) * primal_normal_vert_v2[E2C2VDim(1)]
        ),
        vpfloat,
    )

    nabv_norm_vp = astype(
        (
            u_vert_wp(E2C2V[2]) * primal_normal_vert_v1[E2C2VDim(2)]
            + v_vert_wp(E2C2V[2]) * primal_normal_vert_v2[E2C2VDim(2)]
            + u_vert_wp(E2C2V[3]) * primal_normal_vert_v1[E2C2VDim(3)]
            + v_vert_wp(E2C2V[3]) * primal_normal_vert_v2[E2C2VDim(3)]
        ),
        vpfloat,
    )
    nabv_tang_wp, nabv_norm_wp = astype((nabv_tang_vp, nabv_norm_vp), wpfloat)
    z_nabla4_e2_wp = wpfloat("4.0") * (
        (nabv_norm_wp - wpfloat("2.0") * z_nabla2_e) * (inv_vert_vert_length * inv_vert_vert_length)
        + (nabv_tang_wp - wpfloat("2.0") * z_nabla2_e)
        * (inv_primal_edge_length * inv_primal_edge_length)
    )
    return astype(z_nabla4_e2_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_nabla4(
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_v1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_v2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    z_nabla4_e2: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _calculate_nabla4(
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
        out=z_nabla4_e2,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# @gtx.field_operator
# def _calculate_nabla4_cart(
#     u_vert: gtx.Field[[IDim, JDim, Kolor, dims.KDim], vpfloat],
#     v_vert: gtx.Field[[IDim, JDim, Kolor, dims.KDim], vpfloat],
#     pn_v1: tuple[
#         fa.EdgeKolorField[wpfloat],
#         fa.EdgeKolorField[wpfloat],
#         fa.EdgeKolorField[wpfloat],
#         fa.EdgeKolorField[wpfloat]
#     ],
#     pn_v2: tuple[
#         fa.EdgeKolorField[wpfloat],
#         fa.EdgeKolorField[wpfloat],
#         fa.EdgeKolorField[wpfloat],
#         fa.EdgeKolorField[wpfloat]
#     ],
#     z_nabla2_e: fa.EdgeKolorKField[wpfloat],
#     inv_vert_vert_length: fa.EdgeKolorField[wpfloat],
#     inv_primal_edge_length: fa.EdgeKolorField[wpfloat],
# ) -> fa.EdgeKolorKField[vpfloat]:
#     u_vert_wp, v_vert_wp = astype((u_vert, v_vert), wpfloat)

#     # Kolor 0 (East): V0=(0,0), V1=(0,1), V2=(1,0), V3=(-1,1)
#     tang_k0 = (
#         u_vert_wp * pn_v1[0] + v_vert_wp * pn_v2[0] +
#         u_vert_wp(JDim + 1) * pn_v1[1] + v_vert_wp(JDim + 1) * pn_v2[1]
#     )
#     norm_k0 = (
#         u_vert_wp(IDim + 1) * pn_v1[2] + v_vert_wp(IDim + 1) * pn_v2[2] +
#         u_vert_wp(IDim - 1)(JDim + 1) * pn_v1[3] + v_vert_wp(IDim - 1)(JDim + 1) * pn_v2[3]
#     )

#     # Kolor 1 (NE): V0=(0,0), V1=(1,0), V2=(0,1), V3=(1,-1)
#     tang_k1 = (
#         u_vert_wp(Kolor-1) * pn_v1[0] + v_vert_wp(Kolor-1) * pn_v2[0] +
#         u_vert_wp(IDim + 1)(Kolor-1) * pn_v1[1] + v_vert_wp(IDim + 1)(Kolor-1) * pn_v2[1]
#     )
#     norm_k1 = (
#         u_vert_wp(JDim + 1)(Kolor-1) * pn_v1[2] + v_vert_wp(JDim + 1)(Kolor-1) * pn_v2[2] +
#         u_vert_wp(IDim + 1)(JDim - 1)(Kolor-1) * pn_v1[3] + v_vert_wp(IDim + 1)(JDim - 1)(Kolor-1) * pn_v2[3]
#     )

#     # Kolor 2 (NW): V0=(0,1), V1=(1,0), V2=(0,0), V3=(1,1)
#     tang_k2 = (
#         u_vert_wp(JDim + 1)(Kolor-2) * pn_v1[0] + v_vert_wp(JDim + 1)(Kolor-2) * pn_v2[0] +
#         u_vert_wp(IDim + 1)(Kolor-2) * pn_v1[1] + v_vert_wp(IDim + 1)(Kolor-2) * pn_v2[1]
#     )
#     norm_k2 = (
#         u_vert_wp(Kolor-2) * pn_v1[2] + v_vert_wp(Kolor-2) * pn_v2[2] +
#         u_vert_wp(IDim + 1)(JDim + 1)(Kolor-2) * pn_v1[3] + v_vert_wp(IDim + 1)(JDim + 1)(Kolor-2) * pn_v2[3]
#     )

#     nabv_tang_vp = astype(
#         concat_where(Kolor==0, tang_k0, concat_where(Kolor==1, tang_k1, tang_k2)),
#         vpfloat,
#     )
#     nabv_norm_vp = astype(
#         concat_where(Kolor==0, norm_k0, concat_where(Kolor==1, norm_k1, norm_k2)),
#         vpfloat,
#     )

#     nabv_tang_wp, nabv_norm_wp = astype((nabv_tang_vp, nabv_norm_vp), wpfloat)
#     z_nabla4_e2_wp = wpfloat("4.0") * (
#         (nabv_norm_wp - wpfloat("2.0") * z_nabla2_e) * (inv_vert_vert_length * inv_vert_vert_length)
#         + (nabv_tang_wp - wpfloat("2.0") * z_nabla2_e)
#         * (inv_primal_edge_length * inv_primal_edge_length)
#     )
#     return astype(z_nabla4_e2_wp, vpfloat)


# @gtx.program(grid_type=gtx.GridType.CARTESIAN)
# def calculate_nabla4_cart(
#     u_vert: fa.VertexKolorKField[vpfloat],
#     v_vert: fa.VertexKolorKField[vpfloat],
#     pn_v1: tuple[
#         fa.EdgeKolorField[wpfloat],
#         fa.EdgeKolorField[wpfloat],
#         fa.EdgeKolorField[wpfloat],
#         fa.EdgeKolorField[wpfloat],
#     ],
#     pn_v2: tuple[
#         fa.EdgeKolorField[wpfloat],
#         fa.EdgeKolorField[wpfloat],
#         fa.EdgeKolorField[wpfloat],
#         fa.EdgeKolorField[wpfloat],
#     ],
#     z_nabla2_e: fa.EdgeKolorKField[wpfloat],
#     inv_vert_vert_length: fa.EdgeKolorField[wpfloat],
#     inv_primal_edge_length: fa.EdgeKolorField[wpfloat],
#     z_nabla4_e2: fa.EdgeKolorKField[vpfloat],
#     domain_min_i: gtx.int32,
#     domain_max_i: gtx.int32,
#     domain_min_j: gtx.int32,
#     domain_max_j: gtx.int32,
#     domain_max_kolor: gtx.int32,
#     vertical_start: gtx.int32,
#     vertical_end: gtx.int32,
# ):
#     _calculate_nabla4_cart(
#         u_vert,
#         v_vert,
#         pn_v1,
#         pn_v2,
#         z_nabla2_e,
#         inv_vert_vert_length,
#         inv_primal_edge_length,
#         out=z_nabla4_e2,
#         domain={
#             IDim: (domain_min_i, domain_max_i),
#             JDim: (domain_min_j, domain_max_j),
#             Kolor: (0, domain_max_kolor),
#             dims.KDim: (vertical_start, vertical_end),
#         },
#     )

#     # later for lateral boundary: split into 3 programs, one for every Kolor, because of i,j extents
    