# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _apply_nabla2_to_vn_in_lateral_boundary(
    z_nabla2_e: fa.EdgeKField[wpfloat],
    area_edge: fa.EdgeField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    fac_bdydiff_v: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    vn_wp = vn + (area_edge * fac_bdydiff_v * z_nabla2_e)
    return vn_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_nabla2_to_vn_in_lateral_boundary(
    z_nabla2_e: fa.EdgeKField[wpfloat],
    area_edge: fa.EdgeField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    fac_bdydiff_v: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_nabla2_to_vn_in_lateral_boundary(
        z_nabla2_e,
        area_edge,
        vn,
        fac_bdydiff_v,
        out=vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# @gtx.field_operator
# def _apply_nabla2_to_vn_in_lateral_boundary_cart(
#     z_nabla2_e: fa.EdgeKolorKField[wpfloat],
#     area_edge: fa.EdgeKolorField[wpfloat],
#     vn: fa.EdgeKolorKField[wpfloat],
#     fac_bdydiff_v: wpfloat,
# ) -> fa.EdgeKolorKField[wpfloat]:
#     vn_wp = vn + (area_edge * fac_bdydiff_v * z_nabla2_e)
#     return vn_wp

# @gtx.program(grid_type=gtx.GridType.CARTESIAN)
# def apply_nabla2_to_vn_in_lateral_boundary_cart(
#     z_nabla2_e: fa.EdgeKolorKField[wpfloat],
#     area_edge: fa.EdgeKolorField[wpfloat],
#     vn: fa.EdgeKolorKField[wpfloat],
#     fac_bdydiff_v: wpfloat,
#     domain_min_i: gtx.int32,
#     domain_max_i: gtx.int32,
#     domain_min_j: gtx.int32,
#     domain_max_j: gtx.int32,
#     domain_max_kolor: gtx.int32,
#     vertical_start: gtx.int32,
#     vertical_end: gtx.int32,
# ):
#     _apply_nabla2_to_vn_in_lateral_boundary_cart(
#         z_nabla2_e,
#         area_edge,
#         vn,
#         fac_bdydiff_v,
#         out=vn,  # Updates vn in-place
#         domain={
#             dims.IDim: (domain_min_i, domain_max_i),
#             dims.JDim: (domain_min_j, domain_max_j),
#             dims.Kolor: (0, domain_max_kolor),
#             dims.KDim: (vertical_start, vertical_end),
#         },
#     )