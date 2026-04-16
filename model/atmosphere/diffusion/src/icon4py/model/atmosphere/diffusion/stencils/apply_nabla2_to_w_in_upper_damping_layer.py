# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, broadcast

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _apply_nabla2_to_w_in_upper_damping_layer(
    w: fa.CellKField[wpfloat],
    diff_multfac_n2w: fa.KField[wpfloat],
    cell_area: fa.CellField[wpfloat],
    z_nabla2_c: fa.CellKField[vpfloat],
) -> fa.CellKField[wpfloat]:
    z_nabla2_c_wp = astype(z_nabla2_c, wpfloat)
    cell_area_tmp = broadcast(cell_area, (dims.CellDim, dims.KDim))

    w_wp = w + diff_multfac_n2w * cell_area_tmp * z_nabla2_c_wp
    return w_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_nabla2_to_w_in_upper_damping_layer(
    w: fa.CellKField[wpfloat],
    diff_multfac_n2w: fa.KField[wpfloat],
    cell_area: fa.CellField[wpfloat],
    z_nabla2_c: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_nabla2_to_w_in_upper_damping_layer(
        w,
        diff_multfac_n2w,
        cell_area,
        z_nabla2_c,
        out=w,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

# @gtx.field_operator
# def _apply_nabla2_to_w_in_upper_damping_layer_cart(
#     w: fa.CellKolorKField[wpfloat],
#     diff_multfac_n2w: fa.KField[wpfloat],
#     cell_area: fa.CellKolorField[wpfloat],
#     z_nabla2_c: fa.CellKolorKField[vpfloat],
# ) -> fa.CellKolorKField[wpfloat]:
#     z_nabla2_c_wp = astype(z_nabla2_c, wpfloat)
#     cell_area_tmp = broadcast(cell_area, (dims.IDim, dims.JDim, dims.Kolor, dims.KDim))

#     w_wp = w + diff_multfac_n2w * cell_area_tmp * z_nabla2_c_wp
#     return w_wp

# @gtx.program(grid_type=gtx.GridType.CARTESIAN)
# def apply_nabla2_to_w_in_upper_damping_layer_cart(
#     w: fa.CellKolorKField[wpfloat],
#     diff_multfac_n2w: fa.KField[wpfloat],
#     cell_area: fa.CellKolorField[wpfloat],
#     z_nabla2_c: fa.CellKolorKField[vpfloat],
#     domain_min_i: gtx.int32,
#     domain_max_i: gtx.int32,
#     domain_min_j: gtx.int32,
#     domain_max_j: gtx.int32,
#     domain_max_kolor: gtx.int32,
#     vertical_start: gtx.int32,
#     vertical_end: gtx.int32,
# ):
#     _apply_nabla2_to_w_in_upper_damping_layer_cart(
#         w,
#         diff_multfac_n2w,
#         cell_area,
#         z_nabla2_c,
#         out=w,
#         domain={
#             dims.IDim: (domain_min_i, domain_max_i),
#             dims.JDim: (domain_min_j, domain_max_j),
#             dims.Kolor: (0, domain_max_kolor),
#             dims.KDim: (vertical_start, vertical_end),
#         },
#     )