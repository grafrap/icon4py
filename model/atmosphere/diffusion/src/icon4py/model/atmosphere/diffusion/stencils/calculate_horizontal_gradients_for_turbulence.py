# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, neighbor_sum
from gt4py.next.experimental import concat_where


from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E2CO, C2E2CODim, IDim, JDim, Kolor, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat




@gtx.field_operator
def _calculate_horizontal_gradients_for_turbulence(
    w: fa.CellKField[wpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    dwdx_wp = neighbor_sum(geofac_grg_x * w(C2E2CO), axis=C2E2CODim)
    dwdy_wp = neighbor_sum(geofac_grg_y * w(C2E2CO), axis=C2E2CODim)
    return astype((dwdx_wp, dwdy_wp), vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_horizontal_gradients_for_turbulence(
    w: fa.CellKField[wpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    dwdx: fa.CellKField[vpfloat],
    dwdy: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _calculate_horizontal_gradients_for_turbulence(
        w,
        geofac_grg_x,
        geofac_grg_y,
        out=(dwdx, dwdy),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

@gtx.field_operator
def _calculate_horizontal_gradients_for_turbulence_cart(
    w: fa.CellKolorKField[wpfloat],
    geofac_grg_x: tuple[
        fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat],
    ],
    geofac_grg_y: tuple[
        fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat],
    ],
) -> tuple[fa.CellKolorKField[wpfloat], fa.CellKolorKField[wpfloat]]:
    
    # Kolor 0 (Up Triangle) Neighbors: (0,0,1), (0,-1,1), (-1,0,1)
    w_0_k0 = w(Kolor + 1)
    w_1_k0 = w(JDim - 1)(Kolor + 1)
    w_2_k0 = w(IDim - 1)(Kolor + 1)

    # Kolor 1 (Down Triangle) Neighbors: (0,0,0), (0,1,0), (1,0,0)
    w_0_k1 = w(Kolor - 1)
    w_1_k1 = w(JDim + 1)(Kolor - 1)
    w_2_k1 = w(IDim + 1)(Kolor - 1)

    w_0 = concat_where(Kolor == 0, w_0_k0, w_0_k1)
    w_1 = concat_where(Kolor == 0, w_1_k0, w_1_k1)
    w_2 = concat_where(Kolor == 0, w_2_k0, w_2_k1)

    dwdx_wp = geofac_grg_x[0] * w_0 + geofac_grg_x[1] * w_1 + geofac_grg_x[2] * w_2
    dwdy_wp = geofac_grg_y[0] * w_0 + geofac_grg_y[1] * w_1 + geofac_grg_y[2] * w_2

    return astype((dwdx_wp, dwdy_wp), vpfloat)

@gtx.program(grid_type=gtx.GridType.CARTESIAN)
def calculate_horizontal_gradients_for_turbulence_cart(
    w: gtx.Field[[IDim, JDim, Kolor, KDim], wpfloat],
    geofac_grg_x: tuple[
        fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat], 
    ],
    geofac_grg_y: tuple[
        fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat], 
    ],
    dwdx: gtx.Field[[IDim, JDim, Kolor, KDim], vpfloat],
    dwdy: gtx.Field[[IDim, JDim, Kolor, KDim], vpfloat],
    domain_min_i: gtx.int32, domain_max_i: gtx.int32,
    domain_min_j: gtx.int32, domain_max_j: gtx.int32,
    domain_max_kolor: gtx.int32,
    vertical_start: gtx.int32, vertical_end: gtx.int32,
):
    _calculate_horizontal_gradients_for_turbulence_cart(
        w, geofac_grg_x, geofac_grg_y,
        out=(dwdx, dwdy),
        domain={
            IDim: (domain_min_i, domain_max_i), JDim: (domain_min_j, domain_max_j),
            Kolor: (0, domain_max_kolor), KDim: (vertical_start, vertical_end),
        },
    )