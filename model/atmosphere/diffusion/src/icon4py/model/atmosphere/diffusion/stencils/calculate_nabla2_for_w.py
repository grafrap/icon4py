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
from icon4py.model.common.dimension import C2E2CO, C2E2CODim, Kolor, IDim, JDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _calculate_nabla2_for_w(
    w: fa.CellKField[wpfloat], geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat]
) -> fa.CellKField[vpfloat]:
    z_nabla2_c_wp = neighbor_sum(w(C2E2CO) * geofac_n2s, axis=C2E2CODim)
    return astype(z_nabla2_c_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_nabla2_for_w(
    w: fa.CellKField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    z_nabla2_c: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    # TODO(): replace this by common/math/stencils/compute_nabla2_on_cell_k
    _calculate_nabla2_for_w(
        w,
        geofac_n2s,
        out=z_nabla2_c,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

@gtx.field_operator
def _calculate_nabla2_for_w_cart(
    w: fa.CellKolorKField[wpfloat],
    geofac_n2s: tuple[
        fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat]
    ]
) -> fa.CellKolorKField[vpfloat]:
    
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

    z_nabla2_c_wp = geofac_n2s[0] * w_0 + geofac_n2s[1] * w_1 + geofac_n2s[2] * w_2

    return astype(z_nabla2_c_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.CARTESIAN)
def calculate_nabla2_for_w_cart(
    w: fa.CellKolorKField[wpfloat],
    geofac_n2s: tuple[
        fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat]
    ],
    z_nabla2_c: fa.CellKolorKField[vpfloat],
    domain_min_i: gtx.int32, domain_max_i: gtx.int32,
    domain_min_j: gtx.int32, domain_max_j: gtx.int32,
    domain_max_kolor: gtx.int32,
    vertical_start: gtx.int32, vertical_end: gtx.int32,
):
    _calculate_nabla2_for_w_cart(
        w, geofac_n2s,
        out=z_nabla2_c,
        domain={
            IDim: (domain_min_i, domain_max_i), JDim: (domain_min_j, domain_max_j),
            Kolor: (0, domain_max_kolor), KDim: (vertical_start, vertical_end),
        },
    )