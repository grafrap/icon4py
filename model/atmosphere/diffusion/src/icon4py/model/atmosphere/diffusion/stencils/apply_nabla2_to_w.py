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
def _apply_nabla2_to_w(
    area: fa.CellField[wpfloat],
    z_nabla2_c: fa.CellKField[vpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    w: fa.CellKField[wpfloat],
    diff_multfac_w: wpfloat,
) -> fa.CellKField[wpfloat]:
    z_nabla2_c_wp = astype(z_nabla2_c, wpfloat)

    w_wp = w - diff_multfac_w * (area * area) * neighbor_sum(
        z_nabla2_c_wp(C2E2CO) * geofac_n2s, axis=C2E2CODim
    )
    return w_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_nabla2_to_w(
    area: fa.CellField[wpfloat],
    z_nabla2_c: fa.CellKField[vpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    w: fa.CellKField[wpfloat],
    diff_multfac_w: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_nabla2_to_w(
        area,
        z_nabla2_c,
        geofac_n2s,
        w,
        diff_multfac_w,
        out=w,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _apply_nabla2_to_w_cart(
    area: fa.CellKolorField[wpfloat],
    z_nabla2_c: fa.CellKolorKField[vpfloat],
    geofac_n2s: tuple[
        fa.CellKolorField[wpfloat],
        fa.CellKolorField[wpfloat],
        fa.CellKolorField[wpfloat],
    ],
    w: fa.CellKolorKField[wpfloat],
    diff_multfac_w: wpfloat,
) -> fa.CellKolorKField[wpfloat]:
    z_nabla2_c_wp = astype(z_nabla2_c, wpfloat)

    z_0_k0 = z_nabla2_c_wp(Kolor + 1)
    z_1_k0 = z_nabla2_c_wp(JDim - 1)(Kolor + 1)
    z_2_k0 = z_nabla2_c_wp(IDim - 1)(Kolor + 1)

    z_0_k1 = z_nabla2_c_wp(Kolor - 1)
    z_1_k1 = z_nabla2_c_wp(JDim + 1)(Kolor - 1)
    z_2_k1 = z_nabla2_c_wp(IDim + 1)(Kolor - 1)

    z_0 = concat_where(Kolor == 0, z_0_k0, z_0_k1)
    z_1 = concat_where(Kolor == 0, z_1_k0, z_1_k1)
    z_2 = concat_where(Kolor == 0, z_2_k0, z_2_k1)

    sum_val = geofac_n2s[0] * z_0 + geofac_n2s[1] * z_1 + geofac_n2s[2] * z_2
    return w - diff_multfac_w * (area * area) * sum_val

@gtx.program(grid_type=gtx.GridType.CARTESIAN)
def apply_nabla2_to_w_cart(
    area: fa.CellKolorField[wpfloat],
    z_nabla2_c: fa.CellKolorKField[vpfloat],
    geofac_n2s: tuple[
        fa.CellKolorField[wpfloat],
        fa.CellKolorField[wpfloat],
        fa.CellKolorField[wpfloat],
    ],
    w: fa.CellKolorKField[wpfloat],
    diff_multfac_w: wpfloat,
    domain_min_i: gtx.int32, domain_max_i: gtx.int32,
    domain_min_j: gtx.int32, domain_max_j: gtx.int32,
    domain_max_kolor: gtx.int32,
    vertical_start: gtx.int32, vertical_end: gtx.int32,
):
    _apply_nabla2_to_w_cart(
        area, z_nabla2_c, geofac_n2s, w, diff_multfac_w,
        out=w,
        domain={
            IDim: (domain_min_i, domain_max_i), JDim: (domain_min_j, domain_max_j),
            Kolor: (0, domain_max_kolor), KDim: (vertical_start, vertical_end),
        },
    )