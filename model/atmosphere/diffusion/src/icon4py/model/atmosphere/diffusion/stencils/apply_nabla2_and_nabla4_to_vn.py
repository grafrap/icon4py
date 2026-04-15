# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, broadcast, maximum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _apply_nabla2_and_nabla4_to_vn(
    area_edge: fa.EdgeField[wpfloat],
    kh_smag_e: fa.EdgeKField[vpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    z_nabla4_e2: fa.EdgeKField[vpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    nudgezone_diff: vpfloat,
) -> fa.EdgeKField[wpfloat]:
    kh_smag_e_wp, z_nabla4_e2_wp, nudgezone_diff_wp = astype(
        (kh_smag_e, z_nabla4_e2, nudgezone_diff), wpfloat
    )
    area_edge_broadcast = broadcast(area_edge, (dims.EdgeDim, dims.KDim))

    vn_wp = vn + area_edge * (
        maximum(nudgezone_diff_wp * nudgecoeff_e, kh_smag_e_wp) * z_nabla2_e
        - area_edge_broadcast * diff_multfac_vn * z_nabla4_e2_wp
    )
    return vn_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_nabla2_and_nabla4_to_vn(
    area_edge: fa.EdgeField[wpfloat],
    kh_smag_e: fa.EdgeKField[vpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    z_nabla4_e2: fa.EdgeKField[vpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    nudgezone_diff: vpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_nabla2_and_nabla4_to_vn(
        area_edge,
        kh_smag_e,
        z_nabla2_e,
        z_nabla4_e2,
        diff_multfac_vn,
        nudgecoeff_e,
        vn,
        nudgezone_diff,
        out=vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _apply_nabla2_and_nabla4_to_vn_cart(
    area_edge: fa.EdgeKolorField[wpfloat],
    kh_smag_e: fa.EdgeKolorKField[vpfloat],
    z_nabla2_e: fa.EdgeKolorKField[wpfloat],
    z_nabla4_e2: fa.EdgeKolorKField[vpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeKolorField[wpfloat],
    vn: fa.EdgeKolorKField[wpfloat],
    nudgezone_diff: vpfloat,
    domain_min_i: gtx.int32,
    domain_max_i: gtx.int32,
    domain_min_j: gtx.int32,
    domain_max_j: gtx.int32,
) -> fa.EdgeKolorKField[wpfloat]:
    kh_smag_e_wp, z_nabla4_e2_wp, nudgezone_diff_wp = astype(
        (kh_smag_e, z_nabla4_e2, nudgezone_diff), wpfloat
    )
    area_edge_broadcast = broadcast(area_edge, (dims.IDim, dims.JDim, dims.Kolor, dims.KDim))

    vn_wp = vn + area_edge * (
        maximum(nudgezone_diff_wp * nudgecoeff_e, kh_smag_e_wp) * z_nabla2_e
        - area_edge_broadcast * diff_multfac_vn * z_nabla4_e2_wp
    )
    return vn_wp

@gtx.program(grid_type=gtx.GridType.CARTESIAN)
def apply_nabla2_and_nabla4_to_vn_cart(
    area_edge: fa.EdgeKolorField[wpfloat],
    kh_smag_e: fa.EdgeKolorKField[vpfloat],
    z_nabla2_e: fa.EdgeKolorKField[wpfloat],
    z_nabla4_e2: fa.EdgeKolorKField[vpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeKolorField[wpfloat],
    vn: fa.EdgeKolorKField[wpfloat],
    nudgezone_diff: vpfloat,
    domain_min_i: gtx.int32,
    domain_max_i: gtx.int32,
    domain_min_j: gtx.int32,
    domain_max_j: gtx.int32,
    domain_max_kolor: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _apply_nabla2_and_nabla4_to_vn_cart(
        area_edge,
        kh_smag_e,
        z_nabla2_e,
        z_nabla4_e2,
        diff_multfac_vn,
        nudgecoeff_e,
        vn,
        nudgezone_diff,
        domain_min_i=domain_min_i,
        domain_max_i=domain_max_i,
        domain_min_j=domain_min_j,
        domain_max_j=domain_max_j,
        out=vn,
        domain={
            dims.IDim: (domain_min_i, domain_max_i),
            dims.JDim: (domain_min_j, domain_max_j),
            dims.Kolor: (0, domain_max_kolor),
            dims.KDim: (vertical_start, vertical_end),
        },
    )