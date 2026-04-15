# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_and_nabla4_global_to_vn import (
    _apply_nabla2_and_nabla4_global_to_vn,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_and_nabla4_to_vn import (
    _apply_nabla2_and_nabla4_to_vn,
    _apply_nabla2_and_nabla4_to_vn_cart
)
from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_vn_in_lateral_boundary import (
    _apply_nabla2_to_vn_in_lateral_boundary,
    _apply_nabla2_to_vn_in_lateral_boundary_cart
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla4 import _calculate_nabla4, _calculate_nabla4_cart
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _apply_diffusion_to_vn(
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_v1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_v2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    area_edge: fa.EdgeField[wpfloat],
    kh_smag_e: fa.EdgeKField[vpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    nudgezone_diff: vpfloat,
    fac_bdydiff_v: wpfloat,
    start_2nd_nudge_line_idx_e: gtx.int32,
    limited_area: bool,
) -> fa.EdgeKField[wpfloat]:
    z_nabla4_e2 = _calculate_nabla4(
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
    )

    # TODO(): Use if-else statement instead
    vn = (
        concat_where(
            dims.EdgeDim >= start_2nd_nudge_line_idx_e,
            _apply_nabla2_and_nabla4_to_vn(
                area_edge,
                kh_smag_e,
                z_nabla2_e,
                z_nabla4_e2,
                diff_multfac_vn,
                nudgecoeff_e,
                vn,
                nudgezone_diff,
            ),
            _apply_nabla2_to_vn_in_lateral_boundary(z_nabla2_e, area_edge, vn, fac_bdydiff_v),
        )
        if limited_area
        else concat_where(
            dims.EdgeDim >= start_2nd_nudge_line_idx_e,
            _apply_nabla2_and_nabla4_global_to_vn(
                area_edge,
                kh_smag_e,
                z_nabla2_e,
                z_nabla4_e2,
                diff_multfac_vn,
                vn,
            ),
            vn,
        )
    )

    return vn


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_diffusion_to_vn(
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_v1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_v2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    area_edge: fa.EdgeField[wpfloat],
    kh_smag_e: fa.EdgeKField[vpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    nudgezone_diff: vpfloat,
    fac_bdydiff_v: wpfloat,
    start_2nd_nudge_line_idx_e: gtx.int32,
    limited_area: bool,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_diffusion_to_vn(
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
        area_edge,
        kh_smag_e,
        diff_multfac_vn,
        nudgecoeff_e,
        vn,
        nudgezone_diff,
        fac_bdydiff_v,
        start_2nd_nudge_line_idx_e,
        limited_area,
        out=vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

@gtx.field_operator
def _apply_diffusion_to_vn_cart(
    u_vert: fa.VertexKolorKField[vpfloat],
    v_vert: fa.VertexKolorKField[vpfloat],
    pn_v1: tuple[
        fa.EdgeKolorField[wpfloat],
        fa.EdgeKolorField[wpfloat],
        fa.EdgeKolorField[wpfloat],
        fa.EdgeKolorField[wpfloat],
    ],
    pn_v2: tuple[
        fa.EdgeKolorField[wpfloat],
        fa.EdgeKolorField[wpfloat],
        fa.EdgeKolorField[wpfloat],
        fa.EdgeKolorField[wpfloat],
    ],
    z_nabla2_e: fa.EdgeKolorKField[wpfloat],
    inv_vert_vert_length: fa.EdgeKolorField[wpfloat],
    inv_primal_edge_length: fa.EdgeKolorField[wpfloat],
    area_edge: fa.EdgeKolorField[wpfloat],
    kh_smag_e: fa.EdgeKolorKField[vpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeKolorField[wpfloat],
    vn: fa.EdgeKolorKField[wpfloat],
    nudgezone_diff: vpfloat,
    fac_bdydiff_v: wpfloat,
    domain_min_i: gtx.int32,
    domain_max_i: gtx.int32,
    domain_min_j: gtx.int32,
    domain_max_j: gtx.int32,
) -> fa.EdgeKolorKField[wpfloat]:
    
    # 1. Compute nabla4 across the entire field
    z_nabla4_e2 = _calculate_nabla4_cart(
        u_vert,
        v_vert,
        pn_v1,
        pn_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
    )

    # 2. Evaluate both branches
    vn_interior = _apply_nabla2_and_nabla4_to_vn_cart(
        area_edge,
        kh_smag_e,
        z_nabla2_e,
        z_nabla4_e2,
        diff_multfac_vn,
        nudgecoeff_e,
        vn,
        nudgezone_diff,
        domain_min_i,
        domain_max_i,
        domain_min_j,
        domain_max_j,
    )
    
    vn_boundary = _apply_nabla2_to_vn_in_lateral_boundary_cart(
        z_nabla2_e, area_edge, vn, fac_bdydiff_v
    )

    # 3. Apply Boundary Logic using safe nested concat_where (avoids & AST overloads)
    # Interior is where IDim and JDim are strictly >= 4 and < (max - 4)
    vn_out_j = concat_where(
        (dims.JDim >= domain_min_j + 4) & (dims.JDim < domain_max_j - 4),
        vn_interior,
        vn_boundary
    )
    
    vn_out = concat_where(
        (dims.IDim >= domain_min_i + 4) & (dims.IDim < domain_max_i - 4),
        vn_out_j,
        vn_boundary
    )

    return vn_out


@gtx.program(grid_type=gtx.GridType.CARTESIAN)
def apply_diffusion_to_vn_cart(
    u_vert: fa.VertexKolorKField[vpfloat],
    v_vert: fa.VertexKolorKField[vpfloat],
    pn_v1: tuple[
        fa.EdgeKolorField[wpfloat],
        fa.EdgeKolorField[wpfloat],
        fa.EdgeKolorField[wpfloat],
        fa.EdgeKolorField[wpfloat],
    ],
    pn_v2: tuple[
        fa.EdgeKolorField[wpfloat],
        fa.EdgeKolorField[wpfloat],
        fa.EdgeKolorField[wpfloat],
        fa.EdgeKolorField[wpfloat],
    ],
    z_nabla2_e: fa.EdgeKolorKField[wpfloat],
    inv_vert_vert_length: fa.EdgeKolorField[wpfloat],
    inv_primal_edge_length: fa.EdgeKolorField[wpfloat],
    area_edge: fa.EdgeKolorField[wpfloat],
    kh_smag_e: fa.EdgeKolorKField[vpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeKolorField[wpfloat],
    vn: fa.EdgeKolorKField[wpfloat],
    nudgezone_diff: vpfloat,
    fac_bdydiff_v: wpfloat,
    domain_min_i: gtx.int32,
    domain_max_i: gtx.int32,
    domain_min_j: gtx.int32,
    domain_max_j: gtx.int32,
    domain_max_kolor: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _apply_diffusion_to_vn_cart(
        u_vert,
        v_vert,
        pn_v1,
        pn_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
        area_edge,
        kh_smag_e,
        diff_multfac_vn,
        nudgecoeff_e,
        vn,
        nudgezone_diff,
        fac_bdydiff_v,
        domain_min_i,
        domain_max_i,
        domain_min_j,
        domain_max_j,
        out=vn,  # Updates vn in-place
        domain={
            dims.IDim: (domain_min_i, domain_max_i),
            dims.JDim: (domain_min_j, domain_max_j),
            dims.Kolor: (0, domain_max_kolor),
            dims.KDim: (vertical_start, vertical_end),
        },
    )