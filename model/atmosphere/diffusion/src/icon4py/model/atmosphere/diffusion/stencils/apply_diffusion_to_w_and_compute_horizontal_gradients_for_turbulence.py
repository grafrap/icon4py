# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w import _apply_nabla2_to_w, _apply_nabla2_to_w_cart
from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w_in_upper_damping_layer import (
    _apply_nabla2_to_w_in_upper_damping_layer,
    _apply_nabla2_to_w_in_upper_damping_layer_cart,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_horizontal_gradients_for_turbulence import (
    _calculate_horizontal_gradients_for_turbulence,
    _calculate_horizontal_gradients_for_turbulence_cart,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_w import (
    _calculate_nabla2_for_w,
    _calculate_nabla2_for_w_cart,
)
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import C2E2CODim, CellDim, KDim, Kolor, IDim, JDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence(
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    w_old: fa.CellKField[wpfloat],
    type_shear: gtx.int32,
    dwdx: fa.CellKField[vpfloat],
    dwdy: fa.CellKField[vpfloat],
    diff_multfac_w: wpfloat,
    diff_multfac_n2w: fa.KField[wpfloat],
    nrdmax: gtx.int32,
    interior_idx: gtx.int32,
    halo_idx: gtx.int32,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    dwdx, dwdy = (
        concat_where(
            0 < KDim,
            _calculate_horizontal_gradients_for_turbulence(w_old, geofac_grg_x, geofac_grg_y),
            (dwdx, dwdy),
        )
        if type_shear == 2
        else (dwdx, dwdy)
    )

    z_nabla2_c = _calculate_nabla2_for_w(w_old, geofac_n2s)

    w = concat_where(
        (interior_idx <= CellDim) & (CellDim < halo_idx),
        _apply_nabla2_to_w(area, z_nabla2_c, geofac_n2s, w_old, diff_multfac_w),
        w_old,
    )

    w = concat_where(
        (0 < KDim) & (KDim < nrdmax) & (interior_idx <= CellDim) & (CellDim < halo_idx),
        _apply_nabla2_to_w_in_upper_damping_layer(w, diff_multfac_n2w, area, z_nabla2_c),
        w,
    )

    return w, dwdx, dwdy


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence(
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    w_old: fa.CellKField[wpfloat],
    w: fa.CellKField[wpfloat],
    type_shear: gtx.int32,
    dwdx: fa.CellKField[vpfloat],
    dwdy: fa.CellKField[vpfloat],
    diff_multfac_w: wpfloat,
    diff_multfac_n2w: fa.KField[wpfloat],
    nrdmax: gtx.int32,
    interior_idx: gtx.int32,
    halo_idx: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence(
        area,
        geofac_n2s,
        geofac_grg_x,
        geofac_grg_y,
        w_old,
        type_shear,
        dwdx,
        dwdy,
        diff_multfac_w,
        diff_multfac_n2w,
        nrdmax,
        interior_idx,
        halo_idx,
        out=(w, dwdx, dwdy),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )

# @gtx.field_operator
# def _apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence_cart(
#     area: fa.CellKolorField[wpfloat],
#     geofac_n2s: tuple[fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat]],
#     geofac_grg_x: tuple[fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat]],
#     geofac_grg_y: tuple[fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat]],
#     w_old: fa.CellKolorKField[wpfloat],
#     w: fa.CellKolorKField[wpfloat],
#     type_shear: gtx.int32,
#     dwdx: fa.CellKolorKField[vpfloat],
#     dwdy: fa.CellKolorKField[vpfloat],
#     diff_multfac_w: wpfloat,
#     diff_multfac_n2w: fa.KField[wpfloat],
#     nrdmax: gtx.int32,
#     domain_min_i: gtx.int32, domain_max_i: gtx.int32,
#     domain_min_j: gtx.int32, domain_max_j: gtx.int32,
# ) -> tuple[
#     fa.CellKolorKField[wpfloat],
#     fa.CellKolorKField[vpfloat],
#     fa.CellKolorKField[vpfloat]
# ]:
#     # 1. Compute Horizontal Gradients using w
#     dwdx_new, dwdy_new = _calculate_horizontal_gradients_for_turbulence_cart(w_old, geofac_grg_x, geofac_grg_y)
    
#     # KDim > 0 masking for gradients
#     # zero_vp = vpfloat("0.0")
#     dwdx = concat_where(KDim > 0, dwdx_new, dwdx)
#     dwdy = concat_where(KDim > 0, dwdy_new, dwdy)

#     # 2. Compute Nabla2 for W (using w_old)
#     z_nabla2_c = _calculate_nabla2_for_w_cart(w_old, geofac_n2s)

#     # 3. Apply Nabla2 to W (using w_old)
#     w_nabla2 = _apply_nabla2_to_w_cart(area, z_nabla2_c, geofac_n2s, w_old, diff_multfac_w)

#     # Stage 1: w = w_nabla2 in interior, w_old in margin. 
#     # INLINED BOUNDS to prevent SymRef compiler crash!
#     w_stage1_j = concat_where((JDim >= domain_min_j + 4) & (JDim < domain_max_j - 4), w_nabla2, w_old)
#     w_stage1   = concat_where((IDim >= domain_min_i + 4) & (IDim < domain_max_i - 4), w_stage1_j, w_old)
    
#     # 4. Evaluate Damping Layer branch using w_stage1
#     w_damping = _apply_nabla2_to_w_in_upper_damping_layer_cart(w_stage1, diff_multfac_n2w, area, z_nabla2_c)

#     # Stage 2: w = w_damping in interior AND 0 < KDim < nrdmax, else w_stage1. 
#     # INLINED BOUNDS to prevent SymRef compiler crash!
#     w_out_j = concat_where((JDim >= domain_min_j + 4) & (JDim < domain_max_j - 4) & (KDim > 0) & (KDim < nrdmax), w_damping, w_stage1)
#     w_out   = concat_where((IDim >= domain_min_i + 4) & (IDim < domain_max_i - 4) & (KDim > 0) & (KDim < nrdmax), w_out_j, w_stage1)

#     return w_out, dwdx, dwdy


# @gtx.program(grid_type=gtx.GridType.CARTESIAN)
# def apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence_cart(
#     area: fa.CellKolorField[wpfloat],
#     geofac_n2s: tuple[fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat]],
#     geofac_grg_x: tuple[fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat]],
#     geofac_grg_y: tuple[fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat], fa.CellKolorField[wpfloat]],
#     w_old: fa.CellKolorKField[wpfloat],
#     w: fa.CellKolorKField[wpfloat],
#     type_shear: gtx.int32,
#     dwdx: fa.CellKolorKField[vpfloat],
#     dwdy: fa.CellKolorKField[vpfloat],
#     diff_multfac_w: wpfloat,
#     diff_multfac_n2w: fa.KField[wpfloat],
#     nrdmax: gtx.int32,
#     domain_min_i: gtx.int32, domain_max_i: gtx.int32,
#     domain_min_j: gtx.int32, domain_max_j: gtx.int32,
#     domain_max_kolor: gtx.int32,
#     vertical_start: gtx.int32, vertical_end: gtx.int32,
# ):
#     _apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence_cart(
#         area, geofac_n2s, geofac_grg_x, geofac_grg_y, w_old, w, type_shear, dwdx, dwdy,
#         diff_multfac_w, diff_multfac_n2w, nrdmax,
#         domain_min_i, domain_max_i, domain_min_j, domain_max_j,
#         out=(w, dwdx, dwdy), 
#         domain={
#             IDim: (domain_min_i, domain_max_i), JDim: (domain_min_j, domain_max_j),
#             Kolor: (0, domain_max_kolor), KDim: (vertical_start, vertical_end),
#         },
#     )