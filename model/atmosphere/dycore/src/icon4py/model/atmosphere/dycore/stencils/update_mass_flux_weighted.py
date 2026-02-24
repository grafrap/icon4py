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
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _update_mass_flux_weighted(
    rho_ic: fa.CellKField[wpfloat],
    vwind_expl_wgt: fa.CellField[wpfloat],
    vwind_impl_wgt: fa.CellField[wpfloat],
    w_now: fa.CellKField[wpfloat],
    w_new: fa.CellKField[wpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    mass_flx_ic: fa.CellKField[wpfloat],
    r_nsubsteps: wpfloat,
) -> fa.CellKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_65."""
    w_concorr_c_wp = astype(w_concorr_c, wpfloat)

    mass_flx_ic_wp = mass_flx_ic + (
        r_nsubsteps * rho_ic * (vwind_expl_wgt * w_now + vwind_impl_wgt * w_new - w_concorr_c_wp)
    )
    return mass_flx_ic_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_mass_flux_weighted(
    rho_ic: fa.CellKField[wpfloat],
    vwind_expl_wgt: fa.CellField[wpfloat],
    vwind_impl_wgt: fa.CellField[wpfloat],
    w_now: fa.CellKField[wpfloat],
    w_new: fa.CellKField[wpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    mass_flx_ic: fa.CellKField[wpfloat],
    r_nsubsteps: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _update_mass_flux_weighted(
        rho_ic,
        vwind_expl_wgt,
        vwind_impl_wgt,
        w_now,
        w_new,
        w_concorr_c,
        mass_flx_ic,
        r_nsubsteps,
        out=mass_flx_ic,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

# New function for a structured cartesian grid, which will have a virtual dimension telling if 
# it is a up or down triangle on the grid. 
# q: how can i go from fa.CellKField to something that has a virtual dimension? 
# a: we can define a new type alias for a field with a virtual dimension

IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
# Kolor dim for up/down triangles (size 2)
Kolor = gtx.Dimension("Kolor")

@gtx.field_operator
def _update_mass_flux_weighted_cart(
    rho_ic: gtx.Field[[IDim, JDim, Kolor, dims.KDim], wpfloat],
    vwind_expl_wgt: gtx.Field[[IDim, JDim, Kolor], wpfloat],
    vwind_impl_wgt: gtx.Field[[IDim, JDim, Kolor], wpfloat],
    w_now: gtx.Field[[IDim, JDim, Kolor, dims.KDim], wpfloat],
    w_new: gtx.Field[[IDim, JDim, Kolor, dims.KDim], wpfloat],
    w_concorr_c: gtx.Field[[IDim, JDim, Kolor, dims.KDim], vpfloat],
    mass_flx_ic: gtx.Field[[IDim, JDim, Kolor, dims.KDim], wpfloat],
    r_nsubsteps: wpfloat,
) -> gtx.Field[[IDim, JDim, Kolor, dims.KDim], wpfloat]:
    w_concorr_c_wp = astype(w_concorr_c, wpfloat)

    mass_flx_ic_wp = mass_flx_ic + (
        r_nsubsteps * rho_ic * (vwind_expl_wgt * w_now + vwind_impl_wgt * w_new - w_concorr_c_wp)
    )
    return mass_flx_ic_wp



@gtx.program(grid_type=gtx.GridType.CARTESIAN)
def update_mass_flux_weighted_cart(
    rho_ic: gtx.Field[[IDim, JDim, Kolor, dims.KDim], wpfloat],
    vwind_expl_wgt: gtx.Field[[IDim, JDim, Kolor], wpfloat],
    vwind_impl_wgt: gtx.Field[[IDim, JDim, Kolor], wpfloat],
    w_now: gtx.Field[[IDim, JDim, Kolor, dims.KDim], wpfloat],
    w_new: gtx.Field[[IDim, JDim, Kolor, dims.KDim], wpfloat],
    w_concorr_c: gtx.Field[[IDim, JDim, Kolor, dims.KDim], vpfloat],
    mass_flx_ic: gtx.Field[[IDim, JDim, Kolor, dims.KDim], wpfloat],
    r_nsubsteps: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _update_mass_flux_weighted_cart(
        rho_ic,
        vwind_expl_wgt,
        vwind_impl_wgt,
        w_now,
        w_new,
        w_concorr_c,
        mass_flx_ic,
        r_nsubsteps,
        out=mass_flx_ic,
        domain={
            IDim: (horizontal_start, horizontal_end),
            JDim: (horizontal_start, horizontal_end),
            Kolor: (0, 2),
            dims.KDim: (vertical_start, vertical_end),
        },
    )