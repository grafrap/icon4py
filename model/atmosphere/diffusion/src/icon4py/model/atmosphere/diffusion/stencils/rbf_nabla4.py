# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla4 import _calculate_nabla4
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.interpolation.stencils.mo_intp_rbf_rbf_vec_interpol_vertex import (
    _mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _rbf_nabla4(
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    ptr_coeff_1: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    ptr_coeff_2: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    primal_normal_vert_v1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_v2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
) -> tuple[fa.VertexKField[wpfloat], fa.VertexKField[wpfloat]]:
    z_nabla4_e2 = _calculate_nabla4(
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
    )
    return _mo_intp_rbf_rbf_vec_interpol_vertex(
        astype(z_nabla4_e2, wpfloat), ptr_coeff_1, ptr_coeff_2
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def rbf_nabla4(
    u_vert_out: fa.VertexKField[wpfloat],
    v_vert_out: fa.VertexKField[wpfloat],
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    ptr_coeff_1: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    ptr_coeff_2: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    primal_normal_vert_v1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_v2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _rbf_nabla4(
        u_vert,
        v_vert,
        z_nabla2_e,
        ptr_coeff_1,
        ptr_coeff_2,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        inv_vert_vert_length,
        inv_primal_edge_length,
        out=(u_vert_out, v_vert_out),
        domain={
            dims.VertexDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
