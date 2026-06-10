# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.stencils.rbf_nabla4 import rbf_nabla4
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StandardStaticVariants, StencilTest

from .test_calculate_nabla4 import calculate_nabla4_numpy


def _mo_intp_rbf_vec_interpol_vertex_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    p_e_in: np.ndarray,
    ptr_coeff_1: np.ndarray,
    ptr_coeff_2: np.ndarray,
    horizontal_start: int,
    horizontal_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    v2e = connectivities[dims.V2EDim]
    ptr_coeff_1 = np.expand_dims(ptr_coeff_1, axis=-1)
    p_u_out = np.sum(
        np.where(np.expand_dims(v2e, axis=-1) >= 0, p_e_in[v2e] * ptr_coeff_1, 0.0), axis=1
    )
    ptr_coeff_2 = np.expand_dims(ptr_coeff_2, axis=-1)
    p_v_out = np.sum(
        np.where(np.expand_dims(v2e, axis=-1) >= 0, p_e_in[v2e] * ptr_coeff_2, 0.0), axis=1
    )
    p_u_final_out = np.zeros_like(p_u_out)
    p_v_final_out = np.zeros_like(p_v_out)
    p_u_final_out[horizontal_start:horizontal_end, :] = p_u_out[horizontal_start:horizontal_end, :]
    p_v_final_out[horizontal_start:horizontal_end, :] = p_v_out[horizontal_start:horizontal_end, :]
    return p_u_final_out, p_v_final_out


@pytest.mark.uses_concat_where
@pytest.mark.continuous_benchmarking
class TestRBFNABLA4(StencilTest):
    PROGRAM = rbf_nabla4
    OUTPUTS = ("u_vert_out", "v_vert_out")
    STATIC_PARAMS = {
        # StandardStaticVariants.NONE: (),
        StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "vertical_start",
            "vertical_end",
        ),
        # StandardStaticVariants.COMPILE_TIME_VERTICAL: (
        #     "vertical_start",
        #     "vertical_end",
        # ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        u_vert: np.ndarray,
        v_vert: np.ndarray,
        primal_normal_vert_v1: np.ndarray,
        primal_normal_vert_v2: np.ndarray,
        z_nabla2_e: np.ndarray,
        inv_vert_vert_length: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        ptr_coeff_1: np.ndarray,
        ptr_coeff_2: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        z_nabla4_e2 = calculate_nabla4_numpy(
            connectivities,
            u_vert,
            v_vert,
            primal_normal_vert_v1,
            primal_normal_vert_v2,
            z_nabla2_e,
            inv_vert_vert_length,
            inv_primal_edge_length,
        )
        u_vert_out, v_vert_out = _mo_intp_rbf_vec_interpol_vertex_numpy(
            connectivities,
            z_nabla4_e2,
            ptr_coeff_1,
            ptr_coeff_2,
            horizontal_start,
            horizontal_end,
        )
        return dict(u_vert_out=u_vert_out, v_vert_out=v_vert_out)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        u_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim)
        v_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim)

        primal_normal_vert_v1 = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2VDim)
        primal_normal_vert_v2 = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2VDim)

        inv_vert_vert_length = data_alloc.random_field(grid, dims.EdgeDim)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim)

        z_nabla2_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)

        ptr_coeff_1 = data_alloc.random_field(grid, dims.VertexDim, dims.V2EDim, dtype=wpfloat)
        ptr_coeff_2 = data_alloc.random_field(grid, dims.VertexDim, dims.V2EDim, dtype=wpfloat)

        u_vert_out = data_alloc.zero_field(grid, dims.VertexDim, dims.KDim, dtype=wpfloat)
        v_vert_out = data_alloc.zero_field(grid, dims.VertexDim, dims.KDim, dtype=wpfloat)

        vertex_domain = h_grid.domain(dims.VertexDim)
        horizontal_start = grid.start_index(vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3))
        horizontal_end = grid.end_index(vertex_domain(h_grid.Zone.LOCAL))

        return dict(
            u_vert=u_vert,
            v_vert=v_vert,
            primal_normal_vert_v1=primal_normal_vert_v1,
            primal_normal_vert_v2=primal_normal_vert_v2,
            z_nabla2_e=z_nabla2_e,
            inv_vert_vert_length=inv_vert_vert_length,
            inv_primal_edge_length=inv_primal_edge_length,
            ptr_coeff_1=ptr_coeff_1,
            ptr_coeff_2=ptr_coeff_2,
            u_vert_out=u_vert_out,
            v_vert_out=v_vert_out,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
