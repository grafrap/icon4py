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

from icon4py.model.atmosphere.dycore.stencils.mo_math_divrot_rot_vertex_ri_dsl import (
    mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def mo_math_divrot_rot_vertex_ri_dsl_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray], vec_e: np.ndarray, geofac_rot: np.ndarray,
    horizontal_start: int = 0,
) -> np.ndarray:
    print(f"horizontal_start in the numpy reference: {horizontal_start}")
    v2e = connectivities[dims.V2EDim]
    geofac_rot = np.expand_dims(geofac_rot, axis=-1)
    rot_vec = np.zeros((geofac_rot.shape[0], vec_e.shape[1]), dtype=vpfloat)
    rot_vec[horizontal_start:] = np.sum(np.where((v2e != -1)[:, :, np.newaxis], vec_e[v2e] * geofac_rot, 0), axis=1)[horizontal_start:]
    return rot_vec


class TestMoMathDivrotRotVertexRiDsl(StencilTest):
    PROGRAM = mo_math_divrot_rot_vertex_ri_dsl
    OUTPUTS = ("rot_vec",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        vec_e: np.ndarray,
        geofac_rot: np.ndarray,
        horizontal_start: int = 0,
        **kwargs: Any,
    ) -> dict:
        rot_vec = mo_math_divrot_rot_vertex_ri_dsl_numpy(connectivities, vec_e, geofac_rot, horizontal_start)
        return dict(rot_vec=rot_vec)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        vec_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        geofac_rot = random_field(grid, dims.VertexDim, dims.V2EDim, dtype=wpfloat)
        rot_vec = zero_field(grid, dims.VertexDim, dims.KDim, dtype=vpfloat)
        vertex_domain = h_grid.domain(dims.VertexDim)
        horizontal_start = grid.start_index(vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

        return dict(
            vec_e=vec_e,
            geofac_rot=geofac_rot,
            rot_vec=rot_vec,
            horizontal_start=horizontal_start,
            horizontal_end=gtx.int32(grid.num_vertices),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
