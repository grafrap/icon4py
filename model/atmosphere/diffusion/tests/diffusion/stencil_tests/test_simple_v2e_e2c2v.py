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

from icon4py.model.atmosphere.diffusion.stencils.simple_v2e_e2c2v import simple_v2e_e2c2v
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StandardStaticVariants, StencilTest


@pytest.mark.uses_concat_where
class TestSimpleV2eE2c2v(StencilTest):
    PROGRAM = simple_v2e_e2c2v
    OUTPUTS = ("output",)
    STATIC_PARAMS = {
        StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "vertical_start",
            "vertical_end",
        ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        u_vert: np.ndarray,
        coeff: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        v2e = connectivities[dims.V2EDim]      # (n_vert, 6)
        e2c2v = connectivities[dims.E2C2VDim]  # (n_edges, 4)

        # inner: at each edge, sum vertices at E2C2V slots 0 and 1
        inner_e = u_vert[e2c2v[:, 0]] + u_vert[e2c2v[:, 1]]  # (n_edges, K)

        # outer: at each vertex, V2E reduce with coeff
        coeff_expanded = np.expand_dims(coeff, axis=-1)  # (n_vert, 6, 1)
        result = np.zeros_like(u_vert)
        result[horizontal_start:horizontal_end] = np.sum(
            np.where(
                np.expand_dims(v2e[horizontal_start:horizontal_end], axis=-1) >= 0,
                inner_e[v2e[horizontal_start:horizontal_end]] * coeff_expanded[horizontal_start:horizontal_end],
                0.0,
            ),
            axis=1,
        )
        return {"output": result}

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        u_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim, dtype=wpfloat)
        coeff = data_alloc.random_field(grid, dims.VertexDim, dims.V2EDim, dtype=wpfloat)
        output = data_alloc.zero_field(grid, dims.VertexDim, dims.KDim, dtype=wpfloat)

        vertex_domain = h_grid.domain(dims.VertexDim)
        horizontal_start = grid.start_index(vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3))
        horizontal_end = grid.end_index(vertex_domain(h_grid.Zone.LOCAL))

        return dict(
            u_vert=u_vert,
            coeff=coeff,
            output=output,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
